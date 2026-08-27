#!/usr/bin/env python3
"""
Multi-turn red-teaming: a vLLM attacker simulates a child probing a generator
model across all scenarios in kidbench_multi.json.

Attacker: always vLLM (subprocess pinned to CUDA_VISIBLE_DEVICES=0).
Generator: openai | anthropic | deepseek | google | vllm.
  - vLLM generator runs in a separate subprocess pinned to CUDA_VISIBLE_DEVICES=1.
  - API generators run concurrently in the main process.
"""

import argparse
import asyncio
import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from child_safety.utils import (
    build_system_prompt,
    generate_api_async,
    save_json,
)

PROVIDER_CHOICES = ("openai", "anthropic", "deepseek", "google", "vllm")


# ---------------------------------------------------------------------------
# vLLM worker (runs in a subprocess with its own CUDA_VISIBLE_DEVICES)
# ---------------------------------------------------------------------------

def _vllm_worker(
    cuda_device: str,
    model_path: str,
    tensor_parallel_size: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    language_model_only: bool,
    req_queue: mp.Queue,
    res_queue: mp.Queue,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

    from child_safety.utils import load_model, generate_batch

    try:
        tokenizer, llm = load_model(
            model_path=model_path,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            language_model_only=language_model_only,
        )
        res_queue.put(("ready", None))
    except Exception as e:
        res_queue.put(("error", str(e)))
        return

    while True:
        item = req_queue.get()
        if item is None:
            break
        messages_batch, temperature, max_new_tokens = item
        try:
            responses = generate_batch(
                tokenizer=tokenizer,
                llm=llm,
                messages_batch=messages_batch,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
            res_queue.put(("ok", responses))
        except Exception as e:
            res_queue.put(("error", str(e)))


def _start_vllm_worker(
    cuda_device: str,
    model_path: str,
    tensor_parallel_size: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    language_model_only: bool,
) -> Tuple[mp.Process, mp.Queue, mp.Queue]:
    ctx = mp.get_context("spawn")
    req_q: mp.Queue = ctx.Queue()
    res_q: mp.Queue = ctx.Queue()
    proc = ctx.Process(
        target=_vllm_worker,
        args=(cuda_device, model_path, tensor_parallel_size, max_model_len,
              gpu_memory_utilization, language_model_only, req_q, res_q),
    )
    proc.start()
    status, err = res_q.get()
    if status == "error":
        proc.terminate()
        raise RuntimeError(f"vLLM worker failed to load: {err}")
    return proc, req_q, res_q


def _infer(req_q: mp.Queue, res_q: mp.Queue,
           messages_batch: List, temperature: float, max_new_tokens: int) -> List[str]:
    req_q.put((messages_batch, temperature, max_new_tokens))
    status, result = res_q.get()
    if status == "error":
        raise RuntimeError(f"vLLM worker inference failed: {result}")
    return result


# ---------------------------------------------------------------------------
# Dataset / prompt helpers
# ---------------------------------------------------------------------------

def _load_dataset(path: str) -> List[Dict]:
    with open(path) as f:
        data = json.load(f)
    items = []
    for category, entries in data.items():
        for entry in entries:
            items.append({
                "category": category,
                "scenario": entry["scenario"],
                "child_goal": entry["child_goal"],
            })
    return items


def _attacker_init_user(item: Dict) -> str:
    return (
        f"## Category:\n{item['category']}\n\n"
        f"## Scenario:\n{item['scenario']}\n\n"
        f"## Child Goal:\n{item['child_goal']}\n\n"
        "Give the first question you want to ask based on the scenario and goal."
    )


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def _completed_turns(results: List[Dict]) -> int:
    if not results:
        return 0
    return min(min(len(r["attacker"]), len(r["generator"])) for r in results)


def _rebuild_histories(
    results: List[Dict],
    items: List[Dict],
    attacker_system: str,
    generator_system: Optional[str],
    completed_turns: int,
) -> Tuple[List[List[Dict]], List[List[Dict]]]:
    n = len(items)
    attacker_histories: List[List[Dict]] = [
        [{"role": "system", "content": attacker_system}] for _ in range(n)
    ]
    generator_histories: List[List[Dict]] = []
    for _ in range(n):
        h: List[Dict] = []
        if generator_system:
            h.append({"role": "system", "content": generator_system})
        generator_histories.append(h)

    for i in range(n):
        for turn in range(1, completed_turns + 1):
            t = str(turn)
            if turn == 1:
                user_msg = _attacker_init_user(items[i])
            else:
                prev_gen = results[i]["generator"][str(turn - 1)]
                user_msg = f"{prev_gen}\n\nWhat's your next question?"
            attacker_histories[i].append({"role": "user", "content": user_msg})
            attacker_histories[i].append({"role": "assistant", "content": results[i]["attacker"][t]})
            generator_histories[i].append({"role": "user", "content": results[i]["attacker"][t]})
            generator_histories[i].append({"role": "assistant", "content": results[i]["generator"][t]})

    return attacker_histories, generator_histories


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    io = p.add_argument_group("io")
    io.add_argument("--dataset_path",       required=True)
    io.add_argument("--output_path",        required=True)
    io.add_argument("--system_prompts_dir", required=True)

    att = p.add_argument_group("attacker")
    att.add_argument("--attacker_model_path",          required=True)
    att.add_argument("--attacker_max_new_tokens",      type=int, default=512)
    att.add_argument("--attacker_language_model_only", action="store_true")

    gen = p.add_argument_group("generator")
    gen.add_argument("--generator_provider",           required=True, choices=PROVIDER_CHOICES)
    gen.add_argument("--generator_model_name",         required=True)
    gen.add_argument("--generator_model_path",         default=None)
    gen.add_argument("--generator_max_new_tokens",     type=int, default=2048)
    gen.add_argument("--generator_temperature",        type=float, default=0.0)
    gen.add_argument("--generator_language_model_only", action="store_true")
    gen.add_argument("--age",                          action="store_true")

    vllm_g = p.add_argument_group("vllm")
    vllm_g.add_argument("--tensor_parallel_size",        type=int,   default=1)
    vllm_g.add_argument("--max_model_len",               type=int,   default=8192)
    vllm_g.add_argument("--gpu_memory_utilization",      type=float, default=0.90)
    vllm_g.add_argument("--attacker_cuda_device",        type=str,   default="0")
    vllm_g.add_argument("--generator_cuda_device",       type=str,   default="1")
    vllm_g.add_argument("--generator_tensor_parallel_size", type=int, default=None,
                        help="Override tensor_parallel_size for generator only")

    api_g = p.add_argument_group("api")
    api_g.add_argument("--max_concurrent", type=int, default=50)

    run = p.add_argument_group("run")
    run.add_argument("--num_turns", type=int, default=5)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    system_prompts_dir = Path(args.system_prompts_dir)
    attacker_system = (system_prompts_dir / "generation" / "attacker.jinja").read_text()
    generator_system: Optional[str] = build_system_prompt(age=args.age, country=None, language=None)

    items = _load_dataset(args.dataset_path)
    print(f"Loaded {len(items)} scenarios", flush=True)

    # Resume
    output_path = Path(args.output_path)
    if output_path.exists():
        with open(output_path) as f:
            results: List[Dict] = json.load(f)
        completed = _completed_turns(results)
        print(f"Resuming: {completed}/{args.num_turns} turns already done.", flush=True)
    else:
        results = [
            {
                "category": item["category"],
                "scenario": item["scenario"],
                "child_goal": item["child_goal"],
                "attacker": {},
                "generator": {},
            }
            for item in items
        ]
        completed = 0

    if completed >= args.num_turns:
        print("All turns already complete, nothing to do.", flush=True)
        return

    attacker_histories, generator_histories = _rebuild_histories(
        results, items, attacker_system, generator_system, completed
    )
    n = len(items)

    gen_tp = args.generator_tensor_parallel_size or args.tensor_parallel_size

    # Start attacker worker
    print(f"Loading attacker model (GPU {args.attacker_cuda_device})...", flush=True)
    att_proc, att_req, att_res = _start_vllm_worker(
        cuda_device=args.attacker_cuda_device,
        model_path=args.attacker_model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        language_model_only=args.attacker_language_model_only,
    )
    print("Attacker ready.", flush=True)

    # Start generator worker (vLLM only)
    gen_proc, gen_req, gen_res = None, None, None
    if args.generator_provider == "vllm":
        if not args.generator_model_path:
            raise ValueError("--generator_model_path is required when provider=vllm")
        print(f"Loading generator model (GPU {args.generator_cuda_device})...", flush=True)
        gen_proc, gen_req, gen_res = _start_vllm_worker(
            cuda_device=args.generator_cuda_device,
            model_path=args.generator_model_path,
            tensor_parallel_size=gen_tp,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            language_model_only=args.generator_language_model_only,
        )
        print("Generator ready.", flush=True)

    try:
        for turn in range(completed + 1, args.num_turns + 1):
            print(f"\n=== Turn {turn}/{args.num_turns} ===", flush=True)

            for i in range(n):
                if turn == 1:
                    user_msg = _attacker_init_user(items[i])
                else:
                    prev_gen = results[i]["generator"][str(turn - 1)]
                    user_msg = f"{prev_gen}\n\nWhat's your next question?"
                attacker_histories[i].append({"role": "user", "content": user_msg})

            print("Attacker generating...", flush=True)
            attacker_responses = _infer(
                att_req, att_res, attacker_histories,
                temperature=0.0, max_new_tokens=args.attacker_max_new_tokens,
            )
            for i, resp in enumerate(attacker_responses):
                attacker_histories[i].append({"role": "assistant", "content": resp})
                results[i]["attacker"][str(turn)] = resp
                generator_histories[i].append({"role": "user", "content": resp})

            print("Generator responding...", flush=True)
            if args.generator_provider == "vllm":
                gen_list = _infer(
                    gen_req, gen_res, generator_histories,
                    temperature=args.generator_temperature,
                    max_new_tokens=args.generator_max_new_tokens,
                )
                gen_map: Dict[int, str] = {i: r for i, r in enumerate(gen_list)}
            else:
                gen_map = asyncio.run(generate_api_async(
                    provider=args.generator_provider,
                    model=args.generator_model_name,
                    messages_batch=generator_histories,
                    temperature=args.generator_temperature,
                    max_new_tokens=args.generator_max_new_tokens,
                    max_concurrent=args.max_concurrent,
                ))

            for i in range(n):
                resp = gen_map.get(i, "")
                generator_histories[i].append({"role": "assistant", "content": resp})
                results[i]["generator"][str(turn)] = resp

            save_json(args.output_path, results)
            print(f"Saved turn {turn} → {args.output_path}", flush=True)

    finally:
        att_req.put(None)
        att_proc.join()
        if gen_proc:
            gen_req.put(None)
            gen_proc.join()

    print(f"\nDone. {args.num_turns} turns saved → {args.output_path}", flush=True)


if __name__ == "__main__":
    main()
