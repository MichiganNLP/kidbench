#!/usr/bin/env python3
"""
Iteratively generate and refine gold responses.

Flow per prompt:
  turn 1: generator produces response → judge evaluates
  turn N: if any score < 5 and turns remain → generator refines → judge re-evaluates
  (up to --num_turns total generation turns)

Saves incrementally after each generation and evaluation.
Resume-safe: re-running with an existing output file continues from where it left off.
"""

import argparse
import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from child_safety.utils import (
    generate_api_async,
    generate_batch,
    load_model,
    save_json,
)

EVALUATOR_PROVIDER = "deepseek"
EVALUATOR_MODEL = "deepseek-v4-pro"
JUDGE_MAX_NEW_TOKENS = 1024
BASE_EVAL_METRICS = ["safety", "developmental_fit", "emotional_support", "moral_guidance", "boundary_setting"]
FEEDBACK_MSG = (
    "Update your response based on the evaluation by an expert. "
    "Return only the updated response to the user prompt without any commentary."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _infer_country(path: str) -> Optional[str]:
    c = Path(path).stem.split("_")[-1]
    return None if c == "none" else c


def _load_template(path: Path) -> str:
    return path.read_text().strip()


def _parse_json(text: str) -> Optional[Any]:
    if not text:
        return None
    try:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if m:
            return json.loads(m.group(1))
        return json.loads(text)
    except Exception:
        return None


def _all_perfect(ev: Optional[dict], country: Optional[str]) -> bool:
    if not ev:
        return False
    metrics = BASE_EVAL_METRICS + (["cultural_alignment"] if country else [])
    return all(ev.get(m) == 5 for m in metrics)


def _gen_user_message(rules: str, cultural_values: Optional[str], prompt: str) -> str:
    parts = [f"## CATEGORY-SPECIFIC RULES\n{rules}"]
    if cultural_values:
        parts.append(f"## CULTURAL VALUES\n{cultural_values}")
    parts.append(f"## USER PROMPT\n{prompt}")
    return "\n\n".join(parts)


def _judge_user_message(rules: str, cultural_values: Optional[str], prompt: str, response: str) -> str:
    parts = [f"## CATEGORY-SPECIFIC RULES\n{rules}"]
    if cultural_values:
        parts.append(f"## CULTURAL VALUES\n{cultural_values}")
    parts.append(f"## USER PROMPT\n{prompt}")
    parts.append(f"## LLM RESPONSE\n{response}")
    return "\n\n".join(parts)


def _load_prompts(path: str) -> List[Dict]:
    with open(path) as f:
        data = json.load(f)
    items = []
    for category, entries in data.items():
        for entry in entries:
            items.append({"prompt": entry["prompt"], "category": category})
    return items


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

def _build_state(
    prompts: List[Dict],
    gen_system_prompt: str,
    judge_system_prompt: str,
    category_rules_map: Dict[str, str],
    cultural_values: Optional[str],
) -> List[Dict]:
    state = []
    for item in prompts:
        rules = category_rules_map.get(item["category"], "")
        state.append({
            "prompt": item["prompt"],
            "category": item["category"],
            "response": {},
            "evaluation": {},
            "gen_history": [
                {"role": "system", "content": gen_system_prompt},
                {"role": "user", "content": _gen_user_message(rules, cultural_values, item["prompt"])},
            ],
            "judge_history": [{"role": "system", "content": judge_system_prompt}],
            "done": False,
            "_rules": rules,
        })
    return state


def _restore_state(
    state: List[Dict],
    saved: List[Dict],
    cultural_values: Optional[str],
    num_turns: int,
    country: Optional[str],
) -> None:
    """Restore saved responses/evaluations and rebuild conversation histories."""
    saved_map = {s["prompt"]: s for s in saved}
    for item in state:
        sv = saved_map.get(item["prompt"])
        if not sv:
            continue

        responses   = {int(k): v for k, v in sv.get("response", {}).items()}
        evaluations = {int(k): v for k, v in sv.get("evaluation", {}).items()}
        item["response"]   = responses
        item["evaluation"] = evaluations

        for turn in range(1, num_turns + 1):
            if turn not in responses:
                break

            resp = responses[turn]
            item["gen_history"].append({"role": "assistant", "content": resp})

            # Extend judge history with this response
            if len(item["judge_history"]) == 1:
                jmsg = _judge_user_message(item["_rules"], cultural_values, item["prompt"], resp)
                item["judge_history"].append({"role": "user", "content": jmsg})
            else:
                item["judge_history"].append({
                    "role": "user",
                    "content": (
                        f"## UPDATED RESPONSE\n{resp}\n\n"
                        "Please give evaluation of the updated response. "
                        "Return exactly one valid JSON object and nothing else."
                    ),
                })

            if turn not in evaluations:
                break

            ev = evaluations[turn]
            item["judge_history"].append({"role": "assistant", "content": json.dumps(ev) if ev else ""})

            if _all_perfect(ev, country) or turn == num_turns:
                item["done"] = True
                break

            # Prepare generator for next turn
            item["gen_history"].append({
                "role": "user",
                "content": f"{json.dumps(ev, indent=2) if ev else 'No evaluation.'}\n\n{FEEDBACK_MSG}",
            })


def _to_output(state: List[Dict]) -> List[Dict]:
    return [
        {
            "prompt": s["prompt"],
            "response":   {str(k): v for k, v in s["response"].items()},
            "evaluation": {str(k): v for k, v in s["evaluation"].items()},
        }
        for s in state
    ]


def _save(path: str, state: List[Dict]) -> None:
    save_json(path, _to_output(state))


# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()

    io = p.add_argument_group("io")
    io.add_argument("--prompts_path",      required=True, help="Response file to extract prompts from")
    io.add_argument("--output_path",       required=True)
    io.add_argument("--system_prompts_dir", required=True)

    gen = p.add_argument_group("generator")
    gen.add_argument("--generator_model_name", required=True)
    gen.add_argument("--generator_model_path", required=True)
    gen.add_argument("--generator_provider",   required=True,
                     choices=["vllm", "openai", "anthropic", "google", "deepseek"])
    gen.add_argument("--tensor_parallel_size",    type=int,   default=1)
    gen.add_argument("--max_model_len",           type=int,   default=None)
    gen.add_argument("--gpu_memory_utilization",  type=float, default=0.9)
    gen.add_argument("--language_model_only",     action="store_true")

    params = p.add_argument_group("generation params")
    params.add_argument("--temperature",   type=float, default=0.0)
    params.add_argument("--top_p",         type=float, default=1.0)
    params.add_argument("--max_new_tokens", type=int,  default=2048)
    params.add_argument("--max_concurrent", type=int,  default=10)
    params.add_argument("--num_turns",      type=int,  default=2,
                        help="Total generation turns (turn 1 = initial, turn 2 = first refinement, etc.)")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    prompts_dir = Path(args.system_prompts_dir)
    country = _infer_country(args.prompts_path)
    print(f"Country: {country or 'none'}", flush=True)

    # System prompts
    gen_sp = _load_template(
        prompts_dir / "generation" / ("gold_cultural_single.jinja" if country else "gold_single.jinja")
    )
    judge_sp = _load_template(
        prompts_dir / "evaluation" / ("judge_cultural_gold_single.jinja" if country else "judge_gold_single.jinja")
    )

    # Cultural values
    cultural_values = None
    if country:
        cv_path = prompts_dir / "rules" / "countries" / f"{country}.jinja"
        if cv_path.exists():
            cultural_values = _load_template(cv_path)
        else:
            print(f"Warning: no cultural values at {cv_path}", flush=True)

    # Category rules
    category_rules_map = {
        f.stem: _load_template(f)
        for f in (prompts_dir / "rules" / "categories").glob("*.jinja")
    }

    # Load prompts
    prompts = _load_prompts(args.prompts_path)
    print(f"Loaded {len(prompts)} prompts", flush=True)

    # Build state
    state = _build_state(prompts, gen_sp, judge_sp, category_rules_map, cultural_values)

    # Resume from existing output
    if os.path.exists(args.output_path):
        with open(args.output_path) as f:
            saved = json.load(f)
        _restore_state(state, saved, cultural_values, args.num_turns, country)
        done = sum(1 for s in state if s["done"])
        print(f"Resumed: {done}/{len(state)} already complete", flush=True)

    # Load vLLM model if needed
    tokenizer, llm = None, None
    if args.generator_provider == "vllm":
        tokenizer, llm = load_model(
            model_path=args.generator_model_path,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            language_model_only=args.language_model_only,
        )

    # ── Turn loop ────────────────────────────────────────────────────────────
    for turn in range(1, args.num_turns + 1):
        print(f"\n=== Turn {turn}/{args.num_turns} ===", flush=True)

        # ── Generation ───────────────────────────────────────────────────────
        pending_gen = [i for i, s in enumerate(state)
                       if not s["done"] and turn not in s["response"]]

        if pending_gen:
            print(f"Generating: {len(pending_gen)} prompts...", flush=True)
            messages_batch = [state[i]["gen_history"] for i in pending_gen]

            if args.generator_provider == "vllm":
                responses = generate_batch(
                    tokenizer=tokenizer,
                    llm=llm,
                    messages_batch=messages_batch,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                )
                for idx, resp in zip(pending_gen, responses):
                    state[idx]["response"][turn] = resp
                    state[idx]["gen_history"].append({"role": "assistant", "content": resp})
                _save(args.output_path, state)

            else:
                _turn = turn  # explicit capture for closure

                def on_gen(state_i: int, text: str, t: int = _turn) -> None:
                    state[state_i]["response"][t] = text
                    state[state_i]["gen_history"].append({"role": "assistant", "content": text})
                    _save(args.output_path, state)

                asyncio.run(generate_api_async(
                    provider=args.generator_provider,
                    model=args.generator_model_path,
                    messages_batch=messages_batch,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    indices=pending_gen,
                    on_result=on_gen,
                    max_concurrent=args.max_concurrent,
                ))

        # ── Evaluation ───────────────────────────────────────────────────────
        pending_eval = [
            i for i, s in enumerate(state)
            if not s["done"] and turn in s["response"] and turn not in s["evaluation"]
        ]

        if pending_eval:
            print(f"Evaluating: {len(pending_eval)} responses...", flush=True)

            # Extend each judge history with the new response
            for i in pending_eval:
                s = state[i]
                if len(s["judge_history"]) == 1:
                    # First turn: include full context
                    jmsg = _judge_user_message(
                        s["_rules"], cultural_values, s["prompt"], s["response"][turn]
                    )
                    s["judge_history"].append({"role": "user", "content": jmsg})
                else:
                    # Subsequent turns: just pass updated response
                    s["judge_history"].append({
                        "role": "user",
                        "content": (
                            f"## UPDATED RESPONSE\n{s['response'][turn]}\n\n"
                            "Please give evaluation of the updated response. "
                            "Return exactly one valid JSON object and nothing else."
                        ),
                    })

            judge_messages = [state[i]["judge_history"] for i in pending_eval]
            _turn = turn

            def on_eval(state_i: int, text: str, t: int = _turn) -> None:
                ev = _parse_json(text)
                state[state_i]["evaluation"][t] = ev
                state[state_i]["judge_history"].append({"role": "assistant", "content": text})
                _save(args.output_path, state)

            asyncio.run(generate_api_async(
                provider=EVALUATOR_PROVIDER,
                model=EVALUATOR_MODEL,
                messages_batch=judge_messages,
                temperature=0.0,
                max_new_tokens=JUDGE_MAX_NEW_TOKENS,
                indices=pending_eval,
                on_result=on_eval,
                max_concurrent=args.max_concurrent,
            ))

        # ── Prepare next turn (or mark done) ─────────────────────────────────
        for s in state:
            if s["done"] or turn not in s["evaluation"]:
                continue
            ev = s["evaluation"][turn]
            if turn == args.num_turns or _all_perfect(ev, country):
                s["done"] = True
            else:
                # Add evaluation feedback for generator
                s["gen_history"].append({
                    "role": "user",
                    "content": f"{json.dumps(ev, indent=2) if ev else 'No evaluation.'}\n\n{FEEDBACK_MSG}",
                })

    done = sum(1 for s in state if s["done"])
    print(f"\nDone: {done}/{len(state)}", flush=True)
    _save(args.output_path, state)


if __name__ == "__main__":
    main()
