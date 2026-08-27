#!/usr/bin/env python3
"""
Generates responses for fine-tuned LlamaPlushie models on test_single.jsonl,
producing the same file structure as standard single-turn generation so the
existing evaluation and analysis pipelines work without modification.

Output (under responses/single_turn/{model_name}/):
  without_cues/english_false_none.json   — age off
  without_cues/english_true_none.json    — age on
  with_cues/english_false_none.json      — age off, no country  (cues analysis)
  with_cues/english_true_none.json       — age on,  no country
  with_cues/english_true_china.json
  with_cues/english_true_india.json
  with_cues/english_true_nigeria.json
  with_cues/english_true_pakistan.json
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from child_safety.utils import build_system_prompt, generate_batch, load_model, save_json

ROOT = Path(__file__).parent.parent.parent.parent

EXPERIMENTS: List[Tuple[str, bool, Optional[str], str]] = [
    ("without_cues", False, None,       "english_false_none.json"),
    ("without_cues", True,  None,       "english_true_none.json"),
    ("with_cues",    False, None,       "english_false_none.json"),
    ("with_cues",    True,  None,       "english_true_none.json"),
    ("with_cues",    True,  "china",    "english_true_china.json"),
    ("with_cues",    True,  "india",    "english_true_india.json"),
    ("with_cues",    True,  "nigeria",  "english_true_nigeria.json"),
    ("with_cues",    True,  "pakistan", "english_true_pakistan.json"),
]


def load_test_prompts(test_path: str) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Reads test_single.jsonl and returns two dicts keyed by category:
      without_cues_data — unique without_cues prompts (each appears once per category)
      with_cues_data    — unique with_cues prompts (no-country variant, deduped)
    """
    without_cues_data: Dict[str, List[str]] = {}
    with_cues_data: Dict[str, List[str]] = {}
    seen_woc: set = set()
    seen_wc: set = set()

    with open(test_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            prompt = rec["prompt"]
            category = rec["category"]
            condition = rec["condition"]

            if condition == "without_cues" and prompt not in seen_woc:
                seen_woc.add(prompt)
                without_cues_data.setdefault(category, []).append(prompt)
            elif condition == "with_cues" and rec.get("country") is None and prompt not in seen_wc:
                seen_wc.add(prompt)
                with_cues_data.setdefault(category, []).append(prompt)

    return without_cues_data, with_cues_data


def _flat(dataset: Dict[str, List[str]]) -> Tuple[List[str], List[str]]:
    categories = [cat for cat, prompts in dataset.items() for _ in prompts]
    prompts    = [p   for prompts in dataset.values() for p in prompts]
    return categories, prompts


def _load_existing(output_path: Path, categories: List[str], prompts: List[str]) -> Dict[int, str]:
    if not output_path.exists():
        return {}
    with open(output_path) as f:
        saved = json.load(f)
    existing: Dict[int, str] = {}
    for i, (cat, prompt) in enumerate(zip(categories, prompts)):
        for item in saved.get(cat, []):
            if item["prompt"] == prompt and item.get("response", ""):
                existing[i] = item["response"]
                break
    return existing


def _build_output(categories: List[str], prompts: List[str], results: Dict[int, str]) -> dict:
    output: dict = {}
    for i, (cat, prompt) in enumerate(zip(categories, prompts)):
        output.setdefault(cat, []).append({"prompt": prompt, "response": results.get(i, "")})
    return output


def run_experiment(
    cues_type: str,
    age: bool,
    country: Optional[str],
    filename: str,
    without_cues_data: Dict[str, List[str]],
    with_cues_data: Dict[str, List[str]],
    output_dir: Path,
    tokenizer,
    llm,
    args: argparse.Namespace,
) -> None:
    output_path = output_dir / cues_type / filename
    dataset = without_cues_data if cues_type == "without_cues" else with_cues_data
    categories, all_prompts = _flat(dataset)

    system_prompt = build_system_prompt(age=age, country=country)

    all_messages = []
    for prompt in all_prompts:
        msgs: List[dict] = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": prompt})
        all_messages.append(msgs)

    results = _load_existing(output_path, categories, all_prompts)
    pending_indices = [i for i in range(len(all_messages)) if i not in results]
    pending_messages = [all_messages[i] for i in pending_indices]

    if not pending_messages:
        print(f"  Already complete ({len(all_prompts)} responses) — {output_path.name}")
        return

    if results:
        print(f"  Resuming: {len(results)}/{len(all_prompts)} done — {output_path.name}")

    responses = generate_batch(
        tokenizer=tokenizer,
        llm=llm,
        messages_batch=pending_messages,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
    )
    for i, resp in zip(pending_indices, responses):
        results[i] = resp

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(str(output_path), _build_output(categories, all_prompts, results))
    print(f"  Saved {len(all_prompts)} responses → {output_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name",             required=True)
    p.add_argument("--model_path",             required=True)
    p.add_argument("--test_path",              default=str(ROOT / "dataset/gold/test_single.jsonl"))
    p.add_argument("--output_dir",             default=str(ROOT / "responses/single_turn"))
    p.add_argument("--max_model_len",          type=int,   default=4096)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    p.add_argument("--temperature",            type=float, default=0.0)
    p.add_argument("--top_p",                  type=float, default=1.0)
    p.add_argument("--max_new_tokens",         type=int,   default=2048)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading test prompts from {args.test_path}")
    without_cues_data, with_cues_data = load_test_prompts(args.test_path)
    n_woc = sum(len(v) for v in without_cues_data.values())
    n_wc  = sum(len(v) for v in with_cues_data.values())
    print(f"  without_cues: {n_woc} prompts | with_cues: {n_wc} prompts")

    output_dir = Path(args.output_dir) / args.model_name
    print(f"Output → {output_dir}")

    print(f"Loading model: {args.model_path}")
    tokenizer, llm = load_model(
        model_path=args.model_path,
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        language_model_only=True,
    )

    for cues_type, age, country, filename in EXPERIMENTS:
        label = f"{cues_type} | age={age} | country={country or 'none'}"
        print(f"\n[{label}]")
        run_experiment(
            cues_type=cues_type,
            age=age,
            country=country,
            filename=filename,
            without_cues_data=without_cues_data,
            with_cues_data=with_cues_data,
            output_dir=output_dir,
            tokenizer=tokenizer,
            llm=llm,
            args=args,
        )

    print("\nAll experiments done.")


if __name__ == "__main__":
    main()
