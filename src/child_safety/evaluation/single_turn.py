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
    load_responses,
    save_json,
)


def _infer_country(response_path: str) -> Optional[str]:
    """Infer country from filename: {lang}_{age}_{country}.json. Returns None if country is 'none'."""
    stem = Path(response_path).stem          # e.g. "english_false_pakistan"
    parts = stem.split("_")
    country = parts[-1]                      # last segment
    return None if country == "none" else country


def _load_template(path: Path) -> str:
    with open(path) as f:
        return f.read().strip()


def _build_messages(
    system_prompt: str,
    category_rules: str,
    cultural_values: Optional[str],
    prompt: str,
    response: str,
) -> List[Dict[str, str]]:
    parts = [f"## CATEGORY-SPECIFIC RULES:\n{category_rules}"]
    if cultural_values:
        parts.append(f"## CULTURAL VALUES:\n{cultural_values}")
    parts.append(f"## USER PROMPT:\n{prompt}")
    parts.append(f"## LLM RESPONSE:\n{response}")
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def _parse_json(text: str) -> Optional[Any]:
    try:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if match:
            return json.loads(match.group(1))
        return json.loads(text)
    except Exception:
        return None


def _load_existing(output_path: str) -> Dict[int, str]:
    """Returns {flat_index: raw_evaluation} for already-completed items."""
    if not os.path.exists(output_path):
        return {}
    with open(output_path) as f:
        saved = json.load(f)
    return {i: item["raw_evaluation"] for i, item in enumerate(saved) if item.get("raw_evaluation", "")}


def _build_output(items: List[Dict], results: Dict[int, str]) -> List[Dict]:
    output = []
    for i, item in enumerate(items):
        raw = results.get(i, "")
        output.append({
            "prompt": item["prompt"],
            "response": item["response"],
            "raw_evaluation": raw,
            "json_evaluation": _parse_json(raw) if raw else None,
        })
    return output


def parse_args():
    parser = argparse.ArgumentParser()

    io = parser.add_argument_group("io")
    io.add_argument("--response_path", type=str, required=True)
    io.add_argument("--output_path", type=str, required=True)
    io.add_argument("--system_prompts_dir", type=str, required=True)

    model = parser.add_argument_group("model")
    model.add_argument("--model_name", type=str, required=True)
    model.add_argument("--model_path", type=str, required=True)
    model.add_argument(
        "--provider",
        type=str,
        default="vllm",
        choices=["vllm", "openai", "deepseek", "anthropic", "google"],
    )
    model.add_argument("--tensor_parallel_size", type=int, default=1)
    model.add_argument("--max_model_len", type=int, default=None)
    model.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    model.add_argument("--language_model_only", action="store_true", default=False)

    generation = parser.add_argument_group("generation")
    generation.add_argument("--temperature", type=float, default=0.0)
    generation.add_argument("--top_p", type=float, default=1.0)
    generation.add_argument("--max_new_tokens", type=int, default=1024)
    generation.add_argument("--max_concurrent", type=int, default=10,
                            help="Max concurrent requests for API providers (ignored for vllm)")

    return parser.parse_args()


def main():
    args = parse_args()

    prompts_dir = Path(args.system_prompts_dir)
    country = _infer_country(args.response_path)
    print(f"Inferred country: {country or 'none'}", flush=True)

    # System prompt
    if country:
        system_prompt = _load_template(prompts_dir / "evaluation" / "judge_cultural_single.jinja")
    else:
        system_prompt = _load_template(prompts_dir / "evaluation" / "judge_single.jinja")

    # Cultural values
    cultural_values = None
    if country:
        cultural_path = prompts_dir / "rules" / "countries" / f"{country}.jinja"
        if cultural_path.exists():
            cultural_values = _load_template(cultural_path)
        else:
            print(f"Warning: no cultural values file found at {cultural_path}", flush=True)

    # Load responses
    items = load_responses(args.response_path)
    print(f"Loaded {len(items)} prompt/response pairs.", flush=True)

    # Build judge messages
    all_messages = []
    for item in items:
        rules_path = prompts_dir / "rules" / "categories" / f"{item['category']}.jinja"
        if rules_path.exists():
            category_rules = _load_template(rules_path)
        else:
            print(f"Warning: no category rules found for '{item['category']}'", flush=True)
            category_rules = ""
        all_messages.append(_build_messages(
            system_prompt, category_rules, cultural_values,
            item["prompt"], item["response"],
        ))

    # Generate evaluations
    results = _load_existing(args.output_path)
    if results:
        print(f"Resuming: {len(results)}/{len(items)} already done.", flush=True)

    pending_indices = [i for i in range(len(all_messages)) if i not in results]
    pending_messages = [all_messages[i] for i in pending_indices]

    if not pending_messages:
        print("All evaluations already present, nothing to do.", flush=True)
        return

    print(f"Processing {len(pending_messages)} requests...", flush=True)

    if args.provider == "vllm":
        tokenizer, llm = load_model(
            model_path=args.model_path,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            language_model_only=args.language_model_only,
        )
        raw_evals = generate_batch(
            tokenizer=tokenizer,
            llm=llm,
            messages_batch=pending_messages,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
        )
        for i, r in zip(pending_indices, raw_evals):
            results[i] = r
        save_json(args.output_path, _build_output(items, results))
    else:
        def on_result(orig_i, text):
            results[orig_i] = text
            save_json(args.output_path, _build_output(items, results))

        asyncio.run(generate_api_async(
            provider=args.provider,
            model=args.model_path,
            messages_batch=pending_messages,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            indices=pending_indices,
            on_result=on_result,
            max_concurrent=args.max_concurrent,
        ))


if __name__ == "__main__":
    main()
