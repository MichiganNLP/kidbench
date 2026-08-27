import argparse
import json
import os
from typing import Dict, List, Tuple

from child_safety.utils import (
    build_system_prompt,
    generate_api,
    generate_batch,
    load_dataset,
    load_model,
    save_json,
)


def _load_jsonl_dataset(path: str) -> Dict[str, List[str]]:
    """
    Load prompts from a JSONL file into {category: [prompt]} format.
    Supports records with an explicit 'prompt' + 'category' field
    (e.g. test_single.jsonl) or plain {messages} records (e.g. train.jsonl).
    """
    dataset: Dict[str, List[str]] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            # Prefer explicit 'prompt' field; fall back to first user message
            if "prompt" in rec:
                prompt = rec["prompt"]
            else:
                user_msgs = [m["content"] for m in rec.get("messages", [])
                             if m["role"] == "user"]
                prompt = user_msgs[0] if user_msgs else ""
            if not prompt:
                continue
            category = rec.get("category", "uncategorized")
            dataset.setdefault(category, []).append(prompt)
    return dataset


def _load_existing(output_path, categories, all_prompts):
    """Returns {flat_index: response} for items that have a non-empty saved response."""
    if not os.path.exists(output_path):
        return {}
    import json
    with open(output_path) as f:
        saved = json.load(f)
    existing = {}
    for i, (cat, prompt) in enumerate(zip(categories, all_prompts)):
        for item in saved.get(cat, []):
            if item["prompt"] == prompt and item.get("response", ""):
                existing[i] = item["response"]
                break
    return existing


def _build_output(categories, all_prompts, results):
    output = {}
    for i, (cat, prompt) in enumerate(zip(categories, all_prompts)):
        output.setdefault(cat, []).append({
            "prompt": prompt,
            "response": results.get(i, ""),
        })
    return output


def parse_args():
    parser = argparse.ArgumentParser()

    dataset = parser.add_argument_group("dataset")
    dataset.add_argument("--dataset_path", type=str, required=True)
    dataset.add_argument(
        "--prompt_type",
        type=str,
        default="with_cues",
        choices=["with_cues", "without_cues"],
    )
    dataset.add_argument(
        "--language",
        type=str,
        default="english",
        choices=["english", "urdu", "hindi", "mandarin"],
    )
    dataset.add_argument("--age", action="store_true", default=False)
    dataset.add_argument("--country", type=str, default=None)

    model = parser.add_argument_group("model")
    model.add_argument("--model_name", type=str, required=True)
    model.add_argument("--model_path", type=str, required=True,
                       help="Local path for vllm, or API model ID for other providers")
    model.add_argument(
        "--provider",
        type=str,
        default="vllm",
        choices=["vllm", "openai", "deepseek", "anthropic", "google"],
    )
    # vllm-only options
    model.add_argument("--tensor_parallel_size", type=int, default=1)
    model.add_argument("--max_model_len", type=int, default=None)
    model.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    model.add_argument("--language_model_only", action="store_true", default=False)

    generation = parser.add_argument_group("generation")
    generation.add_argument("--temperature", type=float, default=0.7)
    generation.add_argument("--top_p", type=float, default=1.0)
    generation.add_argument("--max_new_tokens", type=int, default=1024)

    output = parser.add_argument_group("output")
    output.add_argument("--output_path", type=str, required=True)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.dataset_path.endswith(".jsonl"):
        dataset = _load_jsonl_dataset(args.dataset_path)
    else:
        dataset = load_dataset(
            path=args.dataset_path,
            prompt_type=args.prompt_type,
            language=args.language,
        )

    system_prompt = build_system_prompt(age=args.age, country=args.country, language=args.language)

    categories, all_prompts = zip(
        *[(category, prompt) for category, prompts in dataset.items() for prompt in prompts]
    )

    all_messages = []
    for prompt in all_prompts:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        all_messages.append(messages)

    # Resume: load any existing non-empty results
    results = _load_existing(args.output_path, categories, all_prompts)
    if results:
        print(f"Resuming: {len(results)}/{len(all_prompts)} already done.", flush=True)

    pending_indices = [i for i in range(len(all_messages)) if i not in results]
    pending_messages = [all_messages[i] for i in pending_indices]

    if not pending_messages:
        print("All responses already present, nothing to do.", flush=True)
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
        pending_responses = generate_batch(
            tokenizer=tokenizer,
            llm=llm,
            messages_batch=pending_messages,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
        )
        for i, resp in zip(pending_indices, pending_responses):
            results[i] = resp
        save_json(args.output_path, _build_output(categories, all_prompts, results))
    else:
        def on_result(orig_i, text):
            results[orig_i] = text
            save_json(args.output_path, _build_output(categories, all_prompts, results))

        generate_api(
            provider=args.provider,
            model=args.model_path,
            messages_batch=pending_messages,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            indices=pending_indices,
            on_result=on_result,
        )


if __name__ == "__main__":
    main()
