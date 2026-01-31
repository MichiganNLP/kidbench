import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm

from utils import (
    load_dataset,
    load_model,
    generate_batch,
    write_outputs,
)

os.environ.setdefault("VLLM_DISABLE_TQDM", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_messages_batch(prompts: List[str]) -> List[List[Dict[str, str]]]:
    return [[{"role": "user", "content": prompt}] for prompt in prompts]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen-2.5-7B",
        help="Model name for naming the output file.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model checkpoint.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="dataset",
        help="Dataset name (metadata only).",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the dataset JSON file.",
    )
    parser.add_argument(
        "--prompt_type",
        type=str,
        choices=["with_cues", "without_cues"],
        required=True,
        help="Choose which prompt field to use from the dataset.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate per turn.",
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        default=False,
        help="Enable sampling for generation.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Nucleus sampling probability.",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs to use for tensor parallelism.",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=None,
        help="Override vLLM max_model_len to reduce KV cache usage.",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=None,
        help="Override vLLM gpu_memory_utilization (0-1).",
    )
    args = parser.parse_args()

    # Setup output directory
    output_dir = PROJECT_ROOT / "responses" / args.prompt_type
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{args.model_name}.json"

    # Load dataset and model
    dataset = load_dataset(args.dataset_path)
    tokenizer, model = load_model(
        args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    # Flatten prompts while retaining category
    flat_items: List[Tuple[str, str]] = []
    for category, records in dataset.items():
        for record in records:
            if args.prompt_type not in record:
                raise KeyError(
                    f"Prompt type '{args.prompt_type}' missing in category '{category}'."
                )
            flat_items.append((category, record[args.prompt_type]))

    outputs_by_category: Dict[str, List[Dict[str, str]]] = {
        category: [] for category in dataset.keys()
    }

    total_prompts = len(flat_items)
    if total_prompts == 0:
        write_outputs(str(output_file), outputs_by_category)
        return

    with tqdm(
        total=total_prompts,
        desc=f"Generating responses: {args.model_name}",
        file=sys.stdout,
        mininterval=1.0,
        miniters=1,
        dynamic_ncols=True,
        unit="prompt",
    ) as progress_bar:
        prompts = [prompt for _, prompt in flat_items]
        messages_batch = build_messages_batch(prompts)
        responses = generate_batch(
            tokenizer,
            model,
            messages_batch,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        for (category, prompt), response in zip(flat_items, responses):
            outputs_by_category[category].append(
                {"prompt": prompt, "response": response}
            )
        progress_bar.update(total_prompts)

    write_outputs(str(output_file), outputs_by_category)


if __name__ == "__main__":
    main()
