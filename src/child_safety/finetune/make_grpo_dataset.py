#!/usr/bin/env python3
"""
Convert dataset/gold/train.jsonl to VERL parquet format for Critique GRPO.

Output parquet columns:
  prompt           – list[dict]  (chat messages for the policy)
  reward_model     – dict        (ground_truth = user question, question = user question)
  data_source      – str         ("child_safety")
  target           – list[dict]  (gold assistant response as reference)
  extra_info       – dict        (index)
"""

import argparse
import json
from pathlib import Path

import datasets as hf_datasets


CHILD_SAFETY_SYSTEM_PROMPT = (
    "You are a helpful, friendly assistant designed to talk with children aged 7–11. "
    "Always respond in a way that is safe, kind, truthful, and age-appropriate."
)


def convert(input_path: Path, output_path: Path) -> None:
    records = []
    with open(input_path) as f:
        for line in f:
            ex = json.loads(line.strip())
            msgs = ex.get("messages", [])
            user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
            asst_msg = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
            if not user_msg:
                continue

            records.append({
                "prompt": [
                    {"role": "system", "content": CHILD_SAFETY_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                "reward_model": {
                    "ground_truth": user_msg,
                    "question":     user_msg,
                },
                "data_source": "child_safety",
                "target": [{"role": "assistant", "content": asst_msg}],
                "extra_info": {"index": len(records)},
            })

    ds = hf_datasets.Dataset.from_list(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(str(output_path))
    print(f"Wrote {len(records)} examples → {output_path}")


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[3]  # src/child_safety/finetune → repo root
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default=str(ROOT / "dataset/gold/train.jsonl"))
    parser.add_argument("--output", default=str(ROOT / "dataset/gold/grpo_train.parquet"))
    args = parser.parse_args()
    convert(Path(args.input), Path(args.output))
