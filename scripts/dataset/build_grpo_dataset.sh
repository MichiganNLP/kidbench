#!/bin/bash
# Convert the gold SFT dataset into the VERL parquet format used by Critique-GRPO.
#
# Reads:  dataset/gold/train.jsonl
# Writes: dataset/gold/grpo_train.parquet
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

INPUT_PATH="${ROOT_DIR}/dataset/gold/train.jsonl"
OUTPUT_PATH="${ROOT_DIR}/dataset/gold/grpo_train.parquet"

PYTHONPATH="${ROOT_DIR}/src" \
PYTHONUNBUFFERED=1 \
python3 -m child_safety.finetune.make_grpo_dataset \
    --input  "${INPUT_PATH}" \
    --output "${OUTPUT_PATH}"
