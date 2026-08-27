#!/bin/bash
# Build the judge (LlamaKidGuard) SFT dataset from all collected evaluations.
#
# Reads:  dataset/kidbench/kidbench_{single,multi}.json
#         evaluations/{judge}/single_turn/**/*.json
#         evaluations/{judge}/multi_turn/**/*.json
#         gold_responses/single_turn/**/*.json
# Writes: dataset/judge/train.jsonl, dataset/judge/test.jsonl
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

PYTHONPATH="${ROOT_DIR}/src" \
PYTHONUNBUFFERED=1 \
python3 -m child_safety.finetune.judge_dataset
