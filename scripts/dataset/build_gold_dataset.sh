#!/bin/bash
# Build the gold-response SFT dataset from gold_responses/.
#
# Reads:  dataset/kidbench/kidbench_single.json
#         gold_responses/single_turn/{model}/{with_cues,without_cues}/*.json
# Writes: dataset/gold/train.jsonl        (only turns scoring 5/5 on all metrics)
#         dataset/gold/test_single.jsonl  (5 held-out items per category × 7 conditions)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

PYTHONPATH="${ROOT_DIR}/src" \
PYTHONUNBUFFERED=1 \
python3 -m child_safety.finetune.gold_dataset
