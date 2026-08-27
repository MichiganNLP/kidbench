#!/bin/bash
# Generate KIDLlama responses for the held-out single-turn test set.
#
# Replays all 8 single-turn conditions (no-cue, implicit-cue, explicit-age, and
# the 4 country contexts) over dataset/gold/test_single.jsonl and writes them in
# exactly the layout produced by normal single-turn generation, so the same
# evaluation and analysis scripts apply unchanged.
#
# Output: responses/single_turn/{model}/{with_cues,without_cues}/*.json
# Next:   scripts/evaluation/single_turn/deepseek.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

TEMPERATURE=0.0
MAX_NEW_TOKENS=2048
MAX_MODEL_LEN=4096
GPU_MEMORY_UTILIZATION=0.90

# KIDLlama checkpoints to evaluate.
MODELS=(
    "sameearif/LlamaPlushie-3-8B-GRPO"
)

for MODEL in "${MODELS[@]}"; do
    MODEL_NAME="$(basename "${MODEL}")"

    echo "============================================================"
    echo "MODEL=${MODEL_NAME}"
    echo "============================================================"

    PYTHONPATH="${ROOT_DIR}/src" \
    PYTHONUNBUFFERED=1 \
    ${HF_TOKEN:+HF_TOKEN="${HF_TOKEN}"} \
    ${HF_TOKEN:+HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"} \
    python3 -m child_safety.finetune.gold_test \
        --model_name             "${MODEL_NAME}" \
        --model_path             "${MODEL}" \
        --temperature            "${TEMPERATURE}" \
        --max_new_tokens         "${MAX_NEW_TOKENS}" \
        --max_model_len          "${MAX_MODEL_LEN}" \
        --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}"
done

echo "All models done."
