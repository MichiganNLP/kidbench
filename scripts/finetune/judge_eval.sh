#!/bin/bash
# Evaluate KIDGuardLlama checkpoints against the held-out judge test set.
#
# Reports agreement with the DeepSeek-V4-Pro reference judge: Spearman rho,
# quadratic weighted kappa, MAE, exact and within-1 accuracy, plus macro-F1 on
# the coarse 3-way remapping. Multi-turn records are scored with teacher forcing
# (the gold evaluations of earlier turns are given as context).
#
# Input:  dataset/judge/test.jsonl
# Output: responses/judge/{model}/predictions.jsonl and metrics.json
# Next:   the checkpoint comparison table in
#         src/child_safety/analysis/generate_guard_analysis.py
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

TEST_PATH="${ROOT_DIR}/dataset/judge/test.jsonl"
OUTPUT_DIR="${ROOT_DIR}/responses/judge"

MAX_MODEL_LEN=4096
GPU_MEM=0.90
TP_SIZE=1
MAX_NEW_TOKENS=512

# One checkpoint per training epoch. Change HF_NAMESPACE if you retrain.
HF_NAMESPACE="${HF_NAMESPACE:-sameearif}"

for VERSION in 1 2 3; do
    MODEL_NAME="LlamaSproutGuard-3-8B-${VERSION}"
    MODEL_PATH="${HF_NAMESPACE}/LlamaSproutGuard-3-8B-${VERSION}"

    echo "============================================================"
    echo "Model:   ${MODEL_PATH}"
    echo "Test:    ${TEST_PATH}"
    echo "Output:  ${OUTPUT_DIR}/${MODEL_NAME}/"
    echo "============================================================"

    PYTHONPATH="${ROOT_DIR}/src" \
    PYTHONUNBUFFERED=1 \
    python3 -m child_safety.finetune.judge_test \
        --model_name             "${MODEL_NAME}" \
        --model_path             "${MODEL_PATH}" \
        --test_path              "${TEST_PATH}" \
        --output_dir             "${OUTPUT_DIR}" \
        --max_model_len          "${MAX_MODEL_LEN}" \
        --gpu_memory_utilization "${GPU_MEM}" \
        --tensor_parallel_size   "${TP_SIZE}" \
        --max_new_tokens         "${MAX_NEW_TOKENS}" \
        --resume

    echo "Done: ${MODEL_NAME}"
done

echo "All done."
