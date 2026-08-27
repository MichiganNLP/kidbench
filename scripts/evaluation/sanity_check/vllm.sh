#!/bin/bash
# Judge-selection study: score the same 100 prompt-response pairs with a
# candidate judge (vllm), so that judges can be compared for agreement before
# one is chosen to evaluate the full benchmark.
#
# Input:  sanity_check/responses/single_turn/english_both_none.json
# Output: sanity_check/evaluations/single_turn/{judge}.json
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"

RESPONSE_PATH="${ROOT_DIR}/sanity_check/responses/single_turn/english_both_none.json"
SYSTEM_PROMPTS_DIR="${ROOT_DIR}/system_prompts"
TENSOR_PARALLEL_SIZE=2
MAX_NEW_TOKENS=8192
MAX_MODEL_LEN=8192
GPU_MEMORY_UTILIZATION=0.90
TEMPERATURE=0
TOP_P=1

MODELS_DIR="${MODELS_DIR:-${HOME}/models}"

declare -A MODEL_PATHS=(
    ["llama-3.3-70b"]="${MODELS_DIR}/Llama-3.3-70B-Instruct"
    ["qwen-3.6-27b"]="${MODELS_DIR}/Qwen3.6-27B"
    ["gemma-4-31b"]="${MODELS_DIR}/gemma-4-31B-it"
)

for MODEL_NAME in "${!MODEL_PATHS[@]}"; do
    MODEL_PATH="${MODEL_PATHS[$MODEL_NAME]}"
    OUTPUT_PATH="${ROOT_DIR}/sanity_check/evaluations/single_turn/${MODEL_NAME}.json"

    echo "============================================================"
    echo "MODEL=${MODEL_NAME} | PROVIDER=vllm"
    echo "OUTPUT=${OUTPUT_PATH}"
    echo "============================================================"

    PYTHONPATH="${ROOT_DIR}/src" \
    PYTHONUNBUFFERED=1 \
    python3 -m child_safety.evaluation.single_turn \
        --response_path "${RESPONSE_PATH}" \
        --output_path "${OUTPUT_PATH}" \
        --system_prompts_dir "${SYSTEM_PROMPTS_DIR}" \
        --model_name "${MODEL_NAME}" \
        --model_path "${MODEL_PATH}" \
        --provider vllm \
        --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}" \
        --max_model_len "${MAX_MODEL_LEN}" \
        --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
        --temperature "${TEMPERATURE}" \
        --top_p "${TOP_P}" \
        --max_new_tokens "${MAX_NEW_TOKENS}"
done

echo "All models done."
