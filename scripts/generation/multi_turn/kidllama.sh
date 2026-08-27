#!/bin/bash
# Multi-turn red-teaming for the fine-tuned KIDLlama checkpoints.
#
# Same protocol as scripts/generation/multi_turn/vllm.sh, but the responder is
# pulled from the Hugging Face Hub. Set MODELS below to the checkpoints you want
# to evaluate.
#
# Output: responses/multi_turn/{model}/{with_age,without_age}.json
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"

SYSTEM_PROMPTS_DIR="${ROOT_DIR}/system_prompts"
DATASET_PATH="${ROOT_DIR}/dataset/kidbench/kidbench_multi.json"

MODELS_DIR="${MODELS_DIR:-${HOME}/models}"
ATTACKER_MODEL_PATH="${ATTACKER_MODEL_PATH:-${MODELS_DIR}/gemma-4-31B-it-abliterated}"
ATTACKER_MAX_NEW_TOKENS=512
ATTACKER_CUDA_DEVICE="0"

GENERATOR_MAX_NEW_TOKENS=2048
GENERATOR_TEMPERATURE=0.0
GENERATOR_CUDA_DEVICE="1"

MAX_MODEL_LEN=4096
GPU_MEMORY_UTILIZATION=0.90
NUM_TURNS=5

# KIDLlama checkpoints. Update these to your own repos if you retrain.
MODELS=(
    "sameearif/LlamaPlushie-3-8B-GRPO"
)

for MODEL in "${MODELS[@]}"; do
    MODEL_NAME="$(basename "${MODEL}")"

    for AGE_FLAG in "--age" ""; do
        if [ -n "${AGE_FLAG}" ]; then
            OUTPUT_PATH="${ROOT_DIR}/responses/multi_turn/${MODEL_NAME}/with_age.json"
            AGE_LABEL="with_age"
        else
            OUTPUT_PATH="${ROOT_DIR}/responses/multi_turn/${MODEL_NAME}/without_age.json"
            AGE_LABEL="without_age"
        fi

        echo "============================================================"
        echo "MODEL=${MODEL_NAME} | ${AGE_LABEL} | TURNS=${NUM_TURNS}"
        echo "OUTPUT=${OUTPUT_PATH}"
        echo "============================================================"

        PYTHONPATH="${ROOT_DIR}/src" \
        PYTHONUNBUFFERED=1 \
        CUDA_VISIBLE_DEVICES="${ATTACKER_CUDA_DEVICE},${GENERATOR_CUDA_DEVICE}" \
        python3 -m child_safety.generation.multi_turn \
            --dataset_path              "${DATASET_PATH}" \
            --output_path               "${OUTPUT_PATH}" \
            --system_prompts_dir        "${SYSTEM_PROMPTS_DIR}" \
            --attacker_model_path       "${ATTACKER_MODEL_PATH}" \
            --attacker_max_new_tokens   "${ATTACKER_MAX_NEW_TOKENS}" \
            --attacker_language_model_only \
            --attacker_cuda_device      "${ATTACKER_CUDA_DEVICE}" \
            --generator_provider        vllm \
            --generator_model_name      "${MODEL_NAME}" \
            --generator_model_path      "${MODEL}" \
            --generator_max_new_tokens  "${GENERATOR_MAX_NEW_TOKENS}" \
            --generator_temperature     "${GENERATOR_TEMPERATURE}" \
            --generator_cuda_device     "${GENERATOR_CUDA_DEVICE}" \
            --tensor_parallel_size      1 \
            --max_model_len             "${MAX_MODEL_LEN}" \
            --gpu_memory_utilization    "${GPU_MEMORY_UTILIZATION}" \
            --num_turns                 "${NUM_TURNS}" \
            ${AGE_FLAG}
    done
done

echo "All models done."
