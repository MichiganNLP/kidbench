#!/bin/bash
# Multi-turn red-teaming with a hosted deepseek responder.
#
# A child-actor model (Gemma-4-31B abliterated, served locally by vLLM) plays a
# child aged 7-11 across 5 turns; the hosted model under test answers each turn.
# The actor needs one local GPU; the responder is called over the API.
#
# Runs both responder settings: without_age and with_age.
# Output: responses/multi_turn/{model}/{with_age,without_age}.json
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"

SYSTEM_PROMPTS_DIR="${ROOT_DIR}/system_prompts"
DATASET_PATH="${ROOT_DIR}/dataset/kidbench/kidbench_multi.json"

MODELS_DIR="${MODELS_DIR:-${HOME}/models}"
ATTACKER_MODEL_PATH="${ATTACKER_MODEL_PATH:-${MODELS_DIR}/gemma-4-31B-it-abliterated}"

GENERATOR_PROVIDER="deepseek"
GENERATOR_MODEL_NAME="deepseek-v4-flash"

MAX_NEW_TOKENS=2048
ATTACKER_MAX_NEW_TOKENS=8192
GENERATOR_TEMPERATURE=0
NUM_TURNS=5
MAX_CONCURRENT=50

MAX_MODEL_LEN=8192
GPU_MEMORY_UTILIZATION=0.90

for AGE_FLAG in "--age" ""; do
    if [ -n "${AGE_FLAG}" ]; then
        OUTPUT_PATH="${ROOT_DIR}/responses/multi_turn/${GENERATOR_MODEL_NAME}/with_age.json"
        AGE_LABEL="with_age"
    else
        OUTPUT_PATH="${ROOT_DIR}/responses/multi_turn/${GENERATOR_MODEL_NAME}/without_age.json"
        AGE_LABEL="without_age"
    fi

    echo "============================================================"
    echo "GENERATOR=${GENERATOR_MODEL_NAME} | ${AGE_LABEL} | TURNS=${NUM_TURNS}"
    echo "OUTPUT=${OUTPUT_PATH}"
    echo "============================================================"

    PYTHONPATH="${ROOT_DIR}/src" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_VISIBLE_DEVICES="0" \
    python3 -m child_safety.generation.multi_turn \
        --dataset_path             "${DATASET_PATH}" \
        --output_path              "${OUTPUT_PATH}" \
        --system_prompts_dir       "${SYSTEM_PROMPTS_DIR}" \
        --attacker_model_path      "${ATTACKER_MODEL_PATH}" \
        --attacker_max_new_tokens  "${ATTACKER_MAX_NEW_TOKENS}" \
        --attacker_language_model_only \
        --attacker_cuda_device     "0" \
        --generator_provider       "${GENERATOR_PROVIDER}" \
        --generator_model_name     "${GENERATOR_MODEL_NAME}" \
        --generator_max_new_tokens "${MAX_NEW_TOKENS}" \
        --generator_temperature    "${GENERATOR_TEMPERATURE}" \
        --tensor_parallel_size     1 \
        --max_model_len            "${MAX_MODEL_LEN}" \
        --gpu_memory_utilization   "${GPU_MEMORY_UTILIZATION}" \
        --max_concurrent           "${MAX_CONCURRENT}" \
        --num_turns                "${NUM_TURNS}" \
        ${AGE_FLAG}
done

echo "All experiments done."
