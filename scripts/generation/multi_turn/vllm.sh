#!/bin/bash
# Multi-turn red-teaming with an open-weight responder served locally by vLLM.
#
# A child-actor model (Gemma-4-31B abliterated, run with the system prompt in
# system_prompts/generation/attacker.jinja) plays a child aged 7-11 across
# 5 turns; the responder under test answers each turn.
#
# The actor and the responder are loaded in separate subprocesses so they can be
# pinned to different GPUs. Set ATTACKER_CUDA_DEVICE / GENERATOR_CUDA_DEVICE for
# your machine; with a single GPU, set both to "0" and lower
# GPU_MEMORY_UTILIZATION so both models fit.
#
# Runs both responder settings: without_age and with_age.
# Output: responses/multi_turn/{model}/{with_age,without_age}.json
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"

SYSTEM_PROMPTS_DIR="${ROOT_DIR}/system_prompts"
DATASET_PATH="${ROOT_DIR}/dataset/kidbench/kidbench_multi.json"

MODELS_DIR="${MODELS_DIR:-${HOME}/models}"
ATTACKER_MODEL_PATH="${ATTACKER_MODEL_PATH:-${MODELS_DIR}/gemma-4-31B-it-abliterated}"

ATTACKER_MAX_NEW_TOKENS=8192
GENERATOR_MAX_NEW_TOKENS=8192
GENERATOR_TEMPERATURE=0
NUM_TURNS=5

MAX_MODEL_LEN=8192
GPU_MEMORY_UTILIZATION=0.90

ATTACKER_CUDA_DEVICE="0"
GENERATOR_CUDA_DEVICE="1"
GENERATOR_TP_SIZE=1

# Responder name (output folder) -> local path or HF repo id
declare -a GENERATOR_NAMES=(
    "llama-3.3-70b"
)
declare -a GENERATOR_PATHS=(
    "${MODELS_DIR}/Llama-3.3-70B-Instruct"
)
# 1 = pass --generator_language_model_only (text-only multimodal checkpoints)
declare -a GENERATOR_LM_ONLY=(
    "0"
)

for i in "${!GENERATOR_NAMES[@]}"; do
    GENERATOR_MODEL_NAME="${GENERATOR_NAMES[$i]}"
    GENERATOR_MODEL_PATH="${GENERATOR_PATHS[$i]}"
    LM_ONLY_FLAG=""
    [[ "${GENERATOR_LM_ONLY[$i]}" == "1" ]] && LM_ONLY_FLAG="--generator_language_model_only"

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
        CUDA_VISIBLE_DEVICES="${ATTACKER_CUDA_DEVICE},${GENERATOR_CUDA_DEVICE}" \
        python3 -m child_safety.generation.multi_turn \
            --dataset_path                   "${DATASET_PATH}" \
            --output_path                    "${OUTPUT_PATH}" \
            --system_prompts_dir             "${SYSTEM_PROMPTS_DIR}" \
            --attacker_model_path            "${ATTACKER_MODEL_PATH}" \
            --attacker_max_new_tokens        "${ATTACKER_MAX_NEW_TOKENS}" \
            --attacker_language_model_only \
            --attacker_cuda_device           "${ATTACKER_CUDA_DEVICE}" \
            --generator_provider             vllm \
            --generator_model_name           "${GENERATOR_MODEL_NAME}" \
            --generator_model_path           "${GENERATOR_MODEL_PATH}" \
            --generator_max_new_tokens       "${GENERATOR_MAX_NEW_TOKENS}" \
            --generator_temperature          "${GENERATOR_TEMPERATURE}" \
            --generator_cuda_device          "${GENERATOR_CUDA_DEVICE}" \
            --generator_tensor_parallel_size "${GENERATOR_TP_SIZE}" \
            --tensor_parallel_size           1 \
            --max_model_len                  "${MAX_MODEL_LEN}" \
            --gpu_memory_utilization         "${GPU_MEMORY_UTILIZATION}" \
            --num_turns                      "${NUM_TURNS}" \
            ${LM_ONLY_FLAG} \
            ${AGE_FLAG}
    done
done

echo "All experiments done."
