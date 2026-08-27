#!/bin/bash
# Gold-response generation: the critique-revise loop that produces KIDLlama's
# SFT data.
#
# For each benchmark prompt, a strong teacher model (served locally by vLLM)
# writes a child-appropriate response; DeepSeek-V4-Pro then judges it against
# the child-safety rubric and the category / country rules. If any dimension
# scores below 5, the teacher revises using the judge feedback. NUM_TURNS caps
# the number of generation rounds (1 = initial, 2 = one revision).
#
# Only turns scoring 5/5 on every metric are later kept by
# scripts/dataset/build_gold_dataset.sh.
#
# Prompts are read from an existing response file (only its prompts are used,
# never its responses), so run single-turn generation for at least one model first.
#
# Output: gold_responses/single_turn/{teacher}/{with_cues,without_cues}/{file}.json
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"

SYSTEM_PROMPTS_DIR="${ROOT_DIR}/system_prompts"
PROMPTS_SOURCE_MODEL="claude-haiku-4.5"   # any model; only prompts are used, not responses
RESPONSES_DIR="${ROOT_DIR}/responses/single_turn/${PROMPTS_SOURCE_MODEL}"

GENERATOR_PROVIDER="vllm"
TENSOR_PARALLEL_SIZE=2
MAX_MODEL_LEN=8192
GPU_MEMORY_UTILIZATION=0.90

MAX_NEW_TOKENS=8192
TEMPERATURE=0
TOP_P=1
NUM_TURNS=2

MODELS_DIR="${MODELS_DIR:-${HOME}/models}"

# Teacher models used to write gold responses.
declare -a GENERATOR_NAMES=(
    "gemma-4-31B-it"
)

declare -a GENERATOR_PATHS=(
    "${MODELS_DIR}/gemma-4-31B-it"
)

# 1 = pass --language_model_only (for text-only checkpoints of multimodal architectures)
declare -a LANGUAGE_MODEL_ONLY=(
    "1"
)

# 7 experiments (excluding Urdu, Hindi, Mandarin language files)
declare -a EXPERIMENTS=(
    "without_cues/english_false_none.json"
    "without_cues/english_true_none.json"
    "with_cues/english_false_none.json"
    "with_cues/english_true_china.json"
    "with_cues/english_true_india.json"
    "with_cues/english_true_nigeria.json"
    "with_cues/english_true_pakistan.json"
)

for i in "${!GENERATOR_NAMES[@]}"; do
    GENERATOR_MODEL_NAME="${GENERATOR_NAMES[$i]}"
    GENERATOR_MODEL_PATH="${GENERATOR_PATHS[$i]}"
    LM_ONLY_FLAG=""
    [[ "${LANGUAGE_MODEL_ONLY[$i]}" == "1" ]] && LM_ONLY_FLAG="--language_model_only"

    for EXPERIMENT in "${EXPERIMENTS[@]}"; do
        CUES_TYPE="$(dirname "${EXPERIMENT}")"
        FILENAME="$(basename "${EXPERIMENT}")"
        PROMPTS_PATH="${RESPONSES_DIR}/${EXPERIMENT}"
        OUTPUT_PATH="${ROOT_DIR}/gold_responses/single_turn/${GENERATOR_MODEL_NAME}/${CUES_TYPE}/${FILENAME}"

        echo "============================================================"
        echo "GENERATOR=${GENERATOR_MODEL_NAME} | EVALUATOR=deepseek-v4-pro"
        echo "EXPERIMENT=${EXPERIMENT} | TURNS=${NUM_TURNS}"
        echo "OUTPUT=${OUTPUT_PATH}"
        echo "============================================================"

        PYTHONPATH="${ROOT_DIR}/src" \
        PYTHONUNBUFFERED=1 \
        python3 -m child_safety.generation.gold_single_turn \
            --prompts_path "${PROMPTS_PATH}" \
            --output_path "${OUTPUT_PATH}" \
            --system_prompts_dir "${SYSTEM_PROMPTS_DIR}" \
            --generator_model_name "${GENERATOR_MODEL_NAME}" \
            --generator_model_path "${GENERATOR_MODEL_PATH}" \
            --generator_provider "${GENERATOR_PROVIDER}" \
            --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}" \
            --max_model_len "${MAX_MODEL_LEN}" \
            --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
            --temperature "${TEMPERATURE}" \
            --top_p "${TOP_P}" \
            --max_new_tokens "${MAX_NEW_TOKENS}" \
            --num_turns "${NUM_TURNS}" \
            ${LM_ONLY_FLAG}
    done
done

echo "All experiments done."
