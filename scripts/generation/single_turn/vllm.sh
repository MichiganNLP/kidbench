#!/bin/bash
# Single-turn generation with open-weight models served locally by vLLM.
#
# Runs the full single-turn condition grid for every model in MODEL_PATHS:
#   no-cue        : without_cues prompts, no system prompt
#   implicit-cue  : with_cues prompts,    no system prompt
#   explicit-age  : without_cues prompts, "child aged 7-11" system prompt
#   cultural      : with_cues prompts,    age + country system prompt (4 countries)
#   cross-lingual : without_cues prompts in Urdu / Hindi / Mandarin, no system prompt
#
# Output: responses/single_turn/{model}/{with_cues,without_cues}/{lang}_{age}_{country}.json
#
# MODEL_PATHS values may be either a local checkpoint directory (see
# scripts/setup/download_models.sh) or a Hugging Face repo id.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"

DATASET_PATH="${ROOT_DIR}/dataset/kidbench/kidbench_single.json"
TEMPERATURE=0
TOP_P=1.0
MAX_NEW_TOKENS=8192
TENSOR_PARALLEL_SIZE=1
MAX_MODEL_LEN=8192
GPU_MEMORY_UTILIZATION=0.90

MODELS_DIR="${MODELS_DIR:-${HOME}/models}"

# --- Models (output name -> local path or HF repo id) ---
declare -A MODEL_PATHS=(
    ["llama-3.2-3b"]="${MODELS_DIR}/Llama-3.2-3B-Instruct"
    ["llama-3.1-8b"]="${MODELS_DIR}/Llama-3.1-8B-Instruct"
    ["llama-3.3-70b"]="${MODELS_DIR}/Llama-3.3-70B-Instruct"
    ["gemma-3-4b"]="${MODELS_DIR}/gemma-3-4b-it"
    ["gemma-3-12b"]="${MODELS_DIR}/gemma-3-12b-it"
    ["gemma-4-31b"]="${MODELS_DIR}/gemma-4-31B-it"
    ["qwen-3-8b"]="${MODELS_DIR}/Qwen3-8B"
    ["qwen-3.5-4b"]="${MODELS_DIR}/Qwen3.5-4B"
    ["qwen-3.6-27b"]="${MODELS_DIR}/Qwen3.6-27B"
)

# Fine-tuned checkpoints can be listed here too, e.g.
#   ["llamaplushie-3-8b-grpo"]="sameearif/LlamaPlushie-3-8B-GRPO"
# though scripts/finetune/kidllama_eval.sh is the usual path for those.

# Text-only checkpoints of multimodal architectures need --language_model_only.
LM_ONLY_MODELS=("qwen-3.5-4b" "qwen-3.6-27b" "gemma-4-31b")

# --- Countries evaluated for cultural alignment ---
COUNTRIES=("Pakistan" "India" "China" "Nigeria")

is_lm_only() {
    local name="$1"
    for m in "${LM_ONLY_MODELS[@]}"; do [ "$m" = "$name" ] && return 0; done
    return 1
}

run() {
    local MODEL_NAME="$1"
    local MODEL_PATH="$2"
    local PROMPT_TYPE="$3"
    local AGE="$4"
    local COUNTRY="$5"
    local LANG="${6:-english}"
    local LM_ONLY="${7:-false}"

    local COUNTRY_SUFFIX="none"
    [ -n "$COUNTRY" ] && COUNTRY_SUFFIX="${COUNTRY}"

    local OUTPUT_PATH="${ROOT_DIR}/responses/single_turn/${MODEL_NAME}/${PROMPT_TYPE}/${LANG}_${AGE}_${COUNTRY_SUFFIX,,}.json"

    local EXTRA_ARGS=()
    [ "$AGE" = true ] && EXTRA_ARGS+=(--age)
    [ -n "$COUNTRY" ] && EXTRA_ARGS+=(--country "$COUNTRY")
    [ "$LM_ONLY" = true ] && EXTRA_ARGS+=(--language_model_only)

    echo "============================================================"
    echo "MODEL=${MODEL_NAME} | LANG=${LANG} | PROMPT_TYPE=${PROMPT_TYPE} | AGE=${AGE} | COUNTRY=${COUNTRY:-none}"
    echo "OUTPUT=${OUTPUT_PATH}"
    echo "============================================================"

    PYTHONPATH="${ROOT_DIR}/src" \
    PYTHONUNBUFFERED=1 \
    VLLM_CONFIGURE_LOGGING=0 \
    python3 -m child_safety.generation.single_turn \
        --dataset_path "${DATASET_PATH}" \
        --prompt_type "${PROMPT_TYPE}" \
        --language "${LANG}" \
        --model_name "${MODEL_NAME}" \
        --model_path "${MODEL_PATH}" \
        --provider vllm \
        --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}" \
        --max_model_len "${MAX_MODEL_LEN}" \
        --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
        --temperature "${TEMPERATURE}" \
        --top_p "${TOP_P}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --output_path "${OUTPUT_PATH}" \
        "${EXTRA_ARGS[@]}"
}

for MODEL_NAME in "${!MODEL_PATHS[@]}"; do
    MODEL_PATH="${MODEL_PATHS[$MODEL_NAME]}"
    LM_ONLY=false
    is_lm_only "$MODEL_NAME" && LM_ONLY=true

    # implicit cue: with_cues prompts, no age, no country
    run "$MODEL_NAME" "$MODEL_PATH" "with_cues"    false "" "english" "$LM_ONLY"

    # no cue / explicit age: without_cues prompts
    run "$MODEL_NAME" "$MODEL_PATH" "without_cues" false "" "english" "$LM_ONLY"
    run "$MODEL_NAME" "$MODEL_PATH" "without_cues" true  "" "english" "$LM_ONLY"

    # cultural: with_cues prompts, age=true, one run per country
    for COUNTRY in "${COUNTRIES[@]}"; do
        run "$MODEL_NAME" "$MODEL_PATH" "with_cues" true "$COUNTRY" "english" "$LM_ONLY"
    done

    # cross-lingual: without_cues prompts, no age, no country
    for LANG in urdu hindi mandarin; do
        run "$MODEL_NAME" "$MODEL_PATH" "without_cues" false "" "$LANG" "$LM_ONLY"
    done
done

echo "All runs completed."
