#!/bin/bash
# Single-turn generation with a hosted anthropic model.
#
# Runs the full single-turn condition grid (no-cue, implicit-cue, explicit-age,
# 4 country contexts, 3 non-English languages) for one model.
# Requires the matching API key in your environment — see .env.example.
#
# Output: responses/single_turn/{model}/{with_cues,without_cues}/{lang}_{age}_{country}.json
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"

# --- Fixed params ---
DATASET_PATH="${ROOT_DIR}/dataset/kidbench/kidbench_single.json"
TEMPERATURE=0.7
MAX_NEW_TOKENS=8192

MODEL_NAME="claude-haiku-4.5"
MODEL_PATH="claude-haiku-4-5"

# --- Countries evaluated for cultural alignment ---
COUNTRIES=("Pakistan" "India" "China" "Nigeria")

run() {
    local PROMPT_TYPE="$1"
    local AGE="$2"
    local COUNTRY="$3"
    local LANG="${4:-english}"

    local COUNTRY_SUFFIX="none"
    [ -n "$COUNTRY" ] && COUNTRY_SUFFIX="${COUNTRY}"

    local OUTPUT_PATH="${ROOT_DIR}/responses/single_turn/${MODEL_NAME}/${PROMPT_TYPE}/${LANG}_${AGE}_${COUNTRY_SUFFIX,,}.json"

    local EXTRA_ARGS=()
    [ "$AGE" = true ] && EXTRA_ARGS+=(--age)
    [ -n "$COUNTRY" ] && EXTRA_ARGS+=(--country "$COUNTRY")

    echo "============================================================"
    echo "MODEL=${MODEL_NAME} | LANG=${LANG} | PROMPT_TYPE=${PROMPT_TYPE} | AGE=${AGE} | COUNTRY=${COUNTRY:-none}"
    echo "OUTPUT=${OUTPUT_PATH}"
    echo "============================================================"

    PYTHONPATH="${ROOT_DIR}/src" \
    PYTHONUNBUFFERED=1 \
    python3 -m child_safety.generation.single_turn \
        --dataset_path "${DATASET_PATH}" \
        --prompt_type "${PROMPT_TYPE}" \
        --language "${LANG}" \
        --model_name "${MODEL_NAME}" \
        --model_path "${MODEL_PATH}" \
        --provider anthropic \
        --temperature "${TEMPERATURE}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --output_path "${OUTPUT_PATH}" \
        "${EXTRA_ARGS[@]}"
}

# implicit cue: with_cues prompts, no age, no country
run "with_cues"    false ""

# no cue / explicit age: without_cues prompts
run "without_cues" false ""
run "without_cues" true  ""

# cultural: with_cues prompts, age=true, one run per country
for COUNTRY in "${COUNTRIES[@]}"; do
    run "with_cues" true "$COUNTRY"
done

# cross-lingual: without_cues prompts, no age, no country
for LANG in urdu hindi mandarin; do
    run "without_cues" false "" "$LANG"
done

echo "All runs completed."
