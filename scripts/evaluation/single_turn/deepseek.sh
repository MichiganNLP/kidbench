#!/bin/bash
# Score every single-turn response file with the DeepSeek-V4-Pro judge.
#
# For each response file the judge receives the prompt, the model response, the
# scoring rubric, and the category rules; for country-context files it also
# receives that country's cultural rules. The country is inferred from the file
# name ({lang}_{age}_{country}.json).
#
# It emits a 1-5 score per dimension, failure-mode tags, strengths, and an
# improvement string. Runs are resumable: re-running skips finished items.
#
# Input:  responses/single_turn/{model}/{with_cues,without_cues}/*.json
# Output: evaluations/deepseek-v4-pro/single_turn/{model}/{cues}/{file}.json
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"

RESPONSES_DIR="${ROOT_DIR}/responses/single_turn"
SYSTEM_PROMPTS_DIR="${ROOT_DIR}/system_prompts"
MAX_NEW_TOKENS=4096
MAX_CONCURRENT=50
TEMPERATURE=0
TOP_P=1

MODEL_PATH="deepseek-v4-pro"

# Models to evaluate — must match folder names under responses/single_turn/.
MODELS=(
    "llamaplushie-3-8b-grpo"
)

for MODEL_NAME in "${MODELS[@]}"; do
    for CUES_TYPE in with_cues without_cues; do
        CUES_DIR="${RESPONSES_DIR}/${MODEL_NAME}/${CUES_TYPE}"
        if [ ! -d "${CUES_DIR}" ]; then
            continue
        fi

        for RESPONSE_PATH in "${CUES_DIR}"/*.json; do
            [ -f "${RESPONSE_PATH}" ] || continue
            FILENAME="$(basename "${RESPONSE_PATH}")"
            OUTPUT_PATH="${ROOT_DIR}/evaluations/${MODEL_PATH}/single_turn/${MODEL_NAME}/${CUES_TYPE}/${FILENAME}"

            echo "============================================================"
            echo "MODEL=${MODEL_NAME} | CUES=${CUES_TYPE} | FILE=${FILENAME} | JUDGE=${MODEL_PATH}"
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
                --provider deepseek \
                --temperature "${TEMPERATURE}" \
                --top_p "${TOP_P}" \
                --max_new_tokens "${MAX_NEW_TOKENS}" \
                --max_concurrent "${MAX_CONCURRENT}"
        done
    done
done

echo "All done."
