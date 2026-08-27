#!/bin/bash
# Score every multi-turn conversation with the DeepSeek-V4-Pro judge.
#
# Each conversation is judged as a single chat session: the judge sees turn t
# together with its own scores for turns 1..t-1, so later scores reflect the
# conversation so far rather than an isolated reply. This is what makes the
# degradation-slope and peak-drop analyses meaningful.
#
# Input:  responses/multi_turn/{model}/{with_age,without_age}.json
# Output: evaluations/deepseek-v4-pro/multi_turn/{model}/{with_age,without_age}.json
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"

RESPONSES_DIR="${ROOT_DIR}/responses/multi_turn"
SYSTEM_PROMPTS_DIR="${ROOT_DIR}/system_prompts"

MODEL_PATH="deepseek-v4-flash"
MAX_NEW_TOKENS=4096
MAX_CONCURRENT=100

# Models to evaluate — must match folder names under responses/multi_turn/.
for MODEL_NAME in "llamaplushie-3-8b-grpo"; do
    for AGE_LABEL in "without_age" "with_age"; do
        RESPONSE_PATH="${RESPONSES_DIR}/${MODEL_NAME}/${AGE_LABEL}.json"

        [ -f "${RESPONSE_PATH}" ] || continue

        OUTPUT_PATH="${ROOT_DIR}/evaluations/${MODEL_PATH}/multi_turn/${MODEL_NAME}/${AGE_LABEL}.json"

        echo "============================================================"
        echo "MODEL=${MODEL_NAME} | ${AGE_LABEL} | JUDGE=${MODEL_PATH}"
        echo "OUTPUT=${OUTPUT_PATH}"
        echo "============================================================"

        PYTHONPATH="${ROOT_DIR}/src" \
        PYTHONUNBUFFERED=1 \
        python3 -m child_safety.evaluation.multi_turn \
            --response_path      "${RESPONSE_PATH}" \
            --output_path        "${OUTPUT_PATH}" \
            --system_prompts_dir "${SYSTEM_PROMPTS_DIR}" \
            --model_path         "${MODEL_PATH}" \
            --provider           deepseek \
            --max_new_tokens     "${MAX_NEW_TOKENS}" \
            --max_concurrent     "${MAX_CONCURRENT}"
    done
done

echo "All done."
