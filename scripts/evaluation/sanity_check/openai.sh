#!/bin/bash
# Judge-selection study: score the same 100 prompt-response pairs with a
# candidate judge (openai), so that judges can be compared for agreement before
# one is chosen to evaluate the full benchmark.
#
# Input:  sanity_check/responses/single_turn/english_both_none.json
# Output: sanity_check/evaluations/single_turn/{judge}.json
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"

RESPONSE_PATH="${ROOT_DIR}/sanity_check/responses/single_turn/english_both_none.json"
SYSTEM_PROMPTS_DIR="${ROOT_DIR}/system_prompts"
MAX_NEW_TOKENS=4096
MAX_CONCURRENT=10

MODEL_NAME="gpt-5.5"
MODEL_PATH="gpt-5.5"
OUTPUT_PATH="${ROOT_DIR}/sanity_check/evaluations/single_turn/${MODEL_NAME}.json"

echo "============================================================"
echo "MODEL=${MODEL_NAME} | PROVIDER=openai"
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
    --provider openai \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --max_concurrent "${MAX_CONCURRENT}"

echo "Done."
