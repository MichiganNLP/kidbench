#!/bin/bash
# Regenerate every analysis artifact from the evaluation files.
#
# Each script reads evaluations/ (and responses/judge/ for the guard analysis)
# and writes markdown tables plus figures into evaluations/analysis/.
# Nothing here calls a model, so it runs on CPU in a couple of minutes.
#
# Pass a stage name to run just that group, e.g.
#   bash scripts/analysis/run_all.sh cues
# Stages: all (default), cues, language, cultural, multi_turn, models
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
STAGE="${1:-all}"

run() {
    echo "============================================================"
    echo "$1"
    echo "============================================================"
    PYTHONPATH="${ROOT_DIR}/src" PYTHONUNBUFFERED=1 python3 -m "child_safety.analysis.$1"
}

wanted() {
    [ "${STAGE}" = "all" ] || [ "${STAGE}" = "$1" ]
}

case "${STAGE}" in
    all|cues|language|cultural|multi_turn|models) ;;
    *)
        echo "Unknown stage: ${STAGE}"
        echo "Valid stages: all, cues, language, cultural, multi_turn, models"
        exit 1
        ;;
esac

# Cue conditions: no-cue vs implicit-cue vs explicit-age
if wanted cues; then
    run generate_cues
    run generate_cues_stats
fi

# Cross-lingual: English / Mandarin / Hindi / Urdu
if wanted language; then
    run generate_language
    run generate_language_stats
fi

# Cultural alignment: Pakistan / India / China / Nigeria
if wanted cultural; then
    run generate_cultural
    run generate_cultural_stats
fi

# Multi-turn degradation: slope and peak quality drop
if wanted multi_turn; then
    run generate_multi_turn
    run generate_multi_turn_stats
fi

# Fine-tuned models: KIDLlama checkpoints and KIDGuardLlama agreement
if wanted models; then
    run generate_gold_analysis
    run generate_guard_analysis
fi

echo "All analyses written to ${ROOT_DIR}/evaluations/analysis/"
