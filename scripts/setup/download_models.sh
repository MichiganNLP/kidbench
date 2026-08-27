#!/bin/bash
# Download open-weight checkpoints from the Hugging Face Hub into a local
# directory, so that generation/evaluation runs can point --model_path at them.
#
# Usage:
#   MODELS_DIR=/path/to/models bash scripts/setup/download_models.sh
#
# Edit the MODELS map below to choose what to fetch. The key is the local
# directory name, the value is the Hub repo id.
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-${HOME}/models}"

declare -A MODELS=(
    ["Llama-3.2-3B-Instruct"]="meta-llama/Llama-3.2-3B-Instruct"
    ["Llama-3.1-8B-Instruct"]="meta-llama/Llama-3.1-8B-Instruct"
    ["Llama-3.3-70B-Instruct"]="meta-llama/Llama-3.3-70B-Instruct"
    ["gemma-3-4b-it"]="google/gemma-3-4b-it"
    ["gemma-3-12b-it"]="google/gemma-3-12b-it"
    ["Qwen3-8B"]="Qwen/Qwen3-8B"
    # Attacker model used for the multi-turn red-teaming runs:
    ["gemma-4-31B-it-abliterated"]="huihui-ai/Huihui-gemma-4-31B-it-abliterated-v2"
)

mkdir -p "${MODELS_DIR}"

for LOCAL_NAME in "${!MODELS[@]}"; do
    REPO_ID="${MODELS[$LOCAL_NAME]}"
    LOCAL_DIR="${MODELS_DIR}/${LOCAL_NAME}"

    if [ -d "${LOCAL_DIR}" ]; then
        echo "Skipping ${REPO_ID} — already exists at ${LOCAL_DIR}"
        continue
    fi

    echo "Downloading ${REPO_ID} -> ${LOCAL_DIR}"
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='${REPO_ID}', local_dir='${LOCAL_DIR}')
"
    echo "Done: ${REPO_ID}"
done

echo "All models downloaded to ${MODELS_DIR}."
