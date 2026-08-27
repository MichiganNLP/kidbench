#!/bin/bash
# Install the Python dependencies for KIDBench.
#
# Usage:
#   bash scripts/setup/install.sh            # API + analysis dependencies only
#   bash scripts/setup/install.sh --vllm     # also install vLLM (needs a CUDA GPU)
#
# vLLM is only required for running open-weight models locally (generation,
# evaluation with a local judge, and fine-tuned model inference). Everything
# that talks to a hosted API, plus all analysis scripts, works without it.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

WITH_VLLM=0
for arg in "$@"; do
    [ "${arg}" = "--vllm" ] && WITH_VLLM=1
done

echo "[1/2] Installing the child_safety package and its dependencies..."
pip install -e "${ROOT_DIR}"

if [ "${WITH_VLLM}" -eq 1 ]; then
    echo "[2/2] Installing vLLM..."
    pip install uv
    uv pip install vllm --torch-backend=auto --system
else
    echo "[2/2] Skipping vLLM (pass --vllm to install it)."
fi

echo
echo "Done. Next: copy .env.example to .env and fill in your API keys."
