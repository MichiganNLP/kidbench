#!/bin/bash
# Set up the environment for Critique-GRPO fine-tuning (VERL backend).
#
# Run scripts/setup/install.sh --vllm first: this script builds on that
# environment and will not upgrade or reinstall vLLM.
#
# Installs: VERL dependencies (without flash-attn) plus a prebuilt flash-attn wheel.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
VERL_DIR="${ROOT_DIR}/critique-grpo/verl"

# ── 1. vLLM (installed by scripts/setup/install.sh --vllm) ───────────
echo "[1/3] Checking vLLM..."
python3 -c "import vllm; print(f'vLLM {vllm.__version__} already installed, skipping.')" 2>/dev/null \
    || { echo "vLLM not found. Run: bash scripts/setup/install.sh --vllm"; exit 1; }

# ── 2. VERL dependencies (flash-attn excluded — installed separately) ──
echo "[2/3] Installing VERL dependencies..."
grep -v "flash.attn\|flash_attn" "${VERL_DIR}/requirements.txt" | pip install -r /dev/stdin
pip install -e "${VERL_DIR}" --no-deps
pip install word2number math_verify

# ── 3. flash-attn prebuilt wheel ──────────────────────────────────────
echo "[3/3] Installing flash-attn prebuilt wheel..."
CUDA_TAG="cu$(python3 -c "import torch; print(torch.version.cuda.replace('.',''))")"
TORCH_TAG="torch$(python3 -c "import torch; v=torch.__version__.split('+')[0]; parts=v.split('.'); print('.'.join(parts[:2]))")"
PYTHON_TAG="cp$(python3 -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')")"

echo "Detected: ${CUDA_TAG}, ${TORCH_TAG}, ${PYTHON_TAG}"

WHEEL_URL=$(curl -s https://api.github.com/repos/mjun0812/flash-attention-prebuild-wheels/releases \
    | python3 -c "
import json, sys
tag = '${CUDA_TAG}${TORCH_TAG}'
py  = '${PYTHON_TAG}'
releases = json.load(sys.stdin)
urls = [
    a['browser_download_url']
    for r in releases
    for a in r['assets']
    if tag in a['name'] and py in a['name'] and 'linux_x86_64' in a['name']
]
if not urls:
    raise SystemExit(f'No prebuilt wheel found for {tag} {py} linux_x86_64')
print(urls[0])
")

echo "Installing: ${WHEEL_URL}"
pip install "${WHEEL_URL}"

echo "Done. Run: bash scripts/finetune/gold_grpo_verl.sh"
