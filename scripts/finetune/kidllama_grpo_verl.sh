#!/bin/bash
# Stage 2 of KIDLlama: Critique-GRPO fine-tuning with the VERL backend.
# This is the configuration used in the paper.
#
# The guard model (KIDGuardLlama) is served with vLLM as an OpenAI-compatible
# reward server; VERL then runs Critique-GRPO against it and, at the end,
# merges each epoch checkpoint and pushes it to the Hub.
#
# GPU layout is detected automatically:
#   4× GPU: 2 judge replicas (GPU 0+1, ports 8001+8002) + VERL on all 4 GPUs
#   2× GPU: 1 judge (GPU 0, port 8001) + VERL on both GPUs
#   1× GPU: 1 judge (GPU 0) + VERL on GPU 0 (shared)
#
# Setup: bash scripts/setup/install.sh --vllm && bash scripts/setup/install_verl.sh
# Export HF_TOKEN before running (see .env.example).
set -euo pipefail

# ── 0. Clean up any stale Ray cluster / GPU processes from prior runs ─────
echo "[0/4] Stopping stale Ray cluster and GPU processes..."
ray stop --force 2>/dev/null || true
# Give CUDA driver time to release memory before starting fresh
sleep 3

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
VERL_DIR="${ROOT_DIR}/critique-grpo/verl"

: "${HF_TOKEN:?Set HF_TOKEN in your environment (see .env.example)}"

# Policy: the selected KIDLlama SFT checkpoint.
POLICY_MODEL="${POLICY_MODEL:-sameearif/KIDLlama-SFT-Epoch-2}"
# Reward model: the selected KIDGuardLlama checkpoint.
JUDGE_PATH="${JUDGE_PATH:-sameearif/KIDGuardLlama-SFT-Epoch-2}"
TRAIN_PARQUET="${ROOT_DIR}/dataset/gold/grpo_train.parquet"

JUDGE_PORT=8001
JUDGE_PORT_2=8002
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
if [ "${NUM_GPUS}" -ge 4 ]; then
    JUDGE_GPUS="0 1 2 3"   # 4 judge replicas, one per GPU
    TRAIN_GPU=0,1,2,3
    JUDGE_MEM=0.30
    N_GPUS=4
    TP_SIZE=1
    PPO_MICRO_BATCH=8
    TRAIN_BATCH_SIZE=64
    PPO_MINI_BATCH=64
elif [ "${NUM_GPUS}" -ge 2 ]; then
    JUDGE_GPUS="0 1"
    TRAIN_GPU=0,1
    JUDGE_MEM=0.30
    N_GPUS=2
    TP_SIZE=1
    PPO_MICRO_BATCH=8
    TRAIN_BATCH_SIZE=32
    PPO_MINI_BATCH=32
else
    JUDGE_GPUS="0"
    TRAIN_GPU=0
    JUDGE_MEM=0.30
    N_GPUS=1
    TP_SIZE=1
    PPO_MICRO_BATCH=1
    TRAIN_BATCH_SIZE=8
    PPO_MINI_BATCH=8
fi
HUB_REPO="${HUB_REPO:-sameearif/KIDLlama-GRPO}"

echo "============================================================"
echo "Policy:  ${POLICY_MODEL}"
echo "Judge:   ${JUDGE_PATH}"
echo "Dataset: ${TRAIN_PARQUET}"
echo "============================================================"

# ── 1. Build dataset ─────────────────────────────────────────────
if [ ! -f "${TRAIN_PARQUET}" ]; then
    echo "[1/3] Building parquet dataset..."
    bash "${ROOT_DIR}/scripts/dataset/build_grpo_dataset.sh"
else
    echo "[1/3] Dataset already exists at ${TRAIN_PARQUET}, skipping."
fi

# ── 2. Start judge server(s) ──────────────────────────────────────
JUDGE_PIDS=()
JUDGE_URLS=""
PORT=${JUDGE_PORT}
for GPU in ${JUDGE_GPUS}; do
    echo "[2/3] Starting judge on GPU ${GPU}, port ${PORT} (mem=${JUDGE_MEM})..."
    CUDA_VISIBLE_DEVICES="${GPU}" \
    HF_TOKEN="${HF_TOKEN}" \
    HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}" \
    VLLM_LOGGING_LEVEL=WARNING \
    vllm serve "${JUDGE_PATH}" \
        --port "${PORT}" \
        --gpu-memory-utilization "${JUDGE_MEM}" \
        --tensor-parallel-size 1 \
        --max-model-len 8192 \
        --dtype bfloat16 \
        --enforce-eager \
        --served-model-name "${JUDGE_PATH}" \
        --uvicorn-log-level error &
    JUDGE_PIDS+=($!)
    JUDGE_URLS="${JUDGE_URLS:+${JUDGE_URLS},}http://localhost:${PORT}/v1"
    PORT=$((PORT + 1))
done

# Wait until all judge servers are ready
echo "Waiting for judge server(s)..."
for CHECK_PORT in $(seq ${JUDGE_PORT} $((PORT - 1))); do
    for i in $(seq 1 60); do
        if curl -sf "http://localhost:${CHECK_PORT}/health" > /dev/null 2>&1; then
            echo "Judge on port ${CHECK_PORT} ready."
            break
        fi
        sleep 5
    done
done

# ── 3. Run VERL training ──────────────────────────────────────────
echo "[3/3] Starting Critique-GRPO training..."
cleanup() {
    echo "Stopping judge server(s)..."
    for PID in "${JUDGE_PIDS[@]}"; do kill "${PID}" 2>/dev/null || true; done
}
trap cleanup EXIT

cd "${VERL_DIR}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-llamaplushie3_critique_grpo}"
CKPT_DIR="${VERL_DIR}/checkpoints/child_safety_grpo/${EXPERIMENT_NAME}"

CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" \
PYTHONPATH="${VERL_DIR}:${ROOT_DIR}/src" \
PYTHONUNBUFFERED=1 \
HF_TOKEN="${HF_TOKEN}" \
HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}" \
JUDGE_SERVER_URLS="${JUDGE_URLS}" \
JUDGE_MODEL="${JUDGE_PATH}" \
VLLM_LOGGING_LEVEL=WARNING \
VERL_LOGGING_LEVEL=ERROR \
TRANSFORMERS_VERBOSITY=error \
PYTHONWARNINGS=ignore \
python3 -m verl.critique_grpo.critique_main_ppo \
    --config-name child_safety_grpo \
    data.train_files="${TRAIN_PARQUET}" \
    data.val_files="${TRAIN_PARQUET}" \
    actor_rollout_ref.model.path="${POLICY_MODEL}" \
    critic.model.path="${POLICY_MODEL}" \
    critic.model.tokenizer_path="${POLICY_MODEL}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node="${N_GPUS}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${TP_SIZE}" \
    critic.ppo_mini_batch_size="${PPO_MINI_BATCH}" \
    critic.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH}" \
    trainer.total_epochs=1

echo "Training done."

# ── 4. Merge LoRA + push each epoch checkpoint to Hub ─────────────
echo "[4/4] Merging LoRA adapters and pushing epoch checkpoints to Hub..."

# Collect global_step_* dirs sorted numerically (oldest = epoch 1 first)
EPOCH_ACTORS=()
while IFS= read -r step_dir; do
    actor_dir="${step_dir}/actor"
    [ -d "${actor_dir}" ] && EPOCH_ACTORS+=("${actor_dir}")
done < <(ls -d "${CKPT_DIR}/global_step_"* 2>/dev/null | sort -t_ -k3 -n)

if [ ${#EPOCH_ACTORS[@]} -eq 0 ]; then
    echo "WARNING: no actor checkpoints found at ${CKPT_DIR} — skipping Hub push."
else
    EPOCH_NUM=0
    for ACTOR_CKPT in "${EPOCH_ACTORS[@]}"; do
        EPOCH_NUM=$((EPOCH_NUM + 1))
        EPOCH_REPO="${HUB_REPO}-${EPOCH_NUM}"
        echo "  Epoch ${EPOCH_NUM}: ${ACTOR_CKPT} → ${EPOCH_REPO}"
        HF_TOKEN="${HF_TOKEN}" \
        HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}" \
        python3 - <<PYEOF
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

ckpt = Path("${ACTOR_CKPT}")
base = "${POLICY_MODEL}"
repo = "${EPOCH_REPO}"

print(f"Loading base model: {base}")
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map="cpu")
print(f"Loading LoRA adapter from: {ckpt}")
model = PeftModel.from_pretrained(model, str(ckpt))
print("Merging adapter weights...")
model = model.merge_and_unload()
print(f"Pushing to Hub: {repo}")
model.push_to_hub(repo, private=False)
tokenizer = AutoTokenizer.from_pretrained(base)
tokenizer.push_to_hub(repo, private=False)
print("Done.")
PYEOF
    done
fi
