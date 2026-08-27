#!/bin/bash
# Stage 2 of KIDLlama (single-GPU reference implementation): Critique-GRPO with
# KIDGuardLlama as the reward model, implemented directly on top of Unsloth.
#
# Per batch: sample N responses per prompt from the policy, score each with the
# guard model, ask the policy to revise the low-scoring ones using the guard's
# improvement critique, score the revisions, then apply a PPO-clipped update
# with group-normalised advantages.
#
# This version is simple and self-contained but slow. For the configuration used
# in the paper, use scripts/finetune/kidllama_grpo_verl.sh instead.
#
# Export HF_TOKEN before running (see .env.example).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

: "${HF_TOKEN:?Set HF_TOKEN in your environment (see .env.example)}"

# Policy: the selected KIDLlama SFT checkpoint.
MODEL_NAME="${MODEL_NAME:-sameearif/LlamaPlushie-3-8B-2}"
# Reward model: the selected KIDGuardLlama checkpoint.
JUDGE_PATH="${JUDGE_PATH:-sameearif/LlamaSproutGuard-3-8B-2}"
TRAIN_PATH="${ROOT_DIR}/dataset/gold/train.jsonl"
OUTPUT_DIR="${ROOT_DIR}/finetuned_models/grpo"
LOG_DIR="${ROOT_DIR}/finetuning_logs/grpo"
HF_REPO="${HF_REPO:-sameearif/LlamaPlushie-GRPO-3-8B}"

# LoRA
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.0

# Training
MAX_SEQ_LEN=2048
BATCH_SIZE=4        # prompts per update step
N_GEN=8             # responses generated per prompt
LR=1e-6
EPOCHS=2
WARMUP_RATIO=0.05
CLIP_EPS=0.2
SEED=42

# Generation
TEMPERATURE=0.9
MAX_NEW_TOKENS=1024
CRITIQUE_THRESHOLD=0.6   # reward below this triggers critique refinement

# Judge
JUDGE_MAX_SEQ_LEN=4096
JUDGE_MAX_NEW_TOKENS=512
JUDGE_BATCH_SIZE=4

echo "============================================================"
echo "Policy:     ${MODEL_NAME}"
echo "Judge:      ${JUDGE_PATH}"
echo "Train:      ${TRAIN_PATH}"
echo "n_gen:      ${N_GEN}  |  batch: ${BATCH_SIZE}  |  epochs: ${EPOCHS}"
echo "============================================================"

PYTHONPATH="${ROOT_DIR}/src" \
PYTHONUNBUFFERED=1 \
PYTHONWARNINGS="ignore::UserWarning" \
HF_TOKEN="${HF_TOKEN}" \
HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}" \
python3 -m child_safety.finetune.gold_train_grpo \
    --model_name            "${MODEL_NAME}" \
    --judge_path            "${JUDGE_PATH}" \
    --train_path            "${TRAIN_PATH}" \
    --output_dir            "${OUTPUT_DIR}" \
    --log_dir               "${LOG_DIR}" \
    --hf_repo               "${HF_REPO}" \
    --lora_r                "${LORA_R}" \
    --lora_alpha            "${LORA_ALPHA}" \
    --lora_dropout          "${LORA_DROPOUT}" \
    --max_seq_len           "${MAX_SEQ_LEN}" \
    --batch_size            "${BATCH_SIZE}" \
    --n_gen                 "${N_GEN}" \
    --lr                    "${LR}" \
    --epochs                "${EPOCHS}" \
    --warmup_ratio          "${WARMUP_RATIO}" \
    --clip_eps              "${CLIP_EPS}" \
    --seed                  "${SEED}" \
    --temperature           "${TEMPERATURE}" \
    --max_new_tokens        "${MAX_NEW_TOKENS}" \
    --critique_threshold    "${CRITIQUE_THRESHOLD}" \
    --judge_max_seq_len     "${JUDGE_MAX_SEQ_LEN}" \
    --judge_max_new_tokens  "${JUDGE_MAX_NEW_TOKENS}" \
    --judge_batch_size      "${JUDGE_BATCH_SIZE}"

echo "Training done."
