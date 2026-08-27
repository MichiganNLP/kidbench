#!/bin/bash
# Stage 1 of KIDLlama: supervised fine-tuning on the gold responses
# (Unsloth + LoRA).
#
# Trains on dataset/gold/train.jsonl — the teacher responses that scored 5/5 on
# every child-safety dimension (build it with
# scripts/dataset/build_gold_dataset.sh). A merged bfloat16 checkpoint is pushed
# to the Hub after each epoch; the paper selects epoch 2 to initialise
# Critique-GRPO.
#
# Paper configuration (Table 18): Llama-3.1-8B-Instruct base, r=16, alpha=32,
# lr=2e-4, cosine schedule, 3 epochs, effective batch 16, max_seq_len 2048.
#
# Export HF_TOKEN before running (see .env.example).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

: "${HF_TOKEN:?Set HF_TOKEN in your environment (see .env.example)}"

# Base model per the paper (Table 18). Earlier development runs used
# meta-llama/Meta-Llama-3-8B-Instruct.
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}"
TRAIN_PATH="${ROOT_DIR}/dataset/gold/train.jsonl"
OUTPUT_DIR="${ROOT_DIR}/finetuned_models/gold"
LOG_DIR="${ROOT_DIR}/finetuning_logs/gold"
# Destination for the per-epoch checkpoints — change this to your own namespace.
HF_REPO="${HF_REPO:-sameearif/KIDLlama-SFT-Epoch}"

# LoRA
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.0

# Training
MAX_SEQ_LEN=2048
BATCH_SIZE=8
GRAD_ACCUM=2        # effective batch = 8
LR=2e-4
EPOCHS=3
WARMUP_RATIO=0.05
WEIGHT_DECAY=0.01
SEED=42
LOGGING_STEPS=25

echo "============================================================"
echo "Model:      ${MODEL_NAME}"
echo "Train:      ${TRAIN_PATH}"
echo "Output:     ${OUTPUT_DIR}"
echo "Eff. batch: $((BATCH_SIZE * GRAD_ACCUM))"
echo "Epochs:     ${EPOCHS}"
echo "============================================================"

PYTHONPATH="${ROOT_DIR}/src" \
PYTHONUNBUFFERED=1 \
PYTHONWARNINGS="ignore::UserWarning" \
HF_TOKEN="${HF_TOKEN}" \
HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}" \
python3 -m child_safety.finetune.gold_train \
    --model_name     "${MODEL_NAME}" \
    --train_path     "${TRAIN_PATH}" \
    --output_dir     "${OUTPUT_DIR}" \
    --log_dir        "${LOG_DIR}" \
    --lora_r         "${LORA_R}" \
    --lora_alpha     "${LORA_ALPHA}" \
    --lora_dropout   "${LORA_DROPOUT}" \
    --max_seq_len    "${MAX_SEQ_LEN}" \
    --batch_size     "${BATCH_SIZE}" \
    --grad_accum     "${GRAD_ACCUM}" \
    --lr             "${LR}" \
    --epochs         "${EPOCHS}" \
    --warmup_ratio   "${WARMUP_RATIO}" \
    --weight_decay   "${WEIGHT_DECAY}" \
    --seed           "${SEED}" \
    --logging_steps  "${LOGGING_STEPS}" \
    --hf_repo        "${HF_REPO}"

echo "Training done."
