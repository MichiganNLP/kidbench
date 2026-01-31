#!/bin/bash
#SBATCH --account=mihalcea_owned1
#SBATCH --job-name=generate-responses
#SBATCH --mail-user=asamee@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=8g
#SBATCH --time=24:00:00
#SBATCH --partition=spgpu2
#SBATCH --output=/home/%u/%x-%j.log

set -euo pipefail

# Root directory (where you ran sbatch from)
ROOT_DIR="${SLURM_SUBMIT_DIR}"

# --- Singularity image ---
SIF="/scratch/mihalcea_root/mihalcea0/asamee/singularity/trace-verl.sif"

# --- Load singularity module (if your cluster uses modules) ---
module load singularity

# --- Inputs ---
DATASET_PATH="${ROOT_DIR}/dataset/child-safety.json"
PROMPT_TYPES=(
  "with_cues"
  "without_cues"
)
MAX_NEW_TOKENS=2048
TEMPERATURE=0.0
TOP_P=1.0
TENSOR_PARALLEL_SIZE=1
MAX_MODEL_LEN=8192
GPU_MEMORY_UTILIZATION=0.90

# --- Models (name -> path) ---
declare -A MODEL_PATHS=(
  ["Llama-3-8B"]="/scratch/mihalcea_root/mihalcea0/asamee/huggingface/models/Llama-3.1-8B-Instruct"
  ["Gemma-3-12B"]="/scratch/mihalcea_root/mihalcea0/asamee/huggingface/models/Gemma-3-12B-Instruct"
)

echo "ROOT_DIR=${ROOT_DIR}"
echo "DATASET_PATH=${DATASET_PATH}"
echo "SIF=${SIF}"

for MODEL_NAME in "${!MODEL_PATHS[@]}"; do
  MODEL_PATH="${MODEL_PATHS[$MODEL_NAME]}"

  for PROMPT_TYPE in "${PROMPT_TYPES[@]}"; do
    echo "============================================================"
    echo "Running:"
    echo "  MODEL_NAME=${MODEL_NAME}"
    echo "  PROMPT_TYPE=${PROMPT_TYPE}"
    echo "============================================================"

    singularity exec --nv --cleanenv \
      -B "${ROOT_DIR}:${ROOT_DIR}" \
      --env PYTHONUNBUFFERED=1 \
      --env VLLM_DISABLE_CUSTOM_ALLREDUCE=1 \
      --env NCCL_P2P_DISABLE=1 \
      "${SIF}" bash -lc "
        set -euo pipefail
        export VLLM_DISABLE_CUSTOM_ALLREDUCE=1
        export NCCL_P2P_DISABLE=1
        export VLLM_CONFIGURE_LOGGING=0
        python3 --version
        nvidia-smi || true

        python3 '${ROOT_DIR}/src/child-safety/generate_responses.py' \
          --model_name '${MODEL_NAME}' \
          --model_path '${MODEL_PATH}' \
          --dataset_name 'child-safety' \
          --dataset_path '${DATASET_PATH}' \
          --prompt_type '${PROMPT_TYPE}' \
          --max_new_tokens '${MAX_NEW_TOKENS}' \
          --tensor_parallel_size '${TENSOR_PARALLEL_SIZE}' \
          --max_model_len '${MAX_MODEL_LEN}' \
          --gpu_memory_utilization '${GPU_MEMORY_UTILIZATION}' \
          --temperature '${TEMPERATURE}' \
          --top_p '${TOP_P}'
      "
  done
done

echo "All runs completed."
