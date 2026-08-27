# `scripts/`

Runnable entry points for every stage. Each script is a thin wrapper that sets the arguments used in the paper and calls a module from `src/child_safety/` — read the top of a script to see exactly what it runs, and edit the variable block to change models or hyperparameters.

Scripts resolve paths relative to the repository root, so they can be run from anywhere:

```bash
bash scripts/generation/single_turn/openai.sh
```

Load your API keys first:

```bash
cp .env.example .env    # fill it in
set -a && source .env && set +a
```

Every script that calls a model is **resumable** — re-running one continues from wherever it stopped and skips completed work.

---

## Directory map

```
scripts/
├── setup/         install dependencies, download model weights
├── dataset/       build the derived training/test splits
├── generation/    produce model responses
│   ├── single_turn/   one prompt → one response, all conditions
│   ├── multi_turn/    5-turn child-actor simulations
│   └── gold/          teacher critique-revise loop (KIDLlama training data)
├── evaluation/    score responses with the LLM judge
│   ├── single_turn/   the main single-turn judging run
│   ├── multi_turn/    conversation-level judging
│   └── sanity_check/  judge-selection study across 7 candidate judges
├── finetune/      train and evaluate KIDLlama and KIDGuardLlama
└── analysis/      regenerate every table and figure
```

Within `generation/` and `evaluation/`, the file name is the **provider**: `openai.sh`, `anthropic.sh`, `google.sh`, `deepseek.sh` call hosted APIs, and `vllm.sh` runs open-weight models locally.

---

## Typical run order

```bash
# 0. setup
bash scripts/setup/install.sh --vllm
MODELS_DIR=~/models bash scripts/setup/download_models.sh

# 1. generate responses
bash scripts/generation/single_turn/vllm.sh
bash scripts/generation/multi_turn/vllm.sh

# 2. judge them
bash scripts/evaluation/single_turn/deepseek.sh
bash scripts/evaluation/multi_turn/deepseek.sh

# 3. analyze
bash scripts/analysis/run_all.sh

# 4. build training data (needs stage 1 output for the prompts)
bash scripts/generation/gold/vllm.sh
bash scripts/dataset/build_gold_dataset.sh
bash scripts/dataset/build_judge_dataset.sh

# 5. train
bash scripts/finetune/judge_sft.sh
bash scripts/finetune/judge_eval.sh
bash scripts/finetune/kidllama_sft.sh
bash scripts/dataset/build_grpo_dataset.sh
bash scripts/finetune/kidllama_grpo_verl.sh
```

---

## Conventions

**Choosing models.** API generation scripts set `MODEL_NAME` and `MODEL_PATH` near the top — one model per script, so run the script once per model. vLLM scripts use a `MODEL_PATHS` associative array and loop over all of them. Evaluation scripts take a `MODELS` list of folder names under `responses/`.

**Where model weights live.** Local paths default to `${MODELS_DIR:-$HOME/models}`. Override per run:

```bash
MODELS_DIR=/data/models bash scripts/generation/single_turn/vllm.sh
```

Any `MODEL_PATHS` entry can also be a Hugging Face repo id instead of a directory.

**GPUs.** `vllm.sh` scripts expose `TENSOR_PARALLEL_SIZE`, `MAX_MODEL_LEN`, and `GPU_MEMORY_UTILIZATION`. Multi-turn scripts additionally pin the actor and the responder to separate devices via `ATTACKER_CUDA_DEVICE` and `GENERATOR_CUDA_DEVICE`; with one GPU set both to `0` and lower `GPU_MEMORY_UTILIZATION` so both models fit. `--language_model_only` (listed per model in `LM_ONLY_MODELS`) is required for text-only checkpoints of multimodal architectures.

**Secrets.** No script contains a credential. Fine-tuning scripts fail immediately with a clear message if `HF_TOKEN` is unset; provider scripts read the relevant `*_API_KEY` from the environment.

**Hub destinations.** Training scripts push a merged checkpoint per epoch. Point them at your own namespace before running:

```bash
HF_REPO=your-org/KIDGuardLlama-3-8B bash scripts/finetune/judge_sft.sh
```
