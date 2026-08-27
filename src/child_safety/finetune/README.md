# `child_safety.finetune`

Dataset construction, training, and evaluation for the two models introduced in the paper:

- **KIDGuardLlama** — a Llama-3.1-8B guard model trained to reproduce DeepSeek-V4-Pro's child-safety judgments, cheaply enough to use as a reward model.
- **KIDLlama** — a Llama-3.1-8B response model trained to answer children well: SFT on gold responses, then Critique-GRPO with KIDGuardLlama as the reward signal.

| Module | Role | Script |
|---|---|---|
| `gold_dataset.py` | Build KIDLlama's SFT data from `gold_responses/` | `scripts/dataset/build_gold_dataset.sh` |
| `judge_dataset.py` | Build KIDGuardLlama's SFT data from `evaluations/` | `scripts/dataset/build_judge_dataset.sh` |
| `make_grpo_dataset.py` | Convert the SFT data to VERL parquet | `scripts/dataset/build_grpo_dataset.sh` |
| `gold_train.py` | KIDLlama stage 1 (SFT) | `scripts/finetune/kidllama_sft.sh` |
| `gold_train_grpo.py` | KIDLlama stage 2 (Critique-GRPO, Unsloth reference implementation) | `scripts/finetune/kidllama_grpo_unsloth.sh` |
| `gold_test.py` | KIDLlama inference over the held-out test set | `scripts/finetune/kidllama_eval.sh` |
| `judge_train.py` | KIDGuardLlama SFT | `scripts/finetune/judge_sft.sh` |
| `judge_test.py` | KIDGuardLlama agreement metrics | `scripts/finetune/judge_eval.sh` |

The paper's Critique-GRPO run uses the VERL backend driven by `scripts/finetune/kidllama_grpo_verl.sh` and the `critique-grpo/` submodule; `gold_train_grpo.py` is a readable single-GPU implementation of the same algorithm.

> The module names use the internal terms *gold* (the teacher responses KIDLlama is trained on) and *judge* (KIDGuardLlama). Output directories under `responses/` and `evaluations/` use the original checkpoint names — see the mapping table in the top-level README.

---

## Dataset construction

### `gold_dataset.py` → `dataset/gold/`

Reads `gold_responses/single_turn/{teacher}/{cues}/*.json` and keeps **only turns where all five metrics scored 5**. That is the entire quality filter: a revision that still had a 4 anywhere is dropped, and a prompt whose first attempt was already perfect contributes that attempt. Non-English files are skipped. Result: 22,097 training examples.

Training records carry no system prompt, so KIDLlama learns child-appropriate behavior as a default rather than something a prompt has to request:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

For country files the user message is prefixed with `## Country:\n{Country}\n\n## User Message:\n…`, matching the format the model will see at evaluation time.

The test split holds out 5 items per category (seed 42), expanded into the 7 single-turn conditions each — 350 records — and carries `prompt`, `category`, `condition`, `age`, and `country` fields so `gold_test.py` can reconstruct every condition. Held-out prompts are excluded from training in both their `with_cues` and `without_cues` forms.

### `judge_dataset.py` → `dataset/judge/`

Turns every evaluation collected so far into judging-format training data: single-turn evaluations from `evaluations/*/single_turn/`, the judge's scores from the gold-response loop (including the low-scoring first attempts — the guard model needs to see bad responses), and multi-turn evaluations as full multi-turn chats. Result: 67,899 examples.

The held-out split is 4 single-turn items and 2 multi-turn scenarios per category (seed 42), giving 6,224 test records. Both splits use the identical record format, so the test set measures the model's actual judging behavior.

`train.jsonl` is not committed (267 MB, over GitHub's file limit) — regenerate it with `scripts/dataset/build_judge_dataset.sh`.

### `make_grpo_dataset.py` → `dataset/gold/grpo_train.parquet`

Reformats `dataset/gold/train.jsonl` into the columns VERL expects (`prompt`, `reward_model`, `data_source`, `target`, `extra_info`), attaching the child-safety system prompt used during rollout.

---

## Training

### `judge_train.py` — KIDGuardLlama

Unsloth + LoRA SFT with loss on assistant turns only (`train_on_responses_only`). A merged bfloat16 checkpoint is pushed to the Hub at the end of each epoch, so every epoch can be compared on the test set; the paper selects epoch 2. Loss is logged to `{log_dir}/{model}/train_loss.jsonl` and plotted at the end.

The module also contains the scoring utilities used for validation: `parse_scores`, and `compute_metrics`, which reports Spearman ρ, quadratic weighted κ, MAE, exact and within-1 accuracy on the raw 1–5 scale, plus macro-F1 and accuracy on a coarse 3-way remapping (1–2 → 0, 3–4 → 1, 5 → 2).

### `gold_train.py` — KIDLlama stage 1

The same Unsloth + LoRA setup over `dataset/gold/train.jsonl`, 3 epochs, one merged checkpoint per epoch.

### `gold_train_grpo.py` — KIDLlama stage 2

Critique-GRPO, per batch:

1. Sample `--n_gen` responses per prompt from the policy at `--temperature`.
2. Score each with KIDGuardLlama; reward is the mean of the five metrics rescaled from [1, 5] to [0, 1].
3. For responses below `--critique_threshold`, build a critique prompt from the guard's `improvement` string and generate one revision.
4. Score the revisions too.
5. Compute advantages by normalizing rewards within each prompt group — originals and their revisions belong to the same group.
6. Apply a PPO-clipped policy-gradient update over all responses.

The guard model is loaded in 4-bit and frozen. Step-level loss, mean reward, and critique counts are appended to `{log_dir}/train_loss.jsonl`.

---

## Evaluation

### `gold_test.py`

Generates KIDLlama responses for all 8 single-turn conditions over `dataset/gold/test_single.jsonl`, writing them into `responses/single_turn/{model}/…` in exactly the layout normal generation produces — so the standard judge and analysis scripts apply with no changes.

### `judge_test.py`

Scores KIDGuardLlama against the held-out judge test set. Multi-turn records are expanded into one sample per turn under **teacher forcing**: turn *t* is predicted with the *gold* evaluations of turns 1…*t−1* as context, so an early mistake does not cascade and each turn is measured independently.

Reports overall and per-metric agreement with DeepSeek-V4-Pro, along with the parse-failure count, and writes `predictions.jsonl` plus `metrics.json` under `responses/judge/{model}/`. `scripts/finetune/judge_eval.sh` runs this for each epoch checkpoint; `child_safety.analysis.generate_guard_analysis` turns the results into a comparison table.
