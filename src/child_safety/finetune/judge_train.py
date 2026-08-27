#!/usr/bin/env python3
"""
Fine-tunes an LLM judge for child safety evaluation using Unsloth + LoRA/QLoRA.

Metrics evaluated on validation set:
  Raw (1–5): Spearman, Quadratic Weighted Kappa, MAE, Exact Accuracy, Within-1 Accuracy
  Remapped (0/1/2: 1-2→0, 3-4→1, 5→2): Macro-F1, Accuracy

Logs saved to --log_dir:
  metrics.jsonl     — one line per eval step
  train_loss.jsonl  — one line per logging step
  metrics_plot.png  — all eval metrics over steps
  loss_plot.png     — training loss over steps
"""

import unsloth  # must be first — patches transformers/peft before they load
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only

import argparse
import json
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

warnings.filterwarnings("ignore", message=".*right-padding was detected.*")

import numpy as np
import torch
from datasets import Dataset
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, mean_absolute_error
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from trl import SFTConfig, SFTTrainer

METRICS = ["safety", "developmental_fit", "emotional_support", "moral_guidance", "boundary_setting"]
LOWER_IS_BETTER = {"mae"}

EVAL_METRIC_LABELS = {
    "spearman":      "Spearman",
    "qwk":           "QWK",
    "mae":           "MAE",
    "exact_acc":     "Exact Acc",
    "within1_acc":   "Within-1 Acc",
    "macro_f1_remap": "Macro-F1 (remapped)",
    "acc_remap":     "Accuracy (remapped)",
    "parse_rate":    "Parse Rate",
}


# ---------------------------------------------------------------------------
# Score utils
# ---------------------------------------------------------------------------

def remap(v: int) -> int:
    if v <= 2:
        return 0
    if v <= 4:
        return 1
    return 2


def parse_scores(text: str) -> Optional[Dict[str, int]]:
    if not text:
        return None
    try:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        data = json.loads(m.group(1) if m else text)
    except Exception:
        return None
    scores: Dict[str, int] = {}
    for metric in METRICS:
        v = data.get(metric)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            scores[metric] = max(1, min(5, int(round(float(v)))))
    return scores if len(scores) == len(METRICS) else None


def compute_metrics(
    preds: List[Optional[Dict[str, int]]],
    golds: List[Optional[Dict[str, int]]],
) -> Dict[str, float]:
    pred_flat, gold_flat = [], []
    for p, g in zip(preds, golds):
        if p is None or g is None:
            continue
        for m in METRICS:
            if m in p and m in g:
                pred_flat.append(p[m])
                gold_flat.append(g[m])

    parse_rate = sum(1 for p in preds if p is not None) / max(len(preds), 1)
    if not pred_flat:
        return {"parse_rate": parse_rate}

    pred_arr = np.array(pred_flat)
    gold_arr = np.array(gold_flat)

    spearman = float(spearmanr(gold_arr, pred_arr).statistic)
    qwk = float(cohen_kappa_score(gold_arr, pred_arr, weights="quadratic", labels=[1, 2, 3, 4, 5]))
    mae = float(mean_absolute_error(gold_arr, pred_arr))
    exact_acc = float(np.mean(gold_arr == pred_arr))
    within1_acc = float(np.mean(np.abs(gold_arr - pred_arr) <= 1))

    pred_remap = np.array([remap(v) for v in pred_flat])
    gold_remap = np.array([remap(v) for v in gold_flat])
    macro_f1 = float(f1_score(gold_remap, pred_remap, average="macro", labels=[0, 1, 2], zero_division=0))
    remap_acc = float(accuracy_score(gold_remap, pred_remap))

    return {
        "spearman":       spearman,
        "qwk":            qwk,
        "mae":            mae,
        "exact_acc":      exact_acc,
        "within1_acc":    within1_acc,
        "macro_f1_remap": macro_f1,
        "acc_remap":      remap_acc,
        "parse_rate":     parse_rate,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_metrics_plot(metrics_log: List[Dict], log_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not metrics_log:
        return

    steps = [r["step"] for r in metrics_log]
    metric_keys = [k for k in EVAL_METRIC_LABELS if k in metrics_log[0]]

    n_cols = 4
    n_rows = (len(metric_keys) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
    axes = axes.flatten()

    for i, key in enumerate(metric_keys):
        vals = [r.get(key) for r in metrics_log]
        axes[i].plot(steps, vals, marker="o", linewidth=1.5, markersize=3)
        axes[i].set_title(EVAL_METRIC_LABELS[key], fontsize=10)
        axes[i].set_xlabel("Step", fontsize=8)
        axes[i].grid(True, alpha=0.3)

    for j in range(len(metric_keys), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Validation Metrics", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(log_dir / "metrics_plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved metrics plot → {log_dir / 'metrics_plot.png'}")


def save_loss_plot(loss_log: List[Dict], log_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not loss_log:
        return

    steps = [r["step"] for r in loss_log]
    losses = [r["loss"] for r in loss_log]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, losses, linewidth=1.5)
    ax.set_title("Training Loss", fontsize=13, fontweight="bold")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(log_dir / "loss_plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved loss plot → {log_dir / 'loss_plot.png'}")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> Dataset:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return Dataset.from_list(records)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class EvalCallback(TrainerCallback):
    def __init__(self, val_dataset: Dataset, tokenizer, model, args: argparse.Namespace, log_dir: Path):
        self.val_dataset = val_dataset
        self.tokenizer = tokenizer
        self.model = model
        self.args = args
        self.log_dir = log_dir
        self.metrics_log: List[Dict] = []
        self.best_value = float("inf") if args.best_metric in LOWER_IS_BETTER else float("-inf")
        self.best_step = -1
        self.best_dir = Path(args.output_dir) / "best"

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs) -> None:
        if state.global_step > 0 and state.global_step % self.args.eval_steps == 0:
            self._run_eval(state.global_step)

    def _set_padding_side(self, side: str) -> None:
        self.tokenizer.padding_side = side
        # FastLanguageModel wraps a processor; set on the inner tokenizer too
        inner = getattr(self.tokenizer, "tokenizer", None)
        if inner is not None:
            inner.padding_side = side

    def _run_eval(self, step: int) -> None:
        orig_padding_side = self.tokenizer.padding_side
        self._set_padding_side("left")
        FastLanguageModel.for_inference(self.model)

        n = min(len(self.val_dataset), self.args.eval_samples)
        examples = self.val_dataset.select(range(n))
        preds: List[Optional[Dict[str, int]]] = []
        golds: List[Optional[Dict[str, int]]] = []

        print(f"\n[Step {step}] Running eval on {n} examples...", flush=True)
        n_batches = (n + self.args.eval_batch_size - 1) // self.args.eval_batch_size
        for i in tqdm(range(0, n, self.args.eval_batch_size), total=n_batches, desc=f"Eval step {step}", position=0, leave=True):
            batch_msgs = examples[i : i + self.args.eval_batch_size]["messages"]
            prompts = [
                self.tokenizer.apply_chat_template(
                    [m for m in msgs if m["role"] != "assistant"],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                for msgs in batch_msgs
            ]
            inputs = self.tokenizer(
                text=prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.args.max_seq_len,
            ).to(self.model.device)

            n_inp = inputs["input_ids"].shape[1]
            with torch.no_grad():
                out_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.args.eval_max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            for j, msgs in enumerate(batch_msgs):
                generated = self.tokenizer.decode(out_ids[j][n_inp:], skip_special_tokens=True)
                preds.append(parse_scores(generated))
                gold_text = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
                golds.append(parse_scores(gold_text))

        FastLanguageModel.for_training(self.model)
        self._set_padding_side(orig_padding_side)

        metrics = compute_metrics(preds, golds)
        record = {"step": step, **metrics}
        self.metrics_log.append(record)

        # Append to JSONL
        with open(self.log_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")

        # Best model tracking
        metric_val = metrics.get(self.args.best_metric)
        if metric_val is not None:
            lower = self.args.best_metric in LOWER_IS_BETTER
            is_best = metric_val < self.best_value if lower else metric_val > self.best_value
            if is_best:
                self.best_value = metric_val
                self.best_step = step
                self.best_dir.mkdir(parents=True, exist_ok=True)
                self.model.save_pretrained(self.best_dir)
                self.tokenizer.save_pretrained(self.best_dir)
                print(f"  ★ New best {self.args.best_metric}: {metric_val:.4f} — saved to {self.best_dir}")

        print(f"\n[Step {step}] Validation metrics (best {self.args.best_metric} @ step {self.best_step}: {self.best_value:.4f}):")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        print()

        try:
            import wandb
            if wandb.run is not None:
                wandb.log({f"eval/{k}": v for k, v in metrics.items()}, step=step)
                wandb.log({f"eval/best_{self.args.best_metric}": self.best_value}, step=step)
        except ImportError:
            pass


class LossCallback(TrainerCallback):
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.loss_log: List[Dict] = []

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs) -> None:
        if logs and "loss" in logs:
            record = {"step": state.global_step, "loss": logs["loss"]}
            self.loss_log.append(record)
            with open(self.log_dir / "train_loss.jsonl", "a") as f:
                f.write(json.dumps(record) + "\n")


class HFUploadCallback(TrainerCallback):
    def __init__(self, model, tokenizer, hf_repo: str):
        self.model = model
        self.tokenizer = tokenizer
        self.hf_repo = hf_repo

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs) -> None:
        epoch = int(state.epoch)
        repo = f"{self.hf_repo}-{epoch}"
        print(f"\nUploading epoch {epoch} → {repo} (merged bfloat16)...", flush=True)
        self.model.push_to_hub_merged(repo, self.tokenizer, save_method="merged_16bit", private=True)
        print(f"Uploaded → {repo}", flush=True)


# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    io = p.add_argument_group("io")
    io.add_argument("--model_name",  required=True, help="HuggingFace model id or local path")
    io.add_argument("--train_path",  required=True)
    io.add_argument("--output_dir",  required=True)
    io.add_argument("--log_dir",     default="finetuning_logs/judge",
                    help="Directory for metrics JSONL and plots")
    io.add_argument("--hf_repo",     required=True, help="HuggingFace repo, e.g. sameearif/QwenSproutGuard-3.5-4B")

    lora = p.add_argument_group("lora")
    lora.add_argument("--lora_r",       type=int,   default=16)
    lora.add_argument("--lora_alpha",   type=int,   default=32)
    lora.add_argument("--lora_dropout", type=float, default=0.0)

    train_g = p.add_argument_group("training")
    train_g.add_argument("--max_seq_len",    type=int,   default=4096)
    train_g.add_argument("--batch_size",     type=int,   default=2)
    train_g.add_argument("--grad_accum",     type=int,   default=4)
    train_g.add_argument("--lr",             type=float, default=2e-4)
    train_g.add_argument("--epochs",         type=int,   default=2)
    train_g.add_argument("--warmup_ratio",   type=float, default=0.05)
    train_g.add_argument("--weight_decay",   type=float, default=0.01)
    train_g.add_argument("--seed",           type=int,   default=42)

    eval_g = p.add_argument_group("eval / logging")
    eval_g.add_argument("--logging_steps", type=int, default=25)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Nest logs under a model-named subfolder
    model_safe = args.model_name.replace("/", "_")
    log_dir = Path(args.log_dir) / model_safe
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Logs → {log_dir}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_len,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        use_gradient_checkpointing="unsloth",
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    # Ensure all trainable LoRA params are in bfloat16
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.bfloat16)

    train_dataset = load_jsonl(args.train_path)

    def format_example(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        }

    train_dataset = train_dataset.map(format_example, remove_columns=train_dataset.column_names)

    loss_cb = LossCallback(log_dir)
    upload_cb = HFUploadCallback(model, tokenizer, args.hf_repo)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=SFTConfig(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            num_train_epochs=args.epochs,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            lr_scheduler_type="cosine",
            bf16=True,
            fp16=False,
            optim="paged_adamw_8bit",
            max_length=args.max_seq_len,
            dataset_num_proc=2,
            eval_strategy="no",
            save_strategy="epoch",
            logging_steps=args.logging_steps,
            seed=args.seed,
            report_to="none",
            dataset_text_field="text",
        ),
        callbacks=[loss_cb, upload_cb],
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
    )

    trainer.train()

    save_loss_plot(loss_cb.loss_log, log_dir)


if __name__ == "__main__":
    main()
