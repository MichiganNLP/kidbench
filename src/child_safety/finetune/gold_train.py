#!/usr/bin/env python3
"""
Fine-tunes Llama 3 on gold responses using Unsloth + LoRA.
No evaluation — just train and save a checkpoint after each epoch.
"""

import unsloth  # must be first
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*right-padding was detected.*")

import torch
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


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

class LossCallback(TrainerCallback):
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs) -> None:
        if logs and "loss" in logs:
            record = {"step": state.global_step, "loss": logs["loss"]}
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
    io.add_argument("--model_name",  required=True)
    io.add_argument("--train_path",  required=True)
    io.add_argument("--output_dir",  required=True)
    io.add_argument("--log_dir",     default="finetuning_logs/gold")
    io.add_argument("--hf_repo",     required=True, help="HuggingFace repo, e.g. sameearif/LlamaPlushie-3-8B-Instruct")

    lora = p.add_argument_group("lora")
    lora.add_argument("--lora_r",       type=int,   default=16)
    lora.add_argument("--lora_alpha",   type=int,   default=32)
    lora.add_argument("--lora_dropout", type=float, default=0.0)

    train_g = p.add_argument_group("training")
    train_g.add_argument("--max_seq_len",    type=int,   default=2048)
    train_g.add_argument("--batch_size",     type=int,   default=2)
    train_g.add_argument("--grad_accum",     type=int,   default=4)
    train_g.add_argument("--lr",             type=float, default=2e-4)
    train_g.add_argument("--epochs",         type=int,   default=3)
    train_g.add_argument("--warmup_ratio",   type=float, default=0.05)
    train_g.add_argument("--weight_decay",   type=float, default=0.01)
    train_g.add_argument("--seed",           type=int,   default=42)
    train_g.add_argument("--logging_steps",  type=int,   default=25)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Logs → {log_dir}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_len,
        dtype=torch.bfloat16,
        load_in_4bit=False,
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
    print("Training done.")


if __name__ == "__main__":
    main()
