#!/usr/bin/env python3
"""
Critique GRPO fine-tuning for child safety using a fine-tuned judge as reward model.

Algorithm (per batch):
  1. Generate n_gen responses per prompt from the policy (temperature > 0)
  2. Score each (prompt, response) with the judge → reward [0,1] + improvement string
  3. Low-reward responses (< critique_threshold) get a critique refinement:
       build_critique_prompt(prompt, response, improvement) → generate 1 refined response
  4. Score refined responses
  5. GRPO group advantage: (reward - group_mean) / (group_std + eps) per prompt group
  6. PPO-clipped policy gradient loss over all responses (original + refined)
  7. Upload merged model to HF after each epoch

Reward:  mean(safety, developmental_fit, emotional_support, moral_guidance, boundary_setting)
         normalized from [1,5] → [0,1].
Critique: judge's "improvement" field, used to guide the refinement generation.
"""

import unsloth  # must be first
from unsloth import FastLanguageModel

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

ROOT = Path(__file__).parent.parent.parent.parent
SP_DIR = ROOT / "system_prompts"

METRICS = ["safety", "developmental_fit", "emotional_support", "moral_guidance", "boundary_setting"]


# ---------------------------------------------------------------------------
# Judge helpers
# ---------------------------------------------------------------------------

def _load_judge_system_prompt() -> str:
    path = SP_DIR / "evaluation" / "judge_gold_single.jinja"
    return path.read_text().strip()


def _build_judge_user_message(prompt: str, response: str) -> str:
    return f"## USER PROMPT:\n{prompt}\n\n## LLM RESPONSE:\n{response}"


def _parse_judge_output(text: str) -> Tuple[Optional[Dict[str, int]], str]:
    """Returns (scores_dict, improvement_string). scores_dict is None on parse failure."""
    if not text:
        return None, ""
    try:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        data = json.loads(m.group(1) if m else text)
    except Exception:
        return None, ""

    scores: Dict[str, int] = {}
    for metric in METRICS:
        v = data.get(metric)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            scores[metric] = max(1, min(5, int(round(float(v)))))

    improvement = data.get("improvement", "") or ""
    return (scores if len(scores) == len(METRICS) else None), improvement


def _scores_to_reward(scores: Dict[str, int]) -> float:
    """Normalize mean score from [1,5] to [0,1]."""
    return (sum(scores.values()) / len(scores) - 1.0) / 4.0


def judge_score_batch(
    judge,
    judge_tokenizer,
    system_prompt: str,
    prompts: List[str],
    responses: List[str],
    judge_max_seq_len: int,
    judge_max_new_tokens: int = 512,
    batch_size: int = 4,
) -> Tuple[List[float], List[str]]:
    """Score (prompt, response) pairs. Returns (rewards, improvements)."""
    rewards, improvements = [], []
    orig_side = judge_tokenizer.padding_side
    judge_tokenizer.padding_side = "left"

    for i in range(0, len(prompts), batch_size):
        bp = prompts[i : i + batch_size]
        br = responses[i : i + batch_size]

        judge_inputs = [
            judge_tokenizer.apply_chat_template(
                [
                    {"role": "system",    "content": system_prompt},
                    {"role": "user",      "content": _build_judge_user_message(p, r)},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p, r in zip(bp, br)
        ]

        enc = judge_tokenizer(
            judge_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=judge_max_seq_len,
        ).to(judge.device)

        n_inp = enc["input_ids"].shape[1]
        with torch.no_grad():
            out = judge.generate(
                **enc,
                max_new_tokens=judge_max_new_tokens,
                do_sample=False,
                pad_token_id=judge_tokenizer.eos_token_id,
            )

        for j in range(len(bp)):
            generated = judge_tokenizer.decode(out[j][n_inp:], skip_special_tokens=True)
            scores, improvement = _parse_judge_output(generated)
            rewards.append(_scores_to_reward(scores) if scores else 0.0)
            improvements.append(improvement)

    judge_tokenizer.padding_side = orig_side
    return rewards, improvements


# ---------------------------------------------------------------------------
# Critique prompt
# ---------------------------------------------------------------------------

def build_critique_prompt(original_prompt: str, original_response: str, improvement: str) -> str:
    return (
        f"A child asked:\n{original_prompt}\n\n"
        f"A previous response was:\n{original_response}\n\n"
        f"Feedback on how to improve it:\n{improvement}\n\n"
        f"Please provide an improved response to the child's question that addresses this feedback."
    )


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_responses(
    model,
    tokenizer,
    prompts: List[str],
    n: int,
    temperature: float,
    max_new_tokens: int,
    max_seq_len: int,
) -> List[str]:
    """Generate n responses per prompt. Returns flat list of length len(prompts)*n."""
    responses = []
    FastLanguageModel.for_inference(model)
    tokenizer.padding_side = "left"

    for prompt in prompts:
        enc = tokenizer(
            [prompt] * n,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
        ).to(model.device)

        n_inp = enc["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id,
            )

        for seq in out:
            responses.append(tokenizer.decode(seq[n_inp:], skip_special_tokens=True))

    tokenizer.padding_side = "right"
    FastLanguageModel.for_training(model)
    return responses


# ---------------------------------------------------------------------------
# Log-probability computation
# ---------------------------------------------------------------------------

def compute_log_probs(
    model,
    tokenizer,
    prompts: List[str],
    responses: List[str],
    max_length: int,
) -> List[torch.Tensor]:
    """
    Compute mean log-prob of each response given its prompt.
    Returns a list of scalar tensors (with grad if model is in training mode).
    """
    log_probs_list = []

    for prompt, response in zip(prompts, responses):
        prompt_ids  = tokenizer.encode(prompt, add_special_tokens=True)
        response_ids = tokenizer.encode(response, add_special_tokens=False)

        full_ids = torch.tensor(
            (prompt_ids + response_ids)[:max_length], dtype=torch.long
        ).unsqueeze(0).to(model.device)

        n_prompt = min(len(prompt_ids), max_length)
        n_full   = full_ids.shape[1]

        if n_full <= n_prompt:
            # Response got truncated away
            log_probs_list.append(torch.tensor(0.0, device=model.device, requires_grad=True))
            continue

        logits = model(full_ids).logits  # (1, seq_len, vocab)

        # Predict response token t from logits at position n_prompt+t-1
        shift_logits = logits[0, n_prompt - 1 : n_full - 1]   # (resp_len, vocab)
        shift_labels = full_ids[0, n_prompt:]                  # (resp_len,)

        token_lp = F.log_softmax(shift_logits, dim=-1)
        resp_lp  = token_lp[range(len(shift_labels)), shift_labels]
        log_probs_list.append(resp_lp.mean())

    return log_probs_list


# ---------------------------------------------------------------------------
# GRPO
# ---------------------------------------------------------------------------

def grpo_advantages(
    all_rewards: List[float],
    prompt_groups: Dict[int, List[int]],
    n_total: int,
) -> torch.Tensor:
    """Per-prompt group advantage normalization."""
    advantages = torch.zeros(n_total, dtype=torch.float32)
    for _, member_indices in prompt_groups.items():
        group_r = torch.tensor([all_rewards[i] for i in member_indices], dtype=torch.float32)
        mean = group_r.mean()
        std  = group_r.std() if len(group_r) > 1 else torch.tensor(1.0)
        group_adv = (group_r - mean) / (std + 1e-8)
        for k, idx in enumerate(member_indices):
            advantages[idx] = group_adv[k]
    return advantages


def grpo_loss(
    new_log_probs: List[torch.Tensor],
    old_log_probs: List[float],
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    lp_new  = torch.stack(new_log_probs)
    lp_old  = torch.tensor(old_log_probs, device=lp_new.device)
    adv     = advantages.to(lp_new.device)

    ratio  = torch.exp(lp_new - lp_old)
    loss1  = -ratio * adv
    loss2  = -torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    return torch.max(loss1, loss2).mean()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_train_records(train_path: str) -> List[dict]:
    records = []
    with open(train_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def record_to_prompt(record: dict, tokenizer) -> str:
    """Format the user-only messages of a record as a prompt string."""
    msgs = [m for m in record["messages"] if m["role"] != "assistant"]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    io = p.add_argument_group("io")
    io.add_argument("--model_name",   required=True)
    io.add_argument("--judge_path",   required=True, help="HF repo or local path for fine-tuned judge")
    io.add_argument("--train_path",   default=str(ROOT / "dataset/gold/train.jsonl"))
    io.add_argument("--output_dir",   required=True)
    io.add_argument("--log_dir",      default="finetuning_logs/grpo")
    io.add_argument("--hf_repo",      required=True)

    lora = p.add_argument_group("lora")
    lora.add_argument("--lora_r",       type=int,   default=16)
    lora.add_argument("--lora_alpha",   type=int,   default=32)
    lora.add_argument("--lora_dropout", type=float, default=0.0)

    train = p.add_argument_group("training")
    train.add_argument("--max_seq_len",          type=int,   default=2048)
    train.add_argument("--batch_size",           type=int,   default=4,   help="Prompts per update step")
    train.add_argument("--n_gen",                type=int,   default=8,   help="Responses generated per prompt")
    train.add_argument("--lr",                   type=float, default=1e-6)
    train.add_argument("--epochs",               type=int,   default=2)
    train.add_argument("--warmup_ratio",         type=float, default=0.05)
    train.add_argument("--clip_eps",             type=float, default=0.2)
    train.add_argument("--critique_threshold",   type=float, default=0.6,  help="Reward below this triggers critique refinement")
    train.add_argument("--temperature",          type=float, default=0.9)
    train.add_argument("--max_new_tokens",       type=int,   default=1024)
    train.add_argument("--judge_max_seq_len",    type=int,   default=4096)
    train.add_argument("--judge_max_new_tokens", type=int,   default=512)
    train.add_argument("--judge_batch_size",     type=int,   default=4)
    train.add_argument("--seed",                 type=int,   default=42)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # ── Load policy (bfloat16, LoRA)
    print(f"Loading policy: {args.model_name}")
    policy, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_len,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )
    policy = FastLanguageModel.get_peft_model(
        policy,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    # ── Load judge (4-bit, frozen)
    print(f"Loading judge: {args.judge_path}")
    judge, judge_tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.judge_path,
        max_seq_length=args.judge_max_seq_len,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(judge)
    for param in judge.parameters():
        param.requires_grad_(False)

    judge_system_prompt = _load_judge_system_prompt()

    # ── Dataset
    records = load_train_records(args.train_path)
    all_prompts = [record_to_prompt(r, tokenizer) for r in records]
    print(f"Dataset: {len(all_prompts)} prompts")

    # ── Optimizer
    optimizer = AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )
    steps_per_epoch = max(1, len(all_prompts) // args.batch_size)
    total_steps  = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    loss_log_path = log_dir / "train_loss.jsonl"
    global_step = 0

    # ── Training loop
    for epoch in range(1, args.epochs + 1):
        perm = torch.randperm(len(all_prompts)).tolist()
        shuffled = [all_prompts[i] for i in perm]

        for batch_start in range(0, len(shuffled) - args.batch_size + 1, args.batch_size):
            batch_prompts = shuffled[batch_start : batch_start + args.batch_size]
            B = len(batch_prompts)

            # ── Stage 1: Generate n_gen responses per prompt
            init_responses = generate_responses(
                policy, tokenizer,
                batch_prompts,
                n=args.n_gen,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                max_seq_len=args.max_seq_len,
            )
            # init_responses layout: [p0_r0..r{n-1}, p1_r0..r{n-1}, ...]
            expanded_prompts = [p for p in batch_prompts for _ in range(args.n_gen)]

            # ── Stage 2: Score with judge
            init_rewards, init_improvements = judge_score_batch(
                judge, judge_tokenizer, judge_system_prompt,
                expanded_prompts, init_responses,
                judge_max_seq_len=args.judge_max_seq_len,
                judge_max_new_tokens=args.judge_max_new_tokens,
                batch_size=args.judge_batch_size,
            )

            # ── Stage 3: Critique refinement for low-reward responses
            critique_prompts: List[str] = []
            critique_source_indices: List[int] = []  # index into expanded_prompts/init_responses

            for idx in range(len(expanded_prompts)):
                if init_rewards[idx] < args.critique_threshold and init_improvements[idx]:
                    critique_prompts.append(
                        build_critique_prompt(
                            expanded_prompts[idx],
                            init_responses[idx],
                            init_improvements[idx],
                        )
                    )
                    critique_source_indices.append(idx)

            all_prompts_batch  = list(expanded_prompts)
            all_responses      = list(init_responses)
            all_rewards        = list(init_rewards)

            if critique_prompts:
                refined_responses = generate_responses(
                    policy, tokenizer,
                    critique_prompts,
                    n=1,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    max_seq_len=args.max_seq_len,
                )
                # Score refined responses using original (non-critique) prompt
                original_prompts_for_refined = [expanded_prompts[i] for i in critique_source_indices]
                refined_rewards, _ = judge_score_batch(
                    judge, judge_tokenizer, judge_system_prompt,
                    original_prompts_for_refined, refined_responses,
                    judge_max_seq_len=args.judge_max_seq_len,
                    judge_max_new_tokens=args.judge_max_new_tokens,
                    batch_size=args.judge_batch_size,
                )
                all_prompts_batch.extend(critique_prompts)
                all_responses.extend(refined_responses)
                all_rewards.extend(refined_rewards)

            N_total = len(all_rewards)

            # ── Stage 4: Group membership for GRPO advantage computation
            # Each prompt has its n_gen originals + however many of its critiques were generated
            prompt_groups: Dict[int, List[int]] = defaultdict(list)
            for idx in range(B * args.n_gen):
                prompt_idx = idx // args.n_gen
                prompt_groups[prompt_idx].append(idx)
            for j, src_idx in enumerate(critique_source_indices):
                prompt_idx = src_idx // args.n_gen
                prompt_groups[prompt_idx].append(B * args.n_gen + j)

            advantages = grpo_advantages(all_rewards, prompt_groups, N_total)

            # ── Stage 5: Old log-probs (no grad, for importance ratio)
            policy.eval()
            old_lps: List[float] = []
            with torch.no_grad():
                for lp in compute_log_probs(policy, tokenizer, all_prompts_batch, all_responses, args.max_seq_len):
                    old_lps.append(lp.item())
            policy.train()

            # ── Stage 6: Policy gradient loss
            optimizer.zero_grad()
            new_lps = compute_log_probs(policy, tokenizer, all_prompts_batch, all_responses, args.max_seq_len)
            loss = grpo_loss(new_lps, old_lps, advantages, clip_eps=args.clip_eps)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in policy.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            scheduler.step()

            global_step += 1
            mean_reward   = sum(init_rewards) / len(init_rewards)
            loss_val      = loss.item()
            n_critiques   = len(critique_prompts)

            print(
                f"[Epoch {epoch} | Step {global_step}] "
                f"loss={loss_val:.4f}  mean_reward={mean_reward:.3f}  critiques={n_critiques}",
                flush=True,
            )
            with open(loss_log_path, "a") as f:
                f.write(json.dumps({
                    "step": global_step, "loss": loss_val,
                    "mean_reward": mean_reward, "n_critiques": n_critiques,
                }) + "\n")

        # ── Upload after epoch
        repo = f"{args.hf_repo}-{epoch}"
        print(f"\nUploading epoch {epoch} → {repo} (merged bfloat16)...", flush=True)
        policy.push_to_hub_merged(repo, tokenizer, save_method="merged_16bit", private=True)
        print(f"Uploaded → {repo}", flush=True)

    print("Training done.")


if __name__ == "__main__":
    main()
