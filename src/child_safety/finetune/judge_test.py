#!/usr/bin/env python3
"""
Evaluate a fine-tuned judge model on dataset/judge/test.jsonl.

Single-turn records (1 user message): standard eval.
Multi-turn records (N user/assistant pairs): chat-style eval — each turn is
evaluated with the preceding gold evaluations as context (teacher forcing).

Metrics computed over all turns (single + multi-turn flattened):
  Per-metric: Spearman, MAE, Exact Acc, Within-1 Acc
  Overall:    Spearman, QWK, MAE, Exact Acc, Within-1 Acc,
              Macro-F1 (remapped 1-2→0, 3-4→1, 5→2), Accuracy (remapped)
  Parse rate + failure count

Outputs saved to responses/judge/{model_name}/:
  predictions.jsonl  — one line per (record, turn): context + gold + pred + raw
  metrics.json       — all computed metrics
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, mean_absolute_error

from child_safety.utils import generate_batch, load_model

ROOT = Path(__file__).parent.parent.parent.parent

METRICS_CORE = ["safety", "developmental_fit", "emotional_support", "moral_guidance", "boundary_setting"]
METRICS_CULTURAL = METRICS_CORE + ["cultural_alignment"]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _metric_keys(gold: Dict) -> List[str]:
    return METRICS_CULTURAL if "cultural_alignment" in gold else METRICS_CORE


def parse_scores(text: str, metrics: List[str]) -> Optional[Dict[str, int]]:
    if not text:
        return None
    try:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        data = json.loads(m.group(1) if m else text)
    except Exception:
        return None
    scores: Dict[str, int] = {}
    for key in metrics:
        v = data.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            scores[key] = max(1, min(5, int(round(float(v)))))
    return scores if len(scores) == len(metrics) else None


# ---------------------------------------------------------------------------
# Sample extraction
# ---------------------------------------------------------------------------

def extract_samples(records: List[dict]) -> List[dict]:
    """
    Expand each record into one sample per evaluation turn.

    Single-turn: one sample, context = [system, user].
    Multi-turn:  one sample per turn, context = [system, user1, gold_asst1, ..., user_k]
                 (teacher forcing — gold previous evaluations as context).

    Each sample: {rec_idx, turn_idx, n_turns, input_messages, gold}
    """
    samples = []
    for rec_idx, record in enumerate(records):
        msgs = record["messages"]
        system = [msgs[0]] if msgs[0]["role"] == "system" else []
        rest = msgs[len(system):]

        # Collect (user_msg, asst_msg) pairs
        turns: List[Tuple[dict, dict]] = []
        i = 0
        while i < len(rest) - 1:
            if rest[i]["role"] == "user" and rest[i + 1]["role"] == "assistant":
                turns.append((rest[i], rest[i + 1]))
                i += 2
            else:
                i += 1

        context = list(system)
        for turn_idx, (user_msg, asst_msg) in enumerate(turns):
            gold = json.loads(asst_msg["content"])
            samples.append({
                "rec_idx":        rec_idx,
                "turn_idx":       turn_idx,
                "n_turns":        len(turns),
                "input_messages": context + [user_msg],
                "gold":           gold,
            })
            # Add gold evaluation to context for the next turn
            context = context + [user_msg, asst_msg]

    return samples


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def remap(v: int) -> int:
    if v <= 2:
        return 0
    if v <= 4:
        return 1
    return 2


def compute_metrics(
    preds: List[Optional[Dict[str, int]]],
    golds: List[Dict[str, int]],
    per_metric_keys: List[List[str]],
) -> Dict:
    pred_flat, gold_flat = [], []
    per_metric: Dict[str, Dict[str, List]] = {}

    for pred, gold, keys in zip(preds, golds, per_metric_keys):
        if pred is None:
            continue
        for k in keys:
            pv, gv = pred.get(k), gold.get(k)
            if pv is None or gv is None:
                continue
            pred_flat.append(pv)
            gold_flat.append(gv)
            per_metric.setdefault(k, {"pred": [], "gold": []})
            per_metric[k]["pred"].append(pv)
            per_metric[k]["gold"].append(gv)

    n_parsed = sum(1 for p in preds if p is not None)
    n_failed = len(preds) - n_parsed
    parse_rate = n_parsed / max(len(preds), 1)
    result: Dict = {
        "parse_rate": parse_rate,
        "n_parsed":   n_parsed,
        "n_failed":   n_failed,
        "n_total":    len(preds),
    }

    if not pred_flat:
        return result

    pred_arr = np.array(pred_flat)
    gold_arr = np.array(gold_flat)

    result["overall"] = {
        "spearman":       float(spearmanr(gold_arr, pred_arr).statistic),
        "qwk":            float(cohen_kappa_score(gold_arr, pred_arr, weights="quadratic", labels=[1, 2, 3, 4, 5])),
        "mae":            float(mean_absolute_error(gold_arr, pred_arr)),
        "exact_acc":      float(np.mean(gold_arr == pred_arr)),
        "within1_acc":    float(np.mean(np.abs(gold_arr - pred_arr) <= 1)),
        "macro_f1_remap": float(f1_score(
            [remap(v) for v in gold_flat], [remap(v) for v in pred_flat],
            average="macro", labels=[0, 1, 2], zero_division=0,
        )),
        "acc_remap":      float(accuracy_score([remap(v) for v in gold_flat], [remap(v) for v in pred_flat])),
    }

    result["per_metric"] = {}
    for k, arrs in per_metric.items():
        pa, ga = np.array(arrs["pred"]), np.array(arrs["gold"])
        result["per_metric"][k] = {
            "spearman":    float(spearmanr(ga, pa).statistic),
            "mae":         float(mean_absolute_error(ga, pa)),
            "exact_acc":   float(np.mean(ga == pa)),
            "within1_acc": float(np.mean(np.abs(ga - pa) <= 1)),
        }

    return result


def print_metrics(metrics: Dict, n_records: int, n_multiturn: int) -> None:
    n_failed = metrics["n_total"] - metrics["n_parsed"]
    print(f"\n{'='*60}")
    print(f"  Records:  {n_records}  ({n_multiturn} multi-turn, {n_records - n_multiturn} single-turn)")
    print(f"  Turns:    {metrics['n_total']} total eval turns")
    print(f"  Parsed:   {metrics['n_parsed']}/{metrics['n_total']}  ({metrics['parse_rate']:.1%})")
    print(f"  Failed:   {n_failed}/{metrics['n_total']}  ({n_failed/max(metrics['n_total'],1):.1%} parse failures)")

    if "overall" not in metrics:
        print("  No parseable predictions — cannot compute scores.")
        return

    ov = metrics["overall"]
    print(f"\n  Overall (all metrics flattened):")
    print(f"    Spearman:       {ov['spearman']:+.4f}")
    print(f"    QWK:            {ov['qwk']:+.4f}")
    print(f"    MAE:            {ov['mae']:.4f}")
    print(f"    Exact Acc:      {ov['exact_acc']:.1%}")
    print(f"    Within-1 Acc:   {ov['within1_acc']:.1%}")
    print(f"    Macro-F1 (rmp): {ov['macro_f1_remap']:.4f}")
    print(f"    Acc (rmp):      {ov['acc_remap']:.1%}")

    print(f"\n  Per-metric breakdown:")
    hdr = f"    {'Metric':<22}  {'Spearman':>9}  {'MAE':>6}  {'Exact':>7}  {'Within-1':>9}"
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for k, v in metrics["per_metric"].items():
        print(
            f"    {k:<22}  {v['spearman']:>+9.4f}  {v['mae']:>6.4f}"
            f"  {v['exact_acc']:>7.1%}  {v['within1_acc']:>9.1%}"
        )
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name",             required=True, help="Short name used for output dir")
    p.add_argument("--model_path",             required=True, help="HF model ID or local path")
    p.add_argument("--test_path",              default=str(ROOT / "dataset/judge/test.jsonl"))
    p.add_argument("--output_dir",             default=str(ROOT / "responses/judge"))
    p.add_argument("--max_model_len",          type=int,   default=4096)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    p.add_argument("--tensor_parallel_size",   type=int,   default=1)
    p.add_argument("--temperature",            type=float, default=0.0)
    p.add_argument("--top_p",                  type=float, default=1.0)
    p.add_argument("--max_new_tokens",         type=int,   default=512)
    p.add_argument("--resume",                 action="store_true", help="Skip already-completed samples")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading test data: {args.test_path}")
    records = []
    with open(args.test_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    samples = extract_samples(records)
    n_multiturn = sum(1 for r in records if sum(1 for m in r["messages"] if m["role"] == "user") > 1)
    print(f"  {len(records)} records → {len(samples)} eval turns  ({n_multiturn} multi-turn records)")

    out_dir = Path(args.output_dir) / args.model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    metrics_path = out_dir / "metrics.json"

    # Resume: keyed by (rec_idx, turn_idx)
    done: Dict[Tuple[int, int], dict] = {}
    if args.resume and pred_path.exists():
        with open(pred_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    done[(entry["rec_idx"], entry["turn_idx"])] = entry
        print(f"  Resuming: {len(done)} turns already done")

    pending = [s for s in samples if (s["rec_idx"], s["turn_idx"]) not in done]

    if pending:
        print(f"Loading model: {args.model_path}")
        tokenizer, llm = load_model(
            model_path=args.model_path,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )

        print(f"Generating {len(pending)} predictions...")
        raw_outputs = generate_batch(
            tokenizer=tokenizer,
            llm=llm,
            messages_batch=[s["input_messages"] for s in pending],
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
        )

        with open(pred_path, "a") as f_out:
            for sample, raw in zip(pending, raw_outputs):
                gold = sample["gold"]
                keys = _metric_keys(gold)
                pred = parse_scores(raw, keys)
                entry = {
                    "rec_idx":  sample["rec_idx"],
                    "turn_idx": sample["turn_idx"],
                    "n_turns":  sample["n_turns"],
                    "gold":     gold,
                    "pred":     pred,
                    "raw":      raw,
                }
                done[(sample["rec_idx"], sample["turn_idx"])] = entry
                f_out.write(json.dumps(entry) + "\n")

        print(f"Predictions saved → {pred_path}")
    else:
        print("All turns already done — skipping generation.")

    # Compute metrics in original sample order
    all_preds: List[Optional[Dict[str, int]]] = []
    all_golds: List[Dict[str, int]] = []
    all_key_sets: List[List[str]] = []

    for s in samples:
        entry = done[(s["rec_idx"], s["turn_idx"])]
        gold = entry["gold"]
        all_preds.append(entry["pred"])
        all_golds.append({k: gold[k] for k in _metric_keys(gold) if k in gold})
        all_key_sets.append(_metric_keys(gold))

    metrics = compute_metrics(all_preds, all_golds, all_key_sets)
    print_metrics(metrics, len(records), n_multiturn)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved → {metrics_path}")


if __name__ == "__main__":
    main()
