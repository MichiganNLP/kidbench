"""
Generate gold model analysis markdown files:
  evaluations/analysis/gold/gold_cues.md
  evaluations/analysis/gold/gold_cultural.md
  evaluations/analysis/gold/gold_multi_turn.md

Models: llamaplushie-3-8b-sft-1/2/3 + llamaplushie-3-8b-grpo
(GRPO has no multi-turn data — excluded from gold_multi_turn.md)
"""

import json
import pathlib
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
ROOT     = pathlib.Path(__file__).resolve().parents[3]
EVAL_DIR = ROOT / "evaluations" / "deepseek-v4-pro"
ST_DIR   = EVAL_DIR / "single_turn"
MT_DIR   = EVAL_DIR / "multi_turn"
OUT_DIR  = ROOT / "evaluations" / "analysis" / "gold"
DATASET  = ROOT / "dataset" / "kidbench" / "kidbench_single.json"

MODEL_ORDER = [
    "llamaplushie-3-8b-sft-1",
    "llamaplushie-3-8b-sft-2",
    "llamaplushie-3-8b-sft-3",
    "llamaplushie-3-8b-grpo",
]
MODEL_DISPLAY = {
    "llamaplushie-3-8b-sft-1":  "LlamaPlushie SFT-1",
    "llamaplushie-3-8b-sft-2":  "LlamaPlushie SFT-2",
    "llamaplushie-3-8b-sft-3":  "LlamaPlushie SFT-3",
    "llamaplushie-3-8b-grpo":   "LlamaPlushie GRPO",
}
MT_MODEL_ORDER = MODEL_ORDER

METRICS       = ["safety", "developmental_fit", "emotional_support", "moral_guidance", "boundary_setting"]
METRIC_LABELS = ["Safety", "Dev. Fit", "Emotional", "Moral", "Boundary"]

CATEGORY_ORDER = [
    "sexual_content_and_boundaries", "self-harm_and_mental_health",
    "aggression_and_bullying", "moral_reasoning", "physical_health_and_safety",
    "school_conduct_and_integrity", "family_peers_and_relationships",
    "online_safety_and_privacy", "hate_bias_and_identity_attacks",
    "benign_information_seeking",
]
CATEGORY_SHORT = {
    "sexual_content_and_boundaries":  "Sexual",
    "self-harm_and_mental_health":    "Self-Harm",
    "aggression_and_bullying":        "Aggression",
    "moral_reasoning":                "Moral",
    "physical_health_and_safety":     "Health",
    "school_conduct_and_integrity":   "School",
    "family_peers_and_relationships": "Family",
    "online_safety_and_privacy":      "Online",
    "hate_bias_and_identity_attacks": "Hate/Bias",
    "benign_information_seeking":     "Benign",
}

TURNS = [1, 2, 3, 4, 5]

# ── Shared helpers ────────────────────────────────────────────────────────────
def load_json(path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)

def nanmean(vals):
    vals = [v for v in vals if v is not None and not np.isnan(v)]
    return np.mean(vals) if vals else float("nan")

def fmt(v):
    return f"{v:.3f}" if not np.isnan(v) else "N/A"

def build_prompt_to_category():
    with open(DATASET) as f:
        data = json.load(f)
    mapping = {}
    for lang_data in data.values():
        for cat, items in lang_data.items():
            for item in items:
                mapping[item["without_cues"]] = cat
                if "with_cues" in item:
                    mapping[item["with_cues"]] = cat
    return mapping


# ══════════════════════════════════════════════════════════════════════════════
# CUES ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
CONDITIONS = {
    "No Cue":       ("without_cues", "english_false_none.json"),
    "Implicit Cue": ("with_cues",    "english_false_none.json"),
    "Explicit Age": ("without_cues", "english_true_none.json"),
}

def load_cues_condition(folder, filename, prompt_to_cat):
    result = {}
    for model in MODEL_ORDER:
        path = ST_DIR / model / folder / filename
        if not path.exists():
            continue
        items = load_json(path)
        metrics = {m: [] for m in METRICS}
        cats    = {c: [] for c in CATEGORY_ORDER}
        for item in items:
            ev = item.get("json_evaluation")
            if ev is None:
                continue
            row = []
            for m in METRICS:
                v = ev.get(m)
                if v is not None:
                    metrics[m].append(v)
                    row.append(v)
            cat = prompt_to_cat.get(item["prompt"])
            if cat and row:
                cats[cat].append(np.mean(row))
        result[model] = {"metrics": metrics, "categories": cats}
    return result

def cues_metric_table(data):
    header = "| Model | " + " | ".join(METRIC_LABELS) + " | Total |\n"
    sep    = "|---|" + "---|" * (len(METRIC_LABELS) + 1) + "\n"
    rows   = ""
    for model in MODEL_ORDER:
        if model not in data:
            continue
        name = MODEL_DISPLAY[model]
        vals = [nanmean(data[model]["metrics"][m]) for m in METRICS]
        total = nanmean(vals)
        rows += f"| {name} | " + " | ".join(fmt(v) for v in vals) + f" | {fmt(total)} |\n"
    return header + sep + rows

def cues_category_table(data):
    header = "| Model | " + " | ".join(CATEGORY_SHORT[c] for c in CATEGORY_ORDER) + " | Total |\n"
    sep    = "|---|" + "---|" * (len(CATEGORY_ORDER) + 1) + "\n"
    rows   = ""
    for model in MODEL_ORDER:
        if model not in data:
            continue
        name = MODEL_DISPLAY[model]
        vals = [nanmean(data[model]["categories"][c]) for c in CATEGORY_ORDER]
        total = nanmean(vals)
        rows += f"| {name} | " + " | ".join(fmt(v) for v in vals) + f" | {fmt(total)} |\n"
    return header + sep + rows

def generate_gold_cues():
    prompt_to_cat = build_prompt_to_category()
    condition_data = {}
    for cond, (folder, filename) in CONDITIONS.items():
        condition_data[cond] = load_cues_condition(folder, filename, prompt_to_cat)

    md = "# Gold Model — Cues Analysis\n\n"
    for cond in CONDITIONS:
        data = condition_data[cond]
        md += f"## {cond}\n\n"
        md += "### Per Metric\n\n"
        md += cues_metric_table(data) + "\n"
        md += "### Per Category\n\n"
        md += cues_category_table(data) + "\n"

    out = OUT_DIR / "gold_cues.md"
    with open(out, "w") as f:
        f.write(md)
    print(f"Wrote {out}")


# ══════════════════════════════════════════════════════════════════════════════
# CULTURAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
COUNTRIES = {
    "Pakistan": ("with_cues", "english_true_pakistan.json"),
    "India":    ("with_cues", "english_true_india.json"),
    "China":    ("with_cues", "english_true_china.json"),
    "Nigeria":  ("with_cues", "english_true_nigeria.json"),
}

def load_cultural_country(folder, filename, prompt_to_cat):
    result = {}
    for model in MODEL_ORDER:
        path = ST_DIR / model / folder / filename
        if not path.exists():
            continue
        items = load_json(path)
        cats       = {c: [] for c in CATEGORY_ORDER}
        all_scores = []
        for item in items:
            ev = item.get("json_evaluation")
            if ev is None:
                continue
            score = ev.get("cultural_alignment")
            if score is None:
                continue
            all_scores.append(score)
            cat = prompt_to_cat.get(item["prompt"])
            if cat:
                cats[cat].append(score)
        result[model] = {"categories": cats, "all": all_scores}
    return result

def cultural_category_table(data):
    header = "| Model | " + " | ".join(CATEGORY_SHORT[c] for c in CATEGORY_ORDER) + " |\n"
    sep    = "|---|" + "---|" * len(CATEGORY_ORDER) + "\n"
    rows   = ""
    for model in MODEL_ORDER:
        if model not in data:
            continue
        name = MODEL_DISPLAY[model]
        vals = [nanmean(data[model]["categories"][c]) for c in CATEGORY_ORDER]
        rows += f"| {name} | " + " | ".join(fmt(v) for v in vals) + " |\n"
    return header + sep + rows

def generate_gold_cultural():
    prompt_to_cat = build_prompt_to_category()
    country_data = {}
    for country, (folder, fname) in COUNTRIES.items():
        country_data[country] = load_cultural_country(folder, fname, prompt_to_cat)

    md = "# Gold Model — Cultural Analysis\n\n"
    for country in COUNTRIES:
        data = country_data[country]
        md += f"## {country}\n\n"
        md += "### Per Category (Cultural Alignment Score)\n\n"
        md += cultural_category_table(data) + "\n"

    out = OUT_DIR / "gold_cultural.md"
    with open(out, "w") as f:
        f.write(md)
    print(f"Wrote {out}")


# ══════════════════════════════════════════════════════════════════════════════
# MULTI-TURN ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def load_mt_model(model, filename):
    path = MT_DIR / model / filename
    return load_json(path)

def fit_slope(ys):
    xs = np.array(TURNS, dtype=float)
    ys = np.array(ys, dtype=float)
    x_mean, y_mean = xs.mean(), ys.mean()
    beta1 = np.sum((xs - x_mean) * (ys - y_mean)) / np.sum((xs - x_mean) ** 2)
    return -beta1

def per_metric_slopes(items):
    result = {}
    for metric in METRICS:
        series = []
        for t in TURNS:
            scores = []
            for item in items:
                ev_turn = item["evaluations"].get(str(t), {})
                ev = ev_turn.get("json_evaluation") if ev_turn else None
                if ev is None:
                    continue
                v = ev.get(metric)
                if v is not None:
                    scores.append(v)
            series.append(nanmean(scores) if scores else float("nan"))
        result[metric] = fit_slope(series) if any(not np.isnan(v) for v in series) else float("nan")
    return result

def per_category_slopes(items):
    result = {}
    for cat in CATEGORY_ORDER:
        cat_items = [item for item in items if item.get("category") == cat]
        if not cat_items:
            result[cat] = float("nan")
            continue
        series = []
        for t in TURNS:
            scores = []
            for item in cat_items:
                ev_turn = item["evaluations"].get(str(t), {})
                ev = ev_turn.get("json_evaluation") if ev_turn else None
                if not ev:
                    continue
                vals = [ev.get(m) for m in METRICS if ev.get(m) is not None]
                if vals:
                    scores.append(np.mean(vals))
            series.append(nanmean(scores) if scores else float("nan"))
        result[cat] = fit_slope(series)
    return result

def total_slope(items):
    series = []
    for t in TURNS:
        scores = []
        for item in items:
            ev_turn = item["evaluations"].get(str(t), {})
            ev = ev_turn.get("json_evaluation") if ev_turn else None
            if not ev:
                continue
            vals = [ev.get(m) for m in METRICS if ev.get(m) is not None]
            if vals:
                scores.append(np.mean(vals))
        series.append(nanmean(scores) if scores else float("nan"))
    return fit_slope(series)

def peak_degradation(items):
    turn_scores = {}
    for t in TURNS:
        scores = []
        for item in items:
            ev_turn = item["evaluations"].get(str(t), {})
            ev = ev_turn.get("json_evaluation") if ev_turn else None
            if not ev:
                continue
            vals = [ev.get(m) for m in METRICS if ev.get(m) is not None]
            if vals:
                scores.append(np.mean(vals))
        turn_scores[t] = nanmean(scores) if scores else float("nan")
    t1 = turn_scores[1]
    later = {t: v for t, v in turn_scores.items() if t > 1 and not np.isnan(v)}
    if not later:
        return t1, float("nan"), float("nan"), float("nan")
    lowest_t = min(later, key=later.get)
    lowest_score = later[lowest_t]
    return t1, lowest_t, lowest_score, t1 - lowest_score

def mt_slope_metric_table(model_slopes):
    header = "| Model | " + " | ".join(METRIC_LABELS) + " | Total |\n"
    sep    = "|---|" + "---|" * (len(METRIC_LABELS) + 1) + "\n"
    rows   = ""
    for model in MT_MODEL_ORDER:
        if model not in model_slopes:
            continue
        name = MODEL_DISPLAY[model]
        vals = [model_slopes[model]["metrics"].get(m, float("nan")) for m in METRICS]
        total = model_slopes[model].get("total", float("nan"))
        rows += f"| {name} | " + " | ".join(fmt(v) for v in vals) + f" | {fmt(total)} |\n"
    return header + sep + rows

def mt_slope_category_table(model_slopes):
    header = "| Model | " + " | ".join(CATEGORY_SHORT[c] for c in CATEGORY_ORDER) + " |\n"
    sep    = "|---|" + "---|" * len(CATEGORY_ORDER) + "\n"
    rows   = ""
    for model in MT_MODEL_ORDER:
        if model not in model_slopes:
            continue
        name = MODEL_DISPLAY[model]
        vals = [model_slopes[model]["categories"].get(c, float("nan")) for c in CATEGORY_ORDER]
        rows += f"| {name} | " + " | ".join(fmt(v) for v in vals) + " |\n"
    return header + sep + rows

def mt_peak_table(model_peaks):
    header = "| Model | T1 | Lowest Turn | Lowest Score | Delta |\n"
    sep    = "|---|---|---|---|---|\n"
    rows   = ""
    for model in MT_MODEL_ORDER:
        if model not in model_peaks:
            continue
        name = MODEL_DISPLAY[model]
        t1, lt, ls, delta = model_peaks[model]
        rows += (
            f"| {name} | {fmt(t1)} | "
            + (str(int(lt)) if not np.isnan(lt) else "N/A") + f" | {fmt(ls)} | {fmt(delta)} |\n"
        )
    return header + sep + rows

def generate_gold_multi_turn():
    md = "# Gold Model — Multi-Turn Analysis\n\n"

    for condition in ["without_age", "with_age"]:
        filename = f"{condition}.json"
        model_slopes = {}
        model_peaks  = {}

        for model in MT_MODEL_ORDER:
            items = load_mt_model(model, filename)
            if items is None:
                continue
            model_slopes[model] = {
                "metrics":    per_metric_slopes(items),
                "categories": per_category_slopes(items),
                "total":      total_slope(items),
            }
            model_peaks[model] = peak_degradation(items)

        label = condition.replace("_", " ").title()
        md += f"## Quality Degradation Slope (β₁) — Per Metric — {label}\n\n"
        md += mt_slope_metric_table(model_slopes) + "\n"

        md += f"## Quality Degradation Slope (β₁) — Per Category — {label}\n\n"
        md += mt_slope_category_table(model_slopes) + "\n"

        md += f"## Peak Degradation (T1 − min T2..T5) — {label}\n\n"
        md += mt_peak_table(model_peaks) + "\n"

    out = OUT_DIR / "gold_multi_turn.md"
    with open(out, "w") as f:
        f.write(md)
    print(f"Wrote {out}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    generate_gold_cues()
    generate_gold_cultural()
    generate_gold_multi_turn()


if __name__ == "__main__":
    main()
