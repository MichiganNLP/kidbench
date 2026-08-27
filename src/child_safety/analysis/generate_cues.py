"""Generate cues_analysis.md and cues_analysis.png."""

import json
import os
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "figure.dpi": 300,
})
BG = "#FFFFFF"

# ── Config ──────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parents[3]
EVAL_DIR = ROOT / "evaluations" / "deepseek-v4-pro" / "single_turn"
OUT_DIR = ROOT / "evaluations" / "analysis" / "cues"
DATASET = ROOT / "dataset" / "kidbench" / "kidbench_single.json"

MODEL_ORDER = [
    "llama-3.2-3b", "llama-3-8b", "llama-3.3-70b",
    "gemma-3-4b", "gemma-3-12b", "gemma-4-31b",
    "deepseek-v4-flash",
    "qwen-3.1-8b", "qwen-3.5-4b", "qwen-3.6-27b",
    "gemini-3.1-flash-lite", "gpt-5-mini", "claude-haiku-4.5",
]
MODEL_DISPLAY = {
    "llama-3.2-3b": "Llama 3.2-3B", "llama-3-8b": "Llama 3.1-8B",
    "llama-3.3-70b": "Llama 3.3-70B", "gemma-3-4b": "Gemma 3-4B",
    "gemma-3-12b": "Gemma 3-12B", "gemma-4-31b": "Gemma 4-31B",
    "deepseek-v4-flash": "DeepSeek V4 Flash", "qwen-3.1-8b": "Qwen 3-8B",
    "qwen-3.5-4b": "Qwen 3.5-4B", "qwen-3.6-27b": "Qwen 3.6-27B",
    "gemini-3.1-flash-lite": "Gemini 3.1 Flash Lite",
    "gpt-5-mini": "GPT 5 Mini", "claude-haiku-4.5": "Claude Haiku 4.5",
}

METRICS = ["safety", "developmental_fit", "emotional_support", "moral_guidance", "boundary_setting"]
METRIC_LABELS = ["Safety", "Dev. Fit", "Emotional", "Moral", "Boundary"]

CATEGORY_ORDER = [
    "sexual_content_and_boundaries", "self-harm_and_mental_health",
    "aggression_and_bullying", "moral_reasoning", "physical_health_and_safety",
    "school_conduct_and_integrity", "family_peers_and_relationships",
    "online_safety_and_privacy", "hate_bias_and_identity_attacks",
    "benign_information_seeking",
]
CATEGORY_SHORT = {
    "sexual_content_and_boundaries": "Sexual",
    "self-harm_and_mental_health": "Self-Harm",
    "aggression_and_bullying": "Aggression",
    "moral_reasoning": "Moral",
    "physical_health_and_safety": "Health",
    "school_conduct_and_integrity": "School",
    "family_peers_and_relationships": "Family",
    "online_safety_and_privacy": "Online",
    "hate_bias_and_identity_attacks": "Hate/Bias",
    "benign_information_seeking": "Benign",
}

CONDITIONS = {
    "No Cue": ("without_cues", "english_false_none.json"),
    "Implicit Cue": ("with_cues", "english_false_none.json"),
    "Explicit Age": ("without_cues", "english_true_none.json"),
}
COLORS = {
    "No Cue": "#FF495E",
    "Implicit Cue": "#5DADE2",
    "Explicit Age": "#235A61",
}

# ── Build prompt→category map ────────────────────────────────────────────────
def build_prompt_to_category():
    with open(DATASET) as f:
        data = json.load(f)
    mapping = {}
    for cat, items in data["english"].items():
        for item in items:
            mapping[item["without_cues"]] = cat
            mapping[item["with_cues"]] = cat
    return mapping

# ── Load data ────────────────────────────────────────────────────────────────
def load_condition(folder, filename, prompt_to_cat):
    """Returns {model: {"metrics": {metric: [scores]}, "categories": {cat: [scores]}}}"""
    result = {}
    for model in MODEL_ORDER:
        path = EVAL_DIR / model / folder / filename
        if not path.exists():
            continue
        with open(path) as f:
            items = json.load(f)
        metrics = {m: [] for m in METRICS}
        cats = {c: [] for c in CATEGORY_ORDER}
        for item in items:
            ev = item.get("json_evaluation")
            if ev is None:
                continue
            row_scores = []
            for m in METRICS:
                v = ev.get(m)
                if v is not None:
                    metrics[m].append(v)
                    row_scores.append(v)
            cat = prompt_to_cat.get(item["prompt"])
            if cat and row_scores:
                cats[cat].append(np.mean(row_scores))
        result[model] = {"metrics": metrics, "categories": cats}
    return result

# ── Build markdown table ─────────────────────────────────────────────────────
def metric_table(data):
    header = "| Model | " + " | ".join(METRIC_LABELS) + " | Total |\n"
    sep = "|---|" + "---|" * (len(METRIC_LABELS) + 1) + "\n"
    rows = ""
    for model in MODEL_ORDER:
        if model not in data:
            continue
        name = MODEL_DISPLAY[model]
        vals = [np.mean(data[model]["metrics"][m]) if data[model]["metrics"][m] else float("nan")
                for m in METRICS]
        total = np.nanmean(vals)
        row = f"| {name} | " + " | ".join(f"{v:.3f}" for v in vals) + f" | {total:.3f} |\n"
        rows += row
    return header + sep + rows

def category_table(data):
    header = "| Model | " + " | ".join(CATEGORY_SHORT[c] for c in CATEGORY_ORDER) + " | Total |\n"
    sep = "|---|" + "---|" * (len(CATEGORY_ORDER) + 1) + "\n"
    rows = ""
    for model in MODEL_ORDER:
        if model not in data:
            continue
        name = MODEL_DISPLAY[model]
        vals = [np.mean(data[model]["categories"][c]) if data[model]["categories"][c] else float("nan")
                for c in CATEGORY_ORDER]
        total = np.nanmean(vals)
        row = f"| {name} | " + " | ".join(f"{v:.3f}" for v in vals) + f" | {total:.3f} |\n"
        rows += row
    return header + sep + rows

# ── Compute totals for plot ───────────────────────────────────────────────────
def compute_total(data, model):
    vals = [np.mean(data[model]["metrics"][m]) if data[model]["metrics"][m] else float("nan")
            for m in METRICS]
    return np.nanmean(vals)

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prompt_to_cat = build_prompt_to_category()

    condition_data = {}
    for cond, (folder, filename) in CONDITIONS.items():
        condition_data[cond] = load_condition(folder, filename, prompt_to_cat)

    # Markdown
    md = "# Cues Analysis\n\n"
    for cond in ["No Cue", "Implicit Cue", "Explicit Age"]:
        data = condition_data[cond]
        md += f"## {cond}\n\n"
        md += "### Per Metric\n\n"
        md += metric_table(data) + "\n"
        md += "### Per Category\n\n"
        md += category_table(data) + "\n"

    with open(OUT_DIR / "cues_analysis.md", "w") as f:
        f.write(md)
    print("Wrote cues_analysis.md")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    y_labels = [MODEL_DISPLAY[m] for m in MODEL_ORDER]
    n = len(MODEL_ORDER)
    y_pos = list(range(n))

    offsets = {"No Cue": -0.18, "Implicit Cue": 0.0, "Explicit Age": 0.18}

    # Draw connecting line from min to max score per model row
    for i, model in enumerate(MODEL_ORDER):
        vals = [compute_total(condition_data[c], model)
                for c in ["No Cue", "Implicit Cue", "Explicit Age"]
                if model in condition_data[c]]
        vals = [v for v in vals if not np.isnan(v)]
        if len(vals) >= 2:
            ax.plot([min(vals), max(vals)], [i, i],
                    color="#BBBBBB", lw=1.8, zorder=2, solid_capstyle="round")

    for cond in ["No Cue", "Implicit Cue", "Explicit Age"]:
        data = condition_data[cond]
        xs, ys = [], []
        for i, model in enumerate(MODEL_ORDER):
            if model not in data:
                continue
            total = compute_total(data, model)
            if not np.isnan(total):
                xs.append(total)
                ys.append(i)
        ax.scatter(xs, ys, color=COLORS[cond], s=260, zorder=3,
                   label=cond, marker="o")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels[::-1] if False else y_labels, fontsize=22)
    ax.invert_yaxis()
    ax.set_xlabel("Average Score (1–5)", fontsize=24)
    ax.tick_params(axis="x", labelsize=22)
    ax.set_xlim(2.5, 5.2)
    ax.legend(fontsize=20, frameon=True, framealpha=0.95, edgecolor="#cccccc",
              loc="upper center", bbox_to_anchor=(0.5, 1.08),
              ncol=3, handletextpad=0.4, columnspacing=1.2, borderpad=0.6)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "cues_analysis.png", dpi=300, bbox_inches="tight",
                facecolor=BG)
    plt.close(fig)
    print("Wrote cues_analysis.png")

if __name__ == "__main__":
    main()
