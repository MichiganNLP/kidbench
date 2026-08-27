"""Generate cultural_analysis.md and cultural_analysis.png."""

import json
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
OUT_DIR = ROOT / "evaluations" / "analysis" / "cultural"
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

COUNTRIES = {
    "Pakistan": ("with_cues", "english_true_pakistan.json"),
    "India": ("with_cues", "english_true_india.json"),
    "China": ("with_cues", "english_true_china.json"),
    "Nigeria": ("with_cues", "english_true_nigeria.json"),
}
COLORS = {
    "Pakistan": "#01411C",
    "India": "#FF9933",
    "China": "#DE2910",
    "Nigeria": "#80C080",
}

# ── Build prompt→category map ────────────────────────────────────────────────
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

# ── Load data ────────────────────────────────────────────────────────────────
def load_country(folder, filename, prompt_to_cat):
    result = {}
    for model in MODEL_ORDER:
        path = EVAL_DIR / model / folder / filename
        if not path.exists():
            continue
        with open(path) as f:
            items = json.load(f)
        cats = {c: [] for c in CATEGORY_ORDER}
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

# ── Table ────────────────────────────────────────────────────────────────────
def category_table(data):
    header = "| Model | " + " | ".join(CATEGORY_SHORT[c] for c in CATEGORY_ORDER) + " |\n"
    sep = "|---|" + "---|" * len(CATEGORY_ORDER) + "\n"
    rows = ""
    for model in MODEL_ORDER:
        if model not in data:
            continue
        name = MODEL_DISPLAY[model]
        vals = [np.mean(data[model]["categories"][c]) if data[model]["categories"][c] else float("nan")
                for c in CATEGORY_ORDER]
        rows += f"| {name} | " + " | ".join(f"{v:.3f}" for v in vals) + " |\n"
    return header + sep + rows

def compute_total(data, model):
    scores = data[model]["all"]
    return np.mean(scores) if scores else float("nan")

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prompt_to_cat = build_prompt_to_category()

    country_data = {}
    for country, (folder, fname) in COUNTRIES.items():
        country_data[country] = load_country(folder, fname, prompt_to_cat)

    # Markdown
    md = "# Cultural Analysis\n\n"
    for country in ["Pakistan", "India", "China", "Nigeria"]:
        data = country_data[country]
        md += f"## {country}\n\n"
        md += "### Per Category (Cultural Alignment Score)\n\n"
        md += category_table(data) + "\n"

    with open(OUT_DIR / "cultural_analysis.md", "w") as f:
        f.write(md)
    print("Wrote cultural_analysis.md")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    y_labels = [MODEL_DISPLAY[m] for m in MODEL_ORDER]
    n = len(MODEL_ORDER)
    y_pos = list(range(n))

    # Draw connecting line from min to max score per model row
    for i, model in enumerate(MODEL_ORDER):
        vals = [compute_total(country_data[c], model)
                for c in ["Pakistan", "India", "China", "Nigeria"]
                if model in country_data[c]]
        vals = [v for v in vals if not np.isnan(v)]
        if len(vals) >= 2:
            ax.plot([min(vals), max(vals)], [i, i],
                    color="#BBBBBB", lw=1.8, zorder=2, solid_capstyle="round")

    for country in ["Pakistan", "India", "China", "Nigeria"]:
        data = country_data[country]
        xs, ys = [], []
        for i, model in enumerate(MODEL_ORDER):
            if model not in data:
                continue
            total = compute_total(data, model)
            if not np.isnan(total):
                xs.append(total)
                ys.append(i)
        ax.scatter(xs, ys, color=COLORS[country], s=260, zorder=3,
                   label=country, marker="o")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=22)
    ax.invert_yaxis()
    ax.set_xlabel("Cultural Alignment Score (1–5)", fontsize=24)
    ax.tick_params(axis="x", labelsize=22)
    ax.set_xlim(2.3, 5.2)
    ax.legend(fontsize=20, frameon=True, framealpha=0.95, edgecolor="#cccccc",
              loc="upper center", bbox_to_anchor=(0.5, 1.08),
              ncol=4, handletextpad=0.4, columnspacing=1.2, borderpad=0.6)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "cultural_analysis.png", dpi=300, bbox_inches="tight",
                facecolor=BG)
    plt.close(fig)
    print("Wrote cultural_analysis.png")

if __name__ == "__main__":
    main()
