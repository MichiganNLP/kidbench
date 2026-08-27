"""Generate multi_turn_analysis.md and degradation bar charts."""

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
EVAL_DIR = ROOT / "evaluations" / "deepseek-v4-pro" / "multi_turn"
OUT_DIR = ROOT / "evaluations" / "analysis" / "multi_turn"

MODEL_ORDER = [
    "llama-3.2-3b", "llama-3.1-8b", "llama-3.3-70b",
    "gemma-3-4b", "gemma-3-12b", "gemma-4-31b",
    "deepseek-v4-flash",
    "qwen-3-8b", "qwen-3.6-27b",
    "gemini-3.1-flash-lite", "gpt-5-mini", "claude-haiku-4-5",
]
MODEL_DISPLAY = {
    "llama-3.2-3b": "Llama 3.2-3B", "llama-3.1-8b": "Llama 3.1-8B",
    "llama-3.3-70b": "Llama 3.3-70B", "gemma-3-4b": "Gemma 3-4B",
    "gemma-3-12b": "Gemma 3-12B", "gemma-4-31b": "Gemma 4-31B",
    "deepseek-v4-flash": "DeepSeek V4 Flash", "qwen-3-8b": "Qwen 3-8B",
    "qwen-3.6-27b": "Qwen 3.6-27B",
    "gemini-3.1-flash-lite": "Gemini 3.1 Flash Lite",
    "gpt-5-mini": "GPT 5 Mini", "claude-haiku-4-5": "Claude Haiku 4.5",
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

TURNS = [1, 2, 3, 4, 5]

# ── Load data ────────────────────────────────────────────────────────────────
def load_model(model, filename):
    """Returns list of items."""
    path = EVAL_DIR / model / filename
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)

# ── Compute slopes ───────────────────────────────────────────────────────────
def fit_slope(ys):
    """Fit β₁ over turns 1..5. Return D_slope = -β₁."""
    xs = np.array(TURNS, dtype=float)
    ys = np.array(ys, dtype=float)
    # Simple linear regression
    x_mean = xs.mean()
    y_mean = ys.mean()
    beta1 = np.sum((xs - x_mean) * (ys - y_mean)) / np.sum((xs - x_mean) ** 2)
    return -beta1  # positive = degradation

def per_metric_slopes(items):
    """Returns {metric: D_slope}."""
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
            series.append(np.mean(scores) if scores else np.nan)
        if any(not np.isnan(v) for v in series):
            result[metric] = fit_slope(series)
        else:
            result[metric] = np.nan
    return result

def per_category_slopes(items):
    """Returns {category: D_slope} using mean of 5 metrics per item."""
    result = {}
    for cat in CATEGORY_ORDER:
        cat_items = [item for item in items if item.get("category") == cat]
        if not cat_items:
            result[cat] = np.nan
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
            series.append(np.mean(scores) if scores else np.nan)
        result[cat] = fit_slope(series)
    return result

def total_slope(items):
    """D_slope of mean-of-5-metrics over turns."""
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
        series.append(np.mean(scores) if scores else np.nan)
    return fit_slope(series)

def peak_degradation(items):
    """Returns T1 score, lowest turn, lowest score, delta for mean-of-5."""
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
        turn_scores[t] = np.mean(scores) if scores else np.nan
    t1 = turn_scores[1]
    later = {t: v for t, v in turn_scores.items() if t > 1 and not np.isnan(v)}
    if not later:
        return t1, np.nan, np.nan, np.nan
    lowest_t = min(later, key=later.get)
    lowest_score = later[lowest_t]
    delta = t1 - lowest_score
    return t1, lowest_t, lowest_score, delta

# ── Markdown tables ──────────────────────────────────────────────────────────
def slope_metric_table(model_slopes):
    header = "| Model | " + " | ".join(METRIC_LABELS) + " |\n"
    sep = "|---|" + "---|" * len(METRIC_LABELS) + "\n"
    rows = ""
    for model in MODEL_ORDER:
        if model not in model_slopes:
            continue
        name = MODEL_DISPLAY[model]
        vals = [model_slopes[model].get(m, np.nan) for m in METRICS]
        rows += f"| {name} | " + " | ".join(f"{v:.4f}" if not np.isnan(v) else "N/A" for v in vals) + " |\n"
    return header + sep + rows

def slope_category_table(model_cat_slopes):
    header = "| Model | " + " | ".join(CATEGORY_SHORT[c] for c in CATEGORY_ORDER) + " |\n"
    sep = "|---|" + "---|" * len(CATEGORY_ORDER) + "\n"
    rows = ""
    for model in MODEL_ORDER:
        if model not in model_cat_slopes:
            continue
        name = MODEL_DISPLAY[model]
        vals = [model_cat_slopes[model].get(c, np.nan) for c in CATEGORY_ORDER]
        rows += f"| {name} | " + " | ".join(f"{v:.4f}" if not np.isnan(v) else "N/A" for v in vals) + " |\n"
    return header + sep + rows

def peak_table(model_peaks):
    header = "| Model | T1 | Lowest Turn | Lowest Score | Delta |\n"
    sep = "|---|---|---|---|---|\n"
    rows = ""
    for model in MODEL_ORDER:
        if model not in model_peaks:
            continue
        name = MODEL_DISPLAY[model]
        t1, lt, ls, delta = model_peaks[model]
        t1_s = f"{t1:.3f}" if not np.isnan(t1) else "N/A"
        lt_s = str(int(lt)) if not np.isnan(lt) else "N/A"
        ls_s = f"{ls:.3f}" if not np.isnan(ls) else "N/A"
        delta_s = f"{delta:.3f}" if not np.isnan(delta) else "N/A"
        rows += f"| {name} | {t1_s} | {lt_s} | {ls_s} | {delta_s} |\n"
    return header + sep + rows

# ── Two-condition diverging bar chart ────────────────────────────────────────
COL_WITHOUT = "#FF495E"   # coral red  — without_age
COL_WITH    = "#235A61"   # deep teal  — with_age

def plot_two_conditions(vals_without, vals_with, xlabel, out_path):
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    n = len(MODEL_ORDER)
    SPACING = 2.2
    y_pos = [i * SPACING for i in range(n)]
    BAR_H = 0.7
    OFFSET = 0.42

    wa_vals  = [vals_without.get(m, np.nan) for m in MODEL_ORDER]
    age_vals = [vals_with.get(m, np.nan)    for m in MODEL_ORDER]

    # Alternating row backgrounds
    for i, y in enumerate(y_pos):
        color = "#F4F4F4" if i % 2 == 0 else "#FFFFFF"
        ax.axhspan(y - SPACING / 2, y + SPACING / 2, color=color, zorder=0)

    # Without age bars (top)
    ax.barh([y + OFFSET for y in y_pos], wa_vals,
            height=BAR_H, color=COL_WITHOUT, alpha=0.88, zorder=3, label="Without Age")
    # With age bars (bottom)
    ax.barh([y - OFFSET for y in y_pos], age_vals,
            height=BAR_H, color=COL_WITH, alpha=0.88, zorder=3, label="With Age")

    ax.axvline(0, color="#444444", linewidth=1.2, linestyle="--", zorder=4)

    # Annotations
    for i, (wv, av) in enumerate(zip(wa_vals, age_vals)):
        for v, y_off in [(wv, OFFSET), (av, -OFFSET)]:
            if np.isnan(v):
                continue
            if v >= 0:
                x, ha = v + 0.002, "left"
            else:
                x, ha = 0.002, "left"
            ax.text(x, y_pos[i] + y_off, f"{v:.3f}", va="center", ha=ha,
                    fontsize=13, color="#333333", fontweight="bold", fontfamily="DejaVu Sans")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([MODEL_DISPLAY[m] for m in MODEL_ORDER], fontsize=22)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=24, labelpad=10)
    ax.tick_params(axis="x", labelsize=22)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.xaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    ax.legend(fontsize=20, frameon=True, framealpha=0.95, edgecolor="#cccccc",
              loc="upper center", bbox_to_anchor=(0.5, 1.06),
              ncol=2, handletextpad=0.5, columnspacing=1.5, borderpad=0.6)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Wrote {out_path.name}")

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    for condition in ["without_age", "with_age"]:
        filename = f"{condition}.json"
        model_metric_slopes = {}
        model_cat_slopes = {}
        model_peaks = {}
        model_totals = {}
        for model in MODEL_ORDER:
            items = load_model(model, filename)
            if items is None:
                continue
            model_metric_slopes[model] = per_metric_slopes(items)
            model_cat_slopes[model] = per_category_slopes(items)
            model_peaks[model] = peak_degradation(items)
            model_totals[model] = total_slope(items)
        results[condition] = {
            "metric_slopes": model_metric_slopes,
            "cat_slopes": model_cat_slopes,
            "peaks": model_peaks,
            "totals": model_totals,
        }

    # Markdown
    md = "# Multi-Turn Analysis\n\n"
    md += "## Quality Degradation Slope (β₁) — Per Metric\n\n"
    md += "### Without Age\n\n"
    md += slope_metric_table(results["without_age"]["metric_slopes"]) + "\n"
    md += "### With Age\n\n"
    md += slope_metric_table(results["with_age"]["metric_slopes"]) + "\n"

    md += "## Quality Degradation Slope (β₁) — Per Category\n\n"
    md += "### Without Age\n\n"
    md += slope_category_table(results["without_age"]["cat_slopes"]) + "\n"
    md += "### With Age\n\n"
    md += slope_category_table(results["with_age"]["cat_slopes"]) + "\n"

    md += "## Peak Degradation (T1 − min(T2..T5))\n\n"
    md += "### Without Age\n\n"
    md += peak_table(results["without_age"]["peaks"]) + "\n"
    md += "### With Age\n\n"
    md += peak_table(results["with_age"]["peaks"]) + "\n"

    with open(OUT_DIR / "multi_turn_analysis.md", "w") as f:
        f.write(md)
    print("Wrote multi_turn_analysis.md")

    # Plots — 2 charts, each with two bars per model
    peak_delta_without = {m: results["without_age"]["peaks"][m][3]
                          for m in results["without_age"]["peaks"]}
    peak_delta_with    = {m: results["with_age"]["peaks"][m][3]
                          for m in results["with_age"]["peaks"]}

    plot_two_conditions(
        results["without_age"]["totals"],
        results["with_age"]["totals"],
        "Degradation Slope",
        OUT_DIR / "multi_turn_slope.png",
    )
    plot_two_conditions(
        peak_delta_without,
        peak_delta_with,
        "Peak Quality Drop",
        OUT_DIR / "multi_turn_peak.png",
    )

if __name__ == "__main__":
    main()
