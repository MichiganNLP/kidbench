"""Generate evaluations/analysis/gold/guard_analysis.md comparing 3 LlamaPlushieGuard checkpoints."""

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

# ── Config ────────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parents[3]
RESPONSES_DIR = ROOT / "responses" / "judge"
OUT_DIR = ROOT / "evaluations" / "analysis" / "gold"

CHECKPOINTS = {
    "Checkpoint 1": "llamaplushiegaurd-3-8b-1",
    "Checkpoint 2": "llamaplushiegaurd-3-8b-2",
    "Checkpoint 3": "llamaplushiegaurd-3-8b-3",
}

OVERALL_METRICS = ["spearman", "qwk", "mae", "exact_acc", "within1_acc", "macro_f1_remap", "acc_remap"]
OVERALL_LABELS  = ["Spearman ρ", "QWK", "MAE", "Exact Acc", "Within-1 Acc", "Macro F1", "Acc (remap)"]

PER_METRICS = ["safety", "developmental_fit", "emotional_support", "moral_guidance", "boundary_setting", "cultural_alignment"]
PER_LABELS  = ["Safety", "Dev. Fit", "Emotional", "Moral", "Boundary", "Cultural"]

PER_SUBMETS = ["spearman", "mae", "exact_acc", "within1_acc"]
PER_SUBLABELS = ["Spearman ρ", "MAE", "Exact Acc", "Within-1 Acc"]

# Higher = better for most; MAE = lower is better
HIGHER_BETTER = {"spearman": True, "qwk": True, "mae": False,
                 "exact_acc": True, "within1_acc": True,
                 "macro_f1_remap": True, "acc_remap": True}

COL_1 = "#FF495E"
COL_2 = "#5DADE2"
COL_3 = "#235A61"
COLORS = [COL_1, COL_2, COL_3]

# ── Load ──────────────────────────────────────────────────────────────────────
def load_metrics():
    data = {}
    for label, folder in CHECKPOINTS.items():
        path = RESPONSES_DIR / folder / "metrics.json"
        if not path.exists():
            print(f"  WARNING: {path} not found")
            continue
        with open(path) as f:
            data[label] = json.load(f)
    return data


# ── Markdown helpers ──────────────────────────────────────────────────────────
def fmt(v, metric):
    if np.isnan(v):
        return "N/A"
    return f"{v:.4f}"


def overall_table(data):
    ckpts = list(data.keys())
    header = "| Metric | " + " | ".join(ckpts) + " | Best |\n"
    sep = "|---|" + "---|" * (len(ckpts) + 1) + "\n"
    rows = ""
    for m, lbl in zip(OVERALL_METRICS, OVERALL_LABELS):
        vals = [data[c]["overall"].get(m, float("nan")) for c in ckpts]
        best_fn = max if HIGHER_BETTER.get(m, True) else min
        best_val = best_fn(v for v in vals if not np.isnan(v))
        best_ckpt = ckpts[vals.index(best_val)]
        row = f"| {lbl} | " + " | ".join(
            f"**{fmt(v, m)}**" if v == best_val else fmt(v, m) for v in vals
        ) + f" | {best_ckpt} |\n"
        rows += row
    return header + sep + rows


def per_metric_table(data, submetric, sublabel):
    ckpts = list(data.keys())
    header = "| Metric | " + " | ".join(ckpts) + " | Best |\n"
    sep = "|---|" + "---|" * (len(ckpts) + 1) + "\n"
    rows = ""
    for m, lbl in zip(PER_METRICS, PER_LABELS):
        vals = [data[c]["per_metric"].get(m, {}).get(submetric, float("nan")) for c in ckpts]
        best_fn = max if HIGHER_BETTER.get(submetric, True) else min
        valid = [v for v in vals if not np.isnan(v)]
        if not valid:
            rows += f"| {lbl} | " + " | ".join("N/A" for _ in vals) + " | N/A |\n"
            continue
        best_val = best_fn(valid)
        best_ckpt = ckpts[vals.index(best_val)]
        row = f"| {lbl} | " + " | ".join(
            f"**{fmt(v, submetric)}**" if v == best_val else fmt(v, submetric) for v in vals
        ) + f" | {best_ckpt} |\n"
        rows += row
    return header + sep + rows


def parse_table(data):
    ckpts = list(data.keys())
    header = "| | " + " | ".join(ckpts) + " |\n"
    sep = "|---|" + "---|" * len(ckpts) + "\n"
    rows = ""
    for key, lbl in [("n_total", "Total"), ("n_parsed", "Parsed"), ("n_failed", "Failed"), ("parse_rate", "Parse Rate")]:
        vals = [data[c].get(key, float("nan")) for c in ckpts]
        row = f"| {lbl} | "
        if key == "parse_rate":
            row += " | ".join(f"{v:.4f}" if not np.isnan(v) else "N/A" for v in vals)
        else:
            row += " | ".join(f"{int(v):,}" if not np.isnan(v) else "N/A" for v in vals)
        row += " |\n"
        rows += row
    return header + sep + rows


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_overall(data, out_path):
    ckpts = list(data.keys())
    metrics_to_plot = [m for m in OVERALL_METRICS if m != "mae"]
    labels_to_plot  = [l for m, l in zip(OVERALL_METRICS, OVERALL_LABELS) if m != "mae"]

    n_metrics = len(metrics_to_plot)
    x = np.arange(n_metrics)
    width = 0.22

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [5, 1]})
    fig.patch.set_facecolor(BG)

    ax = axes[0]
    ax.set_facecolor(BG)
    for i, (ckpt, col) in enumerate(zip(ckpts, COLORS)):
        vals = [data[ckpt]["overall"].get(m, float("nan")) for m in metrics_to_plot]
        offset = (i - 1) * width
        ax.bar(x + offset, vals, width, color=col, alpha=0.88, label=ckpt, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels_to_plot, fontsize=16)
    ax.set_ylabel("Score", fontsize=18)
    ax.tick_params(axis="y", labelsize=15)
    ax.set_ylim(0, 1.05)
    ax.xaxis.grid(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # MAE in separate panel (lower = better)
    ax2 = axes[1]
    ax2.set_facecolor(BG)
    mae_vals = [data[ckpt]["overall"].get("mae", float("nan")) for ckpt in ckpts]
    for i, (v, col) in enumerate(zip(mae_vals, COLORS)):
        ax2.bar([i], [v], color=col, alpha=0.88, zorder=3)
    ax2.set_xticks([])
    ax2.set_ylabel("MAE (lower = better)", fontsize=14)
    ax2.tick_params(axis="y", labelsize=14)
    ax2.set_xlabel("MAE", fontsize=15)
    ax2.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax2.set_axisbelow(True)
    for spine in ax2.spines.values():
        spine.set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.88) for c in COLORS]
    fig.legend(handles, ckpts, fontsize=16, frameon=True, framealpha=0.95,
               edgecolor="#cccccc", loc="upper center", bbox_to_anchor=(0.5, 1.04),
               ncol=3, handletextpad=0.5, columnspacing=1.2)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Wrote {out_path.name}")


def plot_per_metric(data, submetric, sublabel, out_path):
    ckpts = list(data.keys())
    x = np.arange(len(PER_METRICS))
    width = 0.22

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    for i, (ckpt, col) in enumerate(zip(ckpts, COLORS)):
        vals = [data[ckpt]["per_metric"].get(m, {}).get(submetric, float("nan")) for m in PER_METRICS]
        offset = (i - 1) * width
        ax.bar(x + offset, vals, width, color=col, alpha=0.88, label=ckpt, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(PER_LABELS, fontsize=18)
    ax.set_ylabel(sublabel, fontsize=20)
    ax.tick_params(axis="y", labelsize=16)
    if submetric != "mae":
        ax.set_ylim(0, 1.05)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.legend(fontsize=16, frameon=True, framealpha=0.95, edgecolor="#cccccc",
              loc="upper center", bbox_to_anchor=(0.5, 1.12),
              ncol=3, handletextpad=0.5, columnspacing=1.2)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Wrote {out_path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_metrics()

    if not data:
        print("No metrics found — check RESPONSES_DIR path.")
        return

    # Markdown — overall table only
    md = "# LlamaPlushieGuard Checkpoint Comparison\n\n"
    md += "*Bold = best checkpoint for that metric. MAE: lower is better; all others: higher is better.*\n\n"
    md += overall_table(data) + "\n"

    with open(OUT_DIR / "guard_analysis.md", "w") as f:
        f.write(md)
    print("Wrote guard_analysis.md")


if __name__ == "__main__":
    main()
