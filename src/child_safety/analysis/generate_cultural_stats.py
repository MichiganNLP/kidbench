"""Generate evaluations/analysis/cultural/cultural_stats.md with statistical analysis."""

import json
import pathlib
import numpy as np
from scipy import stats as scipy_stats
from statsmodels.stats.multitest import multipletests

# ── Config ────────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parents[3]
EVAL_DIR = ROOT / "evaluations" / "deepseek-v4-pro" / "single_turn"
DATASET = ROOT / "dataset" / "kidbench" / "kidbench_single.json"
OUT_DIR = ROOT / "evaluations" / "analysis" / "cultural"

MODEL_ORDER = [
    "llama-3.2-3b", "llama-3-8b", "llama-3.3-70b",
    "gemma-3-4b", "gemma-3-12b", "gemma-4-31b",
    "deepseek-v4-flash",
    "qwen-3.1-8b", "qwen-3.5-4b", "qwen-3.6-27b",
    "gemini-3.1-flash-lite", "gpt-5-mini", "claude-haiku-4.5",
]

COUNTRIES = {
    "Pakistan": ("with_cues", "english_true_pakistan.json"),
    "India":    ("with_cues", "english_true_india.json"),
    "China":    ("with_cues", "english_true_china.json"),
    "Nigeria":  ("with_cues", "english_true_nigeria.json"),
}

COUNTRY_ORDER = ["Pakistan", "India", "China", "Nigeria"]

N_BOOT = 10000
SEED = 42


# ── Dataset prompt map ────────────────────────────────────────────────────────
def build_prompt_to_qid():
    """Return dict mapping with_cues prompt text -> qid (category_i)."""
    with open(DATASET) as fh:
        data = json.load(fh)
    prompt_to_qid = {}
    for cat, items in data["english"].items():
        for i, item in enumerate(items):
            qid = f"{cat}_{i}"
            prompt_to_qid[item["with_cues"]] = qid
    return prompt_to_qid


# ── Load country data ─────────────────────────────────────────────────────────
def load_country(folder, filename, prompt_to_qid):
    """
    Load cultural_alignment scores for one country across all models.
    Returns dict: {(model, qid): cultural_alignment_score}
    """
    records = {}
    for model in MODEL_ORDER:
        path = EVAL_DIR / model / folder / filename
        if not path.exists():
            continue
        with open(path) as fh:
            items = json.load(fh)
        for item in items:
            prompt = item.get("prompt", "")
            qid = prompt_to_qid.get(prompt)
            if qid is None:
                continue
            ev = item.get("json_evaluation")
            if ev is None:
                continue
            score = ev.get("cultural_alignment")
            if score is None:
                continue
            records[(model, qid)] = float(score)
    return records


# ── Bootstrap CI ──────────────────────────────────────────────────────────────
def bootstrap_ci_mean(arr, n_boot=N_BOOT, seed=SEED):
    """95% bootstrap CI for the mean of arr."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(arr, dtype=float)
    boot_means = np.array([
        rng.choice(arr, size=len(arr), replace=True).mean()
        for _ in range(n_boot)
    ])
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def bootstrap_ci_mean_diff(diffs, n_boot=N_BOOT, seed=SEED):
    """95% bootstrap CI for the mean of a differences array."""
    return bootstrap_ci_mean(diffs, n_boot=n_boot, seed=seed)


# ── Rank-biserial correlation ─────────────────────────────────────────────────
def rank_biserial(diffs):
    """
    Rank-biserial r for Wilcoxon signed-rank test.
    r = (W_plus - W_minus) / (n*(n+1)/2)
    Computed from the nonzero differences' positive and negative rank sums.
    """
    diffs = np.asarray(diffs, dtype=float)
    nonzero = diffs[diffs != 0]
    n = len(nonzero)
    if n == 0:
        return float("nan")
    ranks = scipy_stats.rankdata(np.abs(nonzero))
    w_plus = float(ranks[nonzero > 0].sum())
    w_minus = float(ranks[nonzero < 0].sum())
    denom = n * (n + 1) / 2.0
    return (w_plus - w_minus) / denom


# ── Per-country summary ───────────────────────────────────────────────────────
def country_summary(records):
    """Compute flat summary stats for all scores in one country dict."""
    scores = np.asarray(list(records.values()), dtype=float)
    n = len(scores)
    mean = float(scores.mean())
    median = float(np.median(scores))
    sd = float(scores.std(ddof=1)) if n > 1 else float("nan")
    ci_lo, ci_hi = bootstrap_ci_mean(scores)
    return {"n": n, "mean": mean, "median": median, "sd": sd,
            "ci_lo": ci_lo, "ci_hi": ci_hi}


# ── Friedman test ─────────────────────────────────────────────────────────────
def friedman_test(country_records):
    """
    Friedman test across 4 countries.
    Groups by (model, qid) requiring all 4 countries present.
    Returns dict with statistic, p, df, n_blocks.
    """
    # Find common (model, qid) keys present in all 4 countries
    key_sets = [set(country_records[c].keys()) for c in COUNTRY_ORDER]
    common_keys = sorted(key_sets[0].intersection(*key_sets[1:]))

    if len(common_keys) < 4:
        return {"statistic": float("nan"), "p": float("nan"), "df": 3,
                "n_blocks": len(common_keys)}

    cols = []
    for country in COUNTRY_ORDER:
        cols.append(np.array([country_records[country][k] for k in common_keys],
                              dtype=float))

    stat, p = scipy_stats.friedmanchisquare(*cols)
    return {
        "statistic": float(stat),
        "p": float(p),
        "df": 3,
        "n_blocks": len(common_keys),
    }


# ── Pairwise comparison ───────────────────────────────────────────────────────
def paired_comparison(records_a, records_b):
    """
    Match by (model, qid). mean_diff = B − A.
    Returns dict of stats or None if fewer than 2 matched pairs.
    """
    common_keys = sorted(set(records_a.keys()) & set(records_b.keys()))
    if len(common_keys) < 2:
        return None
    scores_a = np.array([records_a[k] for k in common_keys], dtype=float)
    scores_b = np.array([records_b[k] for k in common_keys], dtype=float)
    diffs = scores_b - scores_a

    n = len(diffs)
    mean_a = float(scores_a.mean())
    mean_b = float(scores_b.mean())
    mean_diff = float(diffs.mean())
    ci_lo, ci_hi = bootstrap_ci_mean_diff(diffs)

    if np.all(diffs == 0):
        wilcoxon_p = 1.0
    else:
        try:
            _, wilcoxon_p = scipy_stats.wilcoxon(diffs, alternative="two-sided")
        except ValueError:
            wilcoxon_p = float("nan")

    r = rank_biserial(diffs)

    return {
        "n": n,
        "mean_a": mean_a,
        "mean_b": mean_b,
        "mean_diff": mean_diff,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "wilcoxon_p": wilcoxon_p,
        "r": r,
    }


# ── Model-level descriptive summary ──────────────────────────────────────────
def model_level_summary(country_records):
    """
    For each model, compute mean cultural_alignment per country and overall.
    Returns list of dicts with model, per-country means, and overall mean.
    """
    rows = []
    for model in MODEL_ORDER:
        row = {"model": model}
        all_scores = []
        for country in COUNTRY_ORDER:
            scores = [v for (m, _), v in country_records[country].items()
                      if m == model]
            if scores:
                row[country] = float(np.mean(scores))
                all_scores.extend(scores)
            else:
                row[country] = float("nan")
        row["overall"] = float(np.mean(all_scores)) if all_scores else float("nan")
        rows.append(row)
    return rows


def descriptive_highlights(model_rows):
    """Generate descriptive text highlights from model-level summary."""
    lines = []

    # Best model overall
    valid = [(r["model"], r["overall"]) for r in model_rows
             if not np.isnan(r["overall"])]
    if valid:
        best_model, best_val = max(valid, key=lambda x: x[1])
        lines.append(
            f"- **Best model overall**: {best_model} "
            f"(mean cultural_alignment = {best_val:.4f})"
        )

    # Models scoring ≥ 4.0 in ALL 4 countries
    threshold = 4.0
    ge4_models = []
    for r in model_rows:
        if all(not np.isnan(r[c]) and r[c] >= threshold for c in COUNTRY_ORDER):
            ge4_models.append(r["model"])
    if ge4_models:
        lines.append(
            f"- **Models scoring ≥ {threshold} in ALL 4 countries**: "
            + ", ".join(ge4_models)
        )
    else:
        lines.append(f"- **Models scoring ≥ {threshold} in ALL 4 countries**: None")

    # Lowest-performing for Pakistan
    pak_valid = [(r["model"], r["Pakistan"]) for r in model_rows
                 if not np.isnan(r["Pakistan"])]
    if pak_valid:
        worst_pak, worst_pak_val = min(pak_valid, key=lambda x: x[1])
        lines.append(
            f"- **Lowest-performing model for Pakistan**: {worst_pak} "
            f"(mean = {worst_pak_val:.4f})"
        )

    # Lowest-performing for India
    ind_valid = [(r["model"], r["India"]) for r in model_rows
                 if not np.isnan(r["India"])]
    if ind_valid:
        worst_ind, worst_ind_val = min(ind_valid, key=lambda x: x[1])
        lines.append(
            f"- **Lowest-performing model for India**: {worst_ind} "
            f"(mean = {worst_ind_val:.4f})"
        )

    # Country where each model scores highest
    lines.append("- **Country where each model scores highest**:")
    for r in model_rows:
        country_scores = {c: r[c] for c in COUNTRY_ORDER if not np.isnan(r[c])}
        if country_scores:
            best_c = max(country_scores, key=lambda c: country_scores[c])
            lines.append(
                f"  - {r['model']}: {best_c} (mean = {country_scores[best_c]:.4f})"
            )
        else:
            lines.append(f"  - {r['model']}: N/A (no data)")

    return "\n".join(lines)


# ── Markdown formatting ───────────────────────────────────────────────────────
def fmt(v, decimals=4):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v:.{decimals}f}"


def build_markdown(country_records, summaries, friedman_result,
                   pairwise_results, model_rows):
    lines = []
    lines.append("# Cultural Alignment Statistical Analysis\n")

    # ── Per-Country Summary ──
    lines.append("## Per-Country Summary\n")
    lines.append(
        "| Country | n | Mean | Median | SD | CI Lower | CI Upper |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for country in COUNTRY_ORDER:
        s = summaries[country]
        lines.append(
            f"| {country} "
            f"| {s['n']} "
            f"| {fmt(s['mean'])} "
            f"| {fmt(s['median'])} "
            f"| {fmt(s['sd'])} "
            f"| {fmt(s['ci_lo'])} "
            f"| {fmt(s['ci_hi'])} |"
        )
    lines.append("")

    # ── Omnibus Test (Friedman) ──
    lines.append("## Omnibus Test (Friedman)\n")
    fr = friedman_result
    lines.append(
        f"Friedman test across all 4 countries ({COUNTRY_ORDER}), "
        f"grouped by (model, qid) requiring all countries present.\n"
    )
    lines.append(f"- N blocks (matched quadruples): {fr['n_blocks']}")
    lines.append(f"- Degrees of freedom: {fr['df']}")
    lines.append(f"- Statistic: {fmt(fr['statistic'])}")
    lines.append(f"- p-value: {fmt(fr['p'])}")
    lines.append("")

    # ── Pairwise Comparisons ──
    lines.append("## Pairwise Comparisons\n")
    lines.append(
        "Mean Diff = Mean B − Mean A. "
        "Holm correction applied across all 6 pairwise p-values.\n"
    )
    lines.append(
        "| Comparison | n pairs | Mean A | Mean B | Mean Diff "
        "| CI Lower | CI Upper | Wilcoxon p | Holm p | r |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")

    pair_order = [
        ("India vs Pakistan",  "Pakistan", "India"),
        ("China vs Pakistan",  "Pakistan", "China"),
        ("China vs India",     "India",    "China"),
        ("Nigeria vs Pakistan","Pakistan", "Nigeria"),
        ("Nigeria vs India",   "India",    "Nigeria"),
        ("Nigeria vs China",   "China",    "Nigeria"),
    ]
    for label, country_a, country_b in pair_order:
        key = f"{country_b} vs {country_a}"
        res = pairwise_results.get(key)
        if res is None:
            lines.append(
                f"| {label} | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |"
            )
        else:
            lines.append(
                f"| {label} "
                f"| {res['n']} "
                f"| {fmt(res['mean_a'])} "
                f"| {fmt(res['mean_b'])} "
                f"| {fmt(res['mean_diff'])} "
                f"| {fmt(res['ci_lo'])} "
                f"| {fmt(res['ci_hi'])} "
                f"| {fmt(res['wilcoxon_p'])} "
                f"| {fmt(res.get('holm_p', float('nan')))} "
                f"| {fmt(res['r'])} |"
            )
    lines.append("")

    # ── Model-Level Summary ──
    lines.append("## Model-Level Summary\n")
    lines.append(
        "| Model | Pakistan | India | China | Nigeria | Overall |"
    )
    lines.append("|---|---|---|---|---|---|")
    for r in model_rows:
        lines.append(
            f"| {r['model']} "
            f"| {fmt(r['Pakistan'])} "
            f"| {fmt(r['India'])} "
            f"| {fmt(r['China'])} "
            f"| {fmt(r['Nigeria'])} "
            f"| {fmt(r['overall'])} |"
        )
    lines.append("")

    # ── Descriptive Highlights ──
    lines.append("## Descriptive Highlights\n")
    lines.append(descriptive_highlights(model_rows))
    lines.append("")

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    prompt_to_qid = build_prompt_to_qid()

    # Load all country data
    country_records = {}
    for country, (folder, filename) in COUNTRIES.items():
        country_records[country] = load_country(folder, filename, prompt_to_qid)
        n = len(country_records[country])
        print(f"  Loaded {n} records for {country}")

    # Per-country summaries
    summaries = {}
    for country in COUNTRY_ORDER:
        summaries[country] = country_summary(country_records[country])

    # Friedman test
    friedman_result = friedman_test(country_records)
    print(
        f"  Friedman: stat={friedman_result['statistic']:.4f}, "
        f"p={friedman_result['p']:.4g}, "
        f"n_blocks={friedman_result['n_blocks']}"
    )

    # Pairwise comparisons (B vs A, mean_diff = B − A)
    pair_defs = [
        ("India vs Pakistan",   "Pakistan", "India"),
        ("China vs Pakistan",   "Pakistan", "China"),
        ("China vs India",      "India",    "China"),
        ("Nigeria vs Pakistan", "Pakistan", "Nigeria"),
        ("Nigeria vs India",    "India",    "Nigeria"),
        ("Nigeria vs China",    "China",    "Nigeria"),
    ]
    pairwise_raw = {}
    for label, country_a, country_b in pair_defs:
        key = f"{country_b} vs {country_a}"
        pairwise_raw[key] = paired_comparison(
            country_records[country_a], country_records[country_b]
        )

    # Holm correction across all 6 p-values
    p_vals = []
    valid_keys = []
    for label, country_a, country_b in pair_defs:
        key = f"{country_b} vs {country_a}"
        res = pairwise_raw[key]
        if res is not None and not np.isnan(res["wilcoxon_p"]):
            p_vals.append(res["wilcoxon_p"])
            valid_keys.append(key)

    if p_vals:
        _, p_corrected, _, _ = multipletests(p_vals, method="holm")
        for key, hp in zip(valid_keys, p_corrected):
            pairwise_raw[key]["holm_p"] = float(hp)
    for label, country_a, country_b in pair_defs:
        key = f"{country_b} vs {country_a}"
        if pairwise_raw[key] is not None and "holm_p" not in pairwise_raw[key]:
            pairwise_raw[key]["holm_p"] = float("nan")

    # Model-level descriptive summary
    model_rows = model_level_summary(country_records)

    # Build and write markdown
    md = build_markdown(country_records, summaries, friedman_result,
                        pairwise_raw, model_rows)

    out_path = OUT_DIR / "cultural_stats.md"
    with open(out_path, "w") as fh:
        fh.write(md)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
