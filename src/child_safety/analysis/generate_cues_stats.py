"""Generate evaluations/analysis/cues/cues_stats.md with statistical analysis."""

import json
import pathlib
import numpy as np
from scipy import stats as scipy_stats
from statsmodels.stats.multitest import multipletests

# ── Config ───────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parents[3]
EVAL_DIR = ROOT / "evaluations" / "deepseek-v4-pro" / "single_turn"
DATASET = ROOT / "dataset" / "kidbench" / "kidbench_single.json"
OUT_DIR = ROOT / "evaluations" / "analysis" / "cues"

MODEL_ORDER = [
    "llama-3.2-3b", "llama-3-8b", "llama-3.3-70b",
    "gemma-3-4b", "gemma-3-12b", "gemma-4-31b",
    "deepseek-v4-flash",
    "qwen-3.1-8b", "qwen-3.5-4b", "qwen-3.6-27b",
    "gemini-3.1-flash-lite", "gpt-5-mini", "claude-haiku-4.5",
]

CONDITIONS = {
    "no_cue":       ("without_cues", "english_false_none.json"),
    "implicit_cue": ("with_cues",    "english_false_none.json"),
    "explicit_age": ("without_cues", "english_true_none.json"),
}

METRICS = [
    "safety", "developmental_fit", "emotional_support",
    "moral_guidance", "boundary_setting",
]

N_BOOT = 10000
SEED = 42


# ── Dataset prompt maps ───────────────────────────────────────────────────────
def build_prompt_maps():
    """Return (without_to_qid, with_to_qid) dicts mapping prompt text -> qid."""
    with open(DATASET) as fh:
        data = json.load(fh)
    without_to_qid = {}
    with_to_qid = {}
    for cat, items in data["english"].items():
        for i, item in enumerate(items):
            qid = f"{cat}_{i}"
            without_to_qid[item["without_cues"]] = qid
            with_to_qid[item["with_cues"]] = qid
    return without_to_qid, with_to_qid


# ── Score extraction ──────────────────────────────────────────────────────────
def item_score(item):
    """Return mean of METRICS for item, or None if invalid."""
    ev = item.get("json_evaluation")
    if ev is None:
        return None
    vals = []
    for m in METRICS:
        v = ev.get(m)
        if v is None:
            return None
        vals.append(v)
    return float(np.mean(vals))


# ── Load condition data ───────────────────────────────────────────────────────
def load_condition(folder, filename, without_to_qid, with_to_qid):
    """
    Load one condition across all models.
    Returns dict: {(model, qid): score}
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
            # Try without_cues map first, then with_cues map
            qid = without_to_qid.get(prompt) or with_to_qid.get(prompt)
            if qid is None:
                continue
            score = item_score(item)
            if score is None:
                continue
            records[(model, qid)] = score
    return records


# ── Bootstrap CI ─────────────────────────────────────────────────────────────
def bootstrap_ci_mean(arr, n_boot=N_BOOT, seed=SEED):
    """95% bootstrap CI for the mean of arr."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(arr)
    boot_means = np.array([
        rng.choice(arr, size=len(arr), replace=True).mean()
        for _ in range(n_boot)
    ])
    lo = float(np.percentile(boot_means, 2.5))
    hi = float(np.percentile(boot_means, 97.5))
    return lo, hi


def bootstrap_ci_mean_diff(diffs, n_boot=N_BOOT, seed=SEED):
    """95% bootstrap CI for the mean of a differences array."""
    return bootstrap_ci_mean(diffs, n_boot=n_boot, seed=seed)


# ── Rank-biserial correlation from Wilcoxon ──────────────────────────────────
def rank_biserial(diffs):
    """
    Compute rank-biserial r for Wilcoxon signed-rank test.
    r = (W_plus - W_minus) / (n*(n+1)/2)
    where W_plus and W_minus are sums of positive and negative ranks.
    Only non-zero differences are ranked.
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


# ── Per-condition summary ─────────────────────────────────────────────────────
def condition_summary(records):
    """Compute summary stats for a flat list of scores from one condition."""
    scores = list(records.values())
    arr = np.asarray(scores)
    n = len(arr)
    mean = float(arr.mean())
    median = float(np.median(arr))
    sd = float(arr.std(ddof=1))
    ci_lo, ci_hi = bootstrap_ci_mean(arr)
    return {"n": n, "mean": mean, "median": median, "sd": sd,
            "ci_lo": ci_lo, "ci_hi": ci_hi}


# ── Paired comparison ─────────────────────────────────────────────────────────
def paired_comparison(records_a, records_b):
    """
    Match by (model, qid) and compute paired stats.
    B - A is the convention for mean difference.
    Returns dict of stats, or None if fewer than 2 pairs.
    """
    common_keys = set(records_a.keys()) & set(records_b.keys())
    if len(common_keys) < 2:
        return None
    keys = sorted(common_keys)
    scores_a = np.array([records_a[k] for k in keys])
    scores_b = np.array([records_b[k] for k in keys])
    diffs = scores_b - scores_a  # B − A

    n = len(diffs)
    mean_a = float(scores_a.mean())
    mean_b = float(scores_b.mean())
    mean_diff = float(diffs.mean())
    ci_lo, ci_hi = bootstrap_ci_mean_diff(diffs)

    # Wilcoxon signed-rank test (two-sided)
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


# ── Markdown formatting ───────────────────────────────────────────────────────
def fmt(v, decimals=4):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v:.{decimals}f}"


def build_markdown(summaries, paired_results):
    lines = []
    lines.append("# Cues Statistical Analysis\n")

    # Per-condition summary table
    lines.append("## Per-Condition Summary\n")
    lines.append(
        "| Condition | n | Mean | Median | SD | CI 95% Lower | CI 95% Upper |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    cond_display = {
        "no_cue": "No Cue",
        "implicit_cue": "Implicit Cue",
        "explicit_age": "Explicit Age",
    }
    for cond in ["no_cue", "implicit_cue", "explicit_age"]:
        s = summaries[cond]
        row = (
            f"| {cond_display[cond]} "
            f"| {s['n']} "
            f"| {fmt(s['mean'])} "
            f"| {fmt(s['median'])} "
            f"| {fmt(s['sd'])} "
            f"| {fmt(s['ci_lo'])} "
            f"| {fmt(s['ci_hi'])} |"
        )
        lines.append(row)
    lines.append("")

    # Paired comparisons table
    lines.append("## Paired Comparisons\n")
    lines.append(
        "| Comparison | n pairs | Mean A | Mean B | Mean Diff | "
        "CI Lower | CI Upper | Wilcoxon p | Holm p | Rank-biserial r |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")

    comparison_labels = [
        ("implicit_cue vs no_cue",   "Implicit Cue vs No Cue"),
        ("explicit_age vs no_cue",   "Explicit Age vs No Cue"),
        ("explicit_age vs implicit_cue", "Explicit Age vs Implicit Cue"),
    ]
    for key, label in comparison_labels:
        res = paired_results[key]
        if res is None:
            lines.append(f"| {label} | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
            continue
        row = (
            f"| {label} "
            f"| {res['n']} "
            f"| {fmt(res['mean_a'])} "
            f"| {fmt(res['mean_b'])} "
            f"| {fmt(res['mean_diff'])} "
            f"| {fmt(res['ci_lo'])} "
            f"| {fmt(res['ci_hi'])} "
            f"| {fmt(res['wilcoxon_p'])} "
            f"| {fmt(res['holm_p'])} "
            f"| {fmt(res['r'])} |"
        )
        lines.append(row)
    lines.append("")

    return "\n".join(lines)


# ── Interpretation paragraph ──────────────────────────────────────────────────
def build_interpretation(summaries, paired_results):
    alpha = 0.05
    ic_nc = paired_results.get("implicit_cue vs no_cue")
    ea_nc = paired_results.get("explicit_age vs no_cue")
    ea_ic = paired_results.get("explicit_age vs implicit_cue")

    def sig(res):
        if res is None:
            return False
        hp = res.get("holm_p")
        return isinstance(hp, float) and not np.isnan(hp) and hp < alpha

    def direction(res):
        if res is None:
            return "unknown"
        d = res.get("mean_diff", 0)
        return "improvement" if d > 0 else "decrease"

    lines = ["## Interpretation\n"]
    para = []

    # Implicit cue vs no cue
    if ic_nc is not None:
        s = "significantly" if sig(ic_nc) else "not significantly"
        dir_str = direction(ic_nc)
        d = ic_nc["mean_diff"]
        r = ic_nc["r"]
        hp = ic_nc.get("holm_p", float("nan"))
        para.append(
            f"The implicit cue condition (child-voice framing) shows a "
            f"{s} {dir_str} over no cue "
            f"(mean difference = {d:.4f}, rank-biserial r = {r:.4f}, "
            f"Holm-corrected p = {hp:.4f}). "
        )
    else:
        para.append("The implicit cue vs no cue comparison could not be computed. ")

    # Explicit age vs no cue
    if ea_nc is not None:
        s = "significantly" if sig(ea_nc) else "not significantly"
        dir_str = direction(ea_nc)
        d = ea_nc["mean_diff"]
        r = ea_nc["r"]
        hp = ea_nc.get("holm_p", float("nan"))
        para.append(
            f"Explicit age disclosure shows a {s} {dir_str} relative to no cue "
            f"(mean difference = {d:.4f}, rank-biserial r = {r:.4f}, "
            f"Holm-corrected p = {hp:.4f}). "
        )
    else:
        para.append("The explicit age vs no cue comparison could not be computed. ")

    # Explicit age vs implicit cue
    if ea_ic is not None:
        s = "significantly" if sig(ea_ic) else "not significantly"
        dir_str = direction(ea_ic)
        d = ea_ic["mean_diff"]
        r = ea_ic["r"]
        hp = ea_ic.get("holm_p", float("nan"))
        para.append(
            f"Comparing explicit age to implicit cue, the difference is {s} "
            f"(mean difference = {d:.4f}, rank-biserial r = {r:.4f}, "
            f"Holm-corrected p = {hp:.4f}). "
        )
    else:
        para.append("The explicit age vs implicit cue comparison could not be computed. ")

    # Summary
    both_sig = sig(ea_nc) and sig(ea_ic)
    if both_sig:
        para.append(
            "Overall, explicit age disclosure significantly outperforms both the no-cue "
            "and implicit-cue baselines after Holm correction, suggesting that directly "
            "stating the child's age is the most effective contextual signal for eliciting "
            "child-safe responses."
        )
    elif sig(ic_nc) and not sig(ea_ic):
        para.append(
            "Overall, both implicit and explicit age cues provide similar significant "
            "improvements over no cue, with no statistically significant difference "
            "between the two cue types after Holm correction."
        )
    elif not sig(ic_nc) and not sig(ea_nc):
        para.append(
            "Overall, neither cue type produces a statistically significant change in "
            "response quality after Holm correction, suggesting that current models may "
            "not systematically adjust their responses based on implicit or explicit "
            "child-age signals."
        )
    else:
        para.append(
            "Overall, the pattern of results is mixed: not all comparisons reach "
            "significance after Holm correction, and the practical effect sizes "
            "are modest."
        )

    lines.append("".join(para))
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    without_to_qid, with_to_qid = build_prompt_maps()

    # Load all conditions
    condition_records = {}
    for cond, (folder, filename) in CONDITIONS.items():
        condition_records[cond] = load_condition(
            folder, filename, without_to_qid, with_to_qid
        )

    # Per-condition summaries
    summaries = {}
    for cond in CONDITIONS:
        summaries[cond] = condition_summary(condition_records[cond])

    # Paired comparisons (B is the second-named condition)
    pairs_def = [
        ("implicit_cue vs no_cue",       "no_cue",       "implicit_cue"),
        ("explicit_age vs no_cue",        "no_cue",       "explicit_age"),
        ("explicit_age vs implicit_cue",  "implicit_cue", "explicit_age"),
    ]
    paired_raw = {}
    for key, cond_a, cond_b in pairs_def:
        paired_raw[key] = paired_comparison(
            condition_records[cond_a], condition_records[cond_b]
        )

    # Holm correction across all 3 p-values
    p_values = []
    valid_keys = []
    for key, _, _ in pairs_def:
        res = paired_raw[key]
        if res is not None and not np.isnan(res["wilcoxon_p"]):
            p_values.append(res["wilcoxon_p"])
            valid_keys.append(key)

    if p_values:
        reject, p_corrected, _, _ = multipletests(p_values, method="holm")
        for key, hp in zip(valid_keys, p_corrected):
            paired_raw[key]["holm_p"] = float(hp)
    # Set holm_p to nan for pairs that couldn't be computed
    for key, _, _ in pairs_def:
        if paired_raw[key] is not None and "holm_p" not in paired_raw[key]:
            paired_raw[key]["holm_p"] = float("nan")

    # Rename for cleaner lookup in interpretation
    paired_results_named = {}
    for key, _, _ in pairs_def:
        label = {
            "implicit_cue vs no_cue":      "implicit_cue vs no_cue",
            "explicit_age vs no_cue":       "explicit_age vs no_cue",
            "explicit_age vs implicit_cue": "explicit_age vs implicit_cue",
        }[key]
        paired_results_named[label] = paired_raw[key]

    # Build markdown
    md = build_markdown(summaries, paired_raw)
    interp = build_interpretation(summaries, paired_results_named)
    full_md = md + "\n" + interp + "\n"

    out_path = OUT_DIR / "cues_stats.md"
    with open(out_path, "w") as fh:
        fh.write(full_md)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
