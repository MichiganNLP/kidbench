"""Generate evaluations/analysis/multi_turn/multi_turn_stats.md with statistical analysis."""

import json
import pathlib
import numpy as np
import pandas as pd
import warnings
from scipy import stats as scipy_stats
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

# ── Config ────────────────────────────────────────────────────────────────────
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

AGE_SETTINGS = ["without_age", "with_age"]

METRICS = [
    "safety", "developmental_fit", "emotional_support",
    "moral_guidance", "boundary_setting",
]

TURNS = [1, 2, 3, 4, 5]

N_BOOT = 10000
SEED = 42


# ── Score extraction ──────────────────────────────────────────────────────────
def turn_score(json_eval):
    """
    Return mean of METRICS for a single turn's json_evaluation, or None if any
    metric is missing.
    """
    if json_eval is None:
        return None
    vals = []
    for m in METRICS:
        v = json_eval.get(m)
        if v is None:
            return None
        vals.append(float(v))
    return float(np.mean(vals))


# ── Load data ─────────────────────────────────────────────────────────────────
def load_model_age(model, age_setting):
    """
    Load one (model, age_setting) file.
    Returns list of dicts: {conv_id, turn, Q}
    conv_id is 0-based item index.
    """
    path = EVAL_DIR / model / f"{age_setting}.json"
    if not path.exists():
        return []
    with open(path) as fh:
        items = json.load(fh)
    records = []
    for conv_id, item in enumerate(items):
        evals = item.get("evaluations", {})
        for t in TURNS:
            turn_data = evals.get(str(t))
            if turn_data is None:
                continue
            je = turn_data.get("json_evaluation")
            q = turn_score(je)
            if q is None:
                continue
            records.append({"conv_id": conv_id, "turn": t, "Q": q})
    return records


def load_all_data():
    """
    Load all data into a nested structure.
    Returns:
        data[(age_setting, model)] = list of {conv_id, turn, Q}
        combined_df: DataFrame with columns [Q, turn, age_setting, model, conv_id]
    """
    data = {}
    all_rows = []
    for age_setting in AGE_SETTINGS:
        for model in MODEL_ORDER:
            records = load_model_age(model, age_setting)
            data[(age_setting, model)] = records
            age_code = 0 if age_setting == "without_age" else 1
            for r in records:
                all_rows.append({
                    "Q": r["Q"],
                    "turn": r["turn"],
                    "age_setting": age_code,
                    "model": model,
                    "conv_id": f"{model}_{age_setting}_{r['conv_id']}",
                })
    combined_df = pd.DataFrame(all_rows)
    return data, combined_df


# ── Bootstrap CI ──────────────────────────────────────────────────────────────
def bootstrap_ci_mean(arr, n_boot=N_BOOT, seed=SEED):
    """95% bootstrap CI for the mean of arr."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(arr, dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan")
    boot_means = np.array([
        rng.choice(arr, size=len(arr), replace=True).mean()
        for _ in range(n_boot)
    ])
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def bootstrap_ci_mean_diff(diffs, n_boot=N_BOOT, seed=SEED):
    """95% bootstrap CI for the mean of a differences array."""
    return bootstrap_ci_mean(diffs, n_boot=n_boot, seed=seed)


# ── Section 1: Mean Quality@t ─────────────────────────────────────────────────
def compute_mean_quality_per_turn(data):
    """
    For each (age_setting, model, turn), return stats dict.
    Returns: {(age_setting, model, turn): {n, mean, median, std, ci_lo, ci_hi}}
    """
    result = {}
    for age_setting in AGE_SETTINGS:
        for model in MODEL_ORDER:
            records = data.get((age_setting, model), [])
            for t in TURNS:
                qs = [r["Q"] for r in records if r["turn"] == t]
                if not qs:
                    result[(age_setting, model, t)] = None
                    continue
                arr = np.asarray(qs, dtype=float)
                ci_lo, ci_hi = bootstrap_ci_mean(arr)
                result[(age_setting, model, t)] = {
                    "n": len(arr),
                    "mean": float(arr.mean()),
                    "median": float(np.median(arr)),
                    "std": float(arr.std(ddof=1)) if len(arr) > 1 else float("nan"),
                    "ci_lo": ci_lo,
                    "ci_hi": ci_hi,
                }
    return result


# ── Section 2: Degradation slope ─────────────────────────────────────────────
def fit_slope(records, model, age_setting):
    """
    Fit mixed-effects model Q ~ turn with groups=conv_id.
    Falls back to OLS if MixedLM fails.
    Returns dict: {beta_1, d_slope, se, ci_lo, ci_hi, p_value, method}
    """
    if not records:
        return None

    df = pd.DataFrame(records)
    if df.empty or "turn" not in df.columns or "Q" not in df.columns:
        return None

    # Need at least 2 groups with at least 2 observations for MixedLM
    group_counts = df.groupby("conv_id").size()
    n_groups = (group_counts >= 2).sum()

    result = None
    if n_groups >= 2:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                md = smf.mixedlm("Q ~ turn", data=df, groups=df["conv_id"])
                mdf = md.fit(reml=True, method="lbfgs")
            beta_1 = float(mdf.params["turn"])
            se = float(mdf.bse["turn"])
            p_value = float(mdf.pvalues["turn"])
            ci = mdf.conf_int()
            ci_lo = float(ci.loc["turn", 0])
            ci_hi = float(ci.loc["turn", 1])
            # D_slope is the degradation magnitude: -beta_1 means positive = degradation
            d_slope = -beta_1
            result = {
                "beta_1": beta_1,
                "d_slope": d_slope,
                "se": se,
                "ci_lo": -ci_hi,   # CI for D_slope = -CI for beta_1
                "ci_hi": -ci_lo,
                "p_value": p_value,
                "method": "MixedLM",
            }
        except Exception:
            result = None

    if result is None:
        # OLS fallback
        try:
            ols_fit = smf.ols("Q ~ turn", data=df).fit()
            beta_1 = float(ols_fit.params["turn"])
            se = float(ols_fit.bse["turn"])
            p_value = float(ols_fit.pvalues["turn"])
            ci = ols_fit.conf_int()
            ci_lo = float(ci.loc["turn", 0])
            ci_hi = float(ci.loc["turn", 1])
            d_slope = -beta_1
            result = {
                "beta_1": beta_1,
                "d_slope": d_slope,
                "se": se,
                "ci_lo": -ci_hi,
                "ci_hi": -ci_lo,
                "p_value": p_value,
                "method": "OLS",
            }
        except Exception:
            result = None

    return result


def compute_degradation_slopes(data):
    """
    For each (model, age_setting), fit slope.
    Returns: {(age_setting, model): slope_dict}
    """
    slopes = {}
    for age_setting in AGE_SETTINGS:
        for model in MODEL_ORDER:
            records = data.get((age_setting, model), [])
            slopes[(age_setting, model)] = fit_slope(records, model, age_setting)
    return slopes


# ── Section 3: Overall age-setting interaction ────────────────────────────────
def compute_age_interaction(combined_df):
    """
    Fit Q ~ turn * C(age_setting) with mixed effects (1 | model) and (1 | conv_id).
    Uses model as grouping variable; conv_id is a unique string per conversation.

    Returns dict with coefficients and interpretation.
    """
    if combined_df.empty:
        return None

    # Unique conv_id per conversation (already set as model_age_setting_idx)
    result = {}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            md = smf.mixedlm(
                "Q ~ turn * C(age_setting)",
                data=combined_df,
                groups=combined_df["model"],
            )
            mdf = md.fit(reml=True, method="lbfgs")

        params = mdf.params
        pvals = mdf.pvalues
        bse = mdf.bse
        ci = mdf.conf_int()

        # Parameter names from patsy
        # turn coefficient
        turn_key = "turn"
        # age_setting main effect: C(age_setting)[T.1]
        age_key = None
        for k in params.index:
            if "age_setting" in k and "turn" not in k and "Intercept" not in k:
                age_key = k
                break
        # interaction: turn:C(age_setting)[T.1]
        inter_key = None
        for k in params.index:
            if "turn" in k and "age_setting" in k:
                inter_key = k
                break

        beta_turn = float(params.get(turn_key, float("nan")))
        p_turn = float(pvals.get(turn_key, float("nan")))
        se_turn = float(bse.get(turn_key, float("nan")))

        beta_age = float(params.get(age_key, float("nan"))) if age_key else float("nan")
        p_age = float(pvals.get(age_key, float("nan"))) if age_key else float("nan")
        se_age = float(bse.get(age_key, float("nan"))) if age_key else float("nan")

        beta_inter = float(params.get(inter_key, float("nan"))) if inter_key else float("nan")
        p_inter = float(pvals.get(inter_key, float("nan"))) if inter_key else float("nan")
        se_inter = float(bse.get(inter_key, float("nan"))) if inter_key else float("nan")

        # D_slope_without_age = -beta_turn
        # D_slope_with_age    = -(beta_turn + beta_inter)
        d_slope_without = -beta_turn
        d_slope_with = -(beta_turn + beta_inter) if not np.isnan(beta_inter) else float("nan")

        result = {
            "beta_turn": beta_turn,
            "p_turn": p_turn,
            "se_turn": se_turn,
            "beta_age": beta_age,
            "p_age": p_age,
            "se_age": se_age,
            "beta_inter": beta_inter,
            "p_inter": p_inter,
            "se_inter": se_inter,
            "d_slope_without": d_slope_without,
            "d_slope_with": d_slope_with,
            "method": "MixedLM",
            "age_key": age_key,
            "inter_key": inter_key,
        }
    except Exception as e:
        # OLS fallback
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ols_fit = smf.ols(
                    "Q ~ turn * C(age_setting)", data=combined_df
                ).fit()
            params = ols_fit.params
            pvals = ols_fit.pvalues
            bse = ols_fit.bse

            turn_key = "turn"
            age_key = None
            for k in params.index:
                if "age_setting" in k and "turn" not in k and "Intercept" not in k:
                    age_key = k
                    break
            inter_key = None
            for k in params.index:
                if "turn" in k and "age_setting" in k:
                    inter_key = k
                    break

            beta_turn = float(params.get(turn_key, float("nan")))
            p_turn = float(pvals.get(turn_key, float("nan")))
            se_turn = float(bse.get(turn_key, float("nan")))

            beta_age = float(params.get(age_key, float("nan"))) if age_key else float("nan")
            p_age = float(pvals.get(age_key, float("nan"))) if age_key else float("nan")
            se_age = float(bse.get(age_key, float("nan"))) if age_key else float("nan")

            beta_inter = float(params.get(inter_key, float("nan"))) if inter_key else float("nan")
            p_inter = float(pvals.get(inter_key, float("nan"))) if inter_key else float("nan")
            se_inter = float(bse.get(inter_key, float("nan"))) if inter_key else float("nan")

            d_slope_without = -beta_turn
            d_slope_with = -(beta_turn + beta_inter) if not np.isnan(beta_inter) else float("nan")

            result = {
                "beta_turn": beta_turn,
                "p_turn": p_turn,
                "se_turn": se_turn,
                "beta_age": beta_age,
                "p_age": p_age,
                "se_age": se_age,
                "beta_inter": beta_inter,
                "p_inter": p_inter,
                "se_inter": se_inter,
                "d_slope_without": d_slope_without,
                "d_slope_with": d_slope_with,
                "method": "OLS (fallback)",
                "age_key": age_key,
                "inter_key": inter_key,
            }
        except Exception as e2:
            result = {"error": f"Both MixedLM and OLS failed: {e2}"}

    return result


# ── Section 4: Peak quality drop ──────────────────────────────────────────────
def compute_peak_drops(data):
    """
    For each (model, age_setting, conv_id), compute peak_drop = Q_T1 - min(Q_T2..T5).
    Returns: {(age_setting, model): list of peak_drop values, keyed by conv_id}
             as {(age_setting, model): {conv_id: peak_drop}}
    """
    peak_drops = {}
    for age_setting in AGE_SETTINGS:
        for model in MODEL_ORDER:
            records = data.get((age_setting, model), [])
            # Group by conv_id
            by_conv = {}
            for r in records:
                by_conv.setdefault(r["conv_id"], {})[r["turn"]] = r["Q"]

            conv_drops = {}
            for conv_id, turn_map in by_conv.items():
                q1 = turn_map.get(1)
                if q1 is None:
                    continue
                later_qs = [turn_map[t] for t in [2, 3, 4, 5] if t in turn_map]
                if not later_qs:
                    continue
                peak_drop = q1 - min(later_qs)
                conv_drops[conv_id] = peak_drop

            peak_drops[(age_setting, model)] = conv_drops
    return peak_drops


def peak_drop_summary(conv_drops):
    """
    For one (model, age_setting) dict of {conv_id: peak_drop}.
    Returns summary dict.
    """
    if not conv_drops:
        return None
    arr = np.asarray(list(conv_drops.values()), dtype=float)
    n = len(arr)
    mean_pd = float(arr.mean())
    median_pd = float(np.median(arr))
    ci_lo, ci_hi = bootstrap_ci_mean(arr)
    prop_25 = float((arr > 0.25).mean()) * 100.0
    prop_50 = float((arr > 0.50).mean()) * 100.0
    return {
        "n": n,
        "mean": mean_pd,
        "median": median_pd,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "prop_25": prop_25,
        "prop_50": prop_50,
    }


# ── Section 5: Peak drop comparison ──────────────────────────────────────────
def compute_peak_drop_comparison(peak_drops):
    """
    Match by (model, conv_id) between without_age and with_age.
    Mean diff = with_age - without_age.
    Returns summary dict.
    """
    pairs_without = []
    pairs_with = []

    for model in MODEL_ORDER:
        without_drops = peak_drops.get(("without_age", model), {})
        with_drops = peak_drops.get(("with_age", model), {})
        common_convs = sorted(set(without_drops.keys()) & set(with_drops.keys()))
        for conv_id in common_convs:
            pairs_without.append(without_drops[conv_id])
            pairs_with.append(with_drops[conv_id])

    n = len(pairs_without)
    if n < 2:
        return None

    arr_without = np.asarray(pairs_without, dtype=float)
    arr_with = np.asarray(pairs_with, dtype=float)
    diffs = arr_with - arr_without  # with_age - without_age

    mean_without = float(arr_without.mean())
    mean_with = float(arr_with.mean())
    mean_diff = float(diffs.mean())
    ci_lo, ci_hi = bootstrap_ci_mean_diff(diffs)

    if np.all(diffs == 0):
        wilcoxon_p = 1.0
    else:
        try:
            _, wilcoxon_p = scipy_stats.wilcoxon(diffs, alternative="two-sided")
        except ValueError:
            wilcoxon_p = float("nan")

    return {
        "n": n,
        "mean_without": mean_without,
        "mean_with": mean_with,
        "mean_diff": mean_diff,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "wilcoxon_p": wilcoxon_p,
    }


# ── Markdown formatting ───────────────────────────────────────────────────────
def fmt(v, decimals=4):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v:.{decimals}f}"


def fmt_pct(v, decimals=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v:.{decimals}f}"


def build_section1(quality_stats):
    lines = []
    lines.append("## 1. Mean Quality per Turn\n")
    lines.append(
        "Mean Q = mean of (safety, developmental_fit, emotional_support, "
        "moral_guidance, boundary_setting). Showing mean per turn. "
        "95% bootstrap CI based on 10,000 resamples.\n"
    )

    for age_setting in AGE_SETTINGS:
        heading = "Without Age" if age_setting == "without_age" else "With Age"
        lines.append(f"### {heading}\n")
        header = "| Model |" + "".join(f" T{t} mean |" for t in TURNS)
        sep = "|---|" + "".join("---|" for _ in TURNS)
        lines.append(header)
        lines.append(sep)
        for model in MODEL_ORDER:
            cells = [model]
            for t in TURNS:
                s = quality_stats.get((age_setting, model, t))
                if s is None:
                    cells.append("N/A")
                else:
                    cells.append(fmt(s["mean"]))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    return "\n".join(lines)


def build_section2(slopes):
    lines = []
    lines.append("## 2. Degradation Slope (per model)\n")
    lines.append(
        "Mixed-effects model: Q ~ turn, groups=conv_id (OLS fallback if MixedLM fails). "
        "D_slope = -beta_1 (positive = degradation over turns). "
        "CI is 95% for D_slope.\n"
    )

    for age_setting in AGE_SETTINGS:
        heading = "Without Age" if age_setting == "without_age" else "With Age"
        lines.append(f"### {heading}\n")
        lines.append(
            "| Model | beta_1 | D_slope | SE | CI Lower | CI Upper | p-value |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for model in MODEL_ORDER:
            s = slopes.get((age_setting, model))
            if s is None:
                lines.append(f"| {model} | N/A | N/A | N/A | N/A | N/A | N/A |")
            else:
                lines.append(
                    f"| {model} "
                    f"| {fmt(s['beta_1'])} "
                    f"| {fmt(s['d_slope'])} "
                    f"| {fmt(s['se'])} "
                    f"| {fmt(s['ci_lo'])} "
                    f"| {fmt(s['ci_hi'])} "
                    f"| {fmt(s['p_value'])} |"
                )
        lines.append("")

    return "\n".join(lines)


def build_section3(interaction):
    lines = []
    lines.append("## 3. Overall Age-Setting Interaction\n")

    if interaction is None:
        lines.append("Insufficient data to fit interaction model.\n")
        return "\n".join(lines)

    if "error" in interaction:
        lines.append(f"Model fitting failed: {interaction['error']}\n")
        return "\n".join(lines)

    method = interaction.get("method", "unknown")
    lines.append(
        f"Model: Q ~ turn * C(age_setting), mixed effects with (1 | model) "
        f"as grouping variable. Method: {method}.\n"
        f"age_setting is coded 0 = without_age, 1 = with_age.\n"
    )

    age_key_display = interaction.get("age_key", "C(age_setting)[T.1]") or "C(age_setting)[T.1]"
    inter_key_display = interaction.get("inter_key", "turn:C(age_setting)[T.1]") or "turn:C(age_setting)[T.1]"

    lines.append("| Effect | Coefficient | SE | p-value |")
    lines.append("|---|---|---|---|")
    lines.append(
        f"| Main effect of turn (beta_turn) "
        f"| {fmt(interaction['beta_turn'])} "
        f"| {fmt(interaction['se_turn'])} "
        f"| {fmt(interaction['p_turn'])} |"
    )
    lines.append(
        f"| Main effect of age_setting ({age_key_display}) "
        f"| {fmt(interaction['beta_age'])} "
        f"| {fmt(interaction['se_age'])} "
        f"| {fmt(interaction['p_age'])} |"
    )
    lines.append(
        f"| turn × age_setting ({inter_key_display}) "
        f"| {fmt(interaction['beta_inter'])} "
        f"| {fmt(interaction['se_inter'])} "
        f"| {fmt(interaction['p_inter'])} |"
    )
    lines.append("")

    lines.append(
        f"- **D_slope (without_age)** = -beta_turn = {fmt(interaction['d_slope_without'])}"
    )
    lines.append(
        f"- **D_slope (with_age)** = -(beta_turn + beta_inter) = {fmt(interaction['d_slope_with'])}"
    )

    p_inter = interaction.get("p_inter", float("nan"))
    if not np.isnan(p_inter):
        sig_str = "YES (p < 0.05)" if p_inter < 0.05 else "NO (p >= 0.05)"
        lines.append(
            f"- **Slopes differ between age settings** (interaction p = {fmt(p_inter)}): {sig_str}"
        )
    else:
        lines.append("- **Slopes differ between age settings**: N/A (interaction term not estimated)")
    lines.append("")

    return "\n".join(lines)


def build_section4(peak_drops):
    lines = []
    lines.append("## 4. Peak Quality Drop\n")
    lines.append(
        "peak_drop = Q_T1 - min(Q_T2, Q_T3, Q_T4, Q_T5). "
        "Skipped if T1 or all of T2-T5 are missing. "
        "95% bootstrap CI for mean peak_drop. "
        "Proportions are percentages of conversations.\n"
    )

    for age_setting in AGE_SETTINGS:
        heading = "Without Age" if age_setting == "without_age" else "With Age"
        lines.append(f"### {heading}\n")
        lines.append(
            "| Model | Mean | Median | CI Lower | CI Upper | >0.25 (%) | >0.50 (%) |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for model in MODEL_ORDER:
            conv_drops = peak_drops.get((age_setting, model), {})
            s = peak_drop_summary(conv_drops)
            if s is None:
                lines.append(
                    f"| {model} | N/A | N/A | N/A | N/A | N/A | N/A |"
                )
            else:
                lines.append(
                    f"| {model} "
                    f"| {fmt(s['mean'])} "
                    f"| {fmt(s['median'])} "
                    f"| {fmt(s['ci_lo'])} "
                    f"| {fmt(s['ci_hi'])} "
                    f"| {fmt_pct(s['prop_25'])} "
                    f"| {fmt_pct(s['prop_50'])} |"
                )
        lines.append("")

    return "\n".join(lines)


def build_section5(comparison):
    lines = []
    lines.append("## 5. Peak Drop: With Age vs Without Age\n")
    lines.append(
        "Matched by (model, conv_id). "
        "Mean Diff = mean(peak_drop_with_age) - mean(peak_drop_without_age). "
        "Wilcoxon signed-rank test (two-sided). "
        "95% bootstrap CI for mean difference.\n"
    )

    if comparison is None:
        lines.append("Insufficient matched pairs for comparison.\n")
        return "\n".join(lines)

    lines.append(f"- **N matched pairs**: {comparison['n']}")
    lines.append(f"- **Mean peak_drop (without_age)**: {fmt(comparison['mean_without'])}")
    lines.append(f"- **Mean peak_drop (with_age)**: {fmt(comparison['mean_with'])}")
    lines.append(f"- **Mean difference (with_age − without_age)**: {fmt(comparison['mean_diff'])}")
    lines.append(
        f"- **95% bootstrap CI for mean difference**: "
        f"[{fmt(comparison['ci_lo'])}, {fmt(comparison['ci_hi'])}]"
    )
    lines.append(f"- **Wilcoxon p (two-sided)**: {fmt(comparison['wilcoxon_p'])}")
    lines.append("")

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading multi-turn data...")
    data, combined_df = load_all_data()

    for age_setting in AGE_SETTINGS:
        total = sum(len(data[(age_setting, m)]) for m in MODEL_ORDER)
        print(f"  {age_setting}: {total} turn-level records across all models")

    # Section 1
    print("Computing mean quality per turn...")
    quality_stats = compute_mean_quality_per_turn(data)

    # Section 2
    print("Computing degradation slopes...")
    slopes = compute_degradation_slopes(data)

    # Section 3
    print("Fitting overall age-setting interaction model...")
    interaction = compute_age_interaction(combined_df)

    # Section 4
    print("Computing peak quality drops...")
    peak_drops = compute_peak_drops(data)

    # Section 5
    print("Computing peak drop comparison (with_age vs without_age)...")
    comparison = compute_peak_drop_comparison(peak_drops)

    # Build markdown
    lines = []
    lines.append("# Multi-Turn Statistical Analysis\n")

    lines.append(build_section1(quality_stats))
    lines.append(build_section2(slopes))
    lines.append(build_section3(interaction))
    lines.append(build_section4(peak_drops))
    lines.append(build_section5(comparison))

    md = "\n".join(lines)

    out_path = OUT_DIR / "multi_turn_stats.md"
    with open(out_path, "w") as fh:
        fh.write(md)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
