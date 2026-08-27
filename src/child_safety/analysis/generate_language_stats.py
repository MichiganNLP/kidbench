"""Generate evaluations/analysis/language/language_stats.md with statistical analysis."""

import json
import pathlib
import numpy as np
from scipy import stats as scipy_stats
from statsmodels.stats.multitest import multipletests

# ── Config ───────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parents[3]
EVAL_DIR = ROOT / "evaluations" / "deepseek-v4-pro" / "single_turn"
DATASET = ROOT / "dataset" / "kidbench" / "kidbench_single.json"
OUT_DIR = ROOT / "evaluations" / "analysis" / "language"

MODEL_ORDER = [
    "llama-3.2-3b", "llama-3-8b", "llama-3.3-70b",
    "gemma-3-4b", "gemma-3-12b", "gemma-4-31b",
    "deepseek-v4-flash",
    "qwen-3.1-8b", "qwen-3.5-4b", "qwen-3.6-27b",
    "gemini-3.1-flash-lite", "gpt-5-mini", "claude-haiku-4.5",
]

LANGUAGES = {
    "English":  ("without_cues", "english_false_none.json"),
    "Mandarin": ("without_cues", "mandarin_false_none.json"),
    "Hindi":    ("without_cues", "hindi_false_none.json"),
    "Urdu":     ("without_cues", "urdu_false_none.json"),
}

LANG_KEYS = {
    "English":  "english",
    "Mandarin": "mandarin",
    "Hindi":    "hindi",
    "Urdu":     "urdu",
}

METRICS = [
    "safety", "developmental_fit", "emotional_support",
    "moral_guidance", "boundary_setting",
]

N_BOOT = 10000
SEED = 42


# ── Dataset prompt maps ───────────────────────────────────────────────────────
def build_prompt_maps():
    """
    Return dict: {language_name: {prompt_text: qid}}
    qid = f"{category}_{i}" using positional index within each category,
    consistent across all languages.
    """
    with open(DATASET) as fh:
        data = json.load(fh)

    maps = {}
    for lang_display, lang_key in LANG_KEYS.items():
        lang_map = {}
        for cat, items in data[lang_key].items():
            for i, item in enumerate(items):
                qid = f"{cat}_{i}"
                lang_map[item["without_cues"]] = qid
        maps[lang_display] = lang_map
    return maps


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


# ── Load language data ────────────────────────────────────────────────────────
def load_language(folder, filename, prompt_to_qid):
    """
    Load one language condition across all models.
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
            qid = prompt_to_qid.get(prompt)
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
    return bootstrap_ci_mean(diffs, n_boot=n_boot, seed=seed)


# ── Rank-biserial correlation ─────────────────────────────────────────────────
def rank_biserial(diffs):
    """r = (W_plus - W_minus) / (n*(n+1)/2)."""
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


# ── Per-language summary ──────────────────────────────────────────────────────
def language_summary(records):
    scores = list(records.values())
    arr = np.asarray(scores)
    n = len(arr)
    mean = float(arr.mean())
    median = float(np.median(arr))
    sd = float(arr.std(ddof=1))
    ci_lo, ci_hi = bootstrap_ci_mean(arr)
    return {"n": n, "mean": mean, "median": median, "sd": sd,
            "ci_lo": ci_lo, "ci_hi": ci_hi}


# ── Friedman test ─────────────────────────────────────────────────────────────
def friedman_test(all_records):
    """
    Group data by (model, qid); only use groups where all 4 languages present.
    Returns (statistic, p_value, n_groups).
    """
    lang_names = list(LANGUAGES.keys())
    # Collect all (model, qid) keys present in every language
    key_sets = [set(all_records[lang].keys()) for lang in lang_names]
    common_keys = key_sets[0]
    for ks in key_sets[1:]:
        common_keys = common_keys & ks
    common_keys = sorted(common_keys)

    if len(common_keys) < 4:
        return float("nan"), float("nan"), len(common_keys)

    scores_by_lang = {
        lang: np.array([all_records[lang][k] for k in common_keys])
        for lang in lang_names
    }
    stat, p = scipy_stats.friedmanchisquare(
        *[scores_by_lang[lang] for lang in lang_names]
    )
    return float(stat), float(p), len(common_keys)


# ── Paired comparison ─────────────────────────────────────────────────────────
def paired_comparison(records_a, records_b):
    """Match by (model, qid); B − A convention."""
    common_keys = set(records_a.keys()) & set(records_b.keys())
    if len(common_keys) < 2:
        return None
    keys = sorted(common_keys)
    scores_a = np.array([records_a[k] for k in keys])
    scores_b = np.array([records_b[k] for k in keys])
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


# ── Model-level deltas ────────────────────────────────────────────────────────
def model_level_deltas(all_records):
    """
    For each model, compute mean score per language (across all qids for that model).
    Returns {model: {lang: mean_score or None}}.
    """
    result = {}
    lang_names = list(LANGUAGES.keys())
    for model in MODEL_ORDER:
        model_scores = {}
        for lang in lang_names:
            scores = [
                v for (m, q), v in all_records[lang].items() if m == model
            ]
            model_scores[lang] = float(np.mean(scores)) if scores else None
        result[model] = model_scores
    return result


# ── Formatting ────────────────────────────────────────────────────────────────
def fmt(v, decimals=4):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v:.{decimals}f}"


# ── Markdown builder ──────────────────────────────────────────────────────────
def build_markdown(summaries, friedman_stat, friedman_p, friedman_n,
                   paired_results, model_deltas, delta_highlights):
    lines = []
    lines.append("# Language Statistical Analysis\n")

    # Per-language summary
    lines.append("## Per-Language Summary\n")
    lines.append("| Language | n | Mean | Median | SD | CI Lower | CI Upper |")
    lines.append("|---|---|---|---|---|---|---|")
    for lang in LANGUAGES:
        s = summaries[lang]
        row = (
            f"| {lang} "
            f"| {s['n']} "
            f"| {fmt(s['mean'])} "
            f"| {fmt(s['median'])} "
            f"| {fmt(s['sd'])} "
            f"| {fmt(s['ci_lo'])} "
            f"| {fmt(s['ci_hi'])} |"
        )
        lines.append(row)
    lines.append("")

    # Friedman test
    lines.append("## Omnibus Test (Friedman)\n")
    lines.append(
        f"Friedman chi-square test across 4 languages "
        f"(English, Mandarin, Hindi, Urdu), df = 3, "
        f"N = {friedman_n} matched (model, question) groups.\n"
    )
    lines.append("| Statistic | p-value | df | N groups |")
    lines.append("|---|---|---|---|")
    lines.append(
        f"| {fmt(friedman_stat)} | {fmt(friedman_p)} | 3 | {friedman_n} |"
    )
    lines.append("")

    # Pairwise comparisons
    lines.append("## Pairwise Comparisons\n")
    lines.append(
        "| Comparison | n pairs | Mean A | Mean B | Mean Diff | "
        "CI Lower | CI Upper | Wilcoxon p | Holm p | r |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")

    pair_order = [
        "Mandarin vs English",
        "Hindi vs English",
        "Urdu vs English",
        "Hindi vs Mandarin",
        "Urdu vs Mandarin",
        "Urdu vs Hindi",
    ]
    for label in pair_order:
        res = paired_results.get(label)
        if res is None:
            lines.append(
                f"| {label} | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |"
            )
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

    # Model-level deltas table
    lines.append("## Model-Level Deltas (vs English)\n")
    non_en = ["Mandarin", "Hindi", "Urdu"]
    lines.append("| Model | Mandarin Δ | Hindi Δ | Urdu Δ | Avg |Δ| |")
    lines.append("|---|---|---|---|---|")
    for model in MODEL_ORDER:
        d = model_deltas.get(model, {})
        en = d.get("English")
        if en is None:
            continue
        deltas = []
        cells = []
        for lang in non_en:
            lv = d.get(lang)
            if lv is not None and en is not None:
                delta = lv - en
                deltas.append(delta)
                cells.append(fmt(delta))
            else:
                cells.append("N/A")
        avg_abs = float(np.mean([abs(x) for x in deltas])) if deltas else float("nan")
        row = (
            f"| {model} "
            + " ".join(f"| {c} " for c in cells)
            + f"| {fmt(avg_abs)} |"
        )
        lines.append(row)
    lines.append("")

    # Descriptive highlights
    lines.append("## Descriptive Highlights\n")
    h = delta_highlights
    para_parts = []
    if h.get("largest_urdu_drop") is not None:
        para_parts.append(
            f"The largest Urdu performance drop (relative to English) is observed for "
            f"{h['largest_urdu_drop_model']} (Δ = {h['largest_urdu_drop']:.4f}). "
        )
    if h.get("largest_urdu_gain") is not None:
        para_parts.append(
            f"The largest Urdu gain is for "
            f"{h['largest_urdu_gain_model']} (Δ = {h['largest_urdu_gain']:.4f}). "
        )
    if h.get("most_stable_model") is not None:
        para_parts.append(
            f"The most stable model across languages is "
            f"{h['most_stable_model']} (avg |Δ| = {h['most_stable_avg_abs']:.4f}). "
        )
    if h.get("largest_cross_lingual_drop_model") is not None:
        para_parts.append(
            f"The largest average cross-lingual drop is seen for "
            f"{h['largest_cross_lingual_drop_model']} "
            f"(avg Δ = {h['largest_cross_lingual_drop']:.4f}). "
        )
    if h.get("largest_cross_lingual_gain_model") is not None:
        para_parts.append(
            f"The largest average cross-lingual gain is seen for "
            f"{h['largest_cross_lingual_gain_model']} "
            f"(avg Δ = {h['largest_cross_lingual_gain']:.4f})."
        )
    lines.append("".join(para_parts) if para_parts else "No delta data available.")
    lines.append("")

    return "\n".join(lines)


# ── Compute delta highlights ──────────────────────────────────────────────────
def compute_delta_highlights(model_deltas):
    non_en = ["Mandarin", "Hindi", "Urdu"]
    urdu_deltas = []  # (delta, model)
    avg_cross_deltas = []  # (avg_delta, avg_abs_delta, model)

    for model in MODEL_ORDER:
        d = model_deltas.get(model, {})
        en = d.get("English")
        if en is None:
            continue
        lang_deltas = []
        for lang in non_en:
            lv = d.get(lang)
            if lv is not None:
                lang_deltas.append(lv - en)
        if d.get("Urdu") is not None:
            urdu_deltas.append((d["Urdu"] - en, model))
        if lang_deltas:
            avg_delta = float(np.mean(lang_deltas))
            avg_abs = float(np.mean([abs(x) for x in lang_deltas]))
            avg_cross_deltas.append((avg_delta, avg_abs, model))

    result = {}

    if urdu_deltas:
        most_neg = min(urdu_deltas, key=lambda x: x[0])
        most_pos = max(urdu_deltas, key=lambda x: x[0])
        result["largest_urdu_drop"] = most_neg[0]
        result["largest_urdu_drop_model"] = most_neg[1]
        result["largest_urdu_gain"] = most_pos[0]
        result["largest_urdu_gain_model"] = most_pos[1]

    if avg_cross_deltas:
        most_stable = min(avg_cross_deltas, key=lambda x: x[1])
        result["most_stable_model"] = most_stable[2]
        result["most_stable_avg_abs"] = most_stable[1]

        most_neg_avg = min(avg_cross_deltas, key=lambda x: x[0])
        result["largest_cross_lingual_drop"] = most_neg_avg[0]
        result["largest_cross_lingual_drop_model"] = most_neg_avg[2]

        most_pos_avg = max(avg_cross_deltas, key=lambda x: x[0])
        result["largest_cross_lingual_gain"] = most_pos_avg[0]
        result["largest_cross_lingual_gain_model"] = most_pos_avg[2]

    return result


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    prompt_maps = build_prompt_maps()

    # Load all languages
    all_records = {}
    for lang, (folder, filename) in LANGUAGES.items():
        all_records[lang] = load_language(folder, filename, prompt_maps[lang])

    # Per-language summaries
    summaries = {}
    for lang in LANGUAGES:
        summaries[lang] = language_summary(all_records[lang])

    # Friedman test
    friedman_stat, friedman_p, friedman_n = friedman_test(all_records)

    # Pairwise comparisons
    pair_defs = [
        ("Mandarin vs English", "English",  "Mandarin"),
        ("Hindi vs English",    "English",  "Hindi"),
        ("Urdu vs English",     "English",  "Urdu"),
        ("Hindi vs Mandarin",   "Mandarin", "Hindi"),
        ("Urdu vs Mandarin",    "Mandarin", "Urdu"),
        ("Urdu vs Hindi",       "Hindi",    "Urdu"),
    ]
    paired_raw = {}
    for label, lang_a, lang_b in pair_defs:
        paired_raw[label] = paired_comparison(all_records[lang_a], all_records[lang_b])

    # Holm correction across all 6 comparisons
    p_values = []
    valid_labels = []
    for label, _, _ in pair_defs:
        res = paired_raw[label]
        if res is not None and not np.isnan(res["wilcoxon_p"]):
            p_values.append(res["wilcoxon_p"])
            valid_labels.append(label)

    if p_values:
        reject, p_corrected, _, _ = multipletests(p_values, method="holm")
        for label, hp in zip(valid_labels, p_corrected):
            paired_raw[label]["holm_p"] = float(hp)
    for label, _, _ in pair_defs:
        if paired_raw[label] is not None and "holm_p" not in paired_raw[label]:
            paired_raw[label]["holm_p"] = float("nan")

    # Model-level deltas
    model_deltas = model_level_deltas(all_records)
    delta_highlights = compute_delta_highlights(model_deltas)

    # Build and write markdown
    md = build_markdown(
        summaries, friedman_stat, friedman_p, friedman_n,
        paired_raw, model_deltas, delta_highlights
    )

    out_path = OUT_DIR / "language_stats.md"
    with open(out_path, "w") as fh:
        fh.write(md)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
