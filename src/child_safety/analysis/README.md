# `child_safety.analysis`

Turns the judge's evaluation files into the tables, figures, and statistics reported in the paper. Nothing here calls a model — it is pure CPU work over `evaluations/` and `responses/judge/`, and it takes a couple of minutes end to end.

```bash
bash scripts/analysis/run_all.sh              # everything
bash scripts/analysis/run_all.sh cues         # one group: cues|language|cultural|multi_turn|models
```

Each analysis comes in two halves. `generate_*.py` produces the descriptive report and figures; `generate_*_stats.py` produces the inferential tests. All outputs land in `evaluations/analysis/`.

| Module | Output | Contents |
|---|---|---|
| `generate_cues.py` | `cues/cues_analysis.{md,png}` | No-cue vs implicit-cue vs explicit-age, by model, metric, and category |
| `generate_cues_stats.py` | `cues/cues_stats.md` | Paired tests across the three cue conditions |
| `generate_language.py` | `language/language_analysis.{md,png}` | English / Mandarin / Hindi / Urdu |
| `generate_language_stats.py` | `language/language_stats.md` | Omnibus and pairwise language effects |
| `generate_cultural.py` | `cultural/cultural_analysis.{md,png}` | Cultural alignment for Pakistan / India / China / Nigeria |
| `generate_cultural_stats.py` | `cultural/cultural_stats.md` | Omnibus and pairwise country effects |
| `generate_multi_turn.py` | `multi_turn/multi_turn_analysis.md`, `multi_turn_slope.png`, `multi_turn_peak.png` | Turn-wise quality, degradation slope, peak drop |
| `generate_multi_turn_stats.py` | `multi_turn/multi_turn_stats.md` | Mixed-effects turn trends and age-setting interaction |
| `generate_gold_analysis.py` | `gold/gold_cues.md`, `gold_cultural.md`, `gold_multi_turn.md` | KIDLlama checkpoints across all three settings |
| `generate_guard_analysis.py` | `gold/guard_analysis.md` | KIDGuardLlama epoch checkpoints vs DeepSeek-V4-Pro |

---

## How scores are computed

An item's quality score is the mean of the five core metrics (`safety`, `developmental_fit`, `emotional_support`, `moral_guidance`, `boundary_setting`) from its `json_evaluation`. Items with a missing metric or an unparsed evaluation are dropped rather than imputed. Cultural analysis uses the separate `cultural_alignment` dimension instead.

Comparisons are **paired at the item level**, not at the model-average level. Every prompt gets a stable id (`{category}_{index}` from its position in `kidbench_single.json`), which lets the same underlying question be matched across cue conditions, languages, and countries even though the prompt text differs between them. Only ids present in every condition being compared enter a paired test.

## Statistical methods

- **Confidence intervals** — 95% bootstrap CIs over 10,000 resamples, seed 42.
- **Paired comparisons** — two-sided Wilcoxon signed-rank tests on per-item differences.
- **Multiple comparisons** — Holm correction across the full family of pairwise tests in each analysis.
- **Omnibus tests** — Friedman tests over matched examples for the language and country comparisons.
- **Multi-turn trends** — a mixed-effects model `Q ~ turn` with a random intercept per conversation (`statsmodels`), fitted per model and age setting. The reported degradation slope is *D*<sub>slope</sub> = −β₁, so positive values mean quality declines across turns. Peak drop is *Q*₁ − min(*Q*₂…*Q*₅), the worst decline after the first response. A separate model tests the turn × age-setting interaction.

---

## Configuration

Each module has a config block at the top:

- `MODEL_ORDER` — which models appear, and in what order. **A model absent from this list is silently excluded from the analysis**, so add new models here after evaluating them.
- `MODEL_DISPLAY` — display names for tables and figures.
- `CATEGORY_ORDER` / `CATEGORY_SHORT` — category ordering and short labels.
- `EVAL_DIR` — which judge's evaluations to read (`evaluations/deepseek-v4-pro` by default).

Note that folder names differ slightly between the single-turn and multi-turn results (for example `llama-3-8b` vs `llama-3.1-8b`, `claude-haiku-4.5` vs `claude-haiku-4-5`), which is why `MODEL_ORDER` is defined per module rather than shared.

Figures are written at 300 dpi with a white background, sized for the paper.
