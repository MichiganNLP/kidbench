# `child_safety.evaluation`

LLM-as-a-Judge scoring of collected responses. The judge in the paper is DeepSeek-V4-Pro, selected from seven candidates (see `sanity_check/judge_validation.md`), but any supported provider can be used.

| Module | Judges | Script |
|---|---|---|
| `single_turn.py` | One prompt–response pair at a time | `scripts/evaluation/single_turn/deepseek.sh` |
| `multi_turn.py` | A whole conversation, turn by turn, in one chat session | `scripts/evaluation/multi_turn/deepseek.sh` |

Both write `raw_evaluation` (the judge's text) alongside `json_evaluation` (the parsed object, or `null` when parsing fails), so a bad parse is visible and recoverable rather than silently dropped. Both are resumable — the output file is rewritten after every completed item.

---

## `single_turn.py`

The judge message is assembled from four parts:

```
## CATEGORY-SPECIFIC RULES:   system_prompts/rules/categories/{category}.jinja
## CULTURAL VALUES:           system_prompts/rules/countries/{country}.jinja   (country files only)
## USER PROMPT:               the child's question
## LLM RESPONSE:              the model's answer
```

with the rubric itself as the system prompt — `judge_single.jinja`, or `judge_cultural_single.jinja` when a country is involved (that variant adds the `cultural_alignment` dimension).

**The country is inferred from the response filename.** Files are named `{language}_{age}_{country}.json`; the last segment selects the cultural rules, and `none` means no country. Renaming response files changes which rubric is applied.

```bash
PYTHONPATH=src python3 -m child_safety.evaluation.single_turn \
    --response_path responses/single_turn/gpt-5-mini/with_cues/english_true_india.json \
    --output_path evaluations/deepseek-v4-pro/single_turn/gpt-5-mini/with_cues/english_true_india.json \
    --system_prompts_dir system_prompts \
    --model_name deepseek-v4-pro --model_path deepseek-v4-pro --provider deepseek \
    --max_concurrent 50
```

Output is a flat list aligned with the input order:

```json
[
  {
    "prompt": "...",
    "response": "...",
    "raw_evaluation": "```json\n{ ... }\n```",
    "json_evaluation": {
      "safety": 4,
      "developmental_fit": 3,
      "emotional_support": 4,
      "moral_guidance": 4,
      "boundary_setting": 3,
      "failure_modes": ["..."],
      "strengths": ["..."],
      "improvement": "..."
    }
  }
]
```

---

## `multi_turn.py`

Each conversation becomes **one** judge chat session:

```
system     judge rubric + category rules
user       Conversation Step 1: <USER_1>…</USER_1>  <ASSISTANT_1>…</ASSISTANT_1>
assistant  { scores for step 1 }
user       Conversation Step 2: …
assistant  { scores for step 2 }
```

Turn *t* is therefore scored with the judge's own scores for turns 1…*t−1* in context, so a score reflects the conversation so far rather than an isolated reply. This is what the degradation-slope and peak-drop analyses depend on — scoring turns independently would produce different numbers.

Calls *within* a conversation are necessarily sequential; *different* conversations run concurrently, bounded by `--max_concurrent`. On resume, saved turns are replayed into the message history so later turns keep their context.

```bash
PYTHONPATH=src python3 -m child_safety.evaluation.multi_turn \
    --response_path responses/multi_turn/gpt-5-mini/with_age.json \
    --output_path evaluations/deepseek-v4-pro/multi_turn/gpt-5-mini/with_age.json \
    --system_prompts_dir system_prompts \
    --model_path deepseek-v4-pro --provider deepseek \
    --max_concurrent 50
```

Output preserves the full conversation and adds per-turn evaluations:

```json
[
  {
    "category": "...", "scenario": "...", "child_goal": "...",
    "attacker":  { "1": "...", "...": "" },
    "generator": { "1": "...", "...": "" },
    "evaluations": {
      "1": { "raw_evaluation": "...", "json_evaluation": { "safety": 5, "...": 0 } }
    }
  }
]
```

---

## Rubric files

All judge prompts live in `system_prompts/` and are passed via `--system_prompts_dir`:

| File | Used for |
|---|---|
| `evaluation/judge_single.jinja` | single-turn, no country |
| `evaluation/judge_cultural_single.jinja` | single-turn, country context |
| `evaluation/judge_multi.jinja` | multi-turn |
| `evaluation/judge_gold_single.jinja` | gold-response loop, no country |
| `evaluation/judge_cultural_gold_single.jinja` | gold-response loop, country context |
| `rules/categories/{category}.jinja` | per-category judging rules |
| `rules/countries/{country}.jinja` | per-country cultural rules |
