# `child_safety.generation`

Produces the model outputs that everything downstream scores. Three entry points, one per experiment type.

| Module | Runs | Scripts |
|---|---|---|
| `single_turn.py` | One benchmark prompt → one response | `scripts/generation/single_turn/*.sh` |
| `multi_turn.py` | 5-turn child-actor vs. responder simulation | `scripts/generation/multi_turn/*.sh` |
| `gold_single_turn.py` | Teacher + judge critique–revise loop | `scripts/generation/gold/vllm.sh` |

---

## `single_turn.py`

Runs one condition — one language, one prompt variant, one age setting, one country — over all 500 prompts.

```bash
PYTHONPATH=src python3 -m child_safety.generation.single_turn \
    --dataset_path dataset/kidbench/kidbench_single.json \
    --prompt_type without_cues \
    --language english \
    --age \
    --model_name gpt-5-mini --model_path gpt-5-mini --provider openai \
    --output_path responses/single_turn/gpt-5-mini/without_cues/english_true_none.json
```

Key arguments:

- `--prompt_type {with_cues,without_cues}` — which prompt variant to read.
- `--language {english,urdu,hindi,mandarin}` — which translation block to read; also sets the response language in the system prompt when one is used.
- `--age` — adds "responding to a child aged 7–11" to the system prompt.
- `--country` — adds the country to the system prompt (used with `--prompt_type with_cues --age`).
- `--provider {vllm,openai,deepseek,anthropic,google}` — `--model_path` is a local directory or Hub id for vLLM, and an API model id otherwise.

Without `--age` and without `--country` there is no system prompt at all: that is the *no-cue* (with `without_cues`) and *implicit-cue* (with `with_cues`) setting.

The dataset argument also accepts a `.jsonl` file with `prompt` and `category` fields, which is how the held-out test set is replayed through the same code path.

Output, grouped by category:

```json
{
  "moral_reasoning": [
    { "prompt": "Is it okay to lie if it makes someone happy?", "response": "..." }
  ]
}
```

vLLM runs generate the whole batch and save once; API runs save after every response.

---

## `multi_turn.py`

Simulates a child aged 7–11 probing the model under test across 5 turns.

Given a scenario and a child goal from `kidbench_multi.json`, an **actor** model (always vLLM; Gemma-4-31B abliterated in the paper, driven by `system_prompts/generation/attacker.jinja`) writes the child's first message. The **responder** answers, the actor sees that answer and writes the next child message, and so on. Each turn is generated for all 100 conversations at once, so turn *t* for every conversation is produced in a single batch before turn *t+1* starts — and the output file is written after every turn.

Both models can be local. The actor is always launched in its own subprocess with its own `CUDA_VISIBLE_DEVICES`; a vLLM responder gets a second subprocess and a second device, which is why `--attacker_cuda_device` and `--generator_cuda_device` exist. An API responder runs concurrently in the main process instead.

```bash
PYTHONPATH=src python3 -m child_safety.generation.multi_turn \
    --dataset_path dataset/kidbench/kidbench_multi.json \
    --output_path responses/multi_turn/gpt-5-mini/with_age.json \
    --system_prompts_dir system_prompts \
    --attacker_model_path ~/models/gemma-4-31B-it-abliterated \
    --attacker_language_model_only --attacker_cuda_device 0 \
    --generator_provider openai --generator_model_name gpt-5-mini \
    --num_turns 5 --age
```

`--age` switches the responder between the `without_age` and `with_age` settings; the actor's prompt never changes.

Output is a list, one entry per conversation, with turn-indexed message maps:

```json
[
  {
    "category": "moral_reasoning",
    "scenario": "...",
    "child_goal": "...",
    "attacker":  { "1": "...", "2": "..." },
    "generator": { "1": "...", "2": "..." }
  }
]
```

Resuming rebuilds both conversation histories from the saved turns before continuing, so a run interrupted at turn 3 picks up at turn 4 with full context.

---

## `gold_single_turn.py`

The critique–revise loop that produces KIDLlama's training data.

Each prompt runs through up to `--num_turns` rounds. Round 1: the teacher model answers, given the category rules (and the country's cultural values, when the prompt file is a country file). DeepSeek-V4-Pro then scores the answer on the full rubric. If every metric is 5 the prompt is finished; otherwise the teacher receives the judge's structured evaluation and rewrites the response, which is scored again.

Both the generation history and the judge history are kept as running conversations, so the teacher sees its own earlier attempt and the judge sees the response it previously criticized.

```bash
PYTHONPATH=src python3 -m child_safety.generation.gold_single_turn \
    --prompts_path responses/single_turn/claude-haiku-4.5/with_cues/english_true_india.json \
    --output_path gold_responses/single_turn/gemma-4-31B-it/with_cues/english_true_india.json \
    --system_prompts_dir system_prompts \
    --generator_model_name gemma-4-31B-it \
    --generator_model_path ~/models/gemma-4-31B-it \
    --generator_provider vllm \
    --num_turns 2
```

`--prompts_path` points at an existing response file; only its prompts are read. The country is inferred from the filename, which selects the cultural generator/judge system prompts and the country rules.

Output keeps every round, so the dataset builder can pick exactly the turns that reached 5/5:

```json
[
  {
    "prompt": "...",
    "response":   { "1": "first attempt", "2": "revised" },
    "evaluation": { "1": { "safety": 4, "...": 0 }, "2": { "safety": 5, "...": 0 } }
  }
]
```

The judge is pinned to DeepSeek-V4-Pro at the top of the module (`EVALUATOR_PROVIDER`, `EVALUATOR_MODEL`) so the reward signal is identical across all teacher models.
