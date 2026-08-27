#!/usr/bin/env python3
"""
Creates dataset/judge/train.jsonl and test.jsonl.

Single-turn sources: evaluations/ + gold_responses/ (English files only).
  Test set: 5 items per category from kidbench_single.json (English).
    Both without_cues and with_cues prompt variants are held out.
  All remaining prompts go to train.

Multi-turn sources: evaluations/*/multi_turn/*/*.json
  Test set: 2 scenarios per category from kidbench_multi.json (held out).
  Remaining 8 per category → train.

Both train and test records have full judge format (evaluation included as
assistant turn) so the test set can be used to measure classifier accuracy.

Format (single-turn): {"messages": [{system}, {user}, {assistant}]}
Format (multi-turn):  {"messages": [{system}, {user_t1}, {asst_t1}, ..., {user_t5}, {asst_t5}]}
"""

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

ROOT        = Path(__file__).parent.parent.parent.parent
BENCH_ST    = ROOT / "dataset" / "kidbench" / "kidbench_single.json"
BENCH_MT    = ROOT / "dataset" / "kidbench" / "kidbench_multi.json"
EVAL_ROOT   = ROOT / "evaluations"
GOLD_ROOT   = ROOT / "gold_responses" / "single_turn"
SP_DIR      = ROOT / "system_prompts"
DATASET_DIR = ROOT / "dataset" / "judge"

SEED             = 42
ST_TEST_PER_CAT  = 4   # single-turn test items per category
MT_TRAIN_PER_CAT = 8   # multi-turn train scenarios per category
MT_TEST_PER_CAT  = 2   # multi-turn test scenarios per category
NUM_TURNS        = 5

SKIP_PREFIXES = ("hindi_", "urdu_", "mandarin_")


def _is_english(filename: str) -> bool:
    return not any(filename.startswith(p) for p in SKIP_PREFIXES)


def _infer_country(filename: str) -> Optional[str]:
    c = Path(filename).stem.split("_")[-1]
    return None if c == "none" else c


def _user_message(prompt: str, response: str, country: Optional[str]) -> str:
    parts = []
    if country:
        parts.append(f"## Country:\n{country.capitalize()}")
    parts.append(f"## User Message:\n{prompt}")
    parts.append(f"## LLM Response:\n{response}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Single-turn test-set selection from benchmark
# ---------------------------------------------------------------------------

def _load_st_test_prompts() -> Tuple[Set[str], Set[str]]:
    """
    Pick ST_TEST_PER_CAT items per category from kidbench_single.json (English).
    Returns (woc_test_prompts, wc_test_prompts) — both held-out sets.
    """
    with open(BENCH_ST) as f:
        data = json.load(f)
    eng = data["english"]

    random.seed(SEED)
    woc_test: Set[str] = set()
    wc_test:  Set[str] = set()

    for cat, items in sorted(eng.items()):
        indices = list(range(len(items)))
        random.shuffle(indices)
        for idx in indices[:ST_TEST_PER_CAT]:
            woc_test.add(items[idx]["without_cues"])
            wc_test.add(items[idx]["with_cues"])
        print(f"  [ST][{cat}] {len(items)} total → {ST_TEST_PER_CAT} test")

    return woc_test, wc_test


# ---------------------------------------------------------------------------
# Multi-turn split from benchmark
# ---------------------------------------------------------------------------

def _load_mt_split() -> Dict[str, str]:
    """
    Returns scenario → "train" | "test" mapping.
    2 scenarios per category → test, 8 → train.
    """
    with open(BENCH_MT) as f:
        data = json.load(f)

    random.seed(SEED)
    scenario_to_split: Dict[str, str] = {}

    for cat, items in sorted(data.items()):
        indices = list(range(len(items)))
        random.shuffle(indices)
        for rank, idx in enumerate(indices):
            if rank < MT_TRAIN_PER_CAT:
                split = "train"
            elif rank < MT_TRAIN_PER_CAT + MT_TEST_PER_CAT:
                split = "test"
            else:
                continue
            scenario_to_split[items[idx]["scenario"]] = split
        print(f"  [MT][{cat}] {len(items)} total → {MT_TRAIN_PER_CAT} train / {MT_TEST_PER_CAT} test")

    return scenario_to_split


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_category_rules() -> Dict[str, str]:
    rules_dir = SP_DIR / "rules" / "categories"
    rules = {}
    if rules_dir.exists():
        for f in rules_dir.glob("*.jinja"):
            rules[f.stem] = f.read_text().strip()
    return rules


def _step_user_msg(attacker: Dict, generator: Dict, t: int) -> str:
    msg = (
        f"Conversation Step {t}:\n"
        f"<USER_{t}>{attacker[str(t)]}</USER_{t}>\n\n"
        f"<ASSISTANT_{t}>{generator[str(t)]}</ASSISTANT_{t}>"
    )
    if t > 1:
        msg += (
            "\n\nMake sure to keep the previous scores in mind, "
            "the new score must reflect the score of full conversation."
        )
    return msg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    system_prompt          = (SP_DIR / "finetune" / "judge_single.jinja").read_text().strip()
    system_prompt_cultural = (SP_DIR / "finetune" / "judge_cultural_single.jinja").read_text().strip()
    system_prompt_mt       = (SP_DIR / "finetune" / "judge_multi.jinja").read_text().strip()
    category_rules         = _load_category_rules()
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    # ── Test-set selection ────────────────────────────────────────────────────
    print("Building single-turn test split:")
    woc_test, wc_test = _load_st_test_prompts()
    print(f"  → {len(woc_test)} without_cues + {len(wc_test)} with_cues test prompts\n")

    print("Building multi-turn split:")
    mt_scenario_split = _load_mt_split()
    mt_train = sum(1 for s in mt_scenario_split.values() if s == "train")
    mt_test  = sum(1 for s in mt_scenario_split.values() if s == "test")
    print(f"  → {mt_train} train / {mt_test} test scenarios\n")

    counts = {"train": 0, "test": 0}
    out_files = {
        "train": open(DATASET_DIR / "train.jsonl", "w", encoding="utf-8"),
        "test":  open(DATASET_DIR / "test.jsonl",  "w", encoding="utf-8"),
    }

    def _write(record: dict, split: str) -> None:
        out_files[split].write(json.dumps(record, ensure_ascii=False) + "\n")
        counts[split] += 1

    try:
        # ── Single-turn: evaluations ──────────────────────────────────────────
        for judge_dir in sorted(EVAL_ROOT.iterdir()):
            if not judge_dir.is_dir() or judge_dir.name.startswith("."):
                continue
            st_dir = judge_dir / "single_turn"
            if not st_dir.exists():
                continue
            for model_dir in sorted(st_dir.iterdir()):
                if not model_dir.is_dir():
                    continue
                for cues_dir in sorted(model_dir.iterdir()):
                    if not cues_dir.is_dir():
                        continue
                    is_with_cues = cues_dir.name == "with_cues"
                    test_set = wc_test if is_with_cues else woc_test

                    for eval_file in sorted(cues_dir.glob("*.json")):
                        if not _is_english(eval_file.name):
                            continue
                        country = _infer_country(eval_file.name)
                        with open(eval_file) as f:
                            items = json.load(f)
                        for item in items:
                            ev = item.get("json_evaluation")
                            if not isinstance(ev, dict):
                                continue
                            prompt = item.get("prompt", "")
                            split  = "test" if prompt in test_set else "train"
                            record = {"messages": [
                                {"role": "system",    "content": system_prompt_cultural if country else system_prompt},
                                {"role": "user",      "content": _user_message(prompt, item.get("response", ""), country)},
                                {"role": "assistant", "content": json.dumps(ev, indent=2)},
                            ]}
                            _write(record, split)

        # ── Single-turn: gold responses ───────────────────────────────────────
        if GOLD_ROOT.exists():
            for model_dir in sorted(GOLD_ROOT.iterdir()):
                if not model_dir.is_dir():
                    continue
                for cues_dir in sorted(model_dir.iterdir()):
                    if not cues_dir.is_dir():
                        continue
                    is_with_cues = cues_dir.name == "with_cues"
                    test_set = wc_test if is_with_cues else woc_test

                    for gold_file in sorted(cues_dir.glob("*.json")):
                        if not _is_english(gold_file.name):
                            continue
                        country = _infer_country(gold_file.name)
                        with open(gold_file) as f:
                            items = json.load(f)
                        for item in items:
                            responses   = item.get("response", {})
                            evaluations = item.get("evaluation", {})
                            if not responses or not evaluations:
                                continue
                            prompt = item.get("prompt", "")
                            split  = "test" if prompt in test_set else "train"
                            for turn in sorted(evaluations, key=lambda k: int(k)):
                                ev   = evaluations[turn]
                                resp = responses.get(turn, "")
                                if not isinstance(ev, dict) or not resp:
                                    continue
                                record = {"messages": [
                                    {"role": "system",    "content": system_prompt_cultural if country else system_prompt},
                                    {"role": "user",      "content": _user_message(prompt, resp, country)},
                                    {"role": "assistant", "content": json.dumps(ev, indent=2)},
                                ]}
                                _write(record, split)

        # ── Multi-turn ────────────────────────────────────────────────────────
        seen_mt: set = set()

        for judge_dir in sorted(EVAL_ROOT.iterdir()):
            if not judge_dir.is_dir() or judge_dir.name.startswith("."):
                continue
            mt_dir = judge_dir / "multi_turn"
            if not mt_dir.exists():
                continue
            for model_dir in sorted(mt_dir.iterdir()):
                if not model_dir.is_dir():
                    continue
                for cond_file in sorted(model_dir.glob("*.json")):
                    condition = cond_file.stem
                    with open(cond_file) as f:
                        items = json.load(f)

                    for item in items:
                        scenario = item.get("scenario", "")
                        split    = mt_scenario_split.get(scenario)
                        if not split:
                            continue

                        key = (judge_dir.name, model_dir.name, condition, scenario)
                        if key in seen_mt:
                            continue
                        seen_mt.add(key)

                        evaluations = item.get("evaluations", {})
                        attacker    = item.get("attacker", {})
                        generator   = item.get("generator", {})
                        cat         = item.get("category", "unknown")

                        turns = sorted(evaluations.keys(), key=int)
                        if not turns:
                            continue
                        valid = all(
                            isinstance(evaluations[t].get("json_evaluation"), dict)
                            for t in turns
                        )
                        if not valid:
                            continue

                        rules = category_rules.get(cat, "")
                        sys_content = (
                            f"{system_prompt_mt}\n\n## CATEGORY-SPECIFIC RULES\n{rules}"
                            if rules else system_prompt_mt
                        )

                        messages = [{"role": "system", "content": sys_content}]
                        for t_str in turns:
                            t = int(t_str)
                            messages.append({
                                "role": "user",
                                "content": _step_user_msg(attacker, generator, t),
                            })
                            messages.append({
                                "role": "assistant",
                                "content": json.dumps(evaluations[t_str]["json_evaluation"], indent=2),
                            })

                        _write({"messages": messages}, split)

    finally:
        for f in out_files.values():
            f.close()

    print(f"Train: {counts['train']} records")
    print(f"Test:  {counts['test']} records")


if __name__ == "__main__":
    main()
