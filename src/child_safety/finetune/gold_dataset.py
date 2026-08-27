#!/usr/bin/env python3
"""
Creates dataset/gold/train.jsonl and test_single.jsonl.

test_single.jsonl — 5 items per category from kidbench_single.json (English).
  without_cues variant → 2 records (age=False no system, age=True with system)
  with_cues variant    → 5 records (no country + 4 countries, all age=True)
  5 × 10 × 7 = 350 records total.
  Format: {"messages": [{system}?, {user}], "prompt", "category", "condition", "age", "country"}

Train: gold_responses/single_turn/{model}/{cues}/{file}.json
  Only includes response turns where ALL 5 metrics = 5.
  Excludes any prompt that appears in the single-turn test set.
  No system prompt in train records: {"messages": [{user}, {assistant}]}
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from child_safety.utils import build_system_prompt

ROOT         = Path(__file__).parent.parent.parent.parent
BENCH_SINGLE = ROOT / "dataset" / "kidbench" / "kidbench_single.json"
GOLD_ROOT    = ROOT / "gold_responses" / "single_turn"
DATASET_DIR  = ROOT / "dataset" / "gold"

SEED         = 42
TEST_PER_CAT = 5
COUNTRIES    = ["pakistan", "india", "china", "nigeria"]

METRICS       = ["safety", "developmental_fit", "emotional_support", "moral_guidance", "boundary_setting"]
SKIP_PREFIXES = ("hindi_", "urdu_", "mandarin_")


def _is_english(filename: str) -> bool:
    return not any(filename.startswith(p) for p in SKIP_PREFIXES)


def _infer_country(filename: str) -> Optional[str]:
    c = Path(filename).stem.split("_")[-1]
    return None if c == "none" else c


def _all_five(evaluation: dict) -> bool:
    return all(evaluation.get(m) == 5 for m in METRICS)


# ---------------------------------------------------------------------------
# Test-set selection from benchmark
# ---------------------------------------------------------------------------

def _load_test_items() -> List[Dict]:
    """
    Pick TEST_PER_CAT items per category from the English section of
    kidbench_single.json.  Each returned item has:
        {"category", "without_cues_prompt", "with_cues_prompt"}
    """
    with open(BENCH_SINGLE) as f:
        data = json.load(f)
    eng = data["english"]

    random.seed(SEED)
    test_items: List[Dict] = []
    for cat, items in sorted(eng.items()):
        indices = list(range(len(items)))
        random.shuffle(indices)
        for idx in indices[:TEST_PER_CAT]:
            test_items.append({
                "category":            cat,
                "without_cues_prompt": items[idx]["without_cues"],
                "with_cues_prompt":    items[idx]["with_cues"],
            })
        print(f"  [{cat}] {len(items)} total → {TEST_PER_CAT} test")
    return test_items


# ---------------------------------------------------------------------------
# Test record builders
# ---------------------------------------------------------------------------

def _make_without_cues_records(item: Dict) -> List[dict]:
    """2 records: age=False (no system) and age=True."""
    records = []
    prompt = item["without_cues_prompt"]
    for age in [False, True]:
        system = build_system_prompt(age=age, country=None)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        records.append({
            "messages":  messages,
            "prompt":    prompt,
            "category":  item["category"],
            "condition": "without_cues",
            "age":       age,
            "country":   None,
        })
    return records


def _make_with_cues_records(item: Dict) -> List[dict]:
    """5 records: no country + 4 countries, all age=True."""
    records = []
    prompt = item["with_cues_prompt"]

    # No-country variant: age=True system, plain user message
    system = build_system_prompt(age=True, country=None)
    records.append({
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        "prompt":    prompt,
        "category":  item["category"],
        "condition": "with_cues",
        "age":       True,
        "country":   None,
    })

    # 4 country variants
    for country in COUNTRIES:
        system   = build_system_prompt(age=True, country=country)
        user_msg = f"## Country:\n{country.capitalize()}\n\n## User Message:\n{prompt}"
        records.append({
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user_msg},
            ],
            "prompt":    prompt,
            "category":  item["category"],
            "condition": "with_cues",
            "age":       True,
            "country":   country,
        })
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    print("Selecting test items from benchmark:")
    test_items = _load_test_items()

    # Build lookup sets for fast exclusion during train pass
    woc_test_prompts: Set[str] = {it["without_cues_prompt"] for it in test_items}
    wc_test_prompts:  Set[str] = {it["with_cues_prompt"]    for it in test_items}

    print(f"\n{len(test_items)} test items ({len(woc_test_prompts)} unique without_cues, "
          f"{len(wc_test_prompts)} unique with_cues prompts)")

    counts = {"train": 0, "test_single": 0}
    out_files = {
        "train":       open(DATASET_DIR / "train.jsonl",       "w", encoding="utf-8"),
        "test_single": open(DATASET_DIR / "test_single.jsonl", "w", encoding="utf-8"),
    }

    def _write(record: dict, split: str) -> None:
        out_files[split].write(json.dumps(record, ensure_ascii=False) + "\n")
        counts[split] += 1

    # ── Write test_single records ───────────────────────────────────────────
    for item in test_items:
        for rec in _make_without_cues_records(item):
            _write(rec, "test_single")
        for rec in _make_with_cues_records(item):
            _write(rec, "test_single")

    expected_test = len(test_items) * (2 + 5)  # 2 without_cues + 5 with_cues

    # ── Write train records from gold_responses ─────────────────────────────
    seen: set = set()

    try:
        for model_dir in sorted(GOLD_ROOT.iterdir()):
            if not model_dir.is_dir() or model_dir.name.startswith("."):
                continue
            for cues_dir in sorted(model_dir.iterdir()):
                if not cues_dir.is_dir():
                    continue
                is_with_cues = cues_dir.name == "with_cues"
                test_set = wc_test_prompts if is_with_cues else woc_test_prompts

                for gold_file in sorted(cues_dir.glob("*.json")):
                    if not _is_english(gold_file.name):
                        continue
                    country = _infer_country(gold_file.name)

                    with open(gold_file) as f:
                        items = json.load(f)

                    for item in items:
                        prompt      = item.get("prompt", "")
                        responses   = item.get("response", {})
                        evaluations = item.get("evaluation", {})

                        if prompt in test_set:
                            continue

                        for turn, ev in evaluations.items():
                            if not isinstance(ev, dict) or not _all_five(ev):
                                continue
                            response_text = responses.get(turn, "")
                            if not response_text:
                                continue

                            key = (model_dir.name, gold_file.name, prompt, turn)
                            if key in seen:
                                continue
                            seen.add(key)

                            user_msg = (
                                f"## Country:\n{country.capitalize()}\n\n## User Message:\n{prompt}"
                                if country else prompt
                            )
                            _write({"messages": [
                                {"role": "user",      "content": user_msg},
                                {"role": "assistant", "content": response_text},
                            ]}, "train")
    finally:
        for f in out_files.values():
            f.close()

    print(f"\nTrain:       {counts['train']} records")
    print(f"Test single: {counts['test_single']} records  (expected {expected_test})")


if __name__ == "__main__":
    main()
