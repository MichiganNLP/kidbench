#!/usr/bin/env python3
"""
Evaluates multi-turn conversations by running the judge in a multi-turn chat.

For each conversation item, one judge chat session is created:
  system: judge system prompt + category rules
  user:   "Conversation Step 1: <USER_1>...</USER_1>\n\n<ASSISTANT_1>...</ASSISTANT_1>"
  asst:   {scores for step 1}
  user:   "Conversation Step 2: ..."
  asst:   {scores for step 2}
  ...

Calls within an item are sequential (each builds on the judge's prior response).
Items are processed concurrently up to --max_concurrent.

Input:  responses/multi_turn/{model}/{with_age,without_age}.json
Output: evaluations/deepseek-v4-pro/multi_turn/{model}/{with_age,without_age}.json
"""

import argparse
import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from child_safety.utils import save_json

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_template(path: Path) -> str:
    with open(path) as f:
        return f.read().strip()


def _parse_json(text: str) -> Optional[Any]:
    try:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if match:
            return json.loads(match.group(1))
        return json.loads(text)
    except Exception:
        return None


def _step_user_msg(attacker: Dict[str, str], generator: Dict[str, str], t: int) -> str:
    msg = (
        f"Conversation Step {t}:\n"
        f"<USER_{t}>{attacker[str(t)]}</USER_{t}>\n\n"
        f"<ASSISTANT_{t}>{generator[str(t)]}</ASSISTANT_{t}>"
    )
    return msg


# ---------------------------------------------------------------------------
# Async single-call wrappers (reuse client, no semaphore here)
# ---------------------------------------------------------------------------

async def _call_one(provider: str, client: Any, model: str,
                    messages: List[Dict], max_new_tokens: int) -> str:
    from child_safety.utils import _strip_thinking

    if provider == "deepseek":
        resp = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=max_new_tokens,
            extra_body={"thinking": {"type": "disabled"}},
        )
        return _strip_thinking(resp.choices[0].message.content or "")

    elif provider == "openai":
        resp = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=max_new_tokens,
            reasoning_effort="minimal",
        )
        return _strip_thinking(resp.choices[0].message.content or "")

    elif provider == "anthropic":
        system_msgs = [m for m in messages if m["role"] == "system"]
        user_msgs   = [m for m in messages if m["role"] != "system"]
        kwargs: Dict[str, Any] = {
            "model": model,
            "max_tokens": max_new_tokens,
            "thinking": {"type": "disabled"},
            "messages": user_msgs,
        }
        if system_msgs:
            kwargs["system"] = system_msgs[0]["content"]
        resp = await client.messages.create(**kwargs)
        return _strip_thinking(next((b.text for b in resp.content if b.type == "text"), ""))

    elif provider == "google":
        from google.genai import types
        system_msgs = [m for m in messages if m["role"] == "system"]
        user_msgs   = [m for m in messages if m["role"] != "system"]
        contents = [
            {"role": "user" if m["role"] == "user" else "model",
             "parts": [{"text": m["content"]}]}
            for m in user_msgs
        ]
        cfg_kwargs: Dict[str, Any] = {"temperature": 0.0, "max_output_tokens": max_new_tokens}
        if system_msgs:
            cfg_kwargs["system_instruction"] = system_msgs[0]["content"]
        resp = await asyncio.to_thread(
            client.models.generate_content,
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(**cfg_kwargs),
        )
        return _strip_thinking(resp.text or "")

    raise ValueError(f"Unknown provider: {provider}")


# ---------------------------------------------------------------------------
# Per-item evaluator
# ---------------------------------------------------------------------------

async def _evaluate_item(
    i: int,
    item: Dict,
    system_prompt: str,
    category_rules: str,
    existing: Dict[Tuple[int, int], str],
    semaphore: asyncio.Semaphore,
    provider: str,
    client: Any,
    model: str,
    max_new_tokens: int,
    on_turn_done,
) -> None:
    num_turns = len(item["attacker"])
    messages: List[Dict] = [
        {"role": "system", "content": f"{system_prompt}\n\n## CATEGORY-SPECIFIC RULES\n{category_rules}"}
    ]

    for t in range(1, num_turns + 1):
        messages.append({"role": "user", "content": _step_user_msg(item["attacker"], item["generator"], t)})

        if (i, t) in existing:
            # Reconstruct history from saved response so later turns have context
            messages.append({"role": "assistant", "content": existing[(i, t)]})
            continue

        async with semaphore:
            try:
                raw = await _call_one(provider, client, model, messages, max_new_tokens)
            except Exception as e:
                print(f"\n  Warning: item {i} turn {t} failed: {e}", flush=True)
                raw = ""

        messages.append({"role": "assistant", "content": raw})
        on_turn_done(i, t, raw)


# ---------------------------------------------------------------------------
# Output building / resume
# ---------------------------------------------------------------------------

def _load_existing(output_path: str) -> Dict[Tuple[int, int], str]:
    if not os.path.exists(output_path):
        return {}
    with open(output_path) as f:
        saved = json.load(f)
    done: Dict[Tuple[int, int], str] = {}
    for i, item in enumerate(saved):
        for t_str, ev in item.get("evaluations", {}).items():
            if ev.get("raw_evaluation", ""):
                done[(i, int(t_str))] = ev["raw_evaluation"]
    return done


def _build_output(items: List[Dict], results: Dict[Tuple[int, int], str]) -> List[Dict]:
    output = []
    for i, item in enumerate(items):
        evaluations: Dict[str, Dict] = {}
        for t_str in item["attacker"]:
            raw = results.get((i, int(t_str)), "")
            evaluations[t_str] = {
                "raw_evaluation":  raw,
                "json_evaluation": _parse_json(raw) if raw else None,
            }
        output.append({
            "category":    item["category"],
            "scenario":    item["scenario"],
            "child_goal":  item["child_goal"],
            "attacker":    item["attacker"],
            "generator":   item["generator"],
            "evaluations": evaluations,
        })
    return output


# ---------------------------------------------------------------------------
# Args / main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    io = p.add_argument_group("io")
    io.add_argument("--response_path",      required=True)
    io.add_argument("--output_path",        required=True)
    io.add_argument("--system_prompts_dir", required=True)

    model = p.add_argument_group("model")
    model.add_argument("--model_path",     required=True)
    model.add_argument("--provider",       default="deepseek",
                       choices=["openai", "deepseek", "anthropic", "google"])
    model.add_argument("--max_new_tokens", type=int, default=2048)
    model.add_argument("--max_concurrent", type=int, default=50,
                       help="Max items evaluated concurrently")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    prompts_dir = Path(args.system_prompts_dir)
    system_prompt = _load_template(prompts_dir / "evaluation" / "judge_multi.jinja")

    with open(args.response_path) as f:
        items: List[Dict] = json.load(f)
    print(f"Loaded {len(items)} conversations.", flush=True)

    category_rules_map: Dict[int, str] = {}
    for i, item in enumerate(items):
        rules_path = prompts_dir / "rules" / "categories" / f"{item['category']}.jinja"
        if rules_path.exists():
            category_rules_map[i] = _load_template(rules_path)
        else:
            print(f"Warning: no category rules for '{item['category']}'", flush=True)
            category_rules_map[i] = ""

    results = _load_existing(args.output_path)
    num_turns = max(len(item["attacker"]) for item in items)
    total = len(items) * num_turns
    done_count = len(results)
    if done_count:
        print(f"Resuming: {done_count}/{total} turn evaluations already done.", flush=True)

    pending_items = [
        i for i, item in enumerate(items)
        if any((i, t) not in results for t in range(1, len(item["attacker"]) + 1))
    ]
    if not pending_items:
        print("All evaluations already present, nothing to do.", flush=True)
        return
    print(f"Processing {len(pending_items)} conversations...", flush=True)

    # Build client
    if args.provider == "deepseek":
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
    elif args.provider == "openai":
        from openai import AsyncOpenAI
        client = AsyncOpenAI()
    elif args.provider == "anthropic":
        import anthropic
        client = anthropic.AsyncAnthropic()
    elif args.provider == "google":
        from google import genai
        client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    else:
        raise ValueError(f"Unknown provider: {args.provider}")

    completed = [0]

    def on_turn_done(i: int, t: int, raw: str) -> None:
        results[(i, t)] = raw
        save_json(args.output_path, _build_output(items, results))
        completed[0] += 1
        print(f"\r  {completed[0]}/{total} turns done", end="", flush=True)

    semaphore = asyncio.Semaphore(args.max_concurrent)

    async def run_all() -> None:
        tasks = [
            _evaluate_item(
                i, items[i], system_prompt, category_rules_map[i],
                results, semaphore,
                args.provider, client, args.model_path, args.max_new_tokens,
                on_turn_done,
            )
            for i in pending_items
        ]
        await asyncio.gather(*tasks)
        if hasattr(client, "aclose"):
            await client.aclose()

    asyncio.run(run_all())
    print(flush=True)


if __name__ == "__main__":
    main()
