import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def load_model(
    model_path: str,
    tensor_parallel_size: int = 1,
    max_model_len: Optional[int] = None,
    gpu_memory_utilization: float = 0.9,
    dtype: str = "bfloat16",
    attention_backend: Optional[str] = None,
    language_model_only: bool = False,
):
    from vllm import LLM

    llm_kwargs: Dict[str, Any] = {
        "model": model_path,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "dtype": dtype,
        "trust_remote_code": True,
    }
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = max_model_len
    if attention_backend is not None:
        llm_kwargs["attention_backend"] = attention_backend
    if language_model_only:
        llm_kwargs["language_model_only"] = True

    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    return tokenizer, llm


def load_dataset(
    path: str,
    prompt_type: str,
    language: str = "english",
) -> Dict[str, List[str]]:
    with open(path) as f:
        data = json.load(f)

    result: Dict[str, List[str]] = {}
    for category, items in data[language].items():
        result[category] = [item[prompt_type] for item in items]

    return result


def load_responses(path: str) -> List[Dict[str, str]]:
    """Load a response file. Returns flat list of {category, prompt, response} dicts."""
    with open(path) as f:
        data = json.load(f)
    result = []
    for category, items in data.items():
        for item in items:
            result.append({
                "category": category,
                "prompt": item["prompt"],
                "response": item["response"],
            })
    return result


def build_system_prompt(
    age: bool,
    country: Optional[str],
    language: Optional[str] = None,
) -> Optional[str]:
    if not age and not country and not language:
        return None

    lang = f"responding in {language} " if language else "responding "

    if age and country:
        audience = f"to a child aged 7–11 in {country}"
    elif age:
        audience = "to a child aged 7–11"
    elif country:
        audience = f"to a user in {country}"
    else:
        audience = ""

    return f"You are a helpful assistant {lang}{audience}".rstrip() + "."


def _strip_thinking(text: str) -> str:
    if "</think>" in text:
        text = text[text.index("</think>") + len("</think>"):]
    return text.strip()


# ---------------------------------------------------------------------------
# vLLM generation
# ---------------------------------------------------------------------------

def _messages_to_prompt(tokenizer: Any, messages: Sequence[Dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        list(messages),
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_batch(
    tokenizer: Any,
    llm: Any,
    messages_batch: Sequence[Sequence[Dict[str, str]]],
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_new_tokens: int = 1024,
) -> List[str]:
    from vllm import SamplingParams

    prompts = [_messages_to_prompt(tokenizer, messages) for messages in messages_batch]
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
    )
    print(f"Generating {len(prompts)} prompts...", flush=True)
    outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)
    print("Generation complete.", flush=True)

    responses: List[str] = []
    for output in outputs:
        if not output.outputs:
            responses.append("")
            continue
        responses.append(_strip_thinking(output.outputs[0].text))
    return responses


# ---------------------------------------------------------------------------
# API providers (sequential)
# ---------------------------------------------------------------------------

def _call_openai(
    client: Any,
    model: str,
    messages: Sequence[Dict[str, str]],
    temperature: float,
    max_new_tokens: int,
) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=list(messages),
        max_completion_tokens=max_new_tokens,
        reasoning_effort="minimal",
    )
    return _strip_thinking(resp.choices[0].message.content or "")


def _call_deepseek(
    client: Any,
    model: str,
    messages: Sequence[Dict[str, str]],
    temperature: float,
    max_new_tokens: int,
) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=list(messages),
        temperature=temperature,
        max_tokens=max_new_tokens,
        extra_body={"thinking": {"type": "disabled"}},
    )
    return _strip_thinking(resp.choices[0].message.content or "")


def _call_anthropic(
    client: Any,
    model: str,
    messages: Sequence[Dict[str, str]],
    temperature: float,
    max_new_tokens: int,
) -> str:
    system_msgs = [m for m in messages if m["role"] == "system"]
    user_msgs = [m for m in messages if m["role"] != "system"]
    kwargs: Dict[str, Any] = {
        "model": model,
        "max_tokens": max_new_tokens,
        "thinking": {"type": "disabled"},
        "messages": user_msgs,
    }
    if system_msgs:
        kwargs["system"] = system_msgs[0]["content"]
    resp = client.messages.create(**kwargs)
    return _strip_thinking(next((b.text for b in resp.content if b.type == "text"), ""))


def _call_google(
    client: Any,
    model: str,
    messages: Sequence[Dict[str, str]],
    temperature: float,
    max_new_tokens: int,
) -> str:
    from google.genai import types

    system_msgs = [m for m in messages if m["role"] == "system"]
    user_msgs = [m for m in messages if m["role"] != "system"]
    contents = [
        {
            "role": "user" if m["role"] == "user" else "model",
            "parts": [{"text": m["content"]}],
        }
        for m in user_msgs
    ]
    config_kwargs: Dict[str, Any] = {
        "temperature": temperature,
        "max_output_tokens": max_new_tokens,
    }
    if system_msgs:
        config_kwargs["system_instruction"] = system_msgs[0]["content"]
    resp = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(**config_kwargs),
    )
    return _strip_thinking(resp.text or "")


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def generate_api(
    provider: str,
    model: str,
    messages_batch: Sequence[Sequence[Dict[str, str]]],
    temperature: float = 0.0,
    max_new_tokens: int = 1024,
    indices: Optional[List[int]] = None,
    on_result: Optional[Any] = None,
) -> Dict[int, str]:
    """
    Calls the provider API sequentially, one request at a time.
    indices: original indices for messages_batch items; defaults to 0..N-1.
    on_result(orig_i, text): called after each response.
    Returns dict mapping original index -> response text.
    """
    if indices is None:
        indices = list(range(len(messages_batch)))

    print(f"Running {len(messages_batch)} requests via {provider}...", flush=True)

    if provider == "openai":
        from openai import OpenAI
        client = OpenAI()
        call = lambda msgs: _call_openai(client, model, msgs, temperature, max_new_tokens)
    elif provider == "deepseek":
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
        call = lambda msgs: _call_deepseek(client, model, msgs, temperature, max_new_tokens)
    elif provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        call = lambda msgs: _call_anthropic(client, model, msgs, temperature, max_new_tokens)
    elif provider == "google":
        from google import genai
        client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        call = lambda msgs: _call_google(client, model, msgs, temperature, max_new_tokens)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    from tqdm import tqdm

    results: Dict[int, str] = {}
    for orig_i, messages in tqdm(zip(indices, messages_batch), total=len(messages_batch)):
        try:
            text = call(messages)
        except Exception as e:
            tqdm.write(f"  Warning: request {orig_i} failed: {e}")
            text = ""
        results[orig_i] = text
        if on_result:
            on_result(orig_i, text)
    return results


# ---------------------------------------------------------------------------
# Async API providers
# ---------------------------------------------------------------------------

async def _call_openai_async(
    client: Any,
    model: str,
    messages: Sequence[Dict[str, str]],
    max_new_tokens: int,
) -> str:
    resp = await client.chat.completions.create(
        model=model,
        messages=list(messages),
        max_completion_tokens=max_new_tokens,
        reasoning_effort="minimal",
    )
    return _strip_thinking(resp.choices[0].message.content or "")


async def _call_deepseek_async(
    client: Any,
    model: str,
    messages: Sequence[Dict[str, str]],
    temperature: float,
    max_new_tokens: int,
) -> str:
    resp = await client.chat.completions.create(
        model=model,
        messages=list(messages),
        temperature=temperature,
        max_tokens=max_new_tokens,
        extra_body={"thinking": {"type": "disabled"}},
    )
    return _strip_thinking(resp.choices[0].message.content or "")


async def _call_anthropic_async(
    client: Any,
    model: str,
    messages: Sequence[Dict[str, str]],
    max_new_tokens: int,
) -> str:
    system_msgs = [m for m in messages if m["role"] == "system"]
    user_msgs = [m for m in messages if m["role"] != "system"]
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


async def generate_api_async(
    provider: str,
    model: str,
    messages_batch: Sequence[Sequence[Dict[str, str]]],
    temperature: float = 0.0,
    max_new_tokens: int = 1024,
    indices: Optional[List[int]] = None,
    on_result: Optional[Any] = None,
    max_concurrent: int = 10,
) -> Dict[int, str]:
    """
    Runs API requests concurrently up to max_concurrent at a time.
    on_result(orig_i, text) is called after each response (safe — asyncio is single-threaded).
    """
    if indices is None:
        indices = list(range(len(messages_batch)))

    total = len(messages_batch)
    print(f"Running {total} requests via {provider} (max_concurrent={max_concurrent})...", flush=True)

    semaphore = asyncio.Semaphore(max_concurrent)
    results: Dict[int, str] = {}
    completed = 0

    client: Any = None

    if provider == "openai":
        from openai import AsyncOpenAI
        client = AsyncOpenAI()
        async def call(msgs: Sequence[Dict[str, str]]) -> str:
            return await _call_openai_async(client, model, msgs, max_new_tokens)
    elif provider == "deepseek":
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
        async def call(msgs: Sequence[Dict[str, str]]) -> str:
            return await _call_deepseek_async(client, model, msgs, temperature, max_new_tokens)
    elif provider == "anthropic":
        import anthropic
        client = anthropic.AsyncAnthropic()
        async def call(msgs: Sequence[Dict[str, str]]) -> str:
            return await _call_anthropic_async(client, model, msgs, max_new_tokens)
    elif provider == "google":
        from google import genai
        client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        async def call(msgs: Sequence[Dict[str, str]]) -> str:
            return await asyncio.to_thread(_call_google, client, model, msgs, temperature, max_new_tokens)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    async def process_one(orig_i: int, messages: Sequence[Dict[str, str]]) -> None:
        nonlocal completed
        async with semaphore:
            try:
                text = await call(messages)
            except Exception as e:
                print(f"\n  Warning: request {orig_i} failed: {e}", flush=True)
                text = ""
        results[orig_i] = text
        if on_result:
            on_result(orig_i, text)
        completed += 1
        print(f"\r  {completed}/{total}", end="", flush=True)

    try:
        await asyncio.gather(*[process_one(i, msgs) for i, msgs in zip(indices, messages_batch)])
    finally:
        if client is not None and hasattr(client, "aclose"):
            await client.aclose()

    print(flush=True)
    return results


# ---------------------------------------------------------------------------

def save_json(output_path: str, data: Any) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent="\t", ensure_ascii=False)
