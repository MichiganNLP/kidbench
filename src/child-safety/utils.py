import json
import os
from typing import Dict, List, Any

os.environ.setdefault("VLLM_DISABLE_TQDM", "1")


def load_dataset(dataset_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Loads a dataset from a JSON file containing a category -> list mapping.
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model(
    model_path: str,
    tensor_parallel_size: int = 1,
    disable_custom_all_reduce: bool = True,
    max_model_len: int | None = None,
    gpu_memory_utilization: float | None = None,
):
    from vllm import LLM

    llm_kwargs = {
        "model": model_path,
        "dtype": "bfloat16",
        "tensor_parallel_size": tensor_parallel_size,
        "disable_custom_all_reduce": disable_custom_all_reduce,
    }
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = max_model_len
    if gpu_memory_utilization is not None:
        llm_kwargs["gpu_memory_utilization"] = gpu_memory_utilization
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    return tokenizer, llm


def generate_batch(
    tokenizer,
    llm,
    messages_batch: List[List[Dict[str, str]]],
    max_new_tokens: int = 512,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    add_generation_prompt: bool = True,
    continue_final_message: bool = False,
) -> List[str]:
    from vllm import SamplingParams

    prompts = [
        tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            tokenize=False,
        )
        for messages in messages_batch
    ]

    if do_sample:
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
    else:
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=0.0,
            top_p=1.0,
        )

    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    responses = [output.outputs[0].text.strip() for output in outputs]

    for messages, response in zip(messages_batch, responses):
        messages.append({"role": "assistant", "content": response})

    return responses


def write_outputs(output_path: str, data: Dict[str, List[Dict[str, str]]]) -> None:
    """
    Writes responses to a JSON file.
    Preserves newlines in message content.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent="\t")
