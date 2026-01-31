# LLM Child Safety

## ✅ Finished Tasks

2025-11-30
- [x] Create the dataset.
- [x] Manually verify the generated prompts.
- [x] Wrtie the inference script for generating responses.
- [x] Write the system prompt for the LLM judge.

## 🛠️ TODOs
- [ ] Conduct human evaluation of the subset of responses.
- [ ] Run the `generate_responses.py` for all models for without cues prompts.
- [ ] Run the `generate_responses.py` for all models for with cues prompts.
- [ ] Run the `generate_responses.py` for all models for with cues + age prompts.
- [ ] Run the `generate_responses.py` for all models for with cues + age + culture prompts.
- [ ] Run the LLM judge script for all the responses.
- [ ] Run script to generate gold responses using top LLMs.
- [ ] Train a classifier to predict the child-fitness of a response.
- [ ] Fine-tune Llama for children use.
 
## Open Models
- Llama-3-4B, Llama-3-8B, Llama-3-70B
- Gemma-3-3B, Gemma-3-12B, Gemma-3-27B
- Qwen-3-4B, Qwen-3-8B, Qwen-3-14B, Qwen-3-30B
- Mistral-3-3B, Mistral-3-8B, Mistral-3-14B
- DeepSeek-V3, DeepSeek-R1
- GPT-OSS-20B, GPT-OSS-120B

## Close Models
- GPT-5.2
- Gemini-3-Pro
- Claude-4.5-Sonnet

## Instructions

Here we run experiments on greatlakes.

### Copy files in greatlakes
Run `rsync -av --exclude env/ ./ greatlakes:/home/asamee/llm-child-safety/` while inside the `llm-child-safety` folder.

### Setting up the interactive environment

Run `bash scripts/singularity/singularity_bash.sh` to get into the `singularity` (similar to docker) environment.

### Run the jail-breaking script

We can run `bash scripts/get_responses.sh`. The results are stored in (e.g. `/home/asamee/llm-child-safety/responses/`).

### Others
Download models via a one-liner:

`python -c "from huggingface_hub import snapshot_download; snapshot_download(
    repo_id='openai/gpt-oss-20b',
    local_dir='/scratch/mihalcea_root/mihalcea0/asamee/huggingface/models/GPT-OSS-20B',
)"`

Watch the CUDA device usage:

`watch -n 0.5 nvidia-smi`
