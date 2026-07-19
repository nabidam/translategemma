## Virtualenv

```
uv venv .venv
source .venv/bin/activate  # On Linux/macOS
# or .venv\Scripts\activate on Windows

# 1. Install latest PyTorch (by default on Linux this includes CUDA 12.1 or 12.4 bindings)
uv pip install torch torchvision torchaudio

# 2. Install Hugging Face stack, evaluation libraries, and utilities
uv pip install transformers peft trl datasets accelerate bitsandbytes pandas pyyaml pysbd "unbabel-comet>=2.2.0"

# 3. Install MetricX directly from the Google Research repository (required for the evaluator)
uv pip install "git+https://github.com/google-research/metricx.git"

python -c "from accelerate import Accelerator; print(Accelerator().state)"
```

## Generate DPO data using gemini api

```
pip install google-genai aiohttp pandas
export GEMINI_API_KEY="your_api_key_here"
```

## Evaluation

```
pip install unbabel-comet transformers datasets accelerate
```

## vLLM

```
python -m vllm.entrypoints.openai.api_server \
    --model Infomaniak-AI/vllm-translategemma-12b-it \
    --enable-lora \
    --lora-modules farsi-science=/path/to/your/final_farsi_adapter \
    --max-lora-rank 64 \
    --host 0.0.0.0 \
    --port 8000
```
