# vLLM Compatibility and TranslateGemma Serving Guide

Status: Verified Reference
Target Engine: `vllm/vllm-openai:v0.13.0`

## 1. Engine Compatibility Summary

TranslateGemma-12B-IT is built on the Gemma 2 architecture with multi-query attention, local-global sliding window attention layers, and specific vocabulary/token configurations.

| Component | Invariant | vLLM 0.13.0 Configuration |
|---|---|---|
| **Model Type** | Gemma2 CausalLM / Merged LoRA | `--model /models/translategemma/current --served-model-name translategemma` |
| **Data Type** | BFloat16 | `--dtype bfloat16` |
| **Stop Tokens** | `<eos>` (1), `<end_of_turn>` (106) | Passed in `generation_config.json` and request `stop: ["<end_of_turn>"]` / `extra_body: {"stop_token_ids": [1, 106]}` |
| **Max Model Length** | 4096 / 8192 | `--max-model-len 4096` |
| **Continuous Batching** | Dynamic paged attention | `--max-num-batched-tokens 8192 --max-num-seqs 128` |
| **Offline Mode** | No Hugging Face Hub calls | `VLLM_OFFLINE=1 HF_HUB_OFFLINE=1` |

## 2. Stop Token Resolution in vLLM

The TranslateGemma adapter SFT conditioning terminates every target sequence with `<end_of_turn>` (token ID `106`). If the serving engine does not recognize token `106` as an EOS stop boundary, generation continues into runaway hallucination or assistant turn restarts.

vLLM 0.13.0 respects the following stop mechanisms:
1. **Model Generation Config**: `generation_config.json` containing `"eos_token_id": [1, 106]`.
2. **Request Stop Strings**: Passing `stop=["<end_of_turn>"]` in `/v1/completions`.
3. **Request Stop Token IDs**: Passing `stop_token_ids=[1, 106]` in the completion request payload.

The FastAPI gateway passes both `stop=["<end_of_turn>"]` and `stop_token_ids=[1, 106]` in every request to guarantee double protection against unstopped generation.

## 3. Recommended Engine Flags

```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model /models/current \
    --served-model-name translategemma \
    --dtype bfloat16 \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
    --max-num-seqs 128 \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.90 \
    --enforce-eager \
    --disable-log-requests \
    --port 8000 \
    --host 0.0.0.0
```

> **Note on `--enforce-eager`**: Gemma 2 models with sliding window attention can encounter CUDA graph capture issues depending on the CUDA runtime / PyTorch version. If CUDA graph capture is stable in production, `--enforce-eager` may be removed for ~5-10% latency improvement.
