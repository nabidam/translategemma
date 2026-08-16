# vLLM Compatibility and TranslateGemma Serving Guide

Status: Target Specification (Host Verification Required via smoke_test.py)
Target Image Tag: `vllm/vllm-openai:v0.13.0`
Recommended Digest Pinning Format: `vllm/vllm-openai:v0.13.0@sha256:<pinned-digest-from-target-gpu-host>`

## 1. Engine Compatibility Specification

TranslateGemma-12B-IT is built on the Gemma 2 architecture with multi-query attention, local-global sliding window attention layers, and specific vocabulary/token configurations.

| Component | Invariant | vLLM 0.13.0 Configuration |
|---|---|---|
| **Model Type** | Gemma2 CausalLM / Merged LoRA | `--model /models/model --served-model-name translategemma` |
| **Data Type** | BFloat16 | `--dtype bfloat16` |
| **Stop Tokens** | `<eos>` (1), `<end_of_turn>` (106) | Configured in `generation_config.json` and request `stop: ["<end_of_turn>"]`, `extra_body: {"stop_token_ids": [1, 106]}` |
| **Max Model Length** | 4096 tokens | `--max-model-len 4096` |
| **Continuous Batching** | Dynamic paged attention | `--max-num-batched-tokens 8192 --max-num-seqs 128` |
| **Offline Mode** | Zero Hugging Face Hub runtime calls | `VLLM_OFFLINE=1 HF_HUB_OFFLINE=1` |

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
    --model /models/model \
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

## 4. Verification Evidence Protocol

Prior to production traffic cutover on the serving machine:
1. Pull the pinned container image and record exact digest:
   ```bash
   docker pull vllm/vllm-openai:v0.13.0
   docker inspect --format='{{index .RepoDigests 0}}' vllm/vllm-openai:v0.13.0
   ```
2. Run `serving/vllm/smoke_test.py` against the running container:
   ```bash
   python serving/vllm/smoke_test.py --base-url http://localhost:8000/v1 --model-name translategemma
   ```
3. Archive smoke test output in `reports/vllm_smoke_test_results.json` along with host GPU, driver, and CUDA versions.
