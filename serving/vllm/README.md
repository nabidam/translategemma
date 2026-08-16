# vLLM Serving Component

Runs the offline, OpenAI-compatible vLLM inference engine for the merged TranslateGemma checkpoint.

## Prerequisites
- NVIDIA GPU with >= 24GB VRAM (e.g. A100, H100, H200, L40S).
- NVIDIA Container Toolkit installed.
- Merged checkpoint artifact mounted read-only.

## Quickstart

1. Configure `.env` from `.env.example`:
```bash
cp serving/vllm/.env.example serving/vllm/.env
```

2. Start the vLLM container:
```bash
docker compose -f serving/vllm/docker-compose.yml up -d
```

3. Run smoke tests:
```bash
python serving/vllm/smoke_test.py --base-url http://localhost:8000/v1 --model-name translategemma
```
