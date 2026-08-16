# TranslateGemma Inference Serving Runbook

Status: Operational Runbook
Date: 2026-08-16

This runbook guides operators through merging, staging, serving, monitoring, and rolling back TranslateGemma adapter weights using vLLM and the FastAPI Gateway across separate fine-tuning and serving environments.

---

## 1. Architecture Summary

```text
Fine-Tune Host:
  Base (BF16) + Adapter  --->  merge_lora_adapter.py  --->  exports/<release_id>/ (safetensors + manifest + SHA256SUMS)
                                                                    │
                                                               (rsync / S3)
                                                                    │
Serving Host:                                                       ▼
  /opt/models/releases/<release_id>/  <---  verify_model_export.py
               │
               │ (symlink: /opt/models/current)
               ▼
  [ Docker: vLLM (v0.13.0) ]  <--- Private Bridge Network --->  [ Docker: Gateway (FastAPI) ]
  (Continuous Batching, BF16)                                  (Admission Control, Validation, /translate)
                                                                            │
                                                                            ▼
                                                                  External HTTPS Clients
```

---

## 2. Fine-Tune Machine: Merge and Export

### 2.1 Merge Adapter into Base Model
```bash
uv run python scripts/merge_lora_adapter.py \
    --base-model google/translategemma-12b-it \
    --adapter checkpoints/sft-translategemma-12b-it \
    --output-dir exports/tg-12b-merged-v1 \
    --release-id tg-12b-merged-v1 \
    --dtype bfloat16 \
    --force
```

### 2.2 Pre-Transfer Verification
```bash
# 1. Verify artifact structure and checksums
python scripts/verify_model_export.py --model-dir exports/tg-12b-merged-v1

# 2. Verify generation parity and stop behavior against PEFT reference
uv run python scripts/verify_merged_checkpoint.py \
    --base-model google/translategemma-12b-it \
    --adapter checkpoints/sft-translategemma-12b-it \
    --merged-model exports/tg-12b-merged-v1 \
    --output-report reports/merge_verification_v1.json
```

---

## 3. Artifact Transfer to Serving Machine

```bash
# Push release directory to serving machine
rsync -avP --checksum \
    exports/tg-12b-merged-v1/ \
    deploy@serving-host:/opt/models/translategemma/releases/tg-12b-merged-v1/
```

---

## 4. Serving Machine: Deployment & Validation

### 4.1 Verify Transferred Artifact
```bash
# On serving machine
python scripts/verify_model_export.py \
    --model-dir /opt/models/translategemma/releases/tg-12b-merged-v1
```

### 4.2 Activate Model Release via Symlink
```bash
ln -sfn /opt/models/translategemma/releases/tg-12b-merged-v1 /opt/models/translategemma/current
```

### 4.3 Configure Serving Environment
```bash
cd serving/
cp vllm/.env.example .env
# Edit .env if necessary (set MODEL_DIR=/opt/models/translategemma/current, CUDA_VISIBLE_DEVICES=0)
```

### 4.4 Start Containers
```bash
docker compose -f docker-compose.yml up -d --build
```

### 4.5 Post-Deployment Smoke Verification
```bash
# 1. Health check gateway
curl -f http://localhost:8080/health-check
# Expected: {"translator": "OK"}

# 2. Check model info
curl -f http://localhost:8080/model-info

# 3. Test single translation
curl -X POST http://localhost:8080/translate \
    -H "Content-Type: application/json" \
    -d '{"text": "Photosynthesis produces glucose and oxygen.", "source_lang": "en", "target_lang": "fa"}'

# 4. Test batch translation
curl -X POST http://localhost:8080/translate/batch \
    -H "Content-Type: application/json" \
    -d '{"texts": ["Hello world.", "Machine learning is powerful."], "source_lang": "en", "target_lang": "fa"}'
```

---

## 5. Observability and Monitoring

- **Gateway Metrics**: `http://localhost:8080/metrics`
  - Inspect P50/P95/P99 latency, queue wait time, request throughput, and finish reasons (`stop` vs `length`).
- **Container Logs**:
  ```bash
  docker compose -f serving/docker-compose.yml logs -f --tail=100
  ```

---

## 6. Zero-Downtime Rollback Procedure

If issues arise with the new release:

1. **Repoint Symlink to Previous Known-Good Release**:
   ```bash
   ln -sfn /opt/models/translategemma/releases/tg-12b-merged-v0 /opt/models/translategemma/current
   ```

2. **Restart Serving Stack**:
   ```bash
   docker compose -f serving/docker-compose.yml restart vllm gateway
   ```

3. **Verify Service Health**:
   ```bash
   curl -f http://localhost:8080/health-check
   ```

---

## 7. Troubleshooting Guide

| Symptom | Probable Cause | Corrective Action |
|---|---|---|
| Gateway returns `503 Service Unavailable` | vLLM is still initializing weights into VRAM | Wait for vLLM container healthcheck to pass (`docker ps`). |
| Gateway returns `429 Too Many Requests` | In-flight requests or queue depth exceeded limits | Increase `TG_MAX_CONCURRENT_REQUESTS` or enable rate-limiting upstream. |
| Output contains repetitive sentences | Model did not stop on `<end_of_turn>` (token 106) | Run `scripts/verify_model_export.py` to confirm `eos_token_id` contains 106 in `generation_config.json`. |
| vLLM crashes with CUDA OOM on startup | `gpu-memory-utilization` is too high or model context too large | Lower `--gpu-memory-utilization` from 0.90 to 0.85 in `serving/docker-compose.yml`. |
