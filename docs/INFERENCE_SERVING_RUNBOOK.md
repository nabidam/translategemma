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
    --base-revision <40-char base commit sha> \
    --adapter checkpoints/sft-translategemma-12b-it \
    --output-dir exports/tg-12b-merged-v1 \
    --release-id tg-12b-merged-v1 \
    --dtype bfloat16 \
    --trusted-anchor-dir /secure/anchors \
    --force
```

The merge produces an **immutable, inactive** release and writes a detached manifest
anchor to `--trusted-anchor-dir`. It has no activation options: pointers are switched
only by `scripts/promote_model_release.py` (section 4.2). Keep the anchor directory off
the model release path — it is the only thing the serving host trusts.

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

### 4.1 Verify Transferred Artifact Against the External Anchor
```bash
# On serving machine. The anchor was transferred separately, NOT inside the release.
python scripts/verify_model_export.py \
    --model-dir /opt/models/translategemma/releases/tg-12b-merged-v1 \
    --trusted-anchor-file /opt/models/translategemma/anchors/tg-12b-merged-v1.sha256
```

### 4.2 Activate the Release (only path that switches pointers)
Manual `ln -sfn` activation is not a supported procedure: it skips authenticity,
payload, and behavioral gates, and it leaves a window with no active pointer.

```bash
# Gate evidence first: merged quality, degeneration, vLLM stop-token smoke, deployment
# preflight. Archive each report and reference it from the attestation file
# (format documented in scripts/promote_model_release.py).
python scripts/promote_model_release.py promote \
    --release-dir /opt/models/translategemma/releases/tg-12b-merged-v1 \
    --current-symlink /opt/models/translategemma/current \
    --previous-symlink /opt/models/translategemma/previous \
    --trusted-anchor-file /opt/models/translategemma/anchors/tg-12b-merged-v1.sha256 \
    --attestation-file /opt/models/translategemma/attestations/tg-12b-merged-v1.json

# Point the gateway's trust anchor at the newly active release
cp /opt/models/translategemma/anchors/tg-12b-merged-v1.sha256 \
   /opt/models/translategemma/anchors/current.sha256
```

### 4.3 Configure Serving Environment
```bash
cd serving/
cp vllm/.env.example .env
# Required in .env:
#   MODEL_DIR=/opt/models/translategemma/current
#   TRUSTED_ANCHOR_DIR=/opt/models/translategemma/anchors   # mounted read-only at /trust
#   REQUIRE_VERIFIED_MANIFEST=true
#   REQUIRE_VERIFIED_PAYLOAD=true
#   PYTHON_BASE_IMAGE=python:3.11-slim@sha256:<digest>
#   CUDA_VISIBLE_DEVICES=0
```

### 4.4 Deployment Preflight and Start
```bash
docker compose -f docker-compose.yml config > evidence/resolved-compose.yml
docker image inspect "${VLLM_IMAGE}" > evidence/image_inspect.json

python ../scripts/verify_deployment_compatibility.py \
    --resolved-config evidence/resolved-compose.yml \
    --matrix-file vllm_compatibility_matrix.json \
    --image-inspect evidence/image_inspect.json \
    --host-report evidence/host_report.json

docker compose -f docker-compose.yml up -d --build
```

The gateway fails startup and readiness unless the manifest matches the external anchor,
the mounted payload matches that manifest, and the mounted generation config contains
stop token IDs 1 and 106.

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

1. **Roll back through the promotion tool** (re-verifies the target against the manifest
   hash trusted at its own promotion, then atomically swaps `current` and `previous`):
   ```bash
   python scripts/promote_model_release.py rollback \
       --current-symlink /opt/models/translategemma/current \
       --previous-symlink /opt/models/translategemma/previous

   # Repoint the gateway anchor at the release now active
   cp /opt/models/translategemma/anchors/<active-release>.sha256 \
      /opt/models/translategemma/anchors/current.sha256
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
| Gateway refuses to start: `authenticity_status 'colocated_checksum_only'` | No external anchor is configured; the release is only self-signed | Mount the anchor directory read-only at `/trust` and set `TG_TRUSTED_ANCHOR_FILE`, or inject `TG_TRUSTED_MANIFEST_SHA256`. |
| Gateway refuses to start: payload does not match manifest | Mounted release was modified after promotion, or transfer was incomplete | Re-verify with `scripts/verify_model_export.py`, re-transfer the release, and re-promote. |
| `/ready` returns 503 with `payload_verified: false` | Weights/tokenizer changed under a running deployment | Treat as a compromise or corruption incident: roll back and re-verify before serving. |
| vLLM crashes with CUDA OOM on startup | `gpu-memory-utilization` is too high or model context too large | Lower `--gpu-memory-utilization` from 0.90 to 0.85 in `serving/docker-compose.yml`. |
