# TranslateGemma Serving Image and Hardware Compatibility Matrix

## 1. Approved Inference Images

| Service | Image Repository | Tag | Pinned Immutable Digest | Verification Status |
|---|---|---|---|---|
| **vLLM Inference Engine** | `vllm/vllm-openai` | `v0.13.0` | `sha256:7b5cf896b0105374bebb974c0529d47913364f33161c5ca155452f1e29e96ee1` | Approved for Production |
| **FastAPI Serving Gateway** | `translategemma-gateway` | `local-build` | Pinned via `gateway/requirements-lock.txt` | Approved for Production |

## 2. Hardware and Driver Requirements

- **GPU Driver**: NVIDIA Driver `>= 535.104.05` (recommended `>= 550.54.14` or `>= 560+`)
- **CUDA Runtime**: CUDA `>= 12.4` (recommended `12.8`)
- **Target Architectures**:
  - `sm_80` (A100-SXM4-80GB, A100-PCIe-80GB)
  - `sm_89` (NVIDIA L4 24GB, L40S 48GB, RTX 4090 24GB)
  - `sm_90` (NVIDIA H100-SXM5-80GB, H200-141GB)

## 3. Preflight Compatibility Verification

Before starting or updating production services, execute the offline compatibility preflight check:

```bash
python scripts/verify_deployment_compatibility.py \
    --compose-file serving/docker-compose.yml \
    --matrix-file serving/vllm_compatibility_matrix.json
```

## 4. Release Promotion & Rollback Workflow

1. **Export & Merge**:
   ```bash
   python scripts/merge_lora_adapter.py \
       --base-model google/translategemma-12b-it \
       --adapter checkpoints/sft-translategemma-12b-it \
       --output-dir /opt/models/translategemma/releases/tg-merged-v1 \
       --trusted-anchor-dir /opt/models/translategemma/anchors
   ```

2. **Verified Promotion**:
   ```bash
   python scripts/promote_model_release.py promote \
       --release-dir /opt/models/translategemma/releases/tg-merged-v1 \
       --current-symlink /opt/models/translategemma/current \
       --previous-symlink /opt/models/translategemma/previous \
       --trusted-anchor-file /opt/models/translategemma/anchors/tg-merged-v1.sha256
   ```

3. **Instant Rollback**:
   ```bash
   python scripts/promote_model_release.py rollback \
       --current-symlink /opt/models/translategemma/current \
       --previous-symlink /opt/models/translategemma/previous
   ```
