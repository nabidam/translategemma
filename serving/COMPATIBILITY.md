# TranslateGemma Serving Image and Hardware Compatibility Matrix

## 1. Images

| Service | Image Repository | Tag | Pinned Immutable Digest | Verification Status |
|---|---|---|---|---|
| **vLLM Inference Engine** | `vllm/vllm-openai` | `v0.13.0` | `sha256:7b5cf896b0105374bebb974c0529d47913364f33161c5ca155452f1e29e96ee1` | **UNVERIFIED** — no serving-host evidence archived yet |
| **FastAPI Serving Gateway** | `translategemma-gateway` | local build | *(record `docker image inspect` RepoDigest here after the first pinned build)* | **UNVERIFIED** |

`UNVERIFIED` is the honest status until the evidence artifacts in
`serving/vllm_compatibility_matrix.json` exist and hash-match. `scripts/verify_deployment_compatibility.py`
rejects an entry whose status is not `APPROVED_FOR_PRODUCTION` with complete, hash-verified evidence,
so nothing can be deployed on a self-declared approval.

The gateway image builds only from a digest-pinned base and a hash-locked dependency set:

```bash
./scripts/generate_gateway_hash_lock.sh          # writes gateway/requirements-hashes.txt
export PYTHON_BASE_IMAGE=python:3.11-slim@sha256:<digest>
docker compose -f serving/docker-compose.yml build gateway
docker image inspect translategemma-gateway --format '{{index .RepoDigests 0}}'
```

## 2. Hardware and Driver Requirements

- **GPU Driver**: NVIDIA Driver `>= 535.104.05` (recommended `>= 550.54.14` or `>= 560+`)
- **CUDA Runtime**: CUDA `>= 12.4` (recommended `12.8`)
- **Target Architectures**:
  - `sm_80` (A100-SXM4-80GB, A100-PCIe-80GB)
  - `sm_86` (RTX 3090, RTX A6000)
  - `sm_89` (NVIDIA L4 24GB, L40S 48GB, RTX 4090 24GB)
  - `sm_90` (NVIDIA H100-SXM5-80GB, H200-141GB)

## 3. Preflight Compatibility Verification

The preflight resolves environment overrides, parses the effective Compose configuration,
and compares it against the host's real image digest and GPU capabilities. Collect the host
evidence first:

```bash
docker image inspect "${VLLM_IMAGE}" > serving/evidence/image_inspect.json
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader
# Record into serving/evidence/host_report.json:
# {"gpu_name": "...", "compute_capability": "9.0", "driver_version": "...", "cuda_version": "..."}

docker compose -f serving/docker-compose.yml config > serving/evidence/resolved-compose.yml

python scripts/verify_deployment_compatibility.py \
    --resolved-config serving/evidence/resolved-compose.yml \
    --matrix-file serving/vllm_compatibility_matrix.json \
    --image-inspect serving/evidence/image_inspect.json \
    --host-report serving/evidence/host_report.json
```

Promoting an entry to `APPROVED_FOR_PRODUCTION` requires filling every evidence field in the
matrix (verifier identity, date, archived artifact paths with SHA256, and the model manifest
hash the smoke test ran against).

## 4. Release Trust Chain

The gateway runs fail-closed (`TG_REQUIRE_VERIFIED_MANIFEST=true`, `TG_REQUIRE_VERIFIED_PAYLOAD=true`).
It requires:

1. **Authenticity** — `merge_manifest.json` matching an *external* anchor: either
   `TG_TRUSTED_MANIFEST_SHA256` injected from protected deployment configuration, or
   `TG_TRUSTED_ANCHOR_FILE` on the read-only `/trust` mount (host `${TRUSTED_ANCHOR_DIR}`,
   which must live outside the model release directory).
2. **Payload verification** — every mounted config, tokenizer, and weight shard hashed
   against the authenticated manifest inventory and `SHA256SUMS`.
3. **Stop contract agreement** — stop token IDs are read from the mounted
   `generation_config.json`/`tokenizer.json` (never from the manifest), must contain
   `1` and `106`, and must agree with the manifest.

A co-located `merge_manifest.sha256` is integrity evidence only. It never satisfies
required verification, because whoever can rewrite the manifest can rewrite it too.

## 5. Release Promotion & Rollback Workflow

1. **Export & Merge** (produces an immutable, inactive release):
   ```bash
   python scripts/merge_lora_adapter.py \
       --base-model google/translategemma-12b-it \
       --base-revision <40-char commit sha> \
       --adapter checkpoints/sft-translategemma-12b-it \
       --output-dir /opt/models/translategemma/releases/tg-merged-v1 \
       --trusted-anchor-dir /opt/models/translategemma/anchors
   ```
   The merge script never activates a release; it has no symlink options.

2. **Gate evidence** — run the merged-quality, degeneration, vLLM stop-token smoke, and
   deployment preflight checks, archive their reports, and write a release attestation
   binding each result to the release's manifest SHA256 (format documented in
   `scripts/promote_model_release.py`).

3. **Verified Promotion** (external anchor + attestation are both mandatory):
   ```bash
   python scripts/promote_model_release.py promote \
       --release-dir /opt/models/translategemma/releases/tg-merged-v1 \
       --current-symlink /opt/models/translategemma/current \
       --previous-symlink /opt/models/translategemma/previous \
       --trusted-anchor-file /opt/models/translategemma/anchors/tg-merged-v1.sha256 \
       --attestation-file /opt/models/translategemma/attestations/tg-merged-v1.json \
       --expected-image-digest sha256:7b5cf896b0105374bebb974c0529d47913364f33161c5ca155452f1e29e96ee1
   ```
   Each activation is recorded in `release_index.json` beside the pointers.

4. **Instant Rollback** (re-verifies the target against the manifest hash trusted at its
   own promotion, then swaps `current` and `previous` so repeated rollbacks walk history):
   ```bash
   python scripts/promote_model_release.py rollback \
       --current-symlink /opt/models/translategemma/current \
       --previous-symlink /opt/models/translategemma/previous
   ```

5. **Update the gateway anchor** to the newly active release before restarting services:
   ```bash
   cp /opt/models/translategemma/anchors/tg-merged-v1.sha256 \
      /opt/models/translategemma/anchors/current.sha256
   ```
