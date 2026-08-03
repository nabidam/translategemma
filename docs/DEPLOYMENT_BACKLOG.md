# Deployment backlog

## Commit `uv.lock`

`uv.lock` is currently **untracked**. The default `cu128` image builds with
`uv sync --locked`, and `git archive HEAD` — the source snapshot transferred
offline — only exports tracked files. The lockfile therefore exists on whichever
machine last ran `uv lock` and nowhere else, so a rebuild from the shipped
archive resolves nothing and fails.

It also needs regenerating: `liger-kernel` was added to `pyproject.toml` for
`training.use_liger_kernel`, so any pre-existing lock is stale.

- Run `uv lock`, commit the result, and rebuild the image before the next
  transfer.
- Confirm `git ls-files uv.lock` is non-empty in CI or in a staging checklist,
  since the failure mode only appears on the air-gapped host.

## Ship a toolkit-bearing image variant for FlashAttention 3

`model.attn_implementation` is pinned to `sdpa` offline because `flash-attn-3`
compiles against `nvcc`, which `python:3.12-slim` does not carry (§6.7 of the
deployment guide). SDPA already reaches fused kernels on Hopper, so the gap is
small **today**.

It stops being small once sequence packing lands: padding-free batching is
where FlashAttention 3 earns its cost. Do these together, not separately:

- Add a build stage carrying the CUDA toolkit, install the `speed` extra there,
  and copy only the built `flash_attn_3` wheel into the runtime image — a full
  toolkit in the shipped image would add roughly 3 GB. The stage must inherit
  `[tool.uv.extra-build-dependencies]` from `pyproject.toml`; without it the
  build fails on a missing `torch` during PEP 517 isolation.
- Cache the built wheel as a transferable artefact. A 1–3 hour CUDA compile
  repeated on every image rebuild is worse than the problem it solves.
- Pin the wheel to the exact Torch, Python, CUDA and ABI combination it was
  built against, and record it next to the resolved requirements.
- Re-benchmark against `sdpa` on the target GPU before adopting it. Measure
  non-padding tokens/second, not samples/second.

## Replace the temporary dual-CUDA build selection

The default offline image is CUDA 12.8 / PyTorch 2.8.0 (`cu128-py312`), which
works with NVIDIA r570+ drivers. CUDA 13.0 / PyTorch 2.13.0 remains available
as the named `cu130-py312` image for r580+ drivers.

For now, the repository keeps one `pyproject.toml`, representing the default
CUDA 12.8 environment. Its checked-in `uv.lock` is used by the default cu128
Docker build. When `PYTORCH_CUDA=cu130`, the Dockerfile changes the copied
build-time manifest immediately before `uv` resolves dependencies. This keeps
the working project and default developer setup on CUDA 12.8 without
maintaining parallel dependency manifests.

Before treating both variants as long-term supported release artifacts:

- Create independently locked dependency inputs for CUDA 12.8 and CUDA 13.0
  (including `torchvision`, `torchaudio`, and transitive CUDA packages).
- Record and verify the exact wheel versions, hashes, and supported GPU
  architectures for each variant.
- Add TorchVision or TorchAudio only when a supported pipeline feature needs
  them, and pin each to the Torch release it is compiled against.
- Make variant selection declarative in Compose/build tooling rather than
  rewriting the copied manifest in the Dockerfile.
- Validate the full training stack against Transformers v5 before removing the
  `transformers<5.0` compatibility bound.
- Upgrade and validate the COMET/Datasets stack before removing the
  `pyarrow<21.0.0` compatibility bound.
- Upgrade the COMET/TorchMetrics stack before removing the
  `setuptools<82.0.0` compatibility bound.
- Build and smoke-test both images on representative r570 and r580+ hosts, and
  keep their image tarball names versioned by CUDA variant.

## Record measured throughput per GPU

`docs/2026-08-03_training_speed_tier1_tier2_applied.md` sets defaults —
`model.use_4bit: false`, `training.use_liger_kernel: true`,
`training.max_length: 2048` — that are reasoned but not yet measured on
hardware. The per-GPU memory table in §6.1 of the deployment guide is
correspondingly approximate.

- ~~Run the length analysis~~ Done: 2.72M examples, mean 336, p95 1072, max
  6509. `training.max_length` stays 2048 and `training.group_by_length` is now
  on; reasoning recorded in the speed document.
- Run the 20k–50k-row benchmark grid on the H100 host, **at the intended
  `batch_size` rather than the current 4**, then replace the estimates in §6.1
  with observed peak VRAM.
- Decide `training.batch_size` and `training.gradient_checkpointing` from that
  grid. These are the last two settings still at pre-measurement defaults.
- Only after that, size any multi-GPU plan. The pre-fix throughput numbers in
  `2026-08-03_translategemma_training_speed_optimization.md` were taken while
  the model was loading in float32 with an unfused loss, so they overstate the
  hardware needed.
