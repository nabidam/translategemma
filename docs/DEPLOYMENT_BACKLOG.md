# Deployment backlog

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
