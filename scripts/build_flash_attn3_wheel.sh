#!/usr/bin/env bash
#
# Build the FlashAttention 3 wheel that Dockerfile installs when
# INSTALL_FLASH_ATTN3=1.
#
# Why this exists as a separate step rather than a stage in the Dockerfile:
# FA3 compiles a large set of CUDA kernels against nvcc, which takes 1-3 hours
# and needs a ~3 GB CUDA toolkit. Doing that inside the image build would repeat
# the compile on every rebuild and would either bloat the shipped image or
# require a conditional multi-stage COPY. Producing a wheel once makes the
# result a cacheable, transferable artefact -- which is also what
# docs/DEPLOYMENT_BACKLOG.md asks for.
#
# The build runs inside nvidia/cuda:*-devel, so the *build* host needs Docker
# and internet but no GPU and no local toolkit. The wheel it produces needs a
# Hopper GPU (sm_90) only at run time.
#
# Usage:
#   scripts/build_flash_attn3_wheel.sh            # defaults below
#   MAX_JOBS=8 scripts/build_flash_attn3_wheel.sh
#
# Output: wheels/flash_attn_3-3.0.0b1-cp312-cp312-linux_x86_64.whl
set -euo pipefail

# Keep these synchronized with pyproject.toml. TORCH_VERSION must equal the
# runtime pin in [project.dependencies]: FA3 links against Torch's C++ ABI, and
# a mismatch installs cleanly but fails at import with an undefined symbol.
# FA3_TAG must equal the tag in [tool.uv.sources].
TORCH_VERSION="${TORCH_VERSION:-2.8.0}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
FA3_TAG="${FA3_TAG:-v2.8.3.post1}"

# ubuntu22.04 (glibc 2.35), not 24.04, on purpose: the wheel is installed into
# python:3.12-slim-bookworm (glibc 2.36). A wheel linked against a *newer* glibc
# than the runtime image fails to load there.
CUDA_IMAGE="${CUDA_IMAGE:-nvidia/cuda:12.8.1-devel-ubuntu22.04}"

# The compile is memory-hungry per job, not CPU-bound. Unbounded parallelism is
# the usual cause of the build being OOM-killed after an hour of work.
MAX_JOBS="${MAX_JOBS:-4}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$repo_root/wheels"

echo "Building flash-attn-3 $FA3_TAG against torch==$TORCH_VERSION in $CUDA_IMAGE"
echo "MAX_JOBS=$MAX_JOBS -- expect 1-3 hours."

docker run --rm \
    -e TORCH_VERSION -e TORCH_INDEX -e FA3_TAG -e MAX_JOBS \
    -v "$repo_root/wheels:/out" \
    "$CUDA_IMAGE" \
    bash -euo pipefail -c '
        export DEBIAN_FRONTEND=noninteractive
        apt-get update -o Acquire::Retries=5
        apt-get install -y --no-install-recommends \
            -o Acquire::Retries=5 -o Acquire::Queue-Mode=access \
            ca-certificates curl git
        rm -rf /var/lib/apt/lists/*

        curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

        # --seed provides pip: the build itself is driven by pip rather than uv
        # because it needs --no-build-isolation. FA3s setup.py imports torch at
        # module level, so it must run in an environment that already has it;
        # the [tool.uv.extra-build-dependencies] table in pyproject.toml solves
        # the same problem for `uv sync`, and this is its equivalent here.
        uv venv --seed --python 3.12 /opt/build-venv
        export VIRTUAL_ENV=/opt/build-venv
        export PATH=/opt/build-venv/bin:$PATH

        uv pip install "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX}"
        uv pip install packaging wheel ninja setuptools einops

        nvcc --version

        pip wheel --no-build-isolation --no-deps --wheel-dir /out \
            "git+https://github.com/Dao-AILab/flash-attention.git@${FA3_TAG}#subdirectory=hopper"
    '

echo
echo "Wheels now in $repo_root/wheels:"
ls -lh "$repo_root/wheels"/*.whl
echo
echo "Next:"
echo "  IMAGE_TAG=cu128-fa3-py312 INSTALL_FLASH_ATTN3=1 docker compose build trainer"
echo "  docker run --rm -v \"\$PWD/scripts:/scripts:ro\" \\"
echo "      translategemma:cu128-fa3-py312 python /scripts/verify_flash_attn3.py"
echo
echo "Run that verification before exporting the image. This host cannot execute"
echo "the kernels it just built, but it can still catch a glibc or Torch ABI"
echo "mismatch -- which is most of the risk. See OFFLINE_DEPLOYMENT.md 6.7."
