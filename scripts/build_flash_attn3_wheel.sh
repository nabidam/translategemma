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
#   MAX_JOBS=2 scripts/build_flash_attn3_wheel.sh
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
MAX_JOBS="${MAX_JOBS:-2}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$repo_root/wheels"

# Keep downloads, the source checkout, and Ninja objects across disposable
# containers. The key prevents ABI-incompatible build products from being
# reused after changing one of the important pins.
cache_key="${FA3_TAG//\//_}-torch${TORCH_VERSION}-$(basename "$CUDA_IMAGE")-py312"
build_cache="$repo_root/.cache/flash-attn3/$cache_key"
mkdir -p \
    "$build_cache/apt-archives/partial" \
    "$build_cache/apt-lists/partial" \
    "$build_cache/uv" \
    "$build_cache/python"

echo "Building flash-attn-3 $FA3_TAG against torch==$TORCH_VERSION in $CUDA_IMAGE"
echo "MAX_JOBS=$MAX_JOBS -- expect 1-3 hours."
echo "Persistent build cache: $build_cache"

docker run --rm \
    -e "TORCH_VERSION=$TORCH_VERSION" \
    -e "TORCH_INDEX=$TORCH_INDEX" \
    -e "FA3_TAG=$FA3_TAG" \
    -e "MAX_JOBS=$MAX_JOBS" \
    -e FLASH_ATTENTION_FORCE_BUILD=TRUE \
    -e UV_CACHE_DIR=/cache/uv \
    -e UV_PYTHON_INSTALL_DIR=/cache/python \
    -v "$build_cache:/cache" \
    -v "$repo_root/wheels:/out" \
    "$CUDA_IMAGE" \
    bash -euo pipefail -c '
        export DEBIAN_FRONTEND=noninteractive
        apt-get update \
            -o Acquire::Retries=5 \
            -o Dir::Cache::archives=/cache/apt-archives \
            -o Dir::State::lists=/cache/apt-lists
        apt-get install -y --no-install-recommends \
            -o Acquire::Retries=5 -o Acquire::Queue-Mode=access \
            -o Binary::apt::APT::Keep-Downloaded-Packages=true \
            -o Dir::Cache::archives=/cache/apt-archives \
            -o Dir::State::lists=/cache/apt-lists \
            ca-certificates curl git

        if [[ ! -x /cache/bin/uv ]]; then
            mkdir -p /cache/bin
            curl -LsSf https://astral.sh/uv/install.sh \
                | env UV_INSTALL_DIR=/cache/bin sh
        fi
        export PATH=/cache/bin:$PATH

        # --seed provides pip: the build itself is driven by pip rather than uv
        # because it needs --no-build-isolation. FA3s setup.py imports torch at
        # module level, so it must run in an environment that already has it;
        # the [tool.uv.extra-build-dependencies] table in pyproject.toml solves
        # the same problem for `uv sync`, and this is its equivalent here.
        if [[ ! -x /cache/build-venv/bin/python ]]; then
            uv venv --seed --python 3.12 /cache/build-venv
        fi
        export VIRTUAL_ENV=/cache/build-venv
        export PATH=/cache/build-venv/bin:$PATH

        uv pip install "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX}"
        uv pip install packaging wheel ninja setuptools einops

        nvcc --version

        if [[ ! -d /cache/source/.git ]]; then
            git clone --branch "${FA3_TAG}" --depth 1 --recurse-submodules \
                --shallow-submodules \
                https://github.com/Dao-AILab/flash-attention.git /cache/source
        fi
        git -C /cache/source submodule update --init --recursive --depth 1

        # Building a local checkout keeps build/temp.* and its completed object
        # files in /cache/source. Ninja can reuse them after a failed attempt.
        pip wheel --no-build-isolation --no-deps --wheel-dir /out \
            /cache/source/hopper 2>&1 | tee /cache/latest-build.log
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
