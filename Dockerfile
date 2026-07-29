# TranslateGemma offline training/eval image.
#
# Design notes (why it looks like this):
#   * Base is plain python:3.12-slim, NOT pytorch/pytorch or nvidia/cuda. The
#     torch cu130 wheels vendor their own CUDA 13 runtime libraries, so a CUDA
#     base image would only add a second, conflicting toolkit and ~4 GB. The
#     host contributes the driver via nvidia-container-toolkit; nothing else.
#   * Only pyproject.toml is copied at build time. The project source, models,
#     data and outputs are bind-mounted at run time so scripts stay editable and
#     debuggable on the offline host without ever rebuilding the image.
#   * Dependencies are resolved with `uv pip install -r pyproject.toml`, which
#     reads [project.dependencies], [[tool.uv.index]] and [tool.uv.sources]
#     without invoking the build backend (the project itself is not installed).
#
# Target GPUs: sm_89 (RTX 6000 Ada) and sm_120 (RTX 5090 / RTX PRO 6000
# Blackwell). Both are covered by the cu130 wheels pinned in pyproject.toml.

FROM python:3.12-slim-bookworm

# Tag is hardcoded on purpose: variables in a COPY --from stage name are only
# expanded by BuildKit, and expand to empty under the classic builder. uv is a
# build-time installer only -- the environment it produces is frozen into the
# image and recorded in /opt/resolved-requirements.txt -- so pin a full version
# here (e.g. uv:0.9.18) if you want byte-reproducible rebuilds.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Triton JIT-compiles GPU kernels at *runtime* and invokes a C compiler to build
# the launcher stubs, so gcc must be in the image, not just at build time.
# libc6-dev supplies the headers and crt objects it links against; the Python
# headers Triton also needs are already in the official python image.
#
# Kept to two packages, and apt is told to fetch serially with retries: the
# default parallel pipeline is what draws 502s from a proxy under load.
RUN apt-get update -o Acquire::Retries=5 \
    && apt-get install -y --no-install-recommends \
        -o Acquire::Retries=5 \
        -o Acquire::Queue-Mode=access \
        -o Acquire::http::Pipeline-Depth=0 \
        gcc \
        libc6-dev \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    # Generous timeout: the torch wheels are ~3 GB and a slow proxy will
    # otherwise trip uv's default per-request deadline.
    UV_HTTP_TIMEOUT=300 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN uv venv --python 3.12 "$VIRTUAL_ENV"

# Resolve and install project dependencies. WORKDIR matters: uv reads the
# [tool.uv] table (pytorch-cu130 index, torch source pins) from the pyproject
# in the working directory.
WORKDIR /tmp/deps
COPY pyproject.toml /tmp/deps/pyproject.toml
RUN uv pip install -r pyproject.toml

# bitsandbytes must be new enough to ship sm_120 (Blackwell) kernels; without
# this, 4-bit QLoRA fails on RTX 5090 / RTX PRO 6000 with a "no kernel image is
# available for execution on the device" CUDA error.
RUN uv pip install --upgrade "bitsandbytes>=0.48.0"

# MetricX is optional (evaluation.metricx_enabled in config.yaml). Build with
# --build-arg INSTALL_METRICX=0 to skip it and drop the git dependency.
#
# It is vendored as a source checkout on PYTHONPATH rather than pip-installed:
# the repository has no pyproject.toml or setup.py, so `pip install git+...`
# fails outright. Its requirements.txt is deliberately NOT installed either --
# it pins transformers==4.30.2, which would destroy the training environment.
# Only metricx24/models.py is imported, and it works against modern transformers.
#
# Fetched as a commit tarball rather than cloned, so the image needs no git.
ADD https://codeload.github.com/google-research/metricx/tar.gz/fc4978eb064670f7cc33e93ea4f52d38396b8ae6 /tmp/metricx.tar.gz

ARG INSTALL_METRICX=1
RUN if [ "$INSTALL_METRICX" = "1" ]; then \
        mkdir -p /opt/metricx \
        && tar -xzf /tmp/metricx.tar.gz -C /opt/metricx --strip-components=1; \
    fi \
    && rm -f /tmp/metricx.tar.gz
ENV PYTHONPATH=/opt/metricx

# Record the exact resolved environment inside the image. On the offline host,
# `docker run --rm IMAGE cat /opt/resolved-requirements.txt` reproduces the run.
RUN uv pip freeze > /opt/resolved-requirements.txt

# Mount points, created world-writable because the container runs as an
# arbitrary host uid with no /etc/passwd entry. This covers named volumes and
# missing bind targets; a bind mount still inherits the host directory's
# ownership, so that directory must be chown'd to HOST_UID on the host.
RUN mkdir -p /hf_cache /models && chmod 1777 /hf_cache /models

WORKDIR /workspace

# Defaults only; docker-compose.yml sets the offline flags and the real paths.
ENV HF_HOME=/models \
    HUGGINGFACE_HUB_CACHE=/models/hub \
    TOKENIZERS_PARALLELISM=false

CMD ["bash"]
