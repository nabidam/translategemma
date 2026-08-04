# TranslateGemma offline training/eval image.
#
# Design notes (why it looks like this):
#   * Base is plain python:3.12-slim, NOT pytorch/pytorch or nvidia/cuda. The
#     torch CUDA wheels vendor their own CUDA runtime libraries, so a CUDA
#     base image would only add a second, conflicting toolkit and ~4 GB. The
#     host contributes the driver via nvidia-container-toolkit; nothing else.
#   * Only dependency inputs, the optional FA3 wheel, and its verifier are
#     copied at build time. Project source, models, data and outputs are
#     bind-mounted at run time so they stay editable without rebuilding.
#   * The default cu128 image is installed with `uv sync --locked`, so it uses
#     exactly the dependency set validated before staging. The retained cu130
#     image has no independent lockfile yet; see DEPLOYMENT_BACKLOG.md.
#   * FlashAttention 3 is opt-in via INSTALL_FLASH_ATTN3=1 and is installed
#     from a prebuilt wheel, never compiled here. It produces a separate,
#     Hopper-only image tag; the default tag stays FA3-free.
#
# Target GPUs: sm_89 (RTX 6000 Ada), sm_90 (H100 / H100 NVL) and sm_120
# (RTX 5090 / RTX PRO 6000 Blackwell). All are covered by the default cu128
# wheels pinned in pyproject.toml. The named cu130 image remains available via
# PYTORCH_CUDA.

FROM python:3.12-slim-bookworm

# Tag is hardcoded on purpose: variables in a COPY --from stage name are only
# expanded by BuildKit, and expand to empty under the classic builder. uv is a
# build-time installer only -- the environment it produces is frozen into the
# image and recorded in /opt/resolved-requirements.txt -- so pin a full version
# here (e.g. uv:0.9.18) if you want byte-reproducible rebuilds.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Triton JIT-compiles GPU kernels at *runtime* and invokes a C compiler to build
# the launcher stubs, so gcc must be in the image, not just at build time. This
# is load-bearing, not incidental: training.use_liger_kernel puts Triton kernels
# on the forward/backward path of every step.
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
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    # Generous timeout: the torch wheels are ~3 GB and a slow proxy will
    # otherwise trip uv's default per-request deadline.
    UV_HTTP_TIMEOUT=300 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN uv venv --python 3.12 "$VIRTUAL_ENV"

# Resolve and install project dependencies. The project-root uv.lock is
# required: it is the dependency set validated on the staging GPU host. The
# default cu128 build fails if that lock is stale. The retained cu130 build
# transforms the copied manifest and therefore still resolves from it until it
# has a separate lock; see docs/DEPLOYMENT_BACKLOG.md.
#
# WORKDIR matters: uv reads the [tool.uv] table (PyTorch index and source pins)
# from the pyproject in the working directory.
WORKDIR /tmp/deps
COPY pyproject.toml uv.lock /tmp/deps/
ARG PYTORCH_CUDA=cu128
RUN case "$PYTORCH_CUDA" in \
        cu128) uv sync --locked --no-dev --no-install-project ;; \
        cu130) \
            sed -i \
                -e 's/torch==2.8.0/torch==2.13.0/' \
                -e 's/pytorch-cu128/pytorch-cu130/g' \
                -e 's#/whl/cu128#/whl/cu130#' \
                pyproject.toml \
            && uv pip install -r pyproject.toml ;; \
        *) echo "Unsupported PYTORCH_CUDA=$PYTORCH_CUDA (expected cu128 or cu130)" >&2; exit 2 ;; \
    esac

# bitsandbytes>=0.48.0 is declared in pyproject.toml and therefore locked for
# cu128. Older versions lack sm_120 (Blackwell) kernels and fail QLoRA with
# "no kernel image is available for execution on the device".

# FlashAttention 3, installed from a wheel that was fetched or compiled
# *outside* this build by scripts/fetch_flash_attn3_wheel.sh or
# scripts/build_flash_attn3_wheel.sh.
#
# Compiling it here instead would mean a CUDA toolkit (~3 GB) in the build and
# a 1-3 hour nvcc run on every rebuild, to produce a binary that only a Hopper
# GPU can execute. Building it once into a transferable wheel keeps this layer
# at a few seconds and keeps the toolkit out of the shipped image entirely.
#
# --no-index and --no-deps are load-bearing: the wheel is installed *after*
# `uv sync --locked`, so nothing may re-resolve or perturb the locked
# environment. FA3's runtime dependencies (torch, einops, packaging, ninja) are
# already present in the lock, so --no-deps drops nothing.
#
# The wheel is ABI-pinned to the Torch, Python, CUDA and glibc combination it
# was built against. Rebuild it whenever the Torch pin changes.
#
# Produces a separate image; tag it accordingly rather than overwriting the
# default one:
#   IMAGE_TAG=cu128-fa3-py312 INSTALL_FLASH_ATTN3=1 docker compose build trainer
ARG INSTALL_FLASH_ATTN3=0
COPY wheels/ /tmp/wheels/
COPY scripts/verify_flash_attn3.py /tmp/verify_flash_attn3.py
RUN if [ "$INSTALL_FLASH_ATTN3" = "1" ]; then \
        set -- /tmp/wheels/flash_attn_3-*.whl; \
        [ "$#" -eq 1 ] && [ -f "$1" ] \
            || { echo "INSTALL_FLASH_ATTN3=1 requires exactly one flash_attn_3 wheel in wheels/; run scripts/fetch_flash_attn3_wheel.sh or scripts/build_flash_attn3_wheel.sh" >&2; exit 2; }; \
        uv pip install --no-index --no-deps "$1"; \
        python -c 'import importlib.metadata as m, torch; assert torch.__version__.split("+")[0] == "2.8.0", torch.__version__; assert torch.version.cuda == "12.8", torch.version.cuda; assert torch._C._GLIBCXX_USE_CXX11_ABI is True; assert m.version("flash-attn-3").startswith("3.0.0b1"), m.version("flash-attn-3")'; \
        python /tmp/verify_flash_attn3.py; \
    fi \
    && rm -rf /tmp/wheels /tmp/verify_flash_attn3.py

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
