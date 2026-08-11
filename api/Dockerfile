# TranslateGemma serving image.
#
# The build context is api/ alone -- this directory is self-contained and is
# deployed without the rest of the repository:
#
#     docker build -t translategemma-api api/
#
# Design notes (why it looks like this):
#   * Base is python:3.12-slim-bookworm with torch installed from the cu128
#     wheel index, the same combination the trainer image uses. The torch CUDA
#     wheels vendor their own CUDA runtime, so a CUDA base image would add a
#     second, conflicting toolkit and several GB. The host contributes the
#     driver via nvidia-container-toolkit; nothing else.
#   * Dependencies are installed from requirements.txt in a layer that only
#     mentions requirements.txt, so editing the API source never re-resolves or
#     re-downloads them. The uv cache is additionally kept in a BuildKit cache
#     mount, so even a *changed* requirements.txt reuses every wheel it already
#     has -- the ~3 GB torch download happens once per machine, not once per
#     rebuild. Both require BuildKit (default in Docker 23+).
#   * api/prompting.py and api/model_loading.py are copies of the project-root
#     modules of the same name, kept byte-identical by
#     scripts/sync_api_vendored.py and enforced by
#     tests/test_api_vendored_modules.py. They carry the prompt rendering and
#     stop-token contract from docs/2026-08-10_adapter_degeneration_analysis.md,
#     where a mismatch is silent: the model still returns fluent text.
FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Triton JIT-compiles GPU kernels at runtime and shells out to a C compiler to
# build the launcher stubs, so gcc must be present at run time, not just at
# build time. Same reasoning as the trainer image; apt is told to fetch
# serially with retries because the default parallel pipeline is what draws
# 502s from a proxy under load.
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
    # requirements.txt names the cu128 wheel index as an extra index. uv's
    # default first-index strategy would find `torch` on PyPI and stop there,
    # never reaching the CUDA build; this tells it to consider both.
    UV_INDEX_STRATEGY=unsafe-best-match \
    # The torch wheels are ~3 GB and a slow proxy trips uv's default deadline.
    UV_HTTP_TIMEOUT=300 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TOKENIZERS_PARALLELISM=false

RUN uv venv --python 3.12 "$VIRTUAL_ENV"

# Dependency layer. Cached on requirements.txt alone, and backed by a uv cache
# mount that survives across builds.
COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -r /tmp/requirements.txt

# Record the exact resolved environment in the image, as the trainer image does:
#   docker run --rm IMAGE cat /opt/resolved-requirements.txt
RUN uv pip freeze > /opt/resolved-requirements.txt

# Source last: this is the layer that actually changes.
WORKDIR /app
COPY . /app/

EXPOSE 8000

# No weights are baked in, so one image serves any checkpoint. Point
# TG_BASE_MODEL_ID / TG_ADAPTER_PATH at a mounted model tree, and set
# HF_HOME / HUGGINGFACE_HUB_CACHE in compose when serving offline.
CMD ["fastapi", "run", "main.py", "--host", "0.0.0.0", "--port", "8000"]
