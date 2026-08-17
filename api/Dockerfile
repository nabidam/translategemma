# TranslateGemma API gateway image.
#
# This directory is self-contained and is deployed without the rest of the
# repository: copy api/ to the target machine and build from inside it, with
# this directory as the context.
#
#     cd api && docker build -t translategemma-api .
#
# Design notes (why it looks like this):
#   * CPU only. The weights are served by the vLLM container; this image holds
#     a tokenizer and an HTTP client, so there is no torch, no CUDA, and no GPU
#     reservation on the service in docker-compose.yml.
#   * Dependencies are installed from requirements.txt in a layer that only
#     mentions requirements.txt, so editing the API source never re-resolves or
#     re-downloads them. The uv cache is additionally kept in a BuildKit cache
#     mount, so even a *changed* requirements.txt reuses every wheel it already
#     has. Both require BuildKit (default in Docker 23+).
#   * prompting.py here is a copy of the module of the same name at the root of
#     the source repository (which is NOT shipped with this directory). It
#     carries the prompt rendering and stop-token contract from that
#     repository's docs/2026-08-10_adapter_degeneration_analysis.md, where a
#     mismatch is silent: the model still returns fluent text. The copy is kept
#     byte-identical by scripts/sync_api_vendored.py and enforced by
#     tests/test_api_vendored_modules.py; nothing needs to run on the
#     deployment host.
FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
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

# No weights are baked in and none are loaded. Point TG_VLLM_BASE_URL at the
# vLLM server and TG_BASE_MODEL_ID at a mounted copy of the checkpoint it
# serves (read for the tokenizer and chat template only), and set HF_HOME /
# HUGGINGFACE_HUB_CACHE in compose when running offline.
CMD ["fastapi", "run", "main.py", "--host", "0.0.0.0", "--port", "8000"]
