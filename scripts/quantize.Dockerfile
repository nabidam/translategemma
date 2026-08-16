# Quantiser image for scripts/quantize_fp8.py.
#
# Built from the vLLM image rather than from the trainer image, for two reasons:
#
#   * It already carries the exact torch/CUDA build that will serve the result,
#     including the sm_120 (Blackwell) kernels. Quantising against one torch and
#     serving against another is how an FP8 checkpoint ends up technically valid
#     and practically unloadable.
#   * The trainer image installs from a locked environment (`uv sync --locked`).
#     Adding llm-compressor there would re-resolve that lock, which is exactly
#     what the lock exists to prevent.
#
# Build it once, with network access:
#
#     docker compose build quantizer
#
# Pin the base tag to the same version the serving compose files use.
ARG VLLM_IMAGE=vllm/vllm-openai:v0.13.0
FROM ${VLLM_IMAGE}

# llm-compressor brings compressed-tensors, the format vLLM detects from the
# saved config.json. rich and pyyaml are for logging_utils, which the script
# shares with the rest of the repository.
RUN pip install --no-cache-dir \
        "llmcompressor>=0.8,<1.0" \
        "rich>=15.0.0" \
        "pyyaml"

# The base image's entrypoint is `vllm serve`; this image runs one script.
ENTRYPOINT ["python3", "scripts/quantize_fp8.py"]
WORKDIR /workspace
