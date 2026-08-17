# Quantiser image for scripts/quantize_fp8.py.
#
# A separate image, because llm-compressor's supported versions and this
# repository's training lock do not intersect. Measured against
# torch==2.8.0 / transformers==4.57.6:
#
#   llmcompressor 0.8.x   transformers <=4.56.2
#   llmcompressor 0.9.x   transformers <=4.57.3
#   llmcompressor 0.10.x  torch >=2.9.0
#   llmcompressor 0.11.0  torch >=2.10.0
#   llmcompressor 0.12+   transformers >=5.9.0
#
# There is no release that installs into the trainer image. That is not an
# accident on either side: llm-compressor tracks vLLM's release cadence, while
# the training stack is deliberately frozen on the versions the evaluation
# harness was validated against (and pinned below transformers v5, which has a
# new internal contract). Forcing them together would mean moving torch or
# transformers under the trainer, which is the one thing uv.lock exists to
# prevent.
#
# The vLLM image resolves this by construction: it already carries the torch
# that llm-compressor 0.10.x wants, because it is the torch vLLM itself was
# built against.
#
# Build it once, with network access:
#
#     docker compose build quantizer
#
# Keep VLLM_IMAGE and the llmcompressor range moving together: the pin below is
# chosen to match this base image's torch, so bumping one means rechecking the
# other.
ARG VLLM_IMAGE=vllm/vllm-openai:v0.13.0
FROM ${VLLM_IMAGE}

WORKDIR /workspace

# torch is constrained to whatever the base image already has, so a resolution
# that would replace it fails here instead of downloading three gigabytes and
# producing an image whose torch no longer matches its vLLM. rich and pyyaml are
# for logging_utils, shared with the rest of the repository.
RUN pip freeze | grep -E '^torch==' > /tmp/torch-constraint.txt \
    && cat /tmp/torch-constraint.txt \
    && pip install --no-cache-dir --constraint /tmp/torch-constraint.txt \
        "llmcompressor>=0.10,<0.11" \
        "rich>=15.0.0" \
        "pyyaml" \
    && python3 -c 'from llmcompressor import oneshot; from llmcompressor.modifiers.quantization import QuantizationModifier' \
    && rm -f /tmp/torch-constraint.txt

# The base image's entrypoint is `vllm serve`; this image runs one script.
ENTRYPOINT ["python3", "scripts/quantize_fp8.py"]
