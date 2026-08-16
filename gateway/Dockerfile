# The base image must be pinned by digest, not by the mutable python:3.11-slim tag.
# No default is provided on purpose: an unpinned build fails instead of silently
# producing an image nobody can reproduce. Record the digest in serving/COMPATIBILITY.md.
#
#   docker build --build-arg PYTHON_BASE_IMAGE=python:3.11-slim@sha256:<digest> ...
ARG PYTHON_BASE_IMAGE
FROM ${PYTHON_BASE_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN groupadd -r appuser && useradd -r -g appuser appuser

# requirements-hashes.txt carries per-artifact SHA256 hashes, so a compromised or
# republished index file cannot be installed even at an identical version.
# Generate it on a networked host: scripts/generate_gateway_hash_lock.sh
COPY requirements.txt requirements-lock.txt requirements-hashes.txt /app/
RUN pip install --no-cache-dir --require-hashes -r /app/requirements-hashes.txt \
    && pip freeze > /app/resolved-requirements.txt

COPY . /app/

USER appuser

EXPOSE 8080

HEALTHCHECK --interval=10s --timeout=5s --start-period=5s --retries=3 \
    CMD python3 -c "import urllib.request, sys; r = urllib.request.urlopen('http://localhost:8080/ready'); sys.exit(0 if r.status == 200 else 1)" || exit 1

# Single worker per container to guarantee exact in-process admission control and metrics integrity
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
