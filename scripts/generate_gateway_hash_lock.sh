#!/usr/bin/env bash
# Generate gateway/requirements-hashes.txt: the exact-version lock plus per-artifact
# SHA256 hashes, so `pip install --require-hashes` refuses any republished or
# substituted distribution file.
#
# Run on a networked host, then commit the result:
#   ./scripts/generate_gateway_hash_lock.sh
#
# The gateway image build (gateway/Dockerfile) requires this file and installs with
# --require-hashes; it will not build without it.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GATEWAY_DIR="${REPO_ROOT}/gateway"
LOCK_FILE="${GATEWAY_DIR}/requirements-lock.txt"
HASH_FILE="${GATEWAY_DIR}/requirements-hashes.txt"

if [[ ! -f "${LOCK_FILE}" ]]; then
    echo "Missing ${LOCK_FILE}" >&2
    exit 1
fi

# uv resolves and emits hashes for the exact pinned versions already in the lock file.
uv pip compile "${LOCK_FILE}" \
    --generate-hashes \
    --no-header \
    --output-file "${HASH_FILE}"

echo "Wrote ${HASH_FILE}"
echo "Record the resulting gateway image digest in serving/COMPATIBILITY.md after building:"
echo "  docker build --build-arg PYTHON_BASE_IMAGE=python:3.11-slim@sha256:<digest> -t translategemma-gateway:<version> ${GATEWAY_DIR}"
echo "  docker image inspect translategemma-gateway:<version> --format '{{index .RepoDigests 0}}'"
