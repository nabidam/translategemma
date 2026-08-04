#!/usr/bin/env bash
# Download the reviewed community FA3 wheel that exactly matches the default
# runtime: Linux x86_64, CUDA 12.8, Torch 2.8.0, C++11 ABI, and Python >=3.9.
#
# This is the fast path. scripts/build_flash_attn3_wheel.sh remains the
# reproducible official-source fallback.
set -euo pipefail

WHEEL_NAME="flash_attn_3-3.0.0b1+20251110.cu128torch280cxx11abitrue.c8abdd-cp39-abi3-linux_x86_64.whl"
WHEEL_URL="https://github.com/windreamer/flash-attention3-wheels/releases/download/2025.11.10-b4dfcd5/flash_attn_3-3.0.0b1%2B20251110.cu128torch280cxx11abitrue.c8abdd-cp39-abi3-linux_x86_64.whl"
WHEEL_SHA256="6a00f0ba0fd063f228809260c64225cd83aa43ee1c99f7d718f8f104e7e2fd86"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
destination="$repo_root/wheels/$WHEEL_NAME"
partial="$destination.part"
mkdir -p "$repo_root/wheels"

verify_checksum() {
    printf '%s  %s\n' "$WHEEL_SHA256" "$1" | sha256sum --check --status
}

if [[ -f "$destination" ]]; then
    if verify_checksum "$destination"; then
        echo "Already downloaded and checksum verified:"
        echo "  $destination"
        exit 0
    fi
    echo "Existing wheel has the wrong checksum; refusing to overwrite it:" >&2
    echo "  $destination" >&2
    exit 1
fi

echo "Downloading pinned FlashAttention 3 wheel (about 440 MB)."
echo "An interrupted download resumes from $partial"
curl --fail --location \
    --retry 8 --retry-all-errors --retry-delay 5 \
    --connect-timeout 30 --speed-limit 1024 --speed-time 120 \
    --continue-at - --output "$partial" "$WHEEL_URL"

if ! verify_checksum "$partial"; then
    rejected="$partial.bad.$(date -u +%Y%m%dT%H%M%SZ)"
    mv "$partial" "$rejected"
    echo "Downloaded wheel failed SHA-256 verification; preserved it as:" >&2
    echo "  $rejected" >&2
    echo "The next run will download a clean copy." >&2
    exit 1
fi

mv "$partial" "$destination"
echo "Downloaded and checksum verified:"
echo "  $destination"
echo
echo "Next:"
echo "  IMAGE_TAG=cu128-fa3-py312 INSTALL_FLASH_ATTN3=1 docker compose build trainer"
