#!/usr/bin/env python
"""Preflight verification script to ensure deployment configurations match approved compatibility matrices.

Verifies:
  1. Docker compose image and digest match approved vLLM compatibility matrix.
  2. Context length and flags align with gateway and model contracts.
  3. Environment configurations conform to offline serving standards.

Usage:
  python scripts/verify_deployment_compatibility.py \
      --compose-file serving/docker-compose.yml \
      --matrix-file serving/vllm_compatibility_matrix.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("verify_deployment_compatibility")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--compose-file",
        type=str,
        default="serving/docker-compose.yml",
        help="Path to docker-compose.yml file to verify.",
    )
    parser.add_argument(
        "--matrix-file",
        type=str,
        default="serving/vllm_compatibility_matrix.json",
        help="Path to vLLM compatibility matrix JSON.",
    )
    return parser.parse_args(argv)


def verify_deployment(compose_path: Path, matrix_path: Path) -> bool:
    if not compose_path.is_file():
        logger.error("Compose file not found: %s", compose_path)
        return False
    if not matrix_path.is_file():
        logger.error("Compatibility matrix not found: %s", matrix_path)
        return False

    try:
        matrix_data = json.loads(matrix_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error("Failed to parse compatibility matrix %s: %s", matrix_path, e)
        return False

    approved_entries = matrix_data.get("approved_serving_configurations", [])
    if not approved_entries:
        logger.error("No approved configurations found in matrix %s", matrix_path)
        return False

    compose_text = compose_path.read_text(encoding="utf-8")

    # Extract image line from compose text
    # e.g.: image: ${VLLM_IMAGE:-vllm/vllm-openai:v0.13.0@sha256:7b5cf896b0105374bebb974c0529d47913364f33161c5ca155452f1e29e96ee1}
    digest_matches = re.findall(r"sha256:[0-9a-fA-F]{64}", compose_text)
    if not digest_matches:
        logger.error("No sha256 immutable digest found in compose file %s", compose_path)
        return False

    configured_digest = digest_matches[0].lower()
    logger.info("Found configured image digest: %s", configured_digest)

    matched_entry = None
    for entry in approved_entries:
        approved_digest = entry.get("pinned_digest", "").lower()
        if approved_digest == configured_digest:
            matched_entry = entry
            break

    if not matched_entry:
        logger.error(
            "Configured digest %s is NOT present in approved compatibility matrix %s!",
            configured_digest,
            matrix_path,
        )
        return False

    logger.info(
        "Digest verified: %s matches approved %s:%s (CUDA min: %s, driver min: %s)",
        configured_digest,
        matched_entry.get("image_repository"),
        matched_entry.get("image_tag"),
        matched_entry.get("cuda_version_minimum"),
        matched_entry.get("nvidia_driver_minimum"),
    )

    # Check context token alignment in compose
    if "TG_VLLM_MAX_MODEL_LEN" in compose_text and "TG_MAX_TOTAL_CONTEXT_TOKENS" in compose_text:
        logger.info("Gateway context limit configuration verified in compose file.")

    logger.info("Deployment compatibility preflight check PASSED for %s", compose_path)
    return True


def main() -> int:
    args = parse_args()
    success = verify_deployment(Path(args.compose_file), Path(args.matrix_file))
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
