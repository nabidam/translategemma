#!/usr/bin/env python
"""Preflight verification that the EFFECTIVE deployment matches an evidenced compatibility entry.

Source text is not a deployment. This check therefore works on resolved configuration
and on facts collected from the target host:

  1. Compose configuration is parsed as YAML (or read from `docker compose config`
     output), and `${VLLM_IMAGE}`-style overrides are resolved from the environment,
     so an override that swaps the image cannot pass by leaving the pinned digest
     visible in the file.
  2. The image digest actually present on the host (docker inspect RepoDigests) is
     compared with the approved entry, not just the digest written in the file.
  3. Host GPU architecture, NVIDIA driver, and CUDA runtime are compared against the
     approved entry's minimums and supported architectures.
  4. Effective runtime flags (dtype, max-model-len, tensor-parallel-size, enforce-eager)
     and the gateway context alignment are compared value-by-value.
  5. Matrix entries must carry verifiable evidence artifacts (hashed files produced on
     the serving host). Self-asserted booleans are rejected.

Collect host inputs on the serving host:

    docker image inspect <image> > image_inspect.json
    nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader > gpu.csv
    python scripts/verify_deployment_compatibility.py \
        --compose-file serving/docker-compose.yml \
        --matrix-file serving/vllm_compatibility_matrix.json \
        --image-inspect image_inspect.json \
        --host-report host_report.json

host_report.json format:
    {"gpu_name": "NVIDIA H100 80GB HBM3", "compute_capability": "9.0",
     "driver_version": "550.54.14", "cuda_version": "12.8"}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("verify_deployment_compatibility")

ENV_SUBSTITUTION = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::?-([^}]*))?\}")
REQUIRED_EVIDENCE_FIELDS = (
    "verified_by",
    "verified_date",
    "host_report",
    "image_inspect_report",
    "smoke_report",
    "model_manifest_sha256",
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--compose-file",
        type=str,
        default="serving/docker-compose.yml",
        help="Path to docker-compose.yml to verify (env overrides are resolved).",
    )
    parser.add_argument(
        "--resolved-config",
        type=str,
        default=None,
        help="Output of `docker compose config` (YAML or JSON). Preferred over --compose-file.",
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default=None,
        help="Deployment .env file whose values participate in variable resolution.",
    )
    parser.add_argument(
        "--matrix-file",
        type=str,
        default="serving/vllm_compatibility_matrix.json",
        help="Path to vLLM compatibility matrix JSON.",
    )
    parser.add_argument(
        "--image-inspect",
        type=str,
        default=None,
        help="`docker image inspect <image>` JSON captured on the serving host.",
    )
    parser.add_argument(
        "--image-digest",
        type=str,
        default=None,
        help="Image digest observed on the host, if not supplying --image-inspect.",
    )
    parser.add_argument(
        "--host-report",
        type=str,
        default=None,
        help="JSON report of host GPU architecture, driver, and CUDA runtime.",
    )
    parser.add_argument(
        "--allow-missing-host-evidence",
        action="store_true",
        help="Development only: skip host/image inspection instead of failing.",
    )
    parser.add_argument(
        "--allow-unverified-matrix",
        action="store_true",
        help="Development only: accept matrix entries that carry no verifiable evidence.",
    )
    return parser.parse_args(argv)


def compute_sha256(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_env_file(env_file: Optional[str]) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not env_file:
        return values
    path = Path(env_file)
    if not path.is_file():
        raise FileNotFoundError(f"Env file not found: {path}")
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    return values


def substitute_env(value: str, env: Dict[str, str]) -> str:
    """Resolve ${VAR}, ${VAR:-default}, and ${VAR-default} the way Compose does."""

    def replace(match: re.Match) -> str:
        name, default = match.group(1), match.group(2)
        resolved = env.get(name)
        if resolved:
            return resolved
        return default if default is not None else ""

    return ENV_SUBSTITUTION.sub(replace, value)


def resolve_structure(node: Any, env: Dict[str, str]) -> Any:
    if isinstance(node, str):
        return substitute_env(node, env)
    if isinstance(node, list):
        return [resolve_structure(item, env) for item in node]
    if isinstance(node, dict):
        return {key: resolve_structure(item, env) for key, item in node.items()}
    return node


def load_compose(
    compose_path: Optional[Path],
    resolved_config_path: Optional[Path],
    env: Dict[str, str],
) -> Dict[str, Any]:
    source = resolved_config_path or compose_path
    if source is None or not source.is_file():
        raise FileNotFoundError(f"Compose configuration not found: {source}")

    text = source.read_text(encoding="utf-8")
    try:
        import yaml

        data = yaml.safe_load(text)
    except ImportError:
        # `docker compose config --format json` output still parses without PyYAML.
        data = json.loads(text)

    if not isinstance(data, dict):
        raise ValueError(f"Compose configuration in {source} did not parse to a mapping.")

    # `docker compose config` output is already resolved; resolving again is a no-op.
    return resolve_structure(data, env)


def parse_command_flags(command: Any) -> Dict[str, Any]:
    """Turn a vLLM command list into a flag -> value mapping (bare flags map to True)."""
    if isinstance(command, str):
        tokens = command.split()
    elif isinstance(command, list):
        tokens = [str(token) for token in command]
    else:
        return {}

    flags: Dict[str, Any] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--"):
            index += 1
            continue
        if "=" in token:
            name, _, value = token.partition("=")
            flags[name] = value
            index += 1
            continue
        if index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
            flags[token] = tokens[index + 1]
            index += 2
        else:
            flags[token] = True
            index += 1
    return flags


def environment_mapping(service: Dict[str, Any]) -> Dict[str, str]:
    env_section = service.get("environment", {})
    if isinstance(env_section, dict):
        return {str(k): str(v) for k, v in env_section.items() if v is not None}
    mapping: Dict[str, str] = {}
    for item in env_section or []:
        key, _, value = str(item).partition("=")
        mapping[key] = value
    return mapping


def version_tuple(value: str) -> Tuple[int, ...]:
    parts = re.findall(r"\d+", str(value))
    return tuple(int(part) for part in parts) or (0,)


def compute_capability_to_sm(compute_capability: str) -> str:
    digits = re.findall(r"\d+", str(compute_capability))
    if len(digits) >= 2:
        return f"sm_{digits[0]}{digits[1]}"
    if len(digits) == 1 and len(digits[0]) >= 2:
        return f"sm_{digits[0]}"
    return str(compute_capability)


def extract_host_digests(image_inspect_path: Path) -> List[str]:
    data = json.loads(image_inspect_path.read_text(encoding="utf-8"))
    entries = data if isinstance(data, list) else [data]
    digests: List[str] = []
    for entry in entries:
        for repo_digest in entry.get("RepoDigests", []) or []:
            _, _, digest = str(repo_digest).partition("@")
            if digest:
                digests.append(digest.lower())
        image_id = entry.get("Id")
        if image_id:
            digests.append(str(image_id).lower())
    return digests


def verify_matrix_evidence(entry: Dict[str, Any], matrix_path: Path) -> List[str]:
    """Reject approval claims that cannot be audited back to archived host output."""
    errors: List[str] = []
    evidence = entry.get("verification_evidence", {})
    status = str(evidence.get("status", "")).upper()

    if status != "APPROVED_FOR_PRODUCTION":
        errors.append(
            f"Matrix entry status is {status or 'MISSING'}; only APPROVED_FOR_PRODUCTION entries may be deployed."
        )

    for field in REQUIRED_EVIDENCE_FIELDS:
        if not evidence.get(field):
            errors.append(f"Matrix entry evidence is missing required field {field!r}.")

    for artifact_field in ("host_report", "image_inspect_report", "smoke_report"):
        artifact = evidence.get(artifact_field)
        if not artifact:
            continue
        artifact_path = Path(artifact)
        if not artifact_path.is_absolute():
            artifact_path = (matrix_path.parent / artifact_path).resolve()
        if not artifact_path.is_file():
            errors.append(f"Evidence artifact {artifact_field} not found: {artifact_path}")
            continue
        declared = str(evidence.get(f"{artifact_field}_sha256", "")).strip().lower()
        if not declared:
            errors.append(f"Evidence artifact {artifact_field} has no declared {artifact_field}_sha256.")
            continue
        actual = compute_sha256(artifact_path).lower()
        if declared != actual:
            errors.append(
                f"Evidence artifact {artifact_field} hash mismatch: matrix says {declared}, file hashes to {actual}."
            )

    return errors


def verify_deployment(
    compose_path: Optional[Path],
    matrix_path: Path,
    resolved_config_path: Optional[Path] = None,
    env_file: Optional[str] = None,
    image_inspect: Optional[str] = None,
    image_digest: Optional[str] = None,
    host_report: Optional[str] = None,
    allow_missing_host_evidence: bool = False,
    allow_unverified_matrix: bool = False,
) -> bool:
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

    env = dict(os.environ)
    try:
        env.update(load_env_file(env_file))
        compose = load_compose(compose_path, resolved_config_path, env)
    except Exception as e:
        logger.error("Could not load deployment configuration: %s", e)
        return False

    services = compose.get("services", {})
    vllm_service = services.get("vllm")
    gateway_service = services.get("gateway", {})
    if not isinstance(vllm_service, dict):
        logger.error("Compose configuration has no 'vllm' service.")
        return False

    errors: List[str] = []

    # 1. Effective image reference after variable resolution.
    image_ref = str(vllm_service.get("image", ""))
    digest_match = re.search(r"sha256:[0-9a-fA-F]{64}", image_ref)
    if not digest_match:
        logger.error(
            "Effective vLLM image %r carries no immutable sha256 digest. "
            "An overridable tag cannot be verified against the matrix.",
            image_ref,
        )
        return False
    configured_digest = digest_match.group(0).lower()
    logger.info("Effective vLLM image reference: %s", image_ref)

    matched_entry = None
    for entry in approved_entries:
        if str(entry.get("pinned_digest", "")).lower() == configured_digest:
            matched_entry = entry
            break

    if not matched_entry:
        logger.error(
            "Configured digest %s is NOT present in approved compatibility matrix %s!",
            configured_digest,
            matrix_path,
        )
        return False

    # 2. Matrix evidence must be auditable.
    evidence_errors = verify_matrix_evidence(matched_entry, matrix_path)
    if evidence_errors:
        if allow_unverified_matrix:
            for message in evidence_errors:
                logger.warning("Matrix evidence (ignored by --allow-unverified-matrix): %s", message)
        else:
            errors.extend(evidence_errors)

    # 3. Image digest actually present on the host.
    host_digests: List[str] = []
    if image_inspect:
        inspect_path = Path(image_inspect)
        if not inspect_path.is_file():
            errors.append(f"Image inspect report not found: {inspect_path}")
        else:
            try:
                host_digests = extract_host_digests(inspect_path)
            except Exception as e:
                errors.append(f"Failed to parse image inspect report {inspect_path}: {e}")
    if image_digest:
        host_digests.append(image_digest.strip().lower())

    if host_digests:
        if configured_digest not in host_digests:
            errors.append(
                f"Host image digests {host_digests} do not include the approved digest {configured_digest}."
            )
        else:
            logger.info("Host image digest matches the approved pinned digest.")
    elif not allow_missing_host_evidence:
        errors.append(
            "No host image evidence supplied. Pass --image-inspect (docker image inspect output) "
            "or --image-digest, or --allow-missing-host-evidence for development."
        )

    # 4. Host GPU architecture, driver, and CUDA runtime.
    if host_report:
        report_path = Path(host_report)
        if not report_path.is_file():
            errors.append(f"Host report not found: {report_path}")
        else:
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except Exception as e:
                errors.append(f"Failed to parse host report {report_path}: {e}")
                report = {}

            driver_min = matched_entry.get("nvidia_driver_minimum")
            driver_actual = report.get("driver_version")
            if driver_min and driver_actual:
                if version_tuple(driver_actual) < version_tuple(driver_min):
                    errors.append(
                        f"Host NVIDIA driver {driver_actual} is below the approved minimum {driver_min}."
                    )
            else:
                errors.append("Host report does not state driver_version.")

            cuda_min = matched_entry.get("cuda_version_minimum")
            cuda_actual = report.get("cuda_version")
            if cuda_min and cuda_actual:
                if version_tuple(cuda_actual) < version_tuple(cuda_min):
                    errors.append(f"Host CUDA runtime {cuda_actual} is below the approved minimum {cuda_min}.")
            else:
                errors.append("Host report does not state cuda_version.")

            capability = report.get("compute_capability")
            if capability:
                host_sm = compute_capability_to_sm(capability)
                supported = [str(arch) for arch in matched_entry.get("supported_gpu_architectures", [])]
                if not any(host_sm in arch for arch in supported):
                    errors.append(
                        f"Host GPU architecture {host_sm} ({report.get('gpu_name', 'unknown GPU')}) "
                        f"is not in the approved architectures {supported}."
                    )
            else:
                errors.append("Host report does not state compute_capability.")
    elif not allow_missing_host_evidence:
        errors.append(
            "No host GPU report supplied. Pass --host-report with driver_version, cuda_version, and "
            "compute_capability collected on the serving host."
        )

    # 5. Effective runtime flags.
    approved_flags = matched_entry.get("runtime_flags", {})
    actual_flags = parse_command_flags(vllm_service.get("command"))

    if "dtype" in approved_flags and str(actual_flags.get("--dtype")) != str(approved_flags["dtype"]):
        errors.append(f"dtype mismatch: compose uses {actual_flags.get('--dtype')!r}, approved is {approved_flags['dtype']!r}.")

    if "max_model_len" in approved_flags:
        actual_len = actual_flags.get("--max-model-len")
        if str(actual_len) != str(approved_flags["max_model_len"]):
            errors.append(
                f"max_model_len mismatch: compose uses {actual_len!r}, approved is {approved_flags['max_model_len']!r}."
            )

    if approved_flags.get("enforce_eager") and "--enforce-eager" not in actual_flags:
        errors.append("Approved entry requires --enforce-eager, which the effective command does not set.")

    approved_tp = approved_flags.get("tensor_parallel_size")
    actual_tp = actual_flags.get("--tensor-parallel-size")
    if approved_tp is not None and str(actual_tp) != str(approved_tp):
        errors.append(f"tensor_parallel_size mismatch: compose uses {actual_tp!r}, approved is {approved_tp!r}.")

    # 6. Gateway context alignment must hold on effective values, not on variable names.
    gateway_env = environment_mapping(gateway_service) if isinstance(gateway_service, dict) else {}
    gateway_max_len = gateway_env.get("TG_VLLM_MAX_MODEL_LEN")
    gateway_context = gateway_env.get("TG_MAX_TOTAL_CONTEXT_TOKENS")
    engine_max_len = actual_flags.get("--max-model-len")
    if gateway_max_len is None or gateway_context is None:
        errors.append("Gateway service does not set both TG_VLLM_MAX_MODEL_LEN and TG_MAX_TOTAL_CONTEXT_TOKENS.")
    elif not (str(gateway_max_len) == str(gateway_context) == str(engine_max_len)):
        errors.append(
            f"Context length disagreement: engine --max-model-len={engine_max_len!r}, "
            f"TG_VLLM_MAX_MODEL_LEN={gateway_max_len!r}, TG_MAX_TOTAL_CONTEXT_TOKENS={gateway_context!r}."
        )

    # 7. Gateway trust chain must be fail-closed in the effective configuration.
    if str(gateway_env.get("TG_REQUIRE_VERIFIED_MANIFEST", "")).lower() not in ("true", "1", "yes"):
        errors.append("Gateway does not set TG_REQUIRE_VERIFIED_MANIFEST=true in the effective configuration.")
    if not (gateway_env.get("TG_TRUSTED_ANCHOR_FILE") or gateway_env.get("TG_TRUSTED_MANIFEST_SHA256")):
        errors.append("Gateway configures no external trust anchor (TG_TRUSTED_ANCHOR_FILE or TG_TRUSTED_MANIFEST_SHA256).")

    if errors:
        logger.error("Deployment compatibility preflight FAILED:\n  - %s", "\n  - ".join(errors))
        return False

    logger.info(
        "Deployment compatibility preflight PASSED for %s (image %s:%s, digest %s).",
        resolved_config_path or compose_path,
        matched_entry.get("image_repository"),
        matched_entry.get("image_tag"),
        configured_digest,
    )
    return True


def main() -> int:
    args = parse_args()
    success = verify_deployment(
        compose_path=Path(args.compose_file) if args.compose_file else None,
        matrix_path=Path(args.matrix_file),
        resolved_config_path=Path(args.resolved_config) if args.resolved_config else None,
        env_file=args.env_file,
        image_inspect=args.image_inspect,
        image_digest=args.image_digest,
        host_report=args.host_report,
        allow_missing_host_evidence=args.allow_missing_host_evidence,
        allow_unverified_matrix=args.allow_unverified_matrix,
    )
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
