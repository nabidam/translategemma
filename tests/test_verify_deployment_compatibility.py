"""Tests for the deployment compatibility preflight.

The preflight's job is to reject deployments whose *effective* configuration differs
from an evidenced approval, so these tests exercise env-override resolution, host
evidence, and the refusal to accept self-asserted matrix approval.
"""

import json
from pathlib import Path

from scripts.verify_deployment_compatibility import (
    parse_command_flags,
    substitute_env,
    verify_deployment,
)

APPROVED_DIGEST = "sha256:" + "a" * 64
OTHER_DIGEST = "sha256:" + "b" * 64


def write_compose(path: Path, image: str, max_model_len: str = "4096") -> None:
    path.write_text(
        json.dumps(
            {
                "services": {
                    "vllm": {
                        "image": image,
                        "command": [
                            "--model", "/models/model",
                            "--dtype", "bfloat16",
                            "--tensor-parallel-size", "1",
                            "--max-model-len", max_model_len,
                            "--enforce-eager",
                        ],
                    },
                    "gateway": {
                        "environment": [
                            f"TG_VLLM_MAX_MODEL_LEN={max_model_len}",
                            f"TG_MAX_TOTAL_CONTEXT_TOKENS={max_model_len}",
                            "TG_REQUIRE_VERIFIED_MANIFEST=true",
                            "TG_TRUSTED_ANCHOR_FILE=/trust/current.sha256",
                        ]
                    },
                }
            }
        ),
        encoding="utf-8",
    )


def write_matrix(path: Path, evidence: dict) -> None:
    path.write_text(
        json.dumps(
            {
                "approved_serving_configurations": [
                    {
                        "image_repository": "vllm/vllm-openai",
                        "image_tag": "v0.13.0",
                        "pinned_digest": APPROVED_DIGEST,
                        "cuda_version_minimum": "12.4",
                        "nvidia_driver_minimum": "535.104.05",
                        "supported_gpu_architectures": ["sm_90 (NVIDIA Hopper: H100, H200)"],
                        "runtime_flags": {
                            "dtype": "bfloat16",
                            "max_model_len": 4096,
                            "tensor_parallel_size": 1,
                            "enforce_eager": True,
                        },
                        "verification_evidence": evidence,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def full_evidence(tmp_path: Path) -> dict:
    import hashlib

    evidence = {"status": "APPROVED_FOR_PRODUCTION", "verified_by": "ops", "verified_date": "2026-08-16",
                "model_manifest_sha256": "c" * 64}
    for field in ("host_report", "image_inspect_report", "smoke_report"):
        artifact = tmp_path / f"{field}.json"
        artifact.write_text(json.dumps({"artifact": field}), encoding="utf-8")
        evidence[field] = str(artifact)
        evidence[f"{field}_sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    return evidence


def write_host_inputs(tmp_path: Path, digest: str = APPROVED_DIGEST):
    inspect_path = tmp_path / "image_inspect.json"
    inspect_path.write_text(
        json.dumps([{"Id": "sha256:deadbeef", "RepoDigests": [f"vllm/vllm-openai@{digest}"]}]),
        encoding="utf-8",
    )
    host_path = tmp_path / "host_report.json"
    host_path.write_text(
        json.dumps(
            {
                "gpu_name": "NVIDIA H100 80GB HBM3",
                "compute_capability": "9.0",
                "driver_version": "550.54.14",
                "cuda_version": "12.8",
            }
        ),
        encoding="utf-8",
    )
    return inspect_path, host_path


def test_substitute_env_resolves_compose_defaults():
    assert substitute_env("${VLLM_IMAGE:-fallback}", {}) == "fallback"
    assert substitute_env("${VLLM_IMAGE:-fallback}", {"VLLM_IMAGE": "override"}) == "override"


def test_parse_command_flags_handles_values_and_bare_flags():
    flags = parse_command_flags(["--dtype", "bfloat16", "--enforce-eager", "--max-model-len=4096"])
    assert flags["--dtype"] == "bfloat16"
    assert flags["--enforce-eager"] is True
    assert flags["--max-model-len"] == "4096"


def test_environment_override_of_image_is_resolved_and_rejected(tmp_path, monkeypatch):
    """An override that swaps the image must fail even though the file still names the pinned digest."""
    compose = tmp_path / "compose.yml"
    write_compose(compose, "${VLLM_IMAGE:-vllm/vllm-openai:v0.13.0@" + APPROVED_DIGEST + "}")
    matrix = tmp_path / "matrix.json"
    write_matrix(matrix, full_evidence(tmp_path))
    inspect_path, host_path = write_host_inputs(tmp_path)

    monkeypatch.setenv("VLLM_IMAGE", "vllm/vllm-openai:v0.13.0@" + OTHER_DIGEST)
    assert verify_deployment(
        compose_path=compose,
        matrix_path=matrix,
        image_inspect=str(inspect_path),
        host_report=str(host_path),
    ) is False


def test_unverified_matrix_entry_is_rejected(tmp_path):
    compose = tmp_path / "compose.yml"
    write_compose(compose, "vllm/vllm-openai:v0.13.0@" + APPROVED_DIGEST)
    matrix = tmp_path / "matrix.json"
    write_matrix(matrix, {"status": "UNVERIFIED", "token_parity_verified": True})
    inspect_path, host_path = write_host_inputs(tmp_path)

    assert verify_deployment(
        compose_path=compose,
        matrix_path=matrix,
        image_inspect=str(inspect_path),
        host_report=str(host_path),
    ) is False


def test_missing_host_evidence_is_rejected(tmp_path):
    compose = tmp_path / "compose.yml"
    write_compose(compose, "vllm/vllm-openai:v0.13.0@" + APPROVED_DIGEST)
    matrix = tmp_path / "matrix.json"
    write_matrix(matrix, full_evidence(tmp_path))

    assert verify_deployment(compose_path=compose, matrix_path=matrix) is False


def test_effective_configuration_with_host_evidence_passes(tmp_path):
    compose = tmp_path / "compose.yml"
    write_compose(compose, "vllm/vllm-openai:v0.13.0@" + APPROVED_DIGEST)
    matrix = tmp_path / "matrix.json"
    write_matrix(matrix, full_evidence(tmp_path))
    inspect_path, host_path = write_host_inputs(tmp_path)

    assert verify_deployment(
        compose_path=compose,
        matrix_path=matrix,
        image_inspect=str(inspect_path),
        host_report=str(host_path),
    ) is True


def test_context_length_disagreement_is_rejected(tmp_path):
    compose = tmp_path / "compose.yml"
    write_compose(compose, "vllm/vllm-openai:v0.13.0@" + APPROVED_DIGEST, max_model_len="8192")
    matrix = tmp_path / "matrix.json"
    write_matrix(matrix, full_evidence(tmp_path))
    inspect_path, host_path = write_host_inputs(tmp_path)

    # max_model_len no longer matches the approved runtime flags.
    assert verify_deployment(
        compose_path=compose,
        matrix_path=matrix,
        image_inspect=str(inspect_path),
        host_report=str(host_path),
    ) is False


def test_unsupported_gpu_architecture_is_rejected(tmp_path):
    compose = tmp_path / "compose.yml"
    write_compose(compose, "vllm/vllm-openai:v0.13.0@" + APPROVED_DIGEST)
    matrix = tmp_path / "matrix.json"
    write_matrix(matrix, full_evidence(tmp_path))
    inspect_path, host_path = write_host_inputs(tmp_path)
    host_path.write_text(
        json.dumps(
            {
                "gpu_name": "NVIDIA T4",
                "compute_capability": "7.5",
                "driver_version": "550.54.14",
                "cuda_version": "12.8",
            }
        ),
        encoding="utf-8",
    )

    assert verify_deployment(
        compose_path=compose,
        matrix_path=matrix,
        image_inspect=str(inspect_path),
        host_report=str(host_path),
    ) is False
