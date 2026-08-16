"""Trust-chain tests for the gateway's manifest authenticity and payload verification.

The property under test is that a release directory can never authorize itself:
integrity evidence stored beside the manifest is reported, but only an external
anchor plus a matching payload may satisfy production verification.
"""

import hashlib
import json
from pathlib import Path

from gateway.manifest_verification import (
    AUTHENTICITY_COLOCATED,
    AUTHENTICITY_TRUSTED,
    AUTHENTICITY_UNVERIFIED,
    verify_model_release,
)

PAYLOAD_FILES = {
    "config.json": json.dumps({"model_type": "gemma3"}),
    "generation_config.json": json.dumps({"eos_token_id": [1, 106]}),
    "tokenizer.json": json.dumps({"added_tokens": [{"id": 106, "content": "<end_of_turn>"}]}),
    "model-00001-of-00001.safetensors": "weight-bytes",
}


def build_release(tmp_path: Path, payload_overrides=None, write_colocated_anchor=True) -> Path:
    """Create a structurally valid release directory with manifest, SHA256SUMS, and anchor."""
    model_dir = tmp_path / "release"
    model_dir.mkdir()

    files = dict(PAYLOAD_FILES)
    files.update(payload_overrides or {})

    inventory = []
    checksum_lines = []
    for name, content in sorted(files.items()):
        path = model_dir / name
        path.write_text(content, encoding="utf-8")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        inventory.append({"path": name, "size_bytes": path.stat().st_size, "sha256": digest})
        checksum_lines.append(f"{digest}  {name}")

    (model_dir / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    manifest = {
        "release_id": "tg-test-release",
        "created_at": "2026-08-16T00:00:00+00:00",
        "base_model": "google/translategemma-12b-it",
        "adapter": "/home/operator/checkpoints/sft",
        "stop_token_ids": [1, 106],
        "stop_tokens": ["<eos>", "<end_of_turn>"],
        "file_inventory": inventory,
    }
    manifest_path = model_dir / "merge_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if write_colocated_anchor:
        manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        (model_dir / "merge_manifest.sha256").write_text(
            f"{manifest_hash}  merge_manifest.json\n", encoding="utf-8"
        )

    return model_dir


def manifest_hash_of(model_dir: Path) -> str:
    return hashlib.sha256((model_dir / "merge_manifest.json").read_bytes()).hexdigest()


def test_colocated_checksum_is_integrity_only_never_authenticity(tmp_path):
    model_dir = build_release(tmp_path)

    result = verify_model_release(str(model_dir))

    assert result.integrity_verified is True
    assert result.authenticity_verified is False
    assert result.authenticity_status == AUTHENTICITY_COLOCATED
    assert result.is_trusted is False


def test_colocated_only_release_cannot_satisfy_required_verification(tmp_path):
    """A tampered release that rewrote both manifest and its own checksum stays untrusted."""
    model_dir = build_release(tmp_path)
    manifest_path = model_dir / "merge_manifest.json"
    tampered = json.loads(manifest_path.read_text(encoding="utf-8"))
    tampered["stop_token_ids"] = [1]
    manifest_path.write_text(json.dumps(tampered, indent=2), encoding="utf-8")
    (model_dir / "merge_manifest.sha256").write_text(
        f"{manifest_hash_of(model_dir)}  merge_manifest.json\n", encoding="utf-8"
    )

    result = verify_model_release(str(model_dir))

    assert result.authenticity_status == AUTHENTICITY_COLOCATED
    assert result.authenticity_verified is False
    # This is exactly what the gateway's require_verified_manifest gate compares against.
    assert result.authenticity_status != AUTHENTICITY_TRUSTED


def test_external_anchor_file_authenticates_release(tmp_path):
    model_dir = build_release(tmp_path)
    anchor_dir = tmp_path / "anchors"
    anchor_dir.mkdir()
    anchor_file = anchor_dir / "current.sha256"
    anchor_file.write_text(f"{manifest_hash_of(model_dir)}  merge_manifest.json\n", encoding="utf-8")

    result = verify_model_release(str(model_dir), trusted_anchor_file=str(anchor_file))

    assert result.authenticity_status == AUTHENTICITY_TRUSTED
    assert result.authenticity_verified is True
    assert result.payload_verified is True
    assert result.is_trusted is True


def test_external_anchor_mismatch_is_rejected(tmp_path):
    model_dir = build_release(tmp_path)

    result = verify_model_release(str(model_dir), trusted_manifest_sha256="0" * 64)

    assert result.authenticity_status == AUTHENTICITY_UNVERIFIED
    assert result.authenticity_verified is False
    assert "mismatch" in (result.error or "").lower()


def test_payload_edit_after_promotion_fails_payload_verification(tmp_path):
    """Authenticating the manifest says nothing about the bytes vLLM will load."""
    model_dir = build_release(tmp_path)
    anchor_file = tmp_path / "current.sha256"
    anchor_file.write_text(f"{manifest_hash_of(model_dir)}  merge_manifest.json\n", encoding="utf-8")

    (model_dir / "model-00001-of-00001.safetensors").write_text("altered-weights", encoding="utf-8")

    result = verify_model_release(str(model_dir), trusted_anchor_file=str(anchor_file))

    assert result.authenticity_verified is True
    assert result.payload_verified is False
    assert result.is_trusted is False
    assert "Payload hash mismatch" in (result.error or "")


def test_extra_untracked_payload_file_fails_verification(tmp_path):
    model_dir = build_release(tmp_path)
    anchor_file = tmp_path / "current.sha256"
    anchor_file.write_text(f"{manifest_hash_of(model_dir)}  merge_manifest.json\n", encoding="utf-8")

    (model_dir / "extra-shard.safetensors").write_text("smuggled", encoding="utf-8")

    result = verify_model_release(str(model_dir), trusted_anchor_file=str(anchor_file))

    assert result.payload_verified is False
    assert "absent from the authenticated manifest inventory" in (result.error or "")


def test_runtime_stop_contract_comes_from_mounted_generation_config(tmp_path):
    model_dir = build_release(tmp_path)

    result = verify_model_release(str(model_dir))

    assert result.runtime_stop_token_ids == [1, 106]
    assert result.runtime_stop_tokens[1] == "<end_of_turn>"


def test_missing_turn_end_stop_id_is_reported(tmp_path):
    model_dir = build_release(
        tmp_path,
        payload_overrides={"generation_config.json": json.dumps({"eos_token_id": [1]})},
    )

    result = verify_model_release(str(model_dir))

    assert result.runtime_stop_token_ids == [1]
    assert "missing required stop token IDs [106]" in (result.error or "")
