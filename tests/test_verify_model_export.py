"""Unit tests for scripts/verify_model_export.py."""

import json
from pathlib import Path

from scripts.verify_model_export import (
    check_required_files,
    compute_sha256,
    verify_checksums_and_inventory,
    verify_export,
    verify_generation_config,
    verify_manifest,
    verify_tokenizer_token_mapping,
)


def test_import_script():
    import scripts.verify_model_export as script
    assert hasattr(script, "verify_export")


def test_check_required_files_missing(tmp_path):
    ok, missing = check_required_files(tmp_path)
    assert not ok
    assert "config.json" in missing
    assert "generation_config.json" in missing


def test_verify_generation_config_valid(tmp_path):
    gen_config = tmp_path / "generation_config.json"
    gen_config.write_text(json.dumps({"eos_token_id": [1, 106]}))
    ok, msg = verify_generation_config(gen_config)
    assert ok
    assert "106" in msg


def test_verify_generation_config_missing_turn_end(tmp_path):
    gen_config = tmp_path / "generation_config.json"
    gen_config.write_text(json.dumps({"eos_token_id": [1]}))
    ok, msg = verify_generation_config(gen_config)
    assert not ok
    assert "Missing essential stop token IDs {106}" in msg


def test_verify_tokenizer_token_mapping(tmp_path):
    tok_json = tmp_path / "tokenizer.json"
    tok_json.write_text(json.dumps({
        "added_tokens": [{"id": 106, "content": "<end_of_turn>"}, {"id": 1, "content": "<eos>"}]
    }))
    ok, msg = verify_tokenizer_token_mapping(tmp_path)
    assert ok
    assert "verified" in msg.lower()

    tok_json.write_text(json.dumps({
        "added_tokens": [{"id": 106, "content": "<other_token>"}]
    }))
    ok_bad, msg_bad = verify_tokenizer_token_mapping(tmp_path)
    assert not ok_bad
    assert "Tokenizer mapping check failed" in msg_bad


def test_verify_checksums_and_inventory_traversal_rejection(tmp_path):
    sha256sums = tmp_path / "SHA256SUMS"
    sha256sums.write_text("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  ../outside.txt\n")
    ok, errors = verify_checksums_and_inventory(tmp_path, {"file_inventory": []})
    assert not ok
    assert any("path traversal" in err for err in errors)


def test_verify_manifest_anchor_validation(tmp_path):
    manifest_path = tmp_path / "merge_manifest.json"
    anchor_path = tmp_path / "merge_manifest.sha256"

    manifest_data = {
        "release_id": "test-v1",
        "created_at": "2026-08-16T00:00:00Z",
        "base_model": "google/translategemma-12b-it",
        "adapter": "test-adapter",
        "stop_token_ids": [1, 106],
        "file_inventory": [],
    }
    manifest_bytes = json.dumps(manifest_data, indent=2).encode("utf-8")
    manifest_path.write_bytes(manifest_bytes)

    # 1. Missing anchor -> fail
    ok, err = verify_manifest(manifest_path)
    assert not ok
    assert "anchor not found" in err["error"]

    # 2. Tampered anchor -> fail
    anchor_path.write_text("badhash12345  merge_manifest.json\n")
    ok, err = verify_manifest(manifest_path)
    assert not ok
    assert "anchor mismatch" in err["error"]

    # 3. Valid anchor -> pass
    valid_hash = compute_sha256(manifest_path)
    anchor_path.write_text(f"{valid_hash}  merge_manifest.json\n")
    ok, parsed = verify_manifest(manifest_path)
    assert ok
    assert parsed["release_id"] == "test-v1"


def test_verify_export_end_to_end(tmp_path):
    model_dir = tmp_path / "test_release"
    model_dir.mkdir()

    # Create payload files
    payload_files = {
        "config.json": b'{"model_type": "gemma"}',
        "generation_config.json": b'{"eos_token_id": [1, 106]}',
        "tokenizer.json": json.dumps({"added_tokens": [{"id": 106, "content": "<end_of_turn>"}]}).encode("utf-8"),
        "tokenizer_config.json": b'{}',
        "special_tokens_map.json": b'{}',
        "model-00001-of-00001.safetensors": b"dummy_weights",
    }

    file_inventory = []
    checksum_lines = []

    for name, content in payload_files.items():
        p = model_dir / name
        p.write_bytes(content)
        file_hash = compute_sha256(p)
        file_size = len(content)
        file_inventory.append({
            "path": name,
            "size_bytes": file_size,
            "sha256": file_hash,
        })
        checksum_lines.append(f"{file_hash}  {name}")

    # Write SHA256SUMS covering all payload files
    (model_dir / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n")

    # Write merge_manifest.json containing inventory
    manifest = {
        "release_id": "test-v1",
        "created_at": "2026-08-16T00:00:00Z",
        "base_model": "google/translategemma-12b-it",
        "adapter": "test-adapter",
        "stop_token_ids": [1, 106],
        "file_inventory": file_inventory,
    }
    manifest_path = model_dir / "merge_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Write detached merge_manifest.sha256 anchor
    manifest_hash = compute_sha256(manifest_path)
    (model_dir / "merge_manifest.sha256").write_text(f"{manifest_hash}  merge_manifest.json\n")

    # 1. Verification with local anchor
    assert verify_export(str(model_dir), skip_checksums=False) is True

    # 2. Verification with expected manifest SHA256
    assert verify_export(str(model_dir), expected_manifest_sha256=manifest_hash) is True
    assert verify_export(str(model_dir), expected_manifest_sha256="wronghash123") is False

    # 3. Verification with external trusted anchor file
    ext_anchor_file = tmp_path / "trusted_anchor.sha256"
    ext_anchor_file.write_text(f"{manifest_hash}  merge_manifest.json\n")
    assert verify_export(str(model_dir), trusted_anchor_file=str(ext_anchor_file)) is True

    bad_ext_anchor = tmp_path / "bad_anchor.sha256"
    bad_ext_anchor.write_text("badhash99999  merge_manifest.json\n")
    assert verify_export(str(model_dir), trusted_anchor_file=str(bad_ext_anchor)) is False


