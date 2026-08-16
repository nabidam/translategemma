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
    # Proves Any import and module evaluation passes
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
    # Valid mapping
    tok_json = tmp_path / "tokenizer.json"
    tok_json.write_text(json.dumps({
        "added_tokens": [{"id": 106, "content": "<end_of_turn>"}, {"id": 1, "content": "<eos>"}]
    }))
    ok, msg = verify_tokenizer_token_mapping(tmp_path)
    assert ok
    assert "verified" in msg.lower()

    # Invalid mapping
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


def test_verify_export_end_to_end(tmp_path):
    model_dir = tmp_path / "test_release"
    model_dir.mkdir()

    # Create dummy files
    (model_dir / "config.json").write_text('{"model_type": "gemma"}')
    (model_dir / "generation_config.json").write_text('{"eos_token_id": [1, 106]}')
    (model_dir / "tokenizer.json").write_text(json.dumps({
        "added_tokens": [{"id": 106, "content": "<end_of_turn>"}]
    }))
    (model_dir / "tokenizer_config.json").write_text('{}')
    (model_dir / "special_tokens_map.json").write_text('{}')
    (model_dir / "model-00001-of-00001.safetensors").write_bytes(b"dummy_weights")

    manifest = {
        "release_id": "test-v1",
        "created_at": "2026-08-16T00:00:00Z",
        "base_model": "google/translategemma-12b-it",
        "adapter": "test-adapter",
        "stop_token_ids": [1, 106],
        "file_inventory": [],
    }
    (model_dir / "merge_manifest.json").write_text(json.dumps(manifest))

    # Compute checksums covering every file
    checksum_lines = []
    for f in sorted(model_dir.iterdir()):
        if f.is_file():
            checksum_lines.append(f"{compute_sha256(f)}  {f.name}")
    (model_dir / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n")

    # Update SHA256SUMS itself in the file list
    checksum_lines = []
    for f in sorted(model_dir.iterdir()):
        if f.is_file():
            checksum_lines.append(f"{compute_sha256(f)}  {f.name}")
    (model_dir / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n")

    assert verify_export(str(model_dir), skip_checksums=False) is True
