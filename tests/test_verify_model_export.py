"""Unit tests for scripts/verify_model_export.py."""

import json
from pathlib import Path

from scripts.verify_model_export import (
    check_required_files,
    compute_sha256,
    verify_checksums,
    verify_export,
    verify_generation_config,
    verify_manifest,
)


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


def test_verify_export_end_to_end(tmp_path):
    model_dir = tmp_path / "test_release"
    model_dir.mkdir()

    # Create dummy files
    (model_dir / "config.json").write_text('{"model_type": "gemma"}')
    (model_dir / "generation_config.json").write_text('{"eos_token_id": [1, 106]}')
    (model_dir / "tokenizer.json").write_text('{}')
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

    # Compute checksums
    checksum_lines = []
    for f in model_dir.iterdir():
        if f.is_file():
            checksum_lines.append(f"{compute_sha256(f)}  {f.name}")
    (model_dir / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n")

    assert verify_export(str(model_dir), skip_checksums=False) is True
