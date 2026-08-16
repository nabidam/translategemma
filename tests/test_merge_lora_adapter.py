"""Unit tests for scripts/merge_lora_adapter.py."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.merge_lora_adapter import (
    PROMPT_CONTRACT_VERSION,
    compute_file_sha256,
    get_package_versions,
    parse_args,
    validate_adapter_compatibility,
)


def test_parse_args_defaults():
    args = parse_args([
        "--base-model", "google/translategemma-12b-it",
        "--adapter", "checkpoints/sft-adapter",
        "--output-dir", "exports/test-export",
    ])
    assert args.base_model == "google/translategemma-12b-it"
    assert args.adapter == "checkpoints/sft-adapter"
    assert args.output_dir == "exports/test-export"
    assert args.dtype == "bfloat16"
    assert args.device == "auto"
    assert args.max_shard_size == "5GB"
    assert not args.force
    assert args.release_id is None


def test_compute_file_sha256():
    with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
        f.write(b"Hello TranslateGemma serving")
        f_path = Path(f.name)

    try:
        digest = compute_file_sha256(f_path)
        assert isinstance(digest, str)
        assert len(digest) == 64
        # Expected sha256 for b"Hello TranslateGemma serving"
        import hashlib
        expected = hashlib.sha256(b"Hello TranslateGemma serving").hexdigest()
        assert digest == expected
    finally:
        f_path.unlink()


def test_get_package_versions():
    versions = get_package_versions()
    assert "torch" in versions
    assert "transformers" in versions
    assert "peft" in versions


def test_validate_adapter_compatibility_missing_config(tmp_path):
    adapter_dir = tmp_path / "adapter_empty"
    adapter_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="adapter_config.json not found"):
        validate_adapter_compatibility("google/translategemma-12b-it", adapter_dir)


def test_validate_adapter_compatibility_success(tmp_path):
    adapter_dir = tmp_path / "adapter_valid"
    adapter_dir.mkdir()
    config_file = adapter_dir / "adapter_config.json"
    config_file.write_text(json.dumps({
        "base_model_name_or_path": "google/translategemma-12b-it",
        "peft_type": "LORA",
        "r": 16,
        "lora_alpha": 32,
    }))

    with patch("peft.PeftConfig.from_pretrained") as mock_peft_config:
        mock_obj = MagicMock()
        mock_obj.base_model_name_or_path = "google/translategemma-12b-it"
        mock_peft_config.return_value = mock_obj

        result = validate_adapter_compatibility("google/translategemma-12b-it", adapter_dir)
        assert result == mock_obj
