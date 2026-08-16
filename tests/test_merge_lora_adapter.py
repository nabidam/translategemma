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
    normalize_model_identifier,
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
    assert not args.allow_base_mismatch
    assert args.release_id is None


def test_compute_file_sha256():
    with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
        f.write(b"Hello TranslateGemma serving")
        f_path = Path(f.name)

    try:
        digest = compute_file_sha256(f_path)
        assert isinstance(digest, str)
        assert len(digest) == 64
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


def test_validate_adapter_compatibility_exact_match(tmp_path):
    adapter_dir = tmp_path / "adapter_valid"
    adapter_dir.mkdir()
    config_file = adapter_dir / "adapter_config.json"
    config_file.write_text(json.dumps({
        "base_model_name_or_path": "google/translategemma-12b-it",
        "peft_type": "LORA",
        "r": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "v_proj"],
    }))

    with patch("peft.PeftConfig.from_pretrained") as mock_peft_config:
        mock_obj = MagicMock()
        mock_obj.base_model_name_or_path = "google/translategemma-12b-it"
        mock_obj.peft_type = "LORA"
        mock_obj.r = 16
        mock_obj.lora_alpha = 32
        mock_obj.target_modules = ["q_proj", "v_proj"]
        mock_peft_config.return_value = mock_obj

        peft_cfg, matched, arch = validate_adapter_compatibility("google/translategemma-12b-it", adapter_dir)
        assert peft_cfg == mock_obj
        assert matched is True
        assert arch["r"] == 16


def test_validate_adapter_compatibility_unrelated_repo_rejection(tmp_path):
    adapter_dir = tmp_path / "adapter_mismatch"
    adapter_dir.mkdir()
    config_file = adapter_dir / "adapter_config.json"
    config_file.write_text(json.dumps({
        "base_model_name_or_path": "other-org/translategemma-12b-it",
        "peft_type": "LORA",
    }))

    with patch("peft.PeftConfig.from_pretrained") as mock_peft_config:
        mock_obj = MagicMock()
        mock_obj.base_model_name_or_path = "other-org/translategemma-12b-it"
        mock_obj.peft_type = "LORA"
        mock_obj.target_modules = []
        mock_peft_config.return_value = mock_obj

        # Rejection without override
        with pytest.raises(ValueError, match="does not match requested base model"):
            validate_adapter_compatibility(
                "google/translategemma-12b-it",
                adapter_dir,
                allow_mismatch=False,
            )

        # Rejection with override but missing reason
        with pytest.raises(ValueError, match="requires an explicit --override-reason"):
            validate_adapter_compatibility(
                "google/translategemma-12b-it",
                adapter_dir,
                allow_mismatch=True,
                override_reason=None,
            )

        # Success with explicit override reason
        peft_cfg, matched, _ = validate_adapter_compatibility(
            "google/translategemma-12b-it",
            adapter_dir,
            allow_mismatch=True,
            override_reason="Forked base model with identical weights.",
        )
        assert matched is False


def test_validate_adapter_compatibility_missing_base_identity(tmp_path):
    adapter_dir = tmp_path / "adapter_missing_base"
    adapter_dir.mkdir()
    config_file = adapter_dir / "adapter_config.json"
    config_file.write_text(json.dumps({
        "peft_type": "LORA",
    }))

    with patch("peft.PeftConfig.from_pretrained") as mock_peft_config:
        mock_obj = MagicMock()
        mock_obj.base_model_name_or_path = None
        mock_obj.peft_type = "LORA"
        mock_obj.target_modules = []
        mock_peft_config.return_value = mock_obj

        # Rejection without override
        with pytest.raises(ValueError, match="Adapter config does not specify base_model_name_or_path"):
            validate_adapter_compatibility(
                "google/translategemma-12b-it",
                adapter_dir,
                allow_mismatch=False,
            )

        # Success with override and reason
        peft_cfg, matched, _ = validate_adapter_compatibility(
            "google/translategemma-12b-it",
            adapter_dir,
            allow_mismatch=True,
            override_reason="Legacy fine-tune without base path tag.",
        )
        assert matched is False

