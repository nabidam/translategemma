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


def test_validate_adapter_architecture():
    from scripts.merge_lora_adapter import validate_adapter_architecture

    # Mock base model
    mock_base = MagicMock()
    mock_base.named_modules.return_value = [
        ("model.layers.0.self_attn.q_proj", MagicMock()),
        ("model.layers.0.self_attn.v_proj", MagicMock()),
        ("model.layers.0.self_attn.k_proj", MagicMock()),
        ("model.layers.0.self_attn.o_proj", MagicMock()),
    ]
    mock_base.config.model_type = "gemma2"

    # 1. Matching target modules
    peft_cfg_valid = MagicMock()
    peft_cfg_valid.target_modules = ["q_proj", "v_proj"]
    res = validate_adapter_architecture(mock_base, peft_cfg_valid)
    assert res["validated"] is True
    assert res["target_modules_matched"] == ["q_proj", "v_proj"]

    # 2. Missing target modules without override -> raises
    peft_cfg_bad = MagicMock()
    peft_cfg_bad.target_modules = ["q_proj", "nonexistent_proj"]
    with pytest.raises(ValueError, match="Adapter target modules.*not found"):
        validate_adapter_architecture(mock_base, peft_cfg_bad, allow_mismatch=False)

    # 3. Missing target modules with override -> passes with reason
    res_override = validate_adapter_architecture(
        mock_base,
        peft_cfg_bad,
        allow_mismatch=True,
        override_reason="Experimental architecture test",
    )
    assert res_override["validated"] is False
    assert res_override["target_modules_missing"] == ["nonexistent_proj"]


def test_promote_and_rollback_release_pointers(tmp_path):
    from scripts.promote_model_release import promote_release, rollback_release

    # Create dummy verified release 1
    rel1 = tmp_path / "releases" / "rel1"
    rel1.mkdir(parents=True)
    (rel1 / "config.json").write_text("{}")
    (rel1 / "generation_config.json").write_text(json.dumps({"eos_token_id": [1, 106]}))
    (rel1 / "tokenizer.json").write_text(json.dumps({"added_tokens": [{"id": 106, "content": "<end_of_turn>"}]}))
    (rel1 / "tokenizer_config.json").write_text("{}")
    (rel1 / "special_tokens_map.json").write_text("{}")
    (rel1 / "model.safetensors").write_bytes(b"shard1")
    (rel1 / "merge_manifest.json").write_text(json.dumps({
        "release_id": "rel1", "created_at": "now", "base_model": "base", "adapter": "ad",
        "stop_token_ids": [1, 106], "file_inventory": []
    }))
    (rel1 / "merge_manifest.sha256").write_text(f"{compute_file_sha256(rel1 / 'merge_manifest.json')}  merge_manifest.json\n")
    (rel1 / "SHA256SUMS").write_text("")

    # Create dummy verified release 2
    rel2 = tmp_path / "releases" / "rel2"
    rel2.mkdir(parents=True)
    (rel2 / "config.json").write_text("{}")
    (rel2 / "generation_config.json").write_text(json.dumps({"eos_token_id": [1, 106]}))
    (rel2 / "tokenizer.json").write_text(json.dumps({"added_tokens": [{"id": 106, "content": "<end_of_turn>"}]}))
    (rel2 / "tokenizer_config.json").write_text("{}")
    (rel2 / "special_tokens_map.json").write_text("{}")
    (rel2 / "model.safetensors").write_bytes(b"shard2")
    (rel2 / "merge_manifest.json").write_text(json.dumps({
        "release_id": "rel2", "created_at": "now", "base_model": "base", "adapter": "ad",
        "stop_token_ids": [1, 106], "file_inventory": []
    }))
    (rel2 / "merge_manifest.sha256").write_text(f"{compute_file_sha256(rel2 / 'merge_manifest.json')}  merge_manifest.json\n")
    (rel2 / "SHA256SUMS").write_text("")

    curr_symlink = tmp_path / "current"
    prev_symlink = tmp_path / "previous"

    # Promote rel1
    ok1 = promote_release(rel1, curr_symlink, prev_symlink, skip_checksums=True)
    assert ok1 is True
    assert curr_symlink.resolve() == rel1.resolve()

    # Promote rel2 -> curr points to rel2, prev points to rel1
    ok2 = promote_release(rel2, curr_symlink, prev_symlink, skip_checksums=True)
    assert ok2 is True
    assert curr_symlink.resolve() == rel2.resolve()
    assert prev_symlink.resolve() == rel1.resolve()

    # Rollback -> curr reverts to rel1
    ok_roll = rollback_release(curr_symlink, prev_symlink, skip_checksums=True)
    assert ok_roll is True
    assert curr_symlink.resolve() == rel1.resolve()


