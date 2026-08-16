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


IN_FEATURES = 8
OUT_FEATURES = 16
LORA_RANK = 4


def _build_tiny_base_model():
    """Real modules, because the validation now inspects types and weight dimensions."""
    import torch

    class TinyAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False)
            self.v_proj = torch.nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False)
            self.norm = torch.nn.LayerNorm(OUT_FEATURES)

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = TinyAttention()
            self.config = MagicMock()
            self.config.model_type = "gemma3"

    return TinyModel()


def _write_adapter_tensors(adapter_dir: Path, tensors) -> None:
    import torch

    adapter_dir.mkdir(parents=True, exist_ok=True)
    torch.save(tensors, adapter_dir / "adapter_model.bin")


def _lora_state(module_path: str, a_shape, b_shape):
    import torch

    return {
        f"base_model.model.{module_path}.lora_A.weight": torch.zeros(a_shape),
        f"base_model.model.{module_path}.lora_B.weight": torch.zeros(b_shape),
    }


def test_validate_adapter_architecture_accepts_matching_shapes(tmp_path):
    from scripts.merge_lora_adapter import validate_adapter_architecture

    base = _build_tiny_base_model()
    adapter_dir = tmp_path / "adapter_ok"
    _write_adapter_tensors(
        adapter_dir,
        {
            **_lora_state("self_attn.q_proj", (LORA_RANK, IN_FEATURES), (OUT_FEATURES, LORA_RANK)),
            **_lora_state("self_attn.v_proj", (LORA_RANK, IN_FEATURES), (OUT_FEATURES, LORA_RANK)),
        },
    )

    peft_cfg = MagicMock()
    peft_cfg.target_modules = ["q_proj", "v_proj"]
    peft_cfg.r = LORA_RANK

    result = validate_adapter_architecture(base, peft_cfg, adapter_path=adapter_dir)

    assert result["validated"] is True
    assert result["target_modules_matched"] == ["q_proj", "v_proj"]
    assert result["lora_tensors_checked"] == 4
    assert result["modules_shape_validated"] == 2


def test_validate_adapter_architecture_rejects_dimension_mismatch(tmp_path):
    """A same-named projection of the wrong width must fail before merge_and_unload."""
    from scripts.merge_lora_adapter import validate_adapter_architecture

    base = _build_tiny_base_model()
    adapter_dir = tmp_path / "adapter_wrong_width"
    _write_adapter_tensors(
        adapter_dir,
        _lora_state("self_attn.q_proj", (LORA_RANK, IN_FEATURES * 2), (OUT_FEATURES, LORA_RANK)),
    )

    peft_cfg = MagicMock()
    peft_cfg.target_modules = ["q_proj"]
    peft_cfg.r = LORA_RANK

    with pytest.raises(ValueError, match="dimension mismatches"):
        validate_adapter_architecture(base, peft_cfg, adapter_path=adapter_dir, allow_mismatch=False)


def test_validate_adapter_architecture_rejects_non_linear_target(tmp_path):
    from scripts.merge_lora_adapter import validate_adapter_architecture

    base = _build_tiny_base_model()
    adapter_dir = tmp_path / "adapter_non_linear"
    _write_adapter_tensors(
        adapter_dir,
        _lora_state("self_attn.norm", (LORA_RANK, OUT_FEATURES), (OUT_FEATURES, LORA_RANK)),
    )

    peft_cfg = MagicMock()
    peft_cfg.target_modules = ["norm"]
    peft_cfg.r = LORA_RANK

    with pytest.raises(ValueError, match="incompatible module types"):
        validate_adapter_architecture(base, peft_cfg, adapter_path=adapter_dir, allow_mismatch=False)


def test_validate_adapter_architecture_missing_target_requires_override(tmp_path):
    from scripts.merge_lora_adapter import validate_adapter_architecture

    base = _build_tiny_base_model()
    adapter_dir = tmp_path / "adapter_missing_target"
    _write_adapter_tensors(
        adapter_dir,
        _lora_state("self_attn.q_proj", (LORA_RANK, IN_FEATURES), (OUT_FEATURES, LORA_RANK)),
    )

    peft_cfg = MagicMock()
    peft_cfg.target_modules = ["q_proj", "nonexistent_proj"]
    peft_cfg.r = LORA_RANK

    with pytest.raises(ValueError, match="target modules not found"):
        validate_adapter_architecture(base, peft_cfg, adapter_path=adapter_dir, allow_mismatch=False)

    result = validate_adapter_architecture(
        base,
        peft_cfg,
        adapter_path=adapter_dir,
        allow_mismatch=True,
        override_reason="Experimental architecture test",
    )
    assert result["validated"] is False
    assert result["target_modules_missing"] == ["nonexistent_proj"]


def test_merge_script_has_no_release_activation_path():
    """Activation belongs to promote_model_release.py, which enforces the full gate set."""
    args = parse_args([
        "--base-model", "google/translategemma-12b-it",
        "--adapter", "checkpoints/sft-adapter",
        "--output-dir", "exports/test-export",
    ])
    assert not hasattr(args, "current_symlink")
    assert not hasattr(args, "previous_symlink")


def test_immutable_revision_must_resolve(tmp_path):
    """An unresolvable 40-char SHA must abort rather than be echoed back as provenance."""
    from scripts.merge_lora_adapter import resolve_model_provenance

    sha = "a" * 40
    with patch("huggingface_hub.model_info", side_effect=OSError("offline")), patch(
        "scripts.merge_lora_adapter.resolve_cached_snapshot_commit", return_value=None
    ):
        with pytest.raises(ValueError, match="could not be resolved"):
            resolve_model_provenance("google/translategemma-12b-it", sha)


def test_immutable_revision_resolved_from_local_snapshot():
    from scripts.merge_lora_adapter import resolve_model_provenance

    sha = "b" * 40
    with patch("huggingface_hub.model_info", side_effect=OSError("offline")), patch(
        "scripts.merge_lora_adapter.resolve_cached_snapshot_commit", return_value=sha
    ):
        provenance = resolve_model_provenance("google/translategemma-12b-it", sha)

    assert provenance["resolved_revision"] == sha
    assert provenance["revision_type"] == "commit_sha"
    assert provenance["resolution_source"] == "local_hf_cache_snapshot"


def _make_release(root: Path, release_id: str, shard_bytes: bytes) -> Path:
    release_dir = root / "releases" / release_id
    release_dir.mkdir(parents=True)
    (release_dir / "config.json").write_text("{}")
    (release_dir / "generation_config.json").write_text(json.dumps({"eos_token_id": [1, 106]}))
    (release_dir / "tokenizer.json").write_text(
        json.dumps({"added_tokens": [{"id": 106, "content": "<end_of_turn>"}]})
    )
    (release_dir / "tokenizer_config.json").write_text("{}")
    (release_dir / "special_tokens_map.json").write_text("{}")
    (release_dir / "model.safetensors").write_bytes(shard_bytes)
    (release_dir / "merge_manifest.json").write_text(json.dumps({
        "release_id": release_id, "created_at": "now", "base_model": "base", "adapter": "ad",
        "stop_token_ids": [1, 106], "file_inventory": []
    }))
    (release_dir / "merge_manifest.sha256").write_text(
        f"{compute_file_sha256(release_dir / 'merge_manifest.json')}  merge_manifest.json\n"
    )
    (release_dir / "SHA256SUMS").write_text("")
    return release_dir


def _write_attestation(root: Path, release_dir: Path, name: str) -> Path:
    """Attestation with every required gate passing, bound to this release's manifest."""
    from datetime import datetime, timezone

    evidence_dir = root / "evidence"
    evidence_dir.mkdir(exist_ok=True)
    evidence_file = evidence_dir / f"{name}-report.json"
    evidence_file.write_text(json.dumps({"report": name}))
    evidence_hash = compute_file_sha256(evidence_file)
    now = datetime.now(timezone.utc).isoformat()

    attestation = {
        "manifest_sha256": compute_file_sha256(release_dir / "merge_manifest.json"),
        "release_id": release_dir.name,
        "gates": {
            gate: {
                "passed": True,
                "verified_at": now,
                "evidence": str(evidence_file),
                "evidence_sha256": evidence_hash,
            }
            for gate in ("merged_quality", "degeneration", "vllm_smoke", "deployment_preflight")
        },
    }
    attestation_path = root / f"{name}-attestation.json"
    attestation_path.write_text(json.dumps(attestation, indent=2))
    return attestation_path


def _write_anchor(root: Path, release_dir: Path, name: str) -> Path:
    """External anchor, deliberately stored outside the release directory."""
    anchor_dir = root / "anchors"
    anchor_dir.mkdir(exist_ok=True)
    anchor_path = anchor_dir / f"{name}.sha256"
    anchor_path.write_text(
        f"{compute_file_sha256(release_dir / 'merge_manifest.json')}  merge_manifest.json\n"
    )
    return anchor_path


def test_promotion_requires_external_anchor_and_attestation(tmp_path):
    from scripts.promote_model_release import promote_release

    rel = _make_release(tmp_path, "rel1", b"shard1")
    attestation = _write_attestation(tmp_path, rel, "rel1")
    curr, prev = tmp_path / "current", tmp_path / "previous"

    # Co-located checksum only: no external anchor, so promotion must refuse.
    assert promote_release(
        rel, curr, prev, attestation_file=str(attestation), skip_checksums=True
    ) is False
    assert not curr.exists()

    # External anchor but no attestation: behavioral gates are unproven.
    anchor = _write_anchor(tmp_path, rel, "rel1")
    assert promote_release(
        rel, curr, prev, attestation_file=None, trusted_anchor_file=str(anchor), skip_checksums=True
    ) is False
    assert not curr.exists()


def test_promotion_rejects_attestation_bound_to_another_release(tmp_path):
    from scripts.promote_model_release import promote_release

    rel1 = _make_release(tmp_path, "rel1", b"shard1")
    rel2 = _make_release(tmp_path, "rel2", b"shard2")
    foreign_attestation = _write_attestation(tmp_path, rel1, "rel1")
    anchor2 = _write_anchor(tmp_path, rel2, "rel2")
    curr, prev = tmp_path / "current", tmp_path / "previous"

    assert promote_release(
        rel2,
        curr,
        prev,
        attestation_file=str(foreign_attestation),
        trusted_anchor_file=str(anchor2),
        skip_checksums=True,
    ) is False
    assert not curr.exists()


def test_promotion_rejects_stale_gate_evidence(tmp_path):
    from scripts.promote_model_release import promote_release

    rel = _make_release(tmp_path, "rel1", b"shard1")
    attestation_path = _write_attestation(tmp_path, rel, "rel1")
    attestation = json.loads(attestation_path.read_text())
    attestation["gates"]["vllm_smoke"]["verified_at"] = "2020-01-01T00:00:00+00:00"
    attestation_path.write_text(json.dumps(attestation))
    anchor = _write_anchor(tmp_path, rel, "rel1")
    curr, prev = tmp_path / "current", tmp_path / "previous"

    assert promote_release(
        rel,
        curr,
        prev,
        attestation_file=str(attestation_path),
        trusted_anchor_file=str(anchor),
        skip_checksums=True,
    ) is False


def test_promote_rollback_and_roll_forward_walk_release_history(tmp_path):
    from scripts.promote_model_release import promote_release, rollback_release

    rel1 = _make_release(tmp_path, "rel1", b"shard1")
    rel2 = _make_release(tmp_path, "rel2", b"shard2")
    curr, prev = tmp_path / "current", tmp_path / "previous"

    assert promote_release(
        rel1,
        curr,
        prev,
        attestation_file=str(_write_attestation(tmp_path, rel1, "rel1")),
        trusted_anchor_file=str(_write_anchor(tmp_path, rel1, "rel1")),
        skip_checksums=True,
    ) is True
    assert curr.resolve() == rel1.resolve()

    assert promote_release(
        rel2,
        curr,
        prev,
        attestation_file=str(_write_attestation(tmp_path, rel2, "rel2")),
        trusted_anchor_file=str(_write_anchor(tmp_path, rel2, "rel2")),
        skip_checksums=True,
    ) is True
    assert curr.resolve() == rel2.resolve()
    assert prev.resolve() == rel1.resolve()

    # Rollback to rel1, and the release we left becomes the rollback target.
    assert rollback_release(curr, prev, skip_checksums=True) is True
    assert curr.resolve() == rel1.resolve()
    assert prev.resolve() == rel2.resolve()

    # Rolling again returns to rel2 instead of pinning one stale pointer forever.
    assert rollback_release(curr, prev, skip_checksums=True) is True
    assert curr.resolve() == rel2.resolve()
    assert prev.resolve() == rel1.resolve()

    index = json.loads((tmp_path / "release_index.json").read_text())
    assert [entry["action"] for entry in index["history"]] == ["promote", "promote", "rollback", "rollback"]


def test_rollback_rejects_tampered_previous_release(tmp_path):
    """Rollback re-verifies against the hash trusted at promotion time."""
    from scripts.promote_model_release import promote_release, rollback_release

    rel1 = _make_release(tmp_path, "rel1", b"shard1")
    rel2 = _make_release(tmp_path, "rel2", b"shard2")
    curr, prev = tmp_path / "current", tmp_path / "previous"

    promote_release(
        rel1,
        curr,
        prev,
        attestation_file=str(_write_attestation(tmp_path, rel1, "rel1")),
        trusted_anchor_file=str(_write_anchor(tmp_path, rel1, "rel1")),
        skip_checksums=True,
    )
    promote_release(
        rel2,
        curr,
        prev,
        attestation_file=str(_write_attestation(tmp_path, rel2, "rel2")),
        trusted_anchor_file=str(_write_anchor(tmp_path, rel2, "rel2")),
        skip_checksums=True,
    )

    # Attacker rewrites the old release plus its co-located checksum during an incident.
    manifest = rel1 / "merge_manifest.json"
    manifest.write_text(json.dumps({
        "release_id": "rel1", "created_at": "now", "base_model": "evil", "adapter": "ad",
        "stop_token_ids": [1], "file_inventory": []
    }))
    (rel1 / "merge_manifest.sha256").write_text(f"{compute_file_sha256(manifest)}  merge_manifest.json\n")

    assert rollback_release(curr, prev, skip_checksums=True) is False
    assert curr.resolve() == rel2.resolve()


