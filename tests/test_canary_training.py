from pathlib import Path

import pytest

from canary_config import canary_run_config


def _config():
    return {
        "model": {"output_dir": "./full-output"},
        "training": {"resume_from_checkpoint": True},
        "evaluation": {"run_after_training": True, "output_dir": "./evaluation"},
        "canary": {
            "max_examples": 7,
            "max_steps": 3,
            "output_dir": "./canary-output",
            "run_after_training": False,
            "evaluation_output_dir": "./canary-evaluation",
            "resume_from_checkpoint": None,
        },
    }


def test_canary_config_isolated_and_bounded():
    config = _config()

    canary, max_examples, max_steps = canary_run_config(config)

    assert (max_examples, max_steps) == (7, 3)
    assert canary["model"]["output_dir"] == "./canary-output"
    assert canary["training"]["resume_from_checkpoint"] is None
    assert canary["evaluation"]["run_after_training"] is False
    assert config["model"]["output_dir"] == "./full-output"
    assert config["training"]["resume_from_checkpoint"] is True


def test_canary_requires_positive_example_limit():
    config = _config()
    config["canary"]["max_examples"] = 0

    with pytest.raises(ValueError, match="canary.max_examples"):
        canary_run_config(config)


def test_canary_flag_is_a_pipeline_mode():
    source = Path("train.py").read_text(encoding="utf-8")

    assert 'modes.add_argument("--canary"' in source
    assert "run_pipeline(canary_config, max_examples=max_examples, max_steps=max_steps)" in source
