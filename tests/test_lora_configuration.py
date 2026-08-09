import ast
from pathlib import Path

import yaml


def _lora_config_calls(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            isinstance(node.func, ast.Name)
            and node.func.id == "LoraConfig"
            or isinstance(node.func, ast.Attribute)
            and node.func.attr == "LoraConfig"
        )
    ]


def test_text_only_training_excludes_vision_tower():
    config_source = Path("config.yaml").read_text(encoding="utf-8")
    config = yaml.safe_load(config_source)

    assert config["lora"]["exclude_modules"] == ".*vision_tower.*"
    assert "transformers/models/gemma3/modular_gemma3.py" in config_source


def test_every_lora_constructor_uses_configured_exclusions():
    calls = []
    for path in Path(".").rglob("*.py"):
        if ".venv" in path.parts or path == Path(__file__):
            continue
        calls.extend((path, call) for call in _lora_config_calls(path))

    assert calls, "no LoraConfig constructor found"
    for path, call in calls:
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        exclusion = keywords.get("exclude_modules")
        assert isinstance(exclusion, ast.Call), (
            f"{path}:{call.lineno} must pass the configured LoRA exclusions"
        )
        assert isinstance(exclusion.func, ast.Attribute)
        assert exclusion.func.attr == "get"
        assert exclusion.args and isinstance(exclusion.args[0], ast.Constant)
        assert exclusion.args[0].value == "exclude_modules"


def test_model_setup_rejects_accidental_vision_targets():
    source = Path("train.py").read_text(encoding="utf-8")

    assert 'if "vision_tower" in name' in source
    assert "LoRA unexpectedly targeted vision-tower modules" in source
