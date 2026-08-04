import ast
from pathlib import Path


PROCESSOR_FILES = (
    Path("train.py"),
    Path("evaluate_translations.py"),
    Path("inference.py"),
    Path("scripts/analyze_token_lengths.py"),
)


def _auto_processor_calls(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if (
            node.func.attr == "from_pretrained"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "AutoProcessor"
        ):
            yield node


def test_translategemma_processors_disable_inapplicable_mistral_patch():
    for path in PROCESSOR_FILES:
        calls = list(_auto_processor_calls(path))
        assert calls, f"no AutoProcessor load found in {path}"
        for call in calls:
            keywords = {keyword.arg: keyword.value for keyword in call.keywords if keyword.arg}
            fix_regex = keywords.get("fix_mistral_regex")
            assert isinstance(fix_regex, ast.Constant) and fix_regex.value is False, (
                f"{path}:{call.lineno} must pass fix_mistral_regex=False"
            )


def test_checkpointed_training_disables_nested_text_cache():
    source = Path("train.py").read_text(encoding="utf-8")
    tree = ast.parse(source, filename="train.py")
    assignments = {
        ast.unparse(node.targets[0]): node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and len(node.targets) == 1
    }
    nested = assignments.get("model.config.get_text_config().use_cache")
    assert isinstance(nested, ast.Constant) and nested.value is False


def test_pretokenized_collator_suppresses_inapplicable_padding_advice():
    source = Path("train.py").read_text(encoding="utf-8")
    assert 'deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True' in source
