import ast
from pathlib import Path


def _tensor_chat_template_calls(path):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "apply_chat_template":
            continue
        keywords = {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg}
        return_tensors = keywords.get("return_tensors")
        if isinstance(return_tensors, ast.Constant) and return_tensors.value == "pt":
            yield node, keywords


def test_generation_templates_explicitly_request_tokenization():
    for path in (Path("evaluate_translations.py"), Path("inference.py")):
        calls = list(_tensor_chat_template_calls(path))
        assert calls, f"no tensor-producing chat-template call found in {path}"
        for call, keywords in calls:
            tokenize = keywords.get("tokenize")
            assert isinstance(tokenize, ast.Constant) and tokenize.value is True, (
                f"{path}:{call.lineno} must pass tokenize=True; return_tensors alone "
                "can still return a rendered string"
            )
