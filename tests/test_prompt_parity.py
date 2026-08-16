"""Parity and synchronization tests between canonical prompting and gateway rendering."""

import hashlib
import inspect
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import gateway.prompting as gw_prompting
import prompting as canonical_prompting
from gateway.main import CanonicalPromptRenderer

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_gateway_prompting_is_byte_identical_to_root():
    """Signatures can match while implementations diverge, so compare the bytes themselves.

    gateway/prompting.py is a vendored copy maintained by scripts/sync_api_vendored.py.
    Any drift in the rendering or stop-token logic is silent at run time — it produces
    fluent output from the wrong prefix — so it must fail here instead.
    """
    source = (PROJECT_ROOT / "prompting.py").read_bytes()
    vendored = (PROJECT_ROOT / "gateway" / "prompting.py").read_bytes()
    assert hashlib.sha256(vendored).hexdigest() == hashlib.sha256(source).hexdigest(), (
        "gateway/prompting.py differs from prompting.py. Re-sync with: "
        "uv run python scripts/sync_api_vendored.py"
    )


def test_prompting_module_synchronization():
    """Ensure gateway/prompting.py and root prompting.py maintain exact constant and signature parity."""
    assert gw_prompting.TARGET_BOUNDARY_MARKER == canonical_prompting.TARGET_BOUNDARY_MARKER
    assert gw_prompting.CHAT_TURN_END_TOKEN == canonical_prompting.CHAT_TURN_END_TOKEN

    gw_funcs = [f for f in dir(gw_prompting) if inspect.isfunction(getattr(gw_prompting, f))]
    canonical_funcs = [f for f in dir(canonical_prompting) if inspect.isfunction(getattr(canonical_prompting, f))]

    for func_name in ["as_token_id_set", "resolve_stop_token_ids", "render_training_prompt", "render_training_prompts", "render_inference_prompts"]:
        assert func_name in gw_funcs, f"Function {func_name} missing from gateway.prompting"
        assert func_name in canonical_funcs, f"Function {func_name} missing from canonical prompting"

        gw_sig = inspect.signature(getattr(gw_prompting, func_name))
        can_sig = inspect.signature(getattr(canonical_prompting, func_name))
        assert str(gw_sig) == str(can_sig), f"Signature mismatch for {func_name}: {gw_sig} vs {can_sig}"

        gw_src = inspect.getsource(getattr(gw_prompting, func_name))
        can_src = inspect.getsource(getattr(canonical_prompting, func_name))
        assert gw_src == can_src, f"Implementation drift in {func_name} between gateway and root prompting."


def test_canonical_prompt_renderer_exact_parity_with_processor():
    """Verify CanonicalPromptRenderer produces byte-for-byte identical output to render_training_prompt."""
    user_msg = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "source_lang_code": "en",
                "target_lang_code": "fa",
                "text": "Artificial neural networks process sequential data efficiently.",
            }
        ],
    }

    mock_processor = MagicMock()
    # Mock realistic chat template behavior
    mock_processor.apply_chat_template.side_effect = lambda msgs, tokenize=False, add_generation_prompt=False: (
        f"<start_of_turn>user\n<<<source>>>{msgs[0]['content'][0]['source_lang_code']}"
        f"<<<target>>>{msgs[0]['content'][0]['target_lang_code']}"
        f"<<<text>>>{msgs[0]['content'][0]['text']}<end_of_turn>\n"
        f"<start_of_turn>model\n\n        {msgs[1]['content']}<end_of_turn>\n"
    )

    # 1. Canonical root rendering
    canonical_output = canonical_prompting.render_training_prompt(mock_processor, user_msg)

    # 2. Gateway renderer output
    renderer = CanonicalPromptRenderer(processor_or_tokenizer=mock_processor, allow_fallback=False)
    gateway_output = renderer.render("en", "fa", "Artificial neural networks process sequential data efficiently.")

    assert gateway_output == canonical_output
    assert gateway_output.endswith("\n\n        ")
    assert canonical_prompting.TARGET_BOUNDARY_MARKER not in gateway_output
