"""Parity and synchronization tests between canonical prompting and gateway rendering."""

import inspect
from unittest.mock import MagicMock

import pytest

import gateway.prompting as gw_prompting
import prompting as canonical_prompting
from gateway.main import CanonicalPromptRenderer


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
