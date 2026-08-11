"""Guard the prompt/stop-token contract shared by training and generation.

The 2026-08-10 adapter translated correctly and never stopped because two
invariants were violated independently in each generation entry point:
``add_generation_prompt=True`` is not the prefix SFT trains the assistant turn
to continue, and the decoder's stop set omitted ``<end_of_turn>``. Both now live
in prompting.py, and these tests fail if an entry point starts rendering or
stopping on its own again.
"""

import ast
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prompting import (  # noqa: E402
    CHAT_TURN_END_TOKEN,
    as_token_id_set,
    render_training_prompt,
    resolve_stop_token_ids,
)

# Every module that turns a source segment into model input. prompting.py is
# excluded on purpose: it is the one place allowed to call the template.
GENERATION_ENTRY_POINTS = (
    Path("evaluate_translations.py"),
    Path("inference.py"),
    Path("translation_benchmark/generation.py"),
    # The serving API. It imports its own byte-identical copy of prompting.py
    # (see tests/test_api_vendored_modules.py) because api/ is deployed without
    # the repository, but it is held to the same contract: a server that renders
    # its own prompts would answer differently from the evaluated system without
    # any visible failure.
    Path("api/translator.py"),
)


def _chat_template_calls(path):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr == "apply_chat_template":
            yield node, {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg}


@pytest.mark.parametrize("path", GENERATION_ENTRY_POINTS, ids=str)
def test_entry_points_do_not_render_prompts_themselves(path):
    """No entry point may call the chat template directly.

    add_generation_prompt=True silently drops the assistant-turn indentation the
    SFT rendering emits, so a direct call reintroduces the off-distribution
    prompt without any visible failure.
    """
    offenders = [node.lineno for node, _ in _chat_template_calls(path)]
    assert not offenders, (
        f"{path} calls apply_chat_template directly at line(s) {offenders}. "
        "Use prompting.render_inference_prompts / render_training_prompts so the "
        "prompt matches what train.py conditions the model on."
    )


@pytest.mark.parametrize("path", GENERATION_ENTRY_POINTS, ids=str)
def test_entry_points_pass_an_explicit_stop_set(path):
    """generate() must receive eos_token_id, not inherit a partial stop set."""
    source = Path(path).read_text(encoding="utf-8")
    assert "eos_token_id" in source, (
        f"{path} does not pass an explicit eos_token_id to generate(). A model "
        "config's stop set can omit <end_of_turn>, which every SFT target ends with."
    )


class FakeTokenizer:
    """Minimal tokenizer stand-in: no weights, no network, no hub access."""

    unk_token_id = 3

    def __init__(self, eos_token_id=1, turn_end_id=106):
        self.eos_token_id = eos_token_id
        self._turn_end_id = turn_end_id

    def convert_tokens_to_ids(self, token):
        return self._turn_end_id if token == CHAT_TURN_END_TOKEN else self.unk_token_id


class FakeGenerationConfig:
    def __init__(self, eos_token_id):
        self.eos_token_id = eos_token_id


def test_as_token_id_set_normalizes_every_shape():
    assert as_token_id_set(None) == set()
    assert as_token_id_set(1) == {1}
    assert as_token_id_set([1, 106]) == {1, 106}


def test_resolve_stop_token_ids_adds_the_turn_ender():
    """The exact 2026-08-10 failure: a config-derived stop set of just <eos>."""
    stop_ids = resolve_stop_token_ids(FakeTokenizer(), FakeGenerationConfig(1))
    assert stop_ids == [1, 106]


def test_resolve_stop_token_ids_unions_every_source():
    stop_ids = resolve_stop_token_ids(
        FakeTokenizer(eos_token_id=1),
        FakeGenerationConfig([1, 106]),
        FakeGenerationConfig(42),
    )
    assert stop_ids == [1, 42, 106]


def test_resolve_stop_token_ids_tolerates_an_unknown_turn_ender():
    """A tokenizer without <end_of_turn> must not contribute its unk id."""
    tokenizer = FakeTokenizer(eos_token_id=1, turn_end_id=FakeTokenizer.unk_token_id)
    assert resolve_stop_token_ids(tokenizer, FakeGenerationConfig(1)) == [1]


def test_resolve_stop_token_ids_rejects_an_empty_stop_set():
    tokenizer = FakeTokenizer(eos_token_id=None, turn_end_id=FakeTokenizer.unk_token_id)
    with pytest.raises(ValueError, match="stop token"):
        resolve_stop_token_ids(tokenizer, FakeGenerationConfig(None))


class FakeProcessor:
    """Reproduces the TranslateGemma template's assistant-turn indentation."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        rendered = "".join(
            f"<start_of_turn>{message['role']}\n{message['content']}<end_of_turn>\n"
            for message in messages[:-1] if not add_generation_prompt
        )
        if add_generation_prompt:
            body = "".join(
                f"<start_of_turn>{message['role']}\n{message['content']}<end_of_turn>\n"
                for message in messages
            )
            return f"{body}<start_of_turn>model\n"
        last = messages[-1]
        # The eight spaces are the whole point: Jinja block indentation that
        # add_generation_prompt=True does not emit.
        return f"{rendered}<start_of_turn>{last['role']}\n\n        {last['content']}<end_of_turn>\n"


def test_render_training_prompt_keeps_the_assistant_turn_indentation():
    processor = FakeProcessor()
    user_message = {"role": "user", "content": "hello"}
    prompt = render_training_prompt(processor, user_message)
    assert prompt.endswith("<start_of_turn>assistant\n\n        ")
    generation_prompt = processor.apply_chat_template(
        [user_message], tokenize=False, add_generation_prompt=True
    )
    assert prompt != generation_prompt, (
        "the two renderings must stay distinguishable, or this test proves nothing"
    )


def test_render_training_prompt_requires_the_marker_to_survive():
    class SwallowingProcessor:
        def apply_chat_template(self, messages, **kwargs):
            return "<start_of_turn>model\n"

    with pytest.raises(ValueError, match="boundary marker"):
        render_training_prompt(SwallowingProcessor(), {"role": "user", "content": "hi"})
