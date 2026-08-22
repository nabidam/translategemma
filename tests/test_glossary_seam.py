"""The single application seam, and the guarantee that disabling it is inert."""

import pytest

from glossary.matcher import Term, build_index
from translator import TranslationEngine, TranslationResult


SYSTEM = "adapter"


class FakeSettings:
    """Only what TranslationEngine touches; no vLLM, no tokenizer."""

    max_concurrent_requests = 4
    batch_size = 8
    split_sentences = False
    served_system = SYSTEM


class RecordingEngine(TranslationEngine):
    """A TranslationEngine whose upstream is a recorded list of canned outputs."""

    def __init__(self, outputs):
        super().__init__(FakeSettings())
        self._outputs = outputs
        self.sent: list[list[str]] = []

    @property
    def is_loaded(self):
        return True

    async def _generate(self, segments, system, source_lang, target_lang, max_new_tokens):
        self.sent.append(list(segments))
        return list(self._outputs)


async def test_without_a_glossary_the_result_carries_no_reports():
    engine = RecordingEngine(["ترجمه"])
    result = await engine.translate(
        ["The genome."], SYSTEM, "en", "fa", 128, False, glossary_index=None
    )
    assert isinstance(result, TranslationResult)
    assert result.translations == ["ترجمه"]
    assert result.reports == [None]


async def test_without_a_glossary_raw_translations_equal_translations():
    # The debugging affordance (model output vs. glossary-repaired output) has
    # to hold up even when the glossary is off, where the two are equal by
    # definition -- a caller must never see raw_translation silently absent
    # for a reason other than "the glossary is disabled".
    engine = RecordingEngine(["ترجمه"])
    result = await engine.translate(
        ["The genome."], SYSTEM, "en", "fa", 128, False, glossary_index=None
    )
    assert result.raw_translations == result.translations == ["ترجمه"]


async def test_the_text_sent_upstream_is_unchanged_by_a_glossary():
    # preferred mode repairs output; it must never alter the source text, which
    # is what keeps it incapable of regressing translation quality.
    index = build_index(
        [
            Term(
                entry_id=1, source_term="genome", target_term="ژنوم",
                target_mode="preferred", aliases=("گنوم",), forbidden=(),
                case_sensitive=False, whole_word=True, priority=0,
            )
        ],
        version=3,
    )
    engine = RecordingEngine(["گنوم است."])
    await engine.translate(
        ["The genome."], SYSTEM, "en", "fa", 128, False, glossary_index=index
    )
    assert engine.sent == [["The genome."]]


async def test_an_alias_in_the_output_is_rewritten_and_reported():
    index = build_index(
        [
            Term(
                entry_id=1, source_term="genome", target_term="ژنوم",
                target_mode="preferred", aliases=("گنوم",), forbidden=(),
                case_sensitive=False, whole_word=True, priority=0,
            )
        ],
        version=3,
    )
    engine = RecordingEngine(["گنوم است."])
    result = await engine.translate(
        ["The genome."], SYSTEM, "en", "fa", 128, False, glossary_index=index
    )
    assert "ژنوم" in result.translations[0]
    assert result.reports[0].applied[0].count == 1


async def test_raw_translations_preserve_the_pre_glossary_model_output():
    # This is the field an administrator reads to tell "the model got it
    # wrong" from "the termbase rewrote it wrong". If raw_translations ever
    # collapsed to the post-rewrite text, that diagnosis would be impossible.
    index = build_index(
        [
            Term(
                entry_id=1, source_term="genome", target_term="ژنوم",
                target_mode="preferred", aliases=("گنوم",), forbidden=(),
                case_sensitive=False, whole_word=True, priority=0,
            )
        ],
        version=3,
    )
    engine = RecordingEngine(["گنوم است."])
    result = await engine.translate(
        ["The genome."], SYSTEM, "en", "fa", 128, False, glossary_index=index
    )
    assert result.raw_translations == ["گنوم است."]
    assert result.translations != result.raw_translations
    assert "ژنوم" in result.translations[0]
