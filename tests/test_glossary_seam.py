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


def exact_index(source="Wordomatic", target="Wordomatic"):
    return build_index(
        [
            Term(
                entry_id=1, source_term=source, target_term=target,
                target_mode="exact", aliases=(), forbidden=(),
                case_sensitive=False, whole_word=True, priority=0,
            )
        ],
        version=9,
    )


async def test_exact_mode_hides_the_term_from_the_model():
    """The whole point of exact mode, and the one place the glossary alters input."""
    engine = RecordingEngine(["ما __TG_TERM_000__ را مستقر کردیم."])
    await engine.translate(
        ["We deployed Wordomatic."], SYSTEM, "en", "fa", 128, False,
        glossary_index=exact_index(),
    )
    sent = engine.sent[0][0]
    assert "Wordomatic" not in sent
    assert "__TG_TERM_000__" in sent


async def test_exact_mode_restores_the_agreed_term_verbatim():
    engine = RecordingEngine(["ما __TG_TERM_000__ را مستقر کردیم."])
    result = await engine.translate(
        ["We deployed Wordomatic."], SYSTEM, "en", "fa", 128, False,
        glossary_index=exact_index(),
    )
    assert "Wordomatic" in result.translations[0]
    assert "__TG_TERM_000__" not in result.translations[0]
    applied = result.reports[0].applied
    assert [(a.source_term, a.mode) for a in applied] == [("Wordomatic", "exact")]


class LosingEngine(RecordingEngine):
    """Drops the sentinel on the protected call, succeeds on the retry."""

    def __init__(self):
        super().__init__(["دستگاه مستقر شد."])
        self.calls = 0

    async def _generate(self, segments, system, source_lang, target_lang, max_new_tokens):
        self.calls += 1
        self.sent.append(list(segments))
        if self.calls == 1:
            return ["ما آن را مستقر کردیم."]  # sentinel gone
        return ["ترجمه بدون محافظت."]


async def test_a_lost_sentinel_falls_back_and_is_reported():
    """A mishandled sentinel must not be guessed at, emitted, or silently ignored."""
    engine = LosingEngine()
    result = await engine.translate(
        ["We deployed Wordomatic."], SYSTEM, "en", "fa", 128, False,
        glossary_index=exact_index(),
    )
    # the retry re-sent the ORIGINAL, unprotected text
    assert engine.calls == 2
    assert engine.sent[1] == ["We deployed Wordomatic."]
    assert result.translations[0] == "ترجمه بدون محافظت."
    assert "__TG_TERM" not in result.translations[0]
    misses = result.reports[0].misses
    assert [(m.source_term, m.reason) for m in misses] == [("Wordomatic", "sentinel_lost")]


async def test_a_preferred_only_index_still_sends_the_text_untouched():
    """The guarantee exact mode is carefully scoped not to break."""
    index = build_index(
        [
            Term(
                entry_id=1, source_term="genome", target_term="ژنوم",
                target_mode="preferred", aliases=("گنوم",), forbidden=(),
                case_sensitive=False, whole_word=True, priority=0,
            )
        ],
        version=9,
    )
    engine = RecordingEngine(["گنوم است."])
    await engine.translate(
        ["The genome."], SYSTEM, "en", "fa", 128, False, glossary_index=index
    )
    assert engine.sent == [["The genome."]]
