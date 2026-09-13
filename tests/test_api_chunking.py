"""Budget-aware chunking in api/translator.py: the 400 regression.

A document whose source -- or pysbd's "whole text is one segment" fallback
for a language it has no model for -- exceeds the vLLM context window used to
reach /completions as a single prompt and die with a 400 (observed: 269,395
input tokens against a 130,560 window). The gateway now bounds every prompt
by two ceilings:

* context: rendered prompt + max_new_tokens + reserve <= max_context_tokens;
* output: source tokens <= max_new_tokens // 2, so a chunk's translation
  cannot be silently clipped at the stop.

The hierarchy mirrors quick_pipeline's proven chunker: sentence (pysbd) ->
paragraph/line (uncovered languages) -> hard token-boundary slices, with
greedy re-packing of whole sentences into budget-sized chunks.
"""

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "api") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "api"))

from config import Settings, System  # noqa: E402
from translator import (  # noqa: E402
    SentenceSplitter,
    TranslationEngine,
    fallback_units,
    pack_units,
    token_windows,
)


# ---------------------------------------------------------------------------
# Fakes: one word == one token, so token counts are word counts and the math
# is checkable by hand. The tokenizer is never asked to build padded batches.
# ---------------------------------------------------------------------------


class WordTokenizer:
    def __call__(self, texts, add_special_tokens=False):
        if isinstance(texts, str):
            texts = [texts]
        return {"input_ids": [text.split() for text in texts]}

    def decode(self, token_ids):
        return " ".join(token_ids)


class FakeProcessor:
    def __init__(self):
        self.tokenizer = WordTokenizer()


class StubSegmenter:
    def __init__(self, sentences):
        self._sentences = sentences

    def segment(self, text):
        return [sentence + " " for sentence in self._sentences]


class StubSplitter:
    def __init__(self, segmenters=None):
        self._segmenters = segmenters or {}

    def segmenter(self, language):
        return self._segmenters.get(language)


def make_engine(max_context_tokens=130560, segmenters=None):
    engine = TranslationEngine(Settings())
    engine.processor = FakeProcessor()
    engine.max_context_tokens = max_context_tokens
    engine.splitter = StubSplitter(segmenters)
    return engine


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestPackUnits:
    def test_packs_in_order_within_limit(self):
        units = ["a", "bb", "ccc", "dddd"]
        groups = pack_units(units, [1, 2, 3, 4], 5)
        assert groups == [[0, 1], [2], [3]]

    def test_never_exceeds_limit_and_loses_nothing(self):
        units = ["x"] * 10
        costs = [3] * 10
        groups = pack_units(units, costs, 7)
        assert all(sum(costs[i] for i in group) <= 7 for group in groups)
        assert [i for group in groups for i in group] == list(range(10))

    def test_oversized_unit_isolated(self):
        groups = pack_units(["aa", "bigbigbig", "bb"], [2, 9, 2], 5)
        assert groups == [[0], [1], [2]]

    def test_empty_units_dropped(self):
        assert pack_units(["", "a", "", "b"], [0, 1, 0, 1], 10) == [[1, 3]]

    def test_empty_input(self):
        assert pack_units([], [], 5) == []


class TestFallbackUnits:
    def test_paragraphs_and_lines(self):
        assert fallback_units("one two\nthree four\n\nfive six") == [
            "one two",
            "three four",
            "five six",
        ]

    def test_blank_text(self):
        assert fallback_units("   \n\n  ") == []


class TestTokenWindows:
    def test_windows_bounded_and_lossless(self):
        windows = token_windows(list(range(25)), 10)
        assert [len(w) for w in windows] == [10, 10, 5]
        assert [t for w in windows for t in w] == list(range(25))

    def test_empty(self):
        assert token_windows([], 10) == []


# ---------------------------------------------------------------------------
# Engine: budgets and the full chunk path
# ---------------------------------------------------------------------------


class TestSourceBudget:
    def test_output_budget_binds_for_large_context(self):
        assert make_engine(max_context_tokens=130560)._source_budget(512) == 256

    def test_context_budget_binds_for_small_context(self):
        engine = make_engine(max_context_tokens=1024)
        # 1024 - 512 - 256 = 256 == 512 // 2
        assert engine._source_budget(512) == 256
        # 1024 - 256 - 256 = 512 > 256 // 2 = 128
        assert engine._source_budget(256) == 128

    def test_floor(self):
        assert make_engine(max_context_tokens=128)._source_budget(512) == 16


class TestChunkText:
    SENTENCES = ["one two three", "four five six seven", "eight nine ten eleven"]

    def test_packs_sentences_within_budget(self):
        engine = make_engine(segmenters={"en": StubSegmenter(self.SENTENCES)})
        text = " ".join(self.SENTENCES)
        chunks = engine._chunk_text(text, "en", 10)
        assert chunks == [
            "one two three four five six seven",
            "eight nine ten eleven",
        ]
        assert " ".join(chunks).split() == text.split()

    def test_uncovered_language_falls_back_to_lines(self):
        engine = make_engine()  # no segmenter for any language
        text = "alpha beta gamma\ndelta epsilon zeta\n\neta theta iota"
        chunks = engine._chunk_text(text, "xx", 4)
        assert chunks == ["alpha beta gamma", "delta epsilon zeta", "eta theta iota"]

    def test_giant_unit_hard_sliced_without_loss(self):
        giant = " ".join(f"w{i}" for i in range(25))
        engine = make_engine(segmenters={"en": StubSegmenter([giant])})
        chunks = engine._chunk_text(giant, "en", 10)
        assert [len(c.split()) for c in chunks] == [10, 10, 5]
        assert " ".join(chunks).split() == giant.split()

    def test_long_document_no_chunk_exceeds_budget(self):
        # The shape of the reported failure: one 6337-word document that used
        # to become a single over-context prompt.
        words = [f"word{i}" for i in range(6337)]
        text = " ".join(words)
        engine = make_engine(segmenters={"en": StubSegmenter([text])})
        chunks = engine._chunk_text(text, "en", 256)
        assert all(len(c.split()) <= 256 for c in chunks)
        assert len(chunks) == 25
        assert " ".join(chunks).split() == words

    def test_blank_text(self):
        engine = make_engine()
        assert engine._chunk_text("   \n  ", "en", 10) == []

    def test_missing_pysbd_uses_fallback(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pysbd", None)
        engine = make_engine()
        engine.splitter = SentenceSplitter()
        # Two lines, budget 3: each line alone fits, packed together it does
        # not, so they stay separate.
        chunks = engine._chunk_text("alpha beta\ngamma delta", "en", 3)
        assert chunks == ["alpha beta", "gamma delta"]


class TestTranslateRegression:
    """The reported failure, end to end: a long document in a language pysbd
    0.3.4 has no model for used to reach /completions as one over-context
    prompt (269,395 tokens vs a 130,560 window -> 400)."""

    @staticmethod
    def _run_translate(engine, doc, source_lang, monkeypatch):
        engine._client = object()  # is_loaded
        captured = {}

        async def fake_generate(segments, system, s, t, max_new_tokens):
            captured["segments"] = list(segments)
            return [f"T{i}" for i in range(len(segments))]

        monkeypatch.setattr(engine, "_generate", fake_generate)
        loop = asyncio.new_event_loop()
        try:
            out = loop.run_until_complete(
                engine.translate([doc], System.ADAPTER, source_lang, "fa", 512, True)
            )
        finally:
            loop.close()
        return captured["segments"], out

    def test_uncovered_language_document_stays_within_budget(self, monkeypatch):
        # 'tr' has no pysbd model in 0.3.4; one unbroken line, 3000 words.
        engine = make_engine(max_context_tokens=130560)
        doc = " ".join(f"kelime{i}" for i in range(3000))
        segments, out = self._run_translate(engine, doc, "tr", monkeypatch)
        assert segments
        assert all(len(s.split()) <= 256 for s in segments)
        assert " ".join(segments).split() == doc.split()
        # One output per input text, whatever the chunk count.
        assert len(out) == 1 and out[0] == " ".join(f"T{i}" for i in range(len(segments)))

    def test_covered_language_long_document_packs_and_fits(self, monkeypatch):
        # Real pysbd (en): 600 sentences of ~10 words, 6337-word shape.
        engine = make_engine(max_context_tokens=130560)
        sentences = [
            " ".join(f"word{s}_{i}" for i in range(10)) + "."
            for s in range(600)
        ]
        doc = " ".join(sentences)
        segments, out = self._run_translate(engine, doc, "en", monkeypatch)
        assert segments
        assert all(len(s.split()) <= 256 for s in segments)
        # Packing: far fewer prompts than sentences.
        assert len(segments) < 600
        assert " ".join(segments).split() == doc.split()
        assert len(out) == 1
