"""Structure-preserving, budget-aware chunking in api/translator.py.

Two regressions this module pins down:

* The 400: a document whose source -- or pysbd's "whole text is one segment"
  fallback for a language it has no model for -- exceeds the vLLM context
  window reaches /completions as a single prompt and dies (observed:
  269,395 input tokens against a 130,560 window). Every prompt must now
  obey two ceilings:

    context: rendered prompt + max_new_tokens + reserve <= max_context_tokens
    output:  source tokens <= max_new_tokens // 2 (no silent clipping at
             the stop)

* The flattening: a markdown document used to come back as ONE line because
  the pipeline turned it into a flat sentence list and rejoined with " ".
  Blocks (paragraphs, headings, lists, tables) and their original
  separators must survive: the rejoin is an exact reconstruction with only
  the unit texts replaced by their translations, and code fences / $$ math
  are verbatim -- never sent to the model.
"""

import asyncio
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "api") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "api"))

from config import Settings, System  # noqa: E402
from translator import (  # noqa: E402
    SentenceSplitter,
    TranslationEngine,
    _block_is_structured,
    _block_is_verbatim,
    _sentence_units,
    group_units,
    split_blocks,
    token_windows,
)


# ---------------------------------------------------------------------------
# Fakes: one word == one token, so token counts are word counts and the math
# is checkable by hand. The tokenizer is never asked to build padded batches.
# ---------------------------------------------------------------------------


class WordTokenizer:
    """One token = one word plus its surrounding whitespace, like a real
    BPE tokenizer (where spaces ride on adjacent tokens). That keeps hard
    slice seams joinable: a word is never split, and a space is never lost."""

    def __call__(self, texts, add_special_tokens=False):
        if isinstance(texts, str):
            texts = [texts]
        return {"input_ids": [re.findall(r"\s*\S+|\s+\Z", text) for text in texts]}

    def decode(self, token_ids):
        return "".join(token_ids)


class FakeProcessor:
    def __init__(self):
        self.tokenizer = WordTokenizer()


class StubSegmenter:
    def __init__(self, sentences):
        self._sentences = sentences

    def segment(self, text):
        return list(self._sentences)


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


def U(text, sep="", verbatim=False, packable=True):
    """Unit dict shorthand for group_units tests."""
    return {"text": text, "sep": sep, "verbatim": verbatim, "packable": packable}


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestSplitBlocks:
    def test_roundtrip_preserves_every_character(self):
        text = "para one\n\npara two\n\npara three"
        blocks = split_blocks(text)
        assert "".join(content + sep for content, sep in blocks) == text

    def test_multiple_blank_lines_are_one_separator(self):
        text = "a\n\n\n\nb"
        blocks = split_blocks(text)
        assert blocks == [("a", "\n\n\n\n"), ("b", "")]
        assert "".join(content + sep for content, sep in blocks) == text

    def test_single_newlines_stay_inside_blocks(self):
        text = "line one\nline two\n\nnext"
        blocks = split_blocks(text)
        assert blocks == [("line one\nline two", "\n\n"), ("next", "")]

    def test_leading_and_trailing_blanks(self):
        text = "\n\na\n\n"
        blocks = split_blocks(text)
        assert "".join(content + sep for content, sep in blocks) == text
        assert blocks[0][0] == ""  # leading blank is an (empty) block

    def test_no_blank_lines_single_block(self):
        assert split_blocks("a\nb") == [("a\nb", "")]

    def test_empty(self):
        assert split_blocks("") == [("", "")]


class TestBlockClassifiers:
    def test_fenced_code_is_verbatim(self):
        assert _block_is_verbatim("```python\nprint(1)\n```")
        assert _block_is_verbatim("   ~~~\nstuff\n~~~")

    def test_math_block_is_verbatim(self):
        assert _block_is_verbatim("$$\nE = mc^2\n$$")

    def test_prose_is_not_verbatim(self):
        assert not _block_is_verbatim("Some ordinary paragraph.")

    def test_structured_blocks(self):
        assert _block_is_structured(["- item one", "- item two"])
        assert _block_is_structured(["| a | b |", "| c | d |"])
        assert _block_is_structured(["# Heading", "## Sub"])
        assert _block_is_structured(["1. first", "2) second"])
        assert _block_is_structured(["> quoted line"])
        assert _block_is_structured(["- item", ""])  # empty lines ignored

    def test_prose_block_is_not_structured(self):
        assert not _block_is_structured(["Just a paragraph line."])
        assert not _block_is_structured([])
        assert not _block_is_structured(["", "   "])


class TestSentenceUnits:
    def test_gaps_taken_from_original(self):
        content = "First sentence here. Second follows.\nThird after newline."
        sentences = ["First sentence here.", "Second follows.", "Third after newline."]
        units = _sentence_units(content, sentences)
        assert units == [
            ("First sentence here.", " "),
            ("Second follows.", "\n"),
            ("Third after newline.", ""),
        ]

    def test_unlocatable_sentence_degrades_to_empty_gap(self):
        content = "Alpha beta."
        units = _sentence_units(content, ["normalised alpha beta."])
        assert units == [("normalised alpha beta.", "")]

    def test_repeated_sentence_found_in_order(self):
        content = "Same. Same. Same."
        units = _sentence_units(content, ["Same.", "Same.", "Same."])
        assert [gap for _sent, gap in units] == [" ", " ", ""]


class TestGroupUnits:
    def test_packable_run_packed_with_original_separators(self):
        groups = group_units(
            [U("s1", " "), U("s2", " "), U("s3", "")],
            [2, 2, 2],
            limit=5,
        )
        assert [g["prompt"] for g in groups] == ["s1 s2", "s3"]
        assert [g["sep"] for g in groups] == [" ", ""]
        assert not any(g["verbatim"] for g in groups)

    def test_prompt_excludes_last_trailing_separator(self):
        groups = group_units([U("s1", " "), U("s2", " ")], [2, 2], limit=10)
        assert groups[0]["prompt"] == "s1 s2"
        assert groups[0]["sep"] == " "  # the rejoin owns it

    def test_non_packable_unit_starts_its_own_group(self):
        # A paragraph break: the prompt must never span it.
        groups = group_units(
            [U("para1", "\n\n", packable=False), U("s1", " "), U("s2", "")],
            [3, 2, 2],
            limit=100,
        )
        assert [g["prompt"] for g in groups] == ["para1", "s1 s2"]
        assert [g["sep"] for g in groups] == ["\n\n", ""]

    def test_verbatim_unit_is_own_group_and_flagged(self):
        fence = "```\ncode\n```"
        groups = group_units(
            [U(fence, "\n\n", verbatim=True, packable=False), U("after", "")],
            [3, 1],
            limit=100,
        )
        assert groups[0] == {"prompt": fence, "sep": "\n\n", "verbatim": True}
        assert groups[1] == {"prompt": "after", "sep": "", "verbatim": False}

    def test_budget_respected_with_separators_in_cost(self):
        # costs already include separators; the prompt bound is conservative
        # (the last separator of a group is not in the prompt).
        units = [U(f"u{i}", " ") for i in range(6)]
        units[-1] = U("u5", "")
        groups = group_units(units, [3] * 6, limit=7)
        assert all(len(g["prompt"].split()) + g["sep"].split().__len__() <= 7 for g in groups)
        assert [g["prompt"] for g in groups] == ["u0 u1", "u2 u3", "u4 u5"]

    def test_empty_input(self):
        assert group_units([], [], 5) == []

    def test_rejoin_reconstructs_structure(self):
        doc_units = [
            U("# Title", "\n\n", packable=False),
            U("para one", "\n\n", packable=False),
            U("s1", " "),
            U("s2", " "),
            U("para two", "\n\n", packable=False),
            U("- item", "\n", packable=False),
            U("- item2", "", packable=False),
        ]
        groups = group_units(doc_units, [2] * 7, limit=100)
        out = "".join(
            (f"X[{g['prompt']}]" if not g["verbatim"] else g["prompt"]) + g["sep"]
            for g in groups
        )
        assert out == (
            "X[# Title]\n\nX[para one]\n\nX[s1 s2] X[para two]\n\nX[- item]\nX[- item2]"
        )

    def test_prompt_never_spans_paragraph_break(self):
        # A sentence whose original gap is a blank line is not packable, so
        # no prompt ever contains text from two paragraphs.
        units = [U("s1", "\n\n", packable=False), U("s2", " "), U("s3", "")]
        groups = group_units(units, [2, 2, 2], limit=100)
        assert all("\n\n" not in g["prompt"] for g in groups)
        assert [g["prompt"] for g in groups] == ["s1", "s2 s3"]


class TestTokenWindows:
    def test_windows_bounded_and_lossless(self):
        windows = token_windows(list(range(25)), 10)
        assert [len(w) for w in windows] == [10, 10, 5]
        assert [t for w in windows for t in w] == list(range(25))

    def test_empty(self):
        assert token_windows([], 10) == []


# ---------------------------------------------------------------------------
# Engine: budgets and the structure path
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


class TestStructureText:
    def test_fitting_paragraph_is_one_unit_with_separator(self):
        engine = make_engine()
        units = engine._structure_text("para one\n\npara two", "en", 10)
        assert units == [
            {"text": "para one", "sep": "\n\n", "verbatim": False, "packable": False},
            {"text": "para two", "sep": "", "verbatim": False, "packable": True},
        ]

    def test_code_fence_is_verbatim_and_intact(self):
        engine = make_engine()
        fence = "```python\nx = 1\n```"
        units = engine._structure_text(f"{fence}\n\nafter", "en", 10)
        assert units[0]["verbatim"] is True
        assert units[0]["text"] == fence
        assert units[0]["sep"] == "\n\n"
        assert units[1]["text"] == "after"

    def test_math_block_is_verbatim(self):
        engine = make_engine()
        units = engine._structure_text("$$\nE=mc^2\n$$\n\nprose", "en", 10)
        assert units[0]["verbatim"] is True
        assert units[0]["text"].startswith("$$")

    def test_small_list_translates_as_whole_block(self):
        engine = make_engine()
        block = "- item one\n- item two"
        units = engine._structure_text(block, "en", 10)
        assert len(units) == 1
        assert units[0]["text"] == block
        assert units[0]["sep"] == ""

    def test_over_budget_list_translates_line_by_line(self):
        engine = make_engine()
        block = "alpha beta gamma delta epsilon\nzeta eta theta iota kappa"
        # budget 4: the block (10 words) does not fit, lines (5 words each)
        # do not fit either -> hard-sliced at token boundaries, per line.
        units = engine._structure_text(block, "en", 4)
        texts = [u["text"] for u in units]
        assert all(len(t.split()) <= 4 for t in texts)
        # No word lost, and the newline between the two lines survives.
        assert [w for t in texts for w in t.split()] == block.split()
        joined = "".join(u["text"] + u["sep"] for u in units)
        assert "\n" in joined

    def test_over_budget_prose_uses_sentences_with_original_gaps(self):
        engine = make_engine(
            segmenters={"en": StubSegmenter(["one two three", "four five six"])}
        )
        content = "one two three four five six"
        units = engine._structure_text(content, "en", 3)
        assert [u["text"] for u in units] == ["one two three", "four five six"]
        # The gap between the sentences is the original space.
        assert units[0]["sep"] == " "
        assert units[1]["sep"] == ""

    def test_uncovered_language_falls_back_to_lines(self):
        engine = make_engine()  # no segmenter for any language
        text = "alpha beta gamma delta\n\neta theta iota kappa"
        units = engine._structure_text(text, "xx", 3)
        joined = "".join(u["text"] + u["sep"] for u in units)
        assert joined.split() == text.split()
        assert "\n\n" in joined  # the paragraph break survives

    def test_giant_single_line_hard_sliced_without_loss(self):
        giant = " ".join(f"w{i}" for i in range(25))
        engine = make_engine(segmenters={"en": StubSegmenter([giant])})
        units = engine._structure_text(giant, "en", 10)
        assert all(len(u["text"].split()) <= 10 for u in units)
        assert [w for u in units for w in u["text"].split()] == giant.split()

    def test_blank_text(self):
        engine = make_engine()
        assert engine._structure_text("   \n  ", "en", 10) == []

    def test_missing_pysbd_uses_lines(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pysbd", None)
        engine = make_engine()
        engine.splitter = SentenceSplitter()
        units = engine._structure_text("alpha beta\ngamma delta", "en", 3)
        texts = [u["text"] for u in units]
        assert [w for t in texts for w in t.split()] == "alpha beta gamma delta".split()


# ---------------------------------------------------------------------------
# End-to-end through translate(): the 400 regression and the flattening fix.
# ---------------------------------------------------------------------------


class TestTranslateRegression:
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
        assert len(out) == 1 and out[0] == "".join(f"T{i}" for i in range(len(segments)))

    def test_long_document_packs_and_fits(self, monkeypatch):
        # The 6337-word shape of the reported failure: one unbroken line.
        # No segmenter for the language -> line/hard-slice path.
        engine = make_engine(max_context_tokens=130560)
        sentences = [" ".join(f"word{s}_{i}" for i in range(10)) + "." for s in range(600)]
        doc = " ".join(sentences)
        segments, out = self._run_translate(engine, doc, "en", monkeypatch)
        assert segments
        assert all(len(s.split()) <= 256 for s in segments)
        # Packing: far fewer prompts than sentences.
        assert len(segments) < 600
        assert " ".join(segments).split() == doc.split()
        assert len(out) == 1

    def test_markdown_structure_survives_end_to_end(self, monkeypatch):
        engine = make_engine(max_context_tokens=130560)
        fence = "```python\nx = 1\n```"
        doc = f"# Title\n\nFirst paragraph of the paper.\n\n- item one\n- item two\n\n{fence}\n\nFinal paragraph."
        segments, out = self._run_translate(engine, doc, "en", monkeypatch)
        # Code fences never reach the model.
        assert all("x = 1" not in s for s in segments)
        # No prompt spans a paragraph break.
        assert all("\n\n" not in s for s in segments)
        # The translation keeps the structure: blank lines, headings, list
        # lines, and the verbatim fence all come back in place.
        assert out[0].count("\n\n") == doc.count("\n\n")
        assert fence in out[0]
        # Five translated groups (title, para1, list, para2) -> five T's.
        translated = [part for part in out[0].split("\n\n") if part.startswith("T")]
        assert len(translated) == 4

    def test_split_sentences_off_sends_whole_text(self, monkeypatch):
        engine = make_engine(max_context_tokens=130560)
        engine._client = object()
        captured = {}

        async def fake_generate(segments, system, s, t, max_new_tokens):
            captured["segments"] = list(segments)
            return [f"T{i}" for i in range(len(segments))]

        monkeypatch.setattr(engine, "_generate", fake_generate)
        loop = asyncio.new_event_loop()
        try:
            out = loop.run_until_complete(
                engine.translate(["short text"], System.ADAPTER, "en", "fa", 512, False)
            )
        finally:
            loop.close()
        assert captured["segments"] == ["short text"]
        assert out == ["T0"]
