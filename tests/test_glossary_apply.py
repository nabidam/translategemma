"""Rewriting the model's output to the agreed term, without breaking grammar."""

from glossary.apply import Report, apply_preferred
from glossary.matcher import Term
from glossary.matcher import Span

CANONICAL = "توجه چندپرسشی"
VARIANT = "توجه چندگانه"


def span(
    source="multi-query attention",
    target=CANONICAL,
    aliases=(),
    forbidden=(),
    entry_id=1,
    priority=0,
    # True mirrors the production default (api/glossary/router.py,
    # api/glossary/models.py). apply_preferred does not read this field at
    # all -- target-side matching is always a bare substring scan, which is
    # what makes affixes like a plural "ها" or a prefixed "به" survive around
    # the replaced stem. `whole_word` only constrains source-side matching in
    # matcher.py.
    whole_word=True,
):
    return Span(
        start=0,
        end=len(source),
        term=Term(
            entry_id=entry_id,
            source_term=source,
            target_term=target,
            target_mode="preferred",
            aliases=tuple(aliases),
            forbidden=tuple(forbidden),
            case_sensitive=False,
            whole_word=whole_word,
            priority=priority,
        ),
    )


def test_canonical_already_present_counts_as_applied_and_changes_nothing():
    text = f"مدل به {CANONICAL} متکی است."
    result, report = apply_preferred(text, [span()])
    assert result == text
    assert len(report.applied) == 1
    assert report.applied[0].count == 1
    assert report.misses == ()


def test_alias_is_rewritten_to_the_canonical_term():
    text = f"مدل به {VARIANT} متکی است."
    result, report = apply_preferred(text, [span(aliases=[VARIANT])])
    assert CANONICAL in result
    assert VARIANT not in result
    assert report.applied[0].count == 1


def test_missing_term_is_reported_and_the_translation_is_untouched():
    text = "چیزی دیگر است."
    result, report = apply_preferred(text, [span(aliases=[VARIANT])])
    assert result == text
    assert report.applied == ()
    assert [miss.reason for miss in report.misses] == ["target_not_found"]


def test_alias_matching_ignores_zwnj_and_arabic_letters():
    # The model wrote the alias with a ZWNJ and an Arabic yeh; the entry did not.
    written = VARIANT.replace("ی", "ي") + "‌"
    text = f"مدل {written} است."
    result, report = apply_preferred(text, [span(aliases=[VARIANT])])
    assert CANONICAL in result
    assert report.applied[0].count == 1


def test_persian_affixes_around_an_alias_are_preserved():
    # The spec's affix rule, exercised at the production `whole_word=True`
    # default (see span()'s default above). Alias matching is
    # substring-based on purpose, so a plural suffix or a prefixed
    # preposition survives and only the stem is swapped. Matching whole
    # words instead would either miss the inflected form or overwrite the
    # affix, and the second produces ungrammatical Farsi -- the failure
    # `preferred` mode exists to avoid.
    text = f"به{VARIANT}ها نگاه کن."
    result, report = apply_preferred(text, [span(aliases=[VARIANT])])
    assert result == f"به{CANONICAL}ها نگاه کن."
    assert report.applied[0].count == 1


def test_every_occurrence_of_an_alias_is_rewritten():
    text = f"{VARIANT} و {VARIANT}"
    result, report = apply_preferred(text, [span(aliases=[VARIANT])])
    assert result.count(CANONICAL) == 2
    assert report.applied[0].count == 2


def test_forbidden_rendering_is_reported_even_when_the_term_is_missing():
    text = f"مدل {VARIANT} است."
    _, report = apply_preferred(text, [span(forbidden=[VARIANT])])
    assert len(report.violations) == 1
    assert report.violations[0].forbidden == VARIANT


def test_no_spans_produces_an_empty_report_and_the_original_text():
    assert apply_preferred("unchanged", []) == ("unchanged", Report.empty())


def test_overlapping_aliases_of_the_same_term_do_not_corrupt_the_result():
    # An administrator adds both a long and a short variant of the same term.
    # SHORT is a suffix of LONG, so both match the same occurrence in the
    # output and their ranges overlap. Applying both edits (rather than only
    # the longest, non-overlapping one) splices the replacement into itself.
    short_variant = "چندگانه"
    assert short_variant in VARIANT  # the overlap this test relies on
    text = f"مدل {VARIANT} است."
    result, report = apply_preferred(text, [span(aliases=[VARIANT, short_variant])])
    assert result == f"مدل {CANONICAL} است."
    assert report.applied[0].count == 1


def test_overlapping_aliases_of_different_terms_do_not_corrupt_the_result():
    # Two distinct glossary entries whose aliases happen to overlap in the
    # model's output -- neither contains the other, they share a middle
    # section, the same shape as "AB"/"BC" overlapping in "ABC". The longer
    # match wins the overlap and is applied; the shorter one's hit is
    # claimed by the winner, so that term has nothing applied and is
    # reported as a miss instead of a phantom zero-effect success.
    alias_a = VARIANT[0:8]  # "توجه چند"
    alias_b = VARIANT[5:12]  # "چندگانه" -- overlaps alias_a at "چند"
    assert alias_a != alias_b and alias_a not in alias_b and alias_b not in alias_a
    text = f"مدل {VARIANT} است."
    result, report = apply_preferred(
        text,
        [
            span(source="term1", target="X", aliases=[alias_a], entry_id=1),
            span(source="term2", target="Y", aliases=[alias_b], entry_id=2),
        ],
    )
    assert "X" in result and "Y" not in result
    assert VARIANT not in result
    applied_sources = {a.source_term: a.count for a in report.applied}
    miss_sources = {m.source_term for m in report.misses}
    assert applied_sources == {"term1": 1}
    assert miss_sources == {"term2"}
    # term2's alias text genuinely was in the output -- it just lost the
    # overlap race to term1. That is a different, more specific fact than
    # "never appeared at all", and the reason string an administrator reads
    # in the misses report has to say which one actually happened.
    assert [m.reason for m in report.misses] == ["claimed_by_overlap"]


def test_alias_overlapping_a_different_terms_canonical_text_does_not_overwrite_it():
    # term_a's canonical form is already present in the output. term_b's
    # alias partially overlaps that same range -- neither contains the
    # other, same "AB"/"BC" shape as the alias-vs-alias case above, except
    # one side here is term_a's CANONICAL text rather than an alias. A
    # canonical hit produces no edit of its own (the text is already
    # correct), so unless it also claims its range in the overlap pool,
    # term_b's alias edit is free to splice over term_a's already-correct
    # text -- corrupting the output while term_a is still reported as
    # successfully applied, computed before any edits happened.
    #
    # term_a is given higher priority so the outcome is deterministic
    # (both ranges are 8 folded characters, so length alone would not
    # separate them); priority is the documented tie-break both
    # find_spans and apply_preferred use for exactly this situation.
    canonical_tail = CANONICAL[5:13]  # "چندپرسشی", term_a's canonical text
    overlapping_alias = CANONICAL[0:8]  # "توجه چند" -- overlaps at "چند"
    assert overlapping_alias not in canonical_tail and canonical_tail not in overlapping_alias
    text = f"مدل {CANONICAL} است."
    result, report = apply_preferred(
        text,
        [
            span(source="term_a", target=canonical_tail, aliases=[], entry_id=1, priority=10),
            span(source="term_b", target="Z", aliases=[overlapping_alias], entry_id=2, priority=0),
        ],
    )
    assert result == text
    assert CANONICAL in result
    assert "Z" not in result
    applied_sources = {a.source_term: a.count for a in report.applied}
    miss_sources = {m.source_term for m in report.misses}
    assert applied_sources == {"term_a": 1}
    assert miss_sources == {"term_b"}
    assert [m.reason for m in report.misses] == ["claimed_by_overlap"]


def test_affixed_alias_is_rewritten_at_the_production_whole_word_default():
    # Regression test for the finding-5 revert: an entry built exactly the
    # way the admin API builds one (EntryIn.whole_word defaults to True,
    # api/glossary/router.py:60; Boolean column default=True,
    # api/glossary/models.py:77) must still have its affixed alias rewritten.
    # Before this revert, target-side matching enforced `whole_word`, so this
    # exact case -- the affix rule from
    # test_persian_affixes_around_an_alias_are_preserved, but at the real
    # default instead of a test-only False -- silently failed: applied=False,
    # text unchanged. span()'s default is whole_word=True, so no override is
    # needed here to reproduce the production configuration.
    text = f"به{VARIANT}ها نگاه کن."
    result, report = apply_preferred(text, [span(aliases=[VARIANT])])
    assert result == f"به{CANONICAL}ها نگاه کن."
    assert report.applied[0].count == 1
