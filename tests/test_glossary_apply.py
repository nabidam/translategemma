"""Rewriting the model's output to the agreed term, without breaking grammar."""

from glossary.apply import Report, apply_preferred
from glossary.matcher import Term
from glossary.matcher import Span

CANONICAL = "توجه چندپرسشی"
VARIANT = "توجه چندگانه"


def span(source="multi-query attention", target=CANONICAL, aliases=(), forbidden=()):
    return Span(
        start=0,
        end=len(source),
        term=Term(
            entry_id=1,
            source_term=source,
            target_term=target,
            target_mode="preferred",
            aliases=tuple(aliases),
            forbidden=tuple(forbidden),
            case_sensitive=False,
            whole_word=True,
            priority=0,
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
    # The spec's affix rule. Alias matching is substring-based on purpose, so a
    # plural suffix or a prefixed preposition survives and only the stem is
    # swapped. Matching whole words instead would either miss the inflected
    # form or overwrite the affix, and the second produces ungrammatical Farsi
    # -- the failure `preferred` mode exists to avoid.
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
