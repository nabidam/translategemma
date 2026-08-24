"""Sentinel protection and restoration for `exact` mode.

The property under test is that the model never sees the term and the agreed
rendering comes back character-for-character — and, just as important, that a
sentinel the model failed to carry through is reported rather than emitted.
"""

from glossary.matcher import Span, Term
from glossary.protect import SENTINEL_TEMPLATE, protect, restore, strip_sentinels


def term(source, target, mode="exact", entry_id=1):
    return Term(
        entry_id=entry_id,
        source_term=source,
        target_term=target,
        target_mode=mode,
        aliases=(),
        forbidden=(),
        case_sensitive=False,
        whole_word=True,
        priority=0,
    )


def span_for(text, source, target, mode="exact", entry_id=1):
    start = text.index(source)
    return Span(start=start, end=start + len(source), term=term(source, target, mode, entry_id))


def test_an_exact_term_is_replaced_by_a_sentinel():
    text = "We deployed Wordomatic on the cluster."
    protection = protect(text, [span_for(text, "Wordomatic", "Wordomatic")])
    assert "Wordomatic" not in protection.text
    assert SENTINEL_TEMPLATE.format(index=0) in protection.text


def test_a_preferred_term_is_left_alone():
    """The guarantee every non-exact deployment relies on: input untouched."""
    text = "The genome sequence."
    protection = protect(text, [span_for(text, "genome", "ژنوم", mode="preferred")])
    assert protection.text == text
    assert protection.is_empty


def test_each_occurrence_gets_its_own_sentinel():
    # A repeated sentinel could not distinguish a duplicated occurrence from a
    # dropped one, which is exactly what validation has to detect.
    text = "Wordomatic and Wordomatic again."
    spans = [
        Span(start=0, end=10, term=term("Wordomatic", "Wordomatic")),
        Span(start=15, end=25, term=term("Wordomatic", "Wordomatic")),
    ]
    protection = protect(text, spans)
    assert SENTINEL_TEMPLATE.format(index=0) in protection.text
    assert SENTINEL_TEMPLATE.format(index=1) in protection.text


def test_the_agreed_term_comes_back_verbatim():
    text = "We deployed Wordomatic today."
    protection = protect(text, [span_for(text, "Wordomatic", "Wordomatic")])
    model_output = protection.text.replace("We deployed", "ما").replace("today", "امروز")
    restored, failed = restore(model_output, protection)
    assert failed == []
    assert "Wordomatic" in restored
    assert SENTINEL_TEMPLATE.format(index=0) not in restored


def test_a_dropped_sentinel_is_reported_not_emitted():
    text = "We deployed Wordomatic today."
    protection = protect(text, [span_for(text, "Wordomatic", "Wordomatic")])
    restored, failed = restore("ما امروز مستقر کردیم.", protection)
    assert failed == ["Wordomatic"]
    # nothing was invented, and no sentinel leaked
    assert "Wordomatic" not in restored


def test_a_duplicated_sentinel_is_a_failure():
    """The decoder repeating a sentinel is as wrong as losing it."""
    text = "We deployed Wordomatic today."
    protection = protect(text, [span_for(text, "Wordomatic", "Wordomatic")])
    sentinel = SENTINEL_TEMPLATE.format(index=0)
    restored, failed = restore(f"{sentinel} و {sentinel}", protection)
    assert failed == ["Wordomatic"]


def test_one_lost_sentinel_does_not_discard_the_others():
    text = "Wordomatic and Framewidget."
    spans = [
        Span(start=0, end=10, term=term("Wordomatic", "Wordomatic", entry_id=1)),
        Span(start=15, end=26, term=term("Framewidget", "Framewidget", entry_id=2)),
    ]
    protection = protect(text, spans)
    # the model kept the second sentinel and lost the first
    output = f"چیزی {SENTINEL_TEMPLATE.format(index=1)} است."
    restored, failed = restore(output, protection)
    assert failed == ["Wordomatic"]
    assert "Framewidget" in restored


def test_strip_sentinels_never_lets_one_reach_a_caller():
    text = "We deployed Wordomatic today."
    protection = protect(text, [span_for(text, "Wordomatic", "Wordomatic")])
    leaked = f"ما {SENTINEL_TEMPLATE.format(index=0)} کردیم."
    assert "__TG_TERM" not in strip_sentinels(leaked, protection)


def test_protect_is_a_no_op_without_spans():
    assert protect("unchanged", []).text == "unchanged"
    assert restore("unchanged", protect("unchanged", [])) == ("unchanged", [])
