"""Matching rules: longest wins, boundaries hold, priority breaks ties."""

from glossary.matcher import Term, build_index, find_spans


def term(source, target, **kwargs):
    return Term(
        entry_id=kwargs.pop("entry_id", 1),
        source_term=source,
        target_term=target,
        target_mode=kwargs.pop("target_mode", "preferred"),
        aliases=tuple(kwargs.pop("aliases", ())),
        forbidden=tuple(kwargs.pop("forbidden", ())),
        case_sensitive=kwargs.pop("case_sensitive", False),
        whole_word=kwargs.pop("whole_word", True),
        priority=kwargs.pop("priority", 0),
    )


def test_finds_a_simple_term():
    index = build_index([term("genome", "ژنوم")], version=1)
    spans = find_spans(index, "The genome sequence.")
    assert len(spans) == 1
    assert spans[0].start == 4
    assert spans[0].end == 10


def test_longest_match_wins():
    index = build_index(
        [term("attention", "A", entry_id=1), term("multi-query attention", "B", entry_id=2)],
        version=1,
    )
    spans = find_spans(index, "It uses multi-query attention here.")
    assert len(spans) == 1
    assert spans[0].term.entry_id == 2


def test_word_boundary_prevents_substring_match():
    index = build_index([term("art", "X")], version=1)
    assert find_spans(index, "a partial artifact") == []


def test_whole_word_false_allows_substring_match():
    index = build_index([term("art", "X", whole_word=False)], version=1)
    assert len(find_spans(index, "partial")) == 1


def test_case_insensitive_by_default():
    index = build_index([term("genome", "X")], version=1)
    assert len(find_spans(index, "The Genome.")) == 1


def test_case_sensitive_entry_does_not_match_other_casing():
    index = build_index([term("Genome", "X", case_sensitive=True)], version=1)
    assert find_spans(index, "the genome") == []
    assert len(find_spans(index, "the Genome")) == 1


def test_every_occurrence_is_its_own_span():
    index = build_index([term("genome", "X")], version=1)
    assert len(find_spans(index, "genome and genome")) == 2


def test_priority_breaks_equal_length_ties():
    index = build_index(
        [
            term("gene", "low", entry_id=1, priority=0),
            term("gene", "high", entry_id=2, priority=5, case_sensitive=True),
        ],
        version=1,
    )
    spans = find_spans(index, "the gene here")
    assert len(spans) == 1
    assert spans[0].term.entry_id == 2


def test_spans_are_returned_in_document_order():
    index = build_index([term("alpha", "A", entry_id=1), term("beta", "B", entry_id=2)], version=1)
    spans = find_spans(index, "beta then alpha")
    assert [span.term.entry_id for span in spans] == [2, 1]


def test_an_empty_index_matches_nothing():
    assert find_spans(build_index([], version=1), "anything") == []


def test_leftmost_wins_when_length_and_priority_tie():
    # Without this test, a reversed (or dropped) `span.start` component in the
    # overlap-resolution sort key would still pass every other test in this
    # file. The two terms must differ in case_sensitive: two terms sharing a
    # case mode compile into one alternation, and re.finditer never returns
    # overlapping matches from a single pattern, so a same-pattern pair (e.g.
    # "abc"/"bcd" both case-insensitive) never produces two overlapping
    # candidates in the first place -- it can't exercise the tiebreak. Two
    # different case modes give two independent finditer passes over the same
    # text, which is the only way to get genuinely overlapping, equal-length,
    # equal-priority candidates to resolve between.
    index = build_index(
        [
            term("abc", "A", entry_id=1, whole_word=False, case_sensitive=True),
            term("bcd", "B", entry_id=2, whole_word=False, case_sensitive=False),
        ],
        version=1,
    )
    spans = find_spans(index, "abcd")
    assert len(spans) == 1
    assert spans[0].term.entry_id == 1
    assert spans[0].start == 0
