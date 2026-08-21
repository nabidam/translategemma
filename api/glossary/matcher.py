"""Compile a termbase into one regex and find its non-overlapping matches.

A single compiled alternation, not a term-by-term scan: the alternation is
matched in one pass by CPython's regex engine, and alternatives sorted
longest-first make the engine prefer the longer term wherever two overlap --
which is the longest-match-wins rule, obtained for free rather than
post-processed.

Case-sensitive and case-insensitive entries need different matching, so they
compile into two patterns over the same text. `pyahocorasick` is the drop-in
replacement if term counts ever outgrow this; nothing outside this module knows
which is used.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from .normalize import fold_source

# A term is bounded by anything that is not a word character. Applied to the
# folded source, which is why it can assume ASCII-ish word semantics: the source
# side of this deployment is English.
_BOUNDARY_BEFORE = r"(?<![\w])"
_BOUNDARY_AFTER = r"(?![\w])"


@dataclass(frozen=True)
class Term:
    entry_id: int
    source_term: str
    target_term: str
    target_mode: str
    aliases: tuple[str, ...]
    forbidden: tuple[str, ...]
    case_sensitive: bool
    whole_word: bool
    priority: int


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    term: Term


@dataclass(frozen=True)
class Index:
    """An immutable compiled termbase. Built once, read by many requests."""

    version: int
    terms: tuple[Term, ...]
    _patterns: tuple[tuple[bool, re.Pattern[str], tuple[Term, ...]], ...] = field(
        default=(), repr=False
    )


def _group_name(position: int) -> str:
    return f"t{position}"


def _compile(terms: Sequence[Term], case_sensitive: bool) -> tuple[re.Pattern[str], tuple[Term, ...]] | None:
    """One alternation over terms sharing a case mode, longest alternative first."""
    selected = [term for term in terms if term.case_sensitive is case_sensitive]
    if not selected:
        return None
    # Longest first so the engine prefers "multi-query attention" over
    # "attention"; priority then length-ties, descending, so the higher-priority
    # entry is the earlier alternative.
    selected.sort(key=lambda term: (-len(term.source_term), -term.priority, term.entry_id))
    parts = []
    for position, term in enumerate(selected):
        folded_term, _ = fold_source(term.source_term, case_sensitive)
        body = re.escape(folded_term)
        if term.whole_word:
            body = f"{_BOUNDARY_BEFORE}{body}{_BOUNDARY_AFTER}"
        parts.append(f"(?P<{_group_name(position)}>{body})")
    return re.compile("|".join(parts)), tuple(selected)


def build_index(terms: Sequence[Term], version: int) -> Index:
    """Compile a termbase. Raises re.error only on a term that cannot be escaped."""
    patterns = []
    for case_sensitive in (True, False):
        compiled = _compile(terms, case_sensitive)
        if compiled is not None:
            patterns.append((case_sensitive, compiled[0], compiled[1]))
    return Index(version=version, terms=tuple(terms), _patterns=tuple(patterns))


def find_spans(index: Index, text: str) -> list[Span]:
    """Non-overlapping matches in document order, longest and highest priority first."""
    candidates: list[Span] = []
    for case_sensitive, pattern, ordered_terms in index._patterns:
        folded, offsets = fold_source(text, case_sensitive)
        for match in pattern.finditer(folded):
            position = int(match.lastgroup[1:])  # "t7" -> 7
            term = ordered_terms[position]
            start_folded, end_folded = match.span()
            if start_folded == end_folded:
                continue
            candidates.append(
                Span(
                    start=offsets[start_folded],
                    end=offsets[end_folded - 1] + 1,
                    term=term,
                )
            )

    # Resolve overlaps across the two patterns: longest first, then priority,
    # then leftmost. Claiming greedily in that order leaves the winner and drops
    # anything it covers.
    candidates.sort(key=lambda span: (-(span.end - span.start), -span.term.priority, span.start))
    claimed: list[Span] = []
    for span in candidates:
        if any(span.start < other.end and other.start < span.end for other in claimed):
            continue
        claimed.append(span)
    claimed.sort(key=lambda span: span.start)
    return claimed
