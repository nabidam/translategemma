"""Rewrite a translation so agreed terms appear in their agreed form.

`preferred` mode leaves the source text alone and repairs the output, which is
what keeps the sentence grammatical: the model inflects and orders the sentence
freely, and only the term's stem is replaced. Amazon Translate moved from plain
substitution to this shape for the same reason.

Matching runs on a folded copy of the translation (ZWNJ dropped, Arabic letters
folded to Persian) because the model's spelling of a term and an
administrator's rarely agree on those, while looking identical on screen.
Replacement is applied to the ORIGINAL string using the offset map, so nothing
else in the translation is normalized as a side effect.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from .matcher import Span, Term
from .normalize import fold_target


@dataclass(frozen=True)
class Applied:
    source_term: str
    target_term: str
    mode: str
    count: int


@dataclass(frozen=True)
class Miss:
    source_term: str
    reason: str


@dataclass(frozen=True)
class Violation:
    source_term: str
    forbidden: str


@dataclass(frozen=True)
class Report:
    applied: tuple[Applied, ...]
    misses: tuple[Miss, ...]
    violations: tuple[Violation, ...]

    @classmethod
    def empty(cls) -> "Report":
        return cls(applied=(), misses=(), violations=())

    @property
    def is_empty(self) -> bool:
        return not (self.applied or self.misses or self.violations)


def _find_all(folded_haystack: str, needle: str) -> list[tuple[int, int]]:
    """Every occurrence of a folded needle, as (start, end) in the folded string."""
    folded_needle, _ = fold_target(needle)
    if not folded_needle:
        return []
    found = []
    position = folded_haystack.find(folded_needle)
    while position != -1:
        found.append((position, position + len(folded_needle)))
        position = folded_haystack.find(folded_needle, position + 1)
    return found


def apply_preferred(
    translation: str, spans: Sequence[Span]
) -> tuple[str, Report]:
    """Rewrite aliases to their canonical term. Returns the text and what happened.

    A term whose canonical form is already present counts as applied and is left
    alone. A term with neither its canonical form nor any alias in the output is
    a miss, and the translation is returned unmodified rather than guessed at.
    """
    if not spans:
        return translation, Report.empty()

    folded, offsets = fold_target(translation)
    misses: list[Miss] = []
    violations: list[Violation] = []

    # One entry may match several source spans; each entry is judged once, in
    # first-seen order.
    seen: set[int] = set()
    candidate_terms: list[Term] = []
    canonical_counts: dict[int, int] = {}
    # Alias hits from every term, pooled together: a hit from one term's alias
    # can overlap a hit from another term's alias in the model's output (e.g.
    # one entry's alias is a substring of a different entry's alias), and that
    # is exactly as capable of corrupting the result as two aliases of the
    # same term overlapping. Both need the same overlap resolution, so both
    # are resolved in one pool rather than per term.
    alias_candidates: list[tuple[int, int, Term]] = []

    for span in spans:
        term = span.term
        if term.entry_id in seen:
            continue
        seen.add(term.entry_id)

        for forbidden in term.forbidden:
            if _find_all(folded, forbidden):
                violations.append(Violation(source_term=term.source_term, forbidden=forbidden))

        canonical_hits = _find_all(folded, term.target_term)
        term_alias_hits = []
        for alias in term.aliases:
            term_alias_hits.extend(_find_all(folded, alias))

        if not canonical_hits and not term_alias_hits:
            misses.append(Miss(source_term=term.source_term, reason="target_not_found"))
            continue

        candidate_terms.append(term)
        canonical_counts[term.entry_id] = len(canonical_hits)
        for start, end in term_alias_hits:
            alias_candidates.append((start, end, term))

    # Resolve overlaps across ALL pooled alias hits with the same greedy
    # longest-first claim loop matcher.find_spans uses for source spans:
    # longest match wins, then higher priority, then leftmost; anything the
    # winner overlaps is dropped rather than also rewritten, which is what
    # would corrupt the string (e.g. "AB" and "BC" both claiming the "B" in
    # "ABC").
    alias_candidates.sort(key=lambda c: (-(c[1] - c[0]), -c[2].priority, c[0]))
    claimed: list[tuple[int, int, Term]] = []
    for start, end, term in alias_candidates:
        if any(start < c_end and c_start < end for c_start, c_end, _ in claimed):
            continue
        claimed.append((start, end, term))

    # (start, end) in the ORIGINAL string -> replacement. Collected first and
    # applied right-to-left so earlier edits cannot shift later offsets.
    edits: list[tuple[int, int, str]] = []
    claimed_counts: dict[int, int] = {}
    for start, end, term in claimed:
        claimed_counts[term.entry_id] = claimed_counts.get(term.entry_id, 0) + 1
        edits.append((offsets[start], offsets[end - 1] + 1, term.target_term))

    applied: list[Applied] = []
    for term in candidate_terms:
        # count reflects what was actually applied: canonical hits plus only
        # the alias hits that survived overlap resolution. A term whose sole
        # alias hit was dropped because another term's hit claimed the same
        # text is not "applied" -- nothing was rewritten for it -- so it is
        # reported as a miss rather than as a zero-effect success.
        count = canonical_counts[term.entry_id] + claimed_counts.get(term.entry_id, 0)
        if count == 0:
            misses.append(Miss(source_term=term.source_term, reason="target_not_found"))
            continue
        applied.append(
            Applied(
                source_term=term.source_term,
                target_term=term.target_term,
                mode=term.target_mode,
                count=count,
            )
        )

    result = translation
    for start, end, replacement in sorted(edits, key=lambda edit: edit[0], reverse=True):
        result = result[:start] + replacement + result[end:]

    return result, Report(
        applied=tuple(applied), misses=tuple(misses), violations=tuple(violations)
    )
