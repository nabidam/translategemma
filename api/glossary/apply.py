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

from .matcher import Span, Term, resolve_overlaps
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


@dataclass(frozen=True)
class _Occupant:
    """A range in the folded translation that some term lays claim to.

    A canonical hit is already correct text -- it produces no edit, but it
    still has to occupy its range in the SAME overlap-resolution pool as
    alias hits. Without that, a competing alias edit that overlaps a
    canonical range would win by default and splice over already-correct
    text, corrupting the output while the canonical term is still reported
    as successfully applied.
    """

    start: int
    end: int
    term: Term
    is_edit: bool  # True: an alias hit, needs rewriting. False: a canonical hit, already correct.


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
    # Canonical hits (already correct, no edit needed) and alias hits (need
    # rewriting) from every term, pooled together into one set of ranges
    # competing for the same text. A hit from one term -- canonical or alias
    # -- can overlap a hit from a different term (or a different alias of the
    # SAME term) in the model's output: one entry's alias can be a substring
    # of another's alias, or an alias can overlap where a different term's
    # canonical text already sits. All of that is equally capable of
    # corrupting the result if both were edited/counted independently, so all
    # of it is resolved in one pool rather than per term or per kind.
    occupants: list[_Occupant] = []

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
        for start, end in canonical_hits:
            occupants.append(_Occupant(start=start, end=end, term=term, is_edit=False))
        for start, end in term_alias_hits:
            occupants.append(_Occupant(start=start, end=end, term=term, is_edit=True))

    # Resolve overlaps across ALL pooled occupants with the same greedy
    # longest-first claim loop matcher.find_spans uses for source spans:
    # longest match wins, then higher priority, then leftmost; anything the
    # winner overlaps is dropped -- including a canonical range, which
    # otherwise blocks nothing and lets an overlapping alias edit splice over
    # already-correct text.
    claimed = resolve_overlaps(
        occupants,
        start=lambda occupant: occupant.start,
        end=lambda occupant: occupant.end,
        priority=lambda occupant: occupant.term.priority,
    )

    # (start, end) in the ORIGINAL string -> replacement. Collected first and
    # applied right-to-left so earlier edits cannot shift later offsets.
    edits: list[tuple[int, int, str]] = []
    claimed_counts: dict[int, int] = {}
    for occupant in claimed:
        claimed_counts[occupant.term.entry_id] = claimed_counts.get(occupant.term.entry_id, 0) + 1
        if occupant.is_edit:
            edits.append(
                (offsets[occupant.start], offsets[occupant.end - 1] + 1, occupant.term.target_term)
            )

    applied: list[Applied] = []
    for term in candidate_terms:
        # count reflects what was actually applied: only the canonical and
        # alias occurrences that survived overlap resolution, not the raw
        # pre-resolution tally. A term with real hits that all lost their
        # claim to a competing term is not "applied" -- nothing of it
        # survived in the output -- so it is reported as a miss instead of a
        # zero-effect success, with a reason distinct from "never appeared at
        # all": the text WAS there, it just lost to something else that
        # occupied the same range.
        count = claimed_counts.get(term.entry_id, 0)
        if count == 0:
            misses.append(Miss(source_term=term.source_term, reason="claimed_by_overlap"))
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
