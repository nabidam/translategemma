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

from .matcher import Span
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
    applied: list[Applied] = []
    misses: list[Miss] = []
    violations: list[Violation] = []
    # (start, end) in the ORIGINAL string -> replacement. Collected first and
    # applied right-to-left so earlier edits cannot shift later offsets.
    edits: list[tuple[int, int, str]] = []

    # One entry may match several source spans; each entry is judged once.
    seen: set[int] = set()
    for span in spans:
        term = span.term
        if term.entry_id in seen:
            continue
        seen.add(term.entry_id)

        for forbidden in term.forbidden:
            if _find_all(folded, forbidden):
                violations.append(Violation(source_term=term.source_term, forbidden=forbidden))

        canonical_hits = _find_all(folded, term.target_term)
        alias_hits: list[tuple[int, int]] = []
        for alias in term.aliases:
            alias_hits.extend(_find_all(folded, alias))

        if not canonical_hits and not alias_hits:
            misses.append(Miss(source_term=term.source_term, reason="target_not_found"))
            continue

        for start, end in alias_hits:
            edits.append((offsets[start], offsets[end - 1] + 1, term.target_term))

        applied.append(
            Applied(
                source_term=term.source_term,
                target_term=term.target_term,
                mode=term.target_mode,
                count=len(canonical_hits) + len(alias_hits),
            )
        )

    result = translation
    for start, end, replacement in sorted(edits, key=lambda edit: edit[0], reverse=True):
        result = result[:start] + replacement + result[end:]

    return result, Report(
        applied=tuple(applied), misses=tuple(misses), violations=tuple(violations)
    )
