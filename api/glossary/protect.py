"""Sentinel protection for `exact` mode.

`preferred` mode repairs the model's output and never touches its input, which
is what makes it incapable of regressing translation quality. `exact` mode
makes the opposite trade deliberately: the source term is replaced by a
sentinel before generation and the agreed target is substituted back
afterwards, so the model never sees the term and cannot inflect, translate or
paraphrase it.

That is the only way to guarantee a character-for-character rendering, and it
is why the mode is restricted to terms Persian morphology does not attach to --
product names, organisations, codes, URLs, units. Microsoft ships the
equivalent (Azure's dynamic dictionary) with the same restriction, documented
as "safe only for compound nouns like proper names and product names".

The sentinel format was chosen by measurement against the served checkpoint,
not by taste: `__TG_TERM_000__` survived every frame of the design probe
intact, including a frame containing the same sentinel twice, with no run
hitting max_new_tokens (see docs/2026-08-21_glossary_memory_design.md,
"Sentinel survival"). Changing it re-opens a question that cost real GPU time
to answer.

Nothing here calls the model. Protection produces a rewritten source string;
restoration validates and rewrites the translation. The caller owns generation
and owns the fallback when validation fails.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from .matcher import Span

# Deliberately underscore-delimited ASCII with no sentence-ending punctuation:
# the protected text is handed to pysbd for sentence splitting, and a sentinel
# carrying a period would invent a sentence boundary that the source never had.
SENTINEL_TEMPLATE = "__TG_TERM_{index:03d}__"

EXACT = "exact"


@dataclass(frozen=True)
class Protection:
    """A source text with its `exact` terms replaced by sentinels.

    `restorations` pairs each sentinel with the target term that replaces it
    after generation. `source_terms` is parallel to it, for reporting which
    entry each sentinel came from.
    """

    text: str
    restorations: tuple[tuple[str, str], ...]
    source_terms: tuple[str, ...]

    @property
    def is_empty(self) -> bool:
        return not self.restorations


def protect(text: str, spans: Sequence[Span]) -> Protection:
    """Replace every `exact` span with a unique sentinel.

    Spans are applied right to left so an earlier replacement cannot shift a
    later span's offsets — the same discipline apply.py uses, for the same
    reason. Each occurrence gets its own sentinel even when two spans share a
    term, because validation counts sentinels and a repeated one could not tell
    a duplicated occurrence from a dropped one.
    """
    exact = [span for span in spans if span.term.target_mode == EXACT]
    if not exact:
        return Protection(text=text, restorations=(), source_terms=())

    ordered = sorted(exact, key=lambda span: span.start)
    restorations: list[tuple[str, str]] = []
    source_terms: list[str] = []
    for index, span in enumerate(ordered):
        restorations.append((SENTINEL_TEMPLATE.format(index=index), span.term.target_term))
        source_terms.append(span.term.source_term)

    protected = text
    for index, span in reversed(list(enumerate(ordered))):
        sentinel = restorations[index][0]
        protected = protected[: span.start] + sentinel + protected[span.end :]

    return Protection(
        text=protected,
        restorations=tuple(restorations),
        source_terms=tuple(source_terms),
    )


def restore(translation: str, protection: Protection) -> tuple[str, list[str]]:
    """Substitute the agreed terms back. Returns (text, source_terms_that_failed).

    A sentinel must appear exactly once. Anything else — dropped, duplicated,
    or mangled by the decoder — means the model did not carry it through
    faithfully, and the caller must fall back to an unprotected translation
    rather than emit a sentinel or guess where the term belonged.

    Validation is per sentinel, so one lost sentinel does not discard the
    others: the failures are named, and every sentinel that did survive is
    still restored.
    """
    if protection.is_empty:
        return translation, []

    restored = translation
    failed: list[str] = []
    for index, (sentinel, target_term) in enumerate(protection.restorations):
        if restored.count(sentinel) != 1:
            failed.append(protection.source_terms[index])
            continue
        restored = restored.replace(sentinel, target_term)
    return restored, failed


def strip_sentinels(text: str, protection: Protection) -> str:
    """Remove any sentinel still present, for the give-up path.

    Only reached when the unprotected retry ALSO failed, which should not
    happen — the retry sends the original text, so there is nothing to leak.
    It exists so that a sentinel can never reach a caller even then: returning
    a visible `__TG_TERM_000__` to an end user is worse than returning a
    translation missing its agreed term, which at least reads as language.
    """
    for sentinel, _ in protection.restorations:
        text = text.replace(sentinel, "")
    return " ".join(text.split())
