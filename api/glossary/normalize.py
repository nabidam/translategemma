"""Offset-preserving folding for glossary matching.

Matching runs on a folded form; rewriting happens on the original string. That
only works if every folded character can be traced back to the character it
came from, so every function here returns an offset map alongside the folded
text.

Deliberately NOT unicodedata.normalize("NFKC", ...): NFKC expands some Arabic
presentation forms into several characters, which breaks the one-to-one offset
map this module's callers depend on. The explicit character maps below cover
the cases that actually occur in Persian text from this model, and each one is
length-preserving or a pure deletion.
"""

ZWNJ = "‌"

# Arabic letters that Persian text is routinely typed with, folded to their
# Persian equivalents. Without this, an alias containing the Persian form fails
# to match output containing the Arabic form, and vice versa -- silently, with
# both strings looking identical on screen.
_LETTER_FOLDS = {
    "ي": "ی",  # Arabic yeh      -> Persian yeh
    "ى": "ی",  # alef maksura    -> Persian yeh
    "ك": "ک",  # Arabic kaf      -> Persian keheh
    "ڪ": "ک",  # swash kaf       -> Persian keheh
    "ة": "ه",  # teh marbuta     -> heh
}

# Arabic-Indic (U+0660-0669) and extended Arabic-Indic (U+06F0-06F9) digits.
_DIGIT_FOLDS = {
    **{chr(0x0660 + n): str(n) for n in range(10)},
    **{chr(0x06F0 + n): str(n) for n in range(10)},
}

_TARGET_FOLDS = {**_LETTER_FOLDS, **_DIGIT_FOLDS}

# Dropped rather than folded: ZWNJ is invisible and its presence varies between
# the model's output and an administrator's typing, so it must not decide a
# match. It is only removed from the folded copy; the original keeps it.
# Also strips other invisible separators as deliberate hardening.
_DROPPED = {ZWNJ, "​", "﻿"}


def _fold(text: str, table: dict[str, str], lowercase: bool) -> tuple[str, list[int]]:
    """Fold character by character, recording where each output character began.

    Each output character maps back to its source index, even if a transformation
    (like .lower() on Turkish İ) expands to multiple characters. Multiple output
    characters may share one source index — the span arithmetic in callers still
    resolves to the correct slice of the original string.
    """
    folded: list[str] = []
    offsets: list[int] = []
    for index, character in enumerate(text):
        if character in _DROPPED:
            continue
        replacement = table.get(character, character)
        if lowercase:
            # .lower() is preferred over .casefold() because it expands far less
            # often (e.g., casefold expands German sharp s, but .lower() does not).
            # The per-character loop below makes any remaining expansion safe.
            replacement = replacement.lower()
        for char in replacement:
            folded.append(char)
            offsets.append(index)
    return "".join(folded), offsets


def fold_target(text: str) -> tuple[str, list[int]]:
    """Fold Persian output for alias matching. Never lowercased: Persian has no case."""
    return _fold(text, _TARGET_FOLDS, lowercase=False)


def fold_source(text: str, case_sensitive: bool) -> tuple[str, list[int]]:
    """Fold source text for term matching, lowercasing unless the entry forbids it."""
    return _fold(text, _LETTER_FOLDS, lowercase=not case_sensitive)
