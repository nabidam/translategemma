# Glossary Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the TranslateGemma gateway an admin-managed termbase that rewrites
agreed terms in served translations, shipping dark behind an environment flag.

**Architecture:** A new `api/glossary/` package. Pure functions do the matching
and rewriting; a service object holds an immutable per-`(src, tgt, domain)`
snapshot built from SQLite and swapped atomically on admin writes. Exactly one
call site in `translator.py` applies it. With `TG_GLOSSARY_ENABLED=false` the
service is never constructed and the request path does not branch on it.

**Tech Stack:** FastAPI, Pydantic v2, SQLAlchemy 2.x async + aiosqlite, stdlib
`re`, pytest.

**Spec:** `docs/2026-08-21_glossary_memory_design.md`

## Scope

This plan covers **phase 0 and phase 1 only** — the shippable core:
`preferred` mode, admin CRUD, dry-run, the kill switch. Deliberately excluded,
each already scoped in the spec: `exact`/sentinel mode (phase 2),
negative-constraint re-decode (phase 2.5), `suggest` mode (phase 3), train-time
terminology (future).

`target_mode` is still stored on every entry, and writes accept only
`'preferred'` until phase 2 lands. The column exists from day one so phase 2 is
additive rather than a migration.

Two spec items are deliberately deferred for delivery speed, both additive
later and neither load-bearing for correctness:

- **TSV/CSV bulk import.** Entries go in one at a time through `POST /entries`
  until someone has enough of them for import to be worth building. The
  version-per-import rule in the spec applies when it is.
- **Soft admin warnings** — long phrases, verb/adjective terms, entry-count
  thresholds. The *hard* validations that prevent silent misbehaviour do ship,
  in Task 8: stopword rejection, duplicate conflict, unknown domain, and
  refusing `target_mode: exact` before it works.

## Global Constraints

- Python 3.12+, modern type hints, `pathlib` over `os.path`.
- `uv` for everything: `uv run pytest`, `uv add`. Never `pip`.
- Pydantic v2 models for all request/response schemas.
- `api/` ships standalone — glossary code may import only from `api/` and
  `api/requirements.txt`. It must never import from the repository root.
- Do not modify `api/prompting.py`. It is a byte-identical vendored copy guarded
  by `tests/test_api_vendored_modules.py`, and the glossary never touches prompt
  rendering.
- Every new dependency goes in `api/requirements.txt` with a pinned range and a
  comment saying why, matching that file's existing style.
- `TG_GLOSSARY_ENABLED` defaults to `false`. No task may change that default.
- Persian normalization must be offset-preserving. Never normalize a string you
  intend to slice by index without carrying an offset map.

## Deviation from the spec, decided here

The spec says Aho-Corasick via `pyahocorasick`. This plan uses a **compiled
`re` alternation** instead: alternatives sorted longest-first give
longest-match-wins directly, `re` is C-speed and single-pass, and it removes a
compiled C dependency from a deployment that runs offline. `pyahocorasick`
stays a drop-in replacement behind `matcher.build_index` if term counts ever
make it necessary. Nothing outside `matcher.py` knows which is used.

---

### Task 1: Test scaffolding and the kill switch

Phase 0 from the spec: the flag and its observability, with nothing behind it.

**Files:**
- Create: `tests/conftest.py`
- Modify: `pyproject.toml` (add a dev dependency group)
- Modify: `api/config.py` (add settings)
- Modify: `api/schemas.py:ModelInfoResponse`
- Modify: `api/main.py:model_info`
- Test: `tests/test_glossary_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Settings.glossary_enabled: bool`, `Settings.glossary_db_url: str`,
  `Settings.admin_api_key: str | None`,
  `Settings.glossary_default_domain: str | None`,
  `Settings.glossary_unknown_domain: str` (`"reject"` | `"fallback"`),
  `Settings.terminology_mode: str`. `ModelInfoResponse.glossary_enabled: bool`.

- [ ] **Step 1: Configure pytest**

The suite runs through `scripts/test_glossary.sh`, which already exists and
supplies its own dependencies in an isolated environment. Do NOT add a
`[dependency-groups]` table or otherwise touch the project's dependency
declarations: the root project pins torch and CUDA wheels that this suite must
never pull, and the installed uv is old enough that edits to that file are
risky.

Add only this to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

- [ ] **Step 2: Add the shared test path fixture**

Create `tests/conftest.py`:

```python
"""Make api/ importable from the test suite.

api/ ships standalone and has no package metadata, so its modules are imported
by putting the directory on sys.path rather than through an installed package.
tests/test_generation_chat_template.py does the same for the repository root.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
API_ROOT = PROJECT_ROOT / "api"

for path in (PROJECT_ROOT, API_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
```

- [ ] **Step 3: Write the failing test**

Create `tests/test_glossary_config.py`:

```python
"""The glossary ships dark. These tests hold that default in place."""

import pytest

from config import Settings


def test_glossary_is_disabled_by_default():
    settings = Settings(_env_file=None)
    assert settings.glossary_enabled is False


def test_enabling_without_an_admin_key_fails_at_startup():
    # The admin router is the only authorization boundary in this deployment,
    # so a missing key must fail loudly at boot rather than at the first write.
    with pytest.raises(ValueError, match="TG_ADMIN_API_KEY"):
        Settings(_env_file=None, glossary_enabled=True, admin_api_key=None)


def test_enabling_with_an_admin_key_is_accepted():
    settings = Settings(_env_file=None, glossary_enabled=True, admin_api_key="secret")
    assert settings.glossary_enabled is True
    assert settings.admin_api_key == "secret"


def test_unknown_domain_policy_defaults_to_reject():
    settings = Settings(_env_file=None)
    assert settings.glossary_unknown_domain == "reject"
```

- [ ] **Step 4: Run the test to verify it fails**

Run: `scripts/test_glossary.sh tests/test_glossary_config.py -v`
Expected: FAIL — `Settings` has no field `glossary_enabled`.

- [ ] **Step 5: Add the settings**

In `api/config.py`, add to `Settings` after the `--- Sentence splitting ---`
block:

```python
    # --- Glossary (optional feature, off by default) ----------------------
    # The kill switch. False means the glossary is not merely bypassed but
    # absent: no database is opened, no admin router is mounted, and the token
    # ids posted to vLLM are identical to those of a build without this
    # feature. A terminology problem in production is one variable and a
    # restart away from being gone.
    glossary_enabled: bool = False
    glossary_db_url: str = "sqlite+aiosqlite:///./data/glossary.db"
    # The whole authorization boundary for the admin routes. No default: a
    # guessable default key on a write endpoint is worse than no feature.
    admin_api_key: str | None = None
    # Applied when a request omits `domain`. Lets a single-field deployment pin
    # its termbase without teaching callers the concept exists.
    glossary_default_domain: str | None = None
    # What to do with a domain name that does not exist. "reject" surfaces a
    # caller's typo as a 404; "fallback" quietly uses the global layer.
    glossary_unknown_domain: str = "reject"
    terminology_mode: str = "enforce"
```

Add to the existing `_convert_empty_str_to_none` validator's field list:

```python
    @field_validator(
        "adapter_path",
        "tokenizer_path",
        "vllm_api_key",
        "admin_api_key",
        "glossary_default_domain",
        mode="before",
    )
```

Add to `_validate_and_resolve`, before `return self`:

```python
        if self.glossary_enabled and not self.admin_api_key:
            raise ValueError(
                "TG_GLOSSARY_ENABLED is true but TG_ADMIN_API_KEY is unset. The admin "
                "routes would be the only unauthenticated write surface on this "
                "gateway. Set a key or disable the glossary."
            )
        if self.glossary_unknown_domain not in ("reject", "fallback"):
            raise ValueError(
                f"TG_GLOSSARY_UNKNOWN_DOMAIN must be 'reject' or 'fallback'; "
                f"got {self.glossary_unknown_domain!r}."
            )
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `scripts/test_glossary.sh tests/test_glossary_config.py -v`
Expected: PASS, 4 tests.

- [ ] **Step 7: Report the flag on /model-info**

In `api/schemas.py`, add to `ModelInfoResponse` after `adapter_path`:

```python
    # Whether the termbase is active on this deployment, so its state is
    # observable without reading the container's environment.
    glossary_enabled: bool
```

In `api/main.py:model_info`, add to the returned `ModelInfoResponse(...)`:

```python
        glossary_enabled=settings.glossary_enabled,
```

- [ ] **Step 8: Commit**

```bash
git add tests/conftest.py tests/test_glossary_config.py pyproject.toml api/config.py api/schemas.py api/main.py
git commit -m "feat(api): add glossary kill switch and settings"
```

---

### Task 2: Persian and source normalization

**Files:**
- Create: `api/glossary/__init__.py`, `api/glossary/normalize.py`
- Test: `tests/test_glossary_normalize.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `fold_source(text: str, case_sensitive: bool) -> tuple[str, list[int]]`,
  `fold_target(text: str) -> tuple[str, list[int]]`. Both return
  `(folded_text, offsets)` where `offsets[i]` is the index in the original
  string of `folded_text[i]`, and `offsets` has the same length as
  `folded_text`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_glossary_normalize.py`:

```python
"""Folding rules for matching. Offsets must survive, or rewriting slices wrong."""

from glossary.normalize import fold_source, fold_target


def test_target_folds_arabic_yeh_and_kaf_to_persian():
    folded, _ = fold_target("يك")  # Arabic yeh, Arabic kaf
    assert folded == "یک"  # Persian yeh, Persian keheh


def test_target_drops_zwnj_but_keeps_offsets_pointing_at_the_original():
    original = "می‌رود"  # mi-ravad, with ZWNJ
    folded, offsets = fold_target(original)
    assert "‌" not in folded
    assert len(folded) == len(offsets)
    # Every folded character still maps back to the character it came from.
    assert all(original[offset] != "‌" for offset in offsets)
    # The character after the dropped ZWNJ maps past it, not onto it.
    assert offsets[2] == 3


def test_target_folds_arabic_indic_digits():
    folded, _ = fold_target("۱٢")  # extended Arabic-Indic 1, Arabic-Indic 2
    assert folded == "12"


def test_source_lowercases_when_not_case_sensitive():
    folded, offsets = fold_source("Genome", case_sensitive=False)
    assert folded == "genome"
    assert offsets == list(range(6))


def test_source_preserves_case_when_case_sensitive():
    folded, _ = fold_source("Genome", case_sensitive=True)
    assert folded == "Genome"


def test_offsets_allow_slicing_the_original_by_a_folded_match():
    original = "The يGenome‌ sequence"
    folded, offsets = fold_source(original, case_sensitive=False)
    start = folded.index("genome")
    end = start + len("genome")
    assert original[offsets[start] : offsets[end - 1] + 1] == "Genome"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `scripts/test_glossary.sh tests/test_glossary_normalize.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'glossary'`.

- [ ] **Step 3: Implement normalization**

Create `api/glossary/__init__.py` (empty file).

Create `api/glossary/normalize.py`:

```python
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
_DROPPED = {ZWNJ, "​", "﻿"}


def _fold(text: str, table: dict[str, str], lowercase: bool) -> tuple[str, list[int]]:
    """Fold character by character, recording where each output character began."""
    folded: list[str] = []
    offsets: list[int] = []
    for index, character in enumerate(text):
        if character in _DROPPED:
            continue
        replacement = table.get(character, character)
        if lowercase:
            # .lower() rather than .casefold(): casefold expands a few
            # characters (German sharp s) and would break the one-to-one map.
            replacement = replacement.lower()
        folded.append(replacement)
        offsets.append(index)
    return "".join(folded), offsets


def fold_target(text: str) -> tuple[str, list[int]]:
    """Fold Persian output for alias matching. Never lowercased: Persian has no case."""
    return _fold(text, _TARGET_FOLDS, lowercase=False)


def fold_source(text: str, case_sensitive: bool) -> tuple[str, list[int]]:
    """Fold source text for term matching, lowercasing unless the entry forbids it."""
    return _fold(text, _LETTER_FOLDS, lowercase=not case_sensitive)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `scripts/test_glossary.sh tests/test_glossary_normalize.py -v`
Expected: PASS, 6 tests.

- [ ] **Step 5: Commit**

```bash
git add api/glossary/__init__.py api/glossary/normalize.py tests/test_glossary_normalize.py
git commit -m "feat(api): add offset-preserving glossary normalization"
```

---

### Task 3: Term matching

**Files:**
- Create: `api/glossary/matcher.py`
- Test: `tests/test_glossary_matcher.py`

**Interfaces:**
- Consumes: `glossary.normalize.fold_source`.
- Produces: `Term` (frozen dataclass: `entry_id: int`, `source_term: str`,
  `target_term: str`, `target_mode: str`, `aliases: tuple[str, ...]`,
  `forbidden: tuple[str, ...]`, `case_sensitive: bool`, `whole_word: bool`,
  `priority: int`), `Span` (frozen: `start: int`, `end: int`, `term: Term`),
  `Index` (frozen: `version: int`, `terms: tuple[Term, ...]`),
  `build_index(terms: Sequence[Term], version: int) -> Index`,
  `find_spans(index: Index, text: str) -> list[Span]`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_glossary_matcher.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `scripts/test_glossary.sh tests/test_glossary_matcher.py -v`
Expected: FAIL — no module `glossary.matcher`.

- [ ] **Step 3: Implement the matcher**

Create `api/glossary/matcher.py`:

```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `scripts/test_glossary.sh tests/test_glossary_matcher.py -v`
Expected: PASS, 10 tests.

- [ ] **Step 5: Commit**

```bash
git add api/glossary/matcher.py tests/test_glossary_matcher.py
git commit -m "feat(api): add glossary term matcher"
```

---

### Task 4: Applying terms to a translation

**Files:**
- Create: `api/glossary/apply.py`
- Test: `tests/test_glossary_apply.py`

**Interfaces:**
- Consumes: `glossary.normalize.fold_target`, `glossary.matcher.Span`.
- Produces: `Applied` (frozen: `source_term`, `target_term`, `mode`, `count: int`),
  `Miss` (frozen: `source_term`, `reason: str`),
  `Violation` (frozen: `source_term`, `forbidden: str`),
  `Report` (frozen: `applied: tuple[Applied, ...]`, `misses: tuple[Miss, ...]`,
  `violations: tuple[Violation, ...]`, `empty` classmethod),
  `apply_preferred(translation: str, spans: Sequence[Span]) -> tuple[str, Report]`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_glossary_apply.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `scripts/test_glossary.sh tests/test_glossary_apply.py -v`
Expected: FAIL — no module `glossary.apply`.

- [ ] **Step 3: Implement application**

Create `api/glossary/apply.py`:

```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `scripts/test_glossary.sh tests/test_glossary_apply.py -v`
Expected: PASS, 8 tests.

- [ ] **Step 5: Commit**

```bash
git add api/glossary/apply.py tests/test_glossary_apply.py
git commit -m "feat(api): add preferred-mode glossary application"
```

---

### Task 5: Persistence

**Files:**
- Create: `api/glossary/models.py`, `api/glossary/store.py`
- Modify: `api/requirements.txt`
- Test: `tests/test_glossary_store.py`

**Interfaces:**
- Consumes: `glossary.matcher.Term`.
- Produces: `Domain`, `Entry` (SQLAlchemy models); `GlossaryStore` with
  `async create_all()`, `async list_domains() -> list[Domain]`,
  `async create_domain(name, src_lang, tgt_lang, description) -> Domain`,
  `async delete_domain(name) -> bool`,
  `async list_entries(domain_name: str | None) -> list[Entry]`,
  `async create_entry(**fields) -> Entry`, `async delete_entry(entry_id) -> bool`,
  `async load_terms(src_lang, tgt_lang, domain_name: str | None) -> tuple[list[Term], int]`
  returning terms and the version to stamp an index with.

- [ ] **Step 1: Add the dependencies**

Append to `api/requirements.txt`:

```
# Termbase persistence (TG_GLOSSARY_ENABLED). SQLite is the right size for one
# gateway and one administrator; the store is written against a repository
# interface so a move to PostgreSQL, required the moment there are replicas,
# does not touch the matching code.
sqlalchemy[asyncio]>=2.0,<3.0
aiosqlite>=0.20,<1.0
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_glossary_store.py`:

```python
"""Persistence, scoping and the layering of a domain over the global termbase."""

import pytest

from glossary.store import GlossaryStore


@pytest.fixture
async def store():
    store = GlossaryStore("sqlite+aiosqlite:///:memory:")
    await store.create_all()
    yield store
    await store.aclose()


async def test_load_terms_returns_global_entries_when_no_domain_is_given(store):
    await store.create_entry(
        domain_name=None, src_lang="en", tgt_lang="fa", source_term="genome", target_term="X"
    )
    terms, version = await store.load_terms("en", "fa", None)
    assert [term.source_term for term in terms] == ["genome"]
    assert version >= 1


async def test_a_domain_layers_on_top_of_global(store):
    await store.create_domain("medical", "en", "fa", "Clinical terms")
    await store.create_entry(
        domain_name=None, src_lang="en", tgt_lang="fa", source_term="genome", target_term="G"
    )
    await store.create_entry(
        domain_name="medical", src_lang="en", tgt_lang="fa", source_term="lesion", target_term="L"
    )
    terms, _ = await store.load_terms("en", "fa", "medical")
    assert sorted(term.source_term for term in terms) == ["genome", "lesion"]


async def test_a_domain_entry_shadows_a_global_entry_for_the_same_term(store):
    await store.create_domain("medical", "en", "fa", None)
    await store.create_entry(
        domain_name=None, src_lang="en", tgt_lang="fa", source_term="culture", target_term="GLOBAL"
    )
    await store.create_entry(
        domain_name="medical", src_lang="en", tgt_lang="fa", source_term="culture", target_term="DOMAIN"
    )
    terms, _ = await store.load_terms("en", "fa", "medical")
    assert len(terms) == 1
    assert terms[0].target_term == "DOMAIN"


async def test_entries_of_another_language_pair_are_not_loaded(store):
    await store.create_entry(
        domain_name=None, src_lang="de", tgt_lang="fr", source_term="genom", target_term="X"
    )
    terms, _ = await store.load_terms("en", "fa", None)
    assert terms == []


async def test_disabled_entries_are_not_loaded(store):
    await store.create_entry(
        domain_name=None, src_lang="en", tgt_lang="fa", source_term="genome",
        target_term="X", enabled=False,
    )
    terms, _ = await store.load_terms("en", "fa", None)
    assert terms == []


async def test_duplicate_source_term_in_the_same_scope_is_rejected(store):
    await store.create_entry(
        domain_name=None, src_lang="en", tgt_lang="fa", source_term="genome", target_term="X"
    )
    with pytest.raises(ValueError, match="already exists"):
        await store.create_entry(
            domain_name=None, src_lang="en", tgt_lang="fa", source_term="genome", target_term="Y"
        )


async def test_creating_an_entry_in_an_unknown_domain_is_rejected(store):
    with pytest.raises(ValueError, match="No such domain"):
        await store.create_entry(
            domain_name="nope", src_lang="en", tgt_lang="fa", source_term="x", target_term="y"
        )


async def test_version_increases_on_every_write(store):
    _, first = await store.load_terms("en", "fa", None)
    await store.create_entry(
        domain_name=None, src_lang="en", tgt_lang="fa", source_term="genome", target_term="X"
    )
    _, second = await store.load_terms("en", "fa", None)
    assert second > first
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `scripts/test_glossary.sh tests/test_glossary_store.py -v`
Expected: FAIL — no module `glossary.store`.

- [ ] **Step 4: Implement the models**

Create `api/glossary/models.py`:

```python
"""Termbase tables.

A domain is a row rather than a free string on the entry: it gives a request's
`domain` a stable identity, an existence check that turns a caller's typo into
a 404 instead of a silently general translation, and somewhere to hang a
per-termbase version.

Global entries are `domain_id IS NULL` rather than a reserved row, so "the
global layer always applies" is a property of the query and cannot be switched
off or deleted by an admin action.
"""

from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Domain(Base):
    __tablename__ = "glossary_domain"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    src_lang: Mapped[str] = mapped_column(String(16))
    tgt_lang: Mapped[str] = mapped_column(String(16))
    description: Mapped[str | None] = mapped_column(String(512), default=None)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )


class Entry(Base):
    __tablename__ = "glossary_entry"
    __table_args__ = (
        UniqueConstraint(
            "domain_id",
            "src_lang",
            "tgt_lang",
            "source_term",
            "case_sensitive",
            name="uq_glossary_entry_scope",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    domain_id: Mapped[int | None] = mapped_column(
        ForeignKey("glossary_domain.id", ondelete="CASCADE"), default=None, index=True
    )
    src_lang: Mapped[str] = mapped_column(String(16), index=True)
    tgt_lang: Mapped[str] = mapped_column(String(16), index=True)
    source_term: Mapped[str] = mapped_column(String(512))
    target_term: Mapped[str] = mapped_column(String(512))
    # Only 'preferred' is accepted until sentinel protection ships. The column
    # exists now so that phase is additive rather than a migration.
    target_mode: Mapped[str] = mapped_column(String(16), default="preferred")
    aliases: Mapped[list] = mapped_column(JSON, default=list)
    forbidden: Mapped[list] = mapped_column(JSON, default=list)
    case_sensitive: Mapped[bool] = mapped_column(Boolean, default=False)
    whole_word: Mapped[bool] = mapped_column(Boolean, default=True)
    priority: Mapped[int] = mapped_column(Integer, default=0)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    notes: Mapped[str | None] = mapped_column(String(1024), default=None)
    created_by: Mapped[str | None] = mapped_column(String(128), default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )


class Revision(Base):
    """A single monotonic counter, bumped on every write.

    A request stamps its response with the value it read, so a served
    translation can be attributed to the exact termbase that produced it.
    """

    __tablename__ = "glossary_revision"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
```

- [ ] **Step 5: Implement the store**

Create `api/glossary/store.py`:

```python
"""Async persistence for the termbase.

Nothing here runs during a translation: `load_terms` is called when a snapshot
is built, and the snapshot answers requests. Querying the database per term per
request is the thing this design exists to avoid.
"""

from sqlalchemy import delete, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from .matcher import Term
from .models import Base, Domain, Entry, Revision


class GlossaryStore:
    def __init__(self, database_url: str):
        self._engine = create_async_engine(database_url, future=True)
        self._session = async_sessionmaker(self._engine, expire_on_commit=False)

    async def create_all(self) -> None:
        async with self._engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        async with self._session() as session:
            existing = await session.scalar(select(Revision).limit(1))
            if existing is None:
                session.add(Revision(id=1, version=1))
                await session.commit()

    async def aclose(self) -> None:
        await self._engine.dispose()

    # ---------------------------------------------------------------- version

    async def _bump(self, session: AsyncSession) -> None:
        await session.execute(update(Revision).values(version=Revision.version + 1))

    async def version(self) -> int:
        async with self._session() as session:
            return await session.scalar(select(Revision.version).limit(1)) or 1

    # ---------------------------------------------------------------- domains

    async def list_domains(self) -> list[Domain]:
        async with self._session() as session:
            result = await session.scalars(select(Domain).order_by(Domain.name))
            return list(result)

    async def get_domain(self, name: str) -> Domain | None:
        async with self._session() as session:
            return await session.scalar(select(Domain).where(Domain.name == name))

    async def create_domain(
        self, name: str, src_lang: str, tgt_lang: str, description: str | None = None
    ) -> Domain:
        async with self._session() as session:
            domain = Domain(
                name=name, src_lang=src_lang, tgt_lang=tgt_lang, description=description
            )
            session.add(domain)
            try:
                await self._bump(session)
                await session.commit()
            except IntegrityError as error:
                await session.rollback()
                raise ValueError(f"Domain {name!r} already exists.") from error
            return domain

    async def delete_domain(self, name: str) -> bool:
        async with self._session() as session:
            result = await session.execute(delete(Domain).where(Domain.name == name))
            await self._bump(session)
            await session.commit()
            return result.rowcount > 0

    # ---------------------------------------------------------------- entries

    async def list_entries(self, domain_name: str | None = None) -> list[Entry]:
        async with self._session() as session:
            statement = select(Entry).order_by(Entry.source_term)
            if domain_name is not None:
                domain = await session.scalar(select(Domain).where(Domain.name == domain_name))
                if domain is None:
                    raise ValueError(f"No such domain: {domain_name!r}")
                statement = statement.where(Entry.domain_id == domain.id)
            result = await session.scalars(statement)
            return list(result)

    async def create_entry(self, domain_name: str | None = None, **fields) -> Entry:
        async with self._session() as session:
            domain_id = None
            if domain_name is not None:
                domain = await session.scalar(select(Domain).where(Domain.name == domain_name))
                if domain is None:
                    raise ValueError(f"No such domain: {domain_name!r}")
                domain_id = domain.id
            entry = Entry(domain_id=domain_id, **fields)
            session.add(entry)
            try:
                await self._bump(session)
                await session.commit()
            except IntegrityError as error:
                await session.rollback()
                raise ValueError(
                    f"An entry for {fields.get('source_term')!r} already exists in this scope."
                ) from error
            return entry

    async def delete_entry(self, entry_id: int) -> bool:
        async with self._session() as session:
            result = await session.execute(delete(Entry).where(Entry.id == entry_id))
            await self._bump(session)
            await session.commit()
            return result.rowcount > 0

    # ----------------------------------------------------------------- terms

    async def load_terms(
        self, src_lang: str, tgt_lang: str, domain_name: str | None
    ) -> tuple[list[Term], int]:
        """Global entries, with a domain's entries layered over them.

        A domain entry shadows a global entry with the same source term: the
        more specific termbase wins, and the global one is not also applied.
        """
        async with self._session() as session:
            version = await session.scalar(select(Revision.version).limit(1)) or 1

            domain_id = None
            if domain_name is not None:
                domain = await session.scalar(
                    select(Domain).where(Domain.name == domain_name, Domain.enabled.is_(True))
                )
                if domain is None:
                    raise ValueError(f"No such domain: {domain_name!r}")
                domain_id = domain.id

            statement = select(Entry).where(
                Entry.src_lang == src_lang,
                Entry.tgt_lang == tgt_lang,
                Entry.enabled.is_(True),
            )
            rows = list(await session.scalars(statement))

        by_source: dict[str, Entry] = {}
        for entry in rows:
            if entry.domain_id is not None and entry.domain_id != domain_id:
                continue
            key = entry.source_term.lower()
            existing = by_source.get(key)
            # A domain row replaces a global row for the same term.
            if existing is None or (existing.domain_id is None and entry.domain_id is not None):
                by_source[key] = entry

        terms = [
            Term(
                entry_id=entry.id,
                source_term=entry.source_term,
                target_term=entry.target_term,
                target_mode=entry.target_mode,
                aliases=tuple(entry.aliases or ()),
                forbidden=tuple(entry.forbidden or ()),
                case_sensitive=entry.case_sensitive,
                whole_word=entry.whole_word,
                priority=entry.priority,
            )
            for entry in by_source.values()
        ]
        return terms, version
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `scripts/test_glossary.sh tests/test_glossary_store.py -v`
Expected: PASS, 8 tests.

- [ ] **Step 7: Commit**

```bash
git add api/glossary/models.py api/glossary/store.py api/requirements.txt tests/test_glossary_store.py
git commit -m "feat(api): add glossary persistence"
```

---

### Task 6: The service — snapshots, domain resolution, the seam

**Files:**
- Create: `api/glossary/service.py`
- Test: `tests/test_glossary_service.py`

**Interfaces:**
- Consumes: `GlossaryStore`, `build_index`, `find_spans`, `apply_preferred`, `Report`.
- Produces: `UnknownDomainError(Exception)` with attribute `available: list[str]`;
  `GlossaryService` with `async start()`, `async aclose()`, `async reload()`,
  `async resolve(src_lang, tgt_lang, domain: str | None) -> Index`,
  `plan(index, text) -> list[Span]`,
  `apply(translation, spans) -> tuple[str, Report]`, and property `version: int`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_glossary_service.py`:

```python
"""Snapshot caching, domain resolution, and the unknown-domain policy."""

import pytest

from glossary.service import GlossaryService, UnknownDomainError


def build(**overrides):
    options = {
        "database_url": "sqlite+aiosqlite:///:memory:",
        "default_domain": None,
        "unknown_domain": "reject",
    }
    options.update(overrides)
    return GlossaryService(**options)


@pytest.fixture
async def service():
    service = build()
    await service.start()
    yield service
    await service.aclose()


async def test_omitting_a_domain_resolves_the_global_layer(service):
    await service.store.create_entry(
        domain_name=None, src_lang="en", tgt_lang="fa", source_term="genome", target_term="X"
    )
    await service.reload()
    index = await service.resolve("en", "fa", None)
    assert [term.source_term for term in index.terms] == ["genome"]


async def test_an_unknown_domain_is_rejected_and_lists_what_exists(service):
    await service.store.create_domain("medical", "en", "fa", None)
    with pytest.raises(UnknownDomainError) as error:
        await service.resolve("en", "fa", "medcial")
    assert error.value.available == ["medical"]


async def test_fallback_policy_uses_the_global_layer_for_an_unknown_domain():
    service = build(unknown_domain="fallback")
    await service.start()
    try:
        index = await service.resolve("en", "fa", "nope")
        assert index.terms == ()
    finally:
        await service.aclose()


async def test_the_default_domain_applies_when_the_request_omits_one():
    service = build(default_domain="medical")
    await service.start()
    try:
        await service.store.create_domain("medical", "en", "fa", None)
        await service.store.create_entry(
            domain_name="medical", src_lang="en", tgt_lang="fa",
            source_term="lesion", target_term="L",
        )
        await service.reload()
        index = await service.resolve("en", "fa", None)
        assert [term.source_term for term in index.terms] == ["lesion"]
    finally:
        await service.aclose()


async def test_snapshots_are_cached_and_reload_replaces_them(service):
    first = await service.resolve("en", "fa", None)
    assert await service.resolve("en", "fa", None) is first
    await service.store.create_entry(
        domain_name=None, src_lang="en", tgt_lang="fa", source_term="genome", target_term="X"
    )
    await service.reload()
    assert await service.resolve("en", "fa", None) is not first


async def test_plan_and_apply_round_trip(service):
    await service.store.create_entry(
        domain_name=None, src_lang="en", tgt_lang="fa",
        source_term="genome", target_term="ژنوم",
        aliases=["گنوم"],
    )
    await service.reload()
    index = await service.resolve("en", "fa", None)
    spans = service.plan(index, "The genome sequence.")
    assert len(spans) == 1
    result, report = service.apply("توالی گنوم.", spans)
    assert "ژنوم" in result
    assert report.applied[0].count == 1
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `scripts/test_glossary.sh tests/test_glossary_service.py -v`
Expected: FAIL — no module `glossary.service`.

- [ ] **Step 3: Implement the service**

Create `api/glossary/service.py`:

```python
"""The glossary's public face: one object the request path talks to.

Snapshots are built per (src_lang, tgt_lang, domain) and cached. They are
immutable, and a reload replaces the cache dictionary wholesale rather than
mutating it, so a request that already holds a snapshot finishes against the
termbase it started with and readers need no lock.
"""

from collections.abc import Sequence

from .apply import Report, apply_preferred
from .matcher import Index, Span, build_index, find_spans
from .store import GlossaryStore


class UnknownDomainError(Exception):
    """A request named a domain that does not exist.

    Carries the valid names so the API can say what the caller could have
    meant. A typo silently answered from the global termbase is the failure
    this whole layer exists to prevent.
    """

    def __init__(self, name: str, available: list[str]):
        super().__init__(f"No such glossary domain: {name!r}. Available: {available}")
        self.name = name
        self.available = available


class GlossaryService:
    def __init__(
        self,
        database_url: str,
        default_domain: str | None = None,
        unknown_domain: str = "reject",
    ):
        self.store = GlossaryStore(database_url)
        self._default_domain = default_domain
        self._unknown_domain = unknown_domain
        self._snapshots: dict[tuple[str, str, str | None], Index] = {}
        self._version = 1

    async def start(self) -> None:
        await self.store.create_all()
        self._version = await self.store.version()

    async def aclose(self) -> None:
        await self.store.aclose()

    @property
    def version(self) -> int:
        return self._version

    async def reload(self) -> int:
        """Drop every cached snapshot. The next request rebuilds what it needs."""
        self._snapshots = {}
        self._version = await self.store.version()
        return self._version

    async def resolve(self, src_lang: str, tgt_lang: str, domain: str | None) -> Index:
        """The snapshot for this request, building and caching it on first use."""
        name = domain if domain is not None else self._default_domain
        key = (src_lang, tgt_lang, name)
        cached = self._snapshots.get(key)
        if cached is not None:
            return cached

        try:
            terms, version = await self.store.load_terms(src_lang, tgt_lang, name)
        except ValueError:
            if self._unknown_domain == "fallback":
                terms, version = await self.store.load_terms(src_lang, tgt_lang, None)
            else:
                available = [row.name for row in await self.store.list_domains()]
                raise UnknownDomainError(name or "", available) from None

        index = build_index(terms, version)
        # Rebind rather than mutate: a concurrent reader either sees the old
        # dictionary or the new one, never a half-updated one.
        self._snapshots = {**self._snapshots, key: index}
        self._version = version
        return index

    def plan(self, index: Index, text: str) -> list[Span]:
        return find_spans(index, text)

    def apply(self, translation: str, spans: Sequence[Span]) -> tuple[str, Report]:
        return apply_preferred(translation, spans)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `scripts/test_glossary.sh tests/test_glossary_service.py -v`
Expected: PASS, 6 tests.

- [ ] **Step 5: Commit**

```bash
git add api/glossary/service.py tests/test_glossary_service.py
git commit -m "feat(api): add glossary service and snapshot cache"
```

---

### Task 7: Wire the seam into translation

The one place the glossary touches the request path. Matching happens on the
full text **before** sentence splitting, so a term straddling a sentence
boundary is not lost.

**Files:**
- Modify: `api/translator.py` (`TranslationEngine.__init__`, `translate`)
- Modify: `api/schemas.py`
- Modify: `api/main.py`
- Test: `tests/test_glossary_seam.py`

**Interfaces:**
- Consumes: `GlossaryService`, `Report`, `Index`.
- Produces: `translator.TranslationResult` (frozen dataclass:
  `translations: list[str]`, `reports: list[Report | None]`);
  `TranslationEngine.translate(...)` now returns `TranslationResult` and accepts
  a keyword-only `glossary_index: Index | None = None`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_glossary_seam.py`:

```python
"""The single application seam, and the guarantee that disabling it is inert."""

import pytest

from glossary.matcher import Term, build_index
from translator import TranslationEngine, TranslationResult


SYSTEM = "adapter"


class FakeSettings:
    """Only what TranslationEngine touches; no vLLM, no tokenizer."""

    max_concurrent_requests = 4
    batch_size = 8
    split_sentences = False
    served_system = SYSTEM


class RecordingEngine(TranslationEngine):
    """A TranslationEngine whose upstream is a recorded list of canned outputs."""

    def __init__(self, outputs):
        super().__init__(FakeSettings())
        self._outputs = outputs
        self.sent: list[list[str]] = []

    @property
    def is_loaded(self):
        return True

    async def _generate(self, segments, system, source_lang, target_lang, max_new_tokens):
        self.sent.append(list(segments))
        return list(self._outputs)


async def test_without_a_glossary_the_result_carries_no_reports():
    engine = RecordingEngine(["ترجمه"])
    result = await engine.translate(
        ["The genome."], SYSTEM, "en", "fa", 128, False, glossary_index=None
    )
    assert isinstance(result, TranslationResult)
    assert result.translations == ["ترجمه"]
    assert result.reports == [None]


async def test_the_text_sent_upstream_is_unchanged_by_a_glossary():
    # preferred mode repairs output; it must never alter the source text, which
    # is what keeps it incapable of regressing translation quality.
    index = build_index(
        [
            Term(
                entry_id=1, source_term="genome", target_term="ژنوم",
                target_mode="preferred", aliases=("گنوم",), forbidden=(),
                case_sensitive=False, whole_word=True, priority=0,
            )
        ],
        version=3,
    )
    engine = RecordingEngine(["گنوم است."])
    await engine.translate(
        ["The genome."], SYSTEM, "en", "fa", 128, False, glossary_index=index
    )
    assert engine.sent == [["The genome."]]


async def test_an_alias_in_the_output_is_rewritten_and_reported():
    index = build_index(
        [
            Term(
                entry_id=1, source_term="genome", target_term="ژنوم",
                target_mode="preferred", aliases=("گنوم",), forbidden=(),
                case_sensitive=False, whole_word=True, priority=0,
            )
        ],
        version=3,
    )
    engine = RecordingEngine(["گنوم است."])
    result = await engine.translate(
        ["The genome."], SYSTEM, "en", "fa", 128, False, glossary_index=index
    )
    assert "ژنوم" in result.translations[0]
    assert result.reports[0].applied[0].count == 1
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `scripts/test_glossary.sh tests/test_glossary_seam.py -v`
Expected: FAIL — cannot import `TranslationResult`.

- [ ] **Step 3: Change the engine**

In `api/translator.py`, add to the imports:

```python
from dataclasses import dataclass

from glossary.apply import Report
from glossary.matcher import Index, Span
from glossary.service import GlossaryService  # noqa: F401  (re-exported for main.py)
```

Add above `class TranslationEngine`:

```python
@dataclass(frozen=True)
class TranslationResult:
    """Translations plus, per text, what the glossary did — or None when off.

    A result object rather than a bare list because the glossary's report has to
    reach the response, and threading it through a second return value would put
    the two out of step on the first refactor.
    """

    translations: list[str]
    reports: list["Report | None"]
```

Replace the body of `TranslationEngine.translate` with:

```python
    async def translate(
        self,
        texts: list[str],
        system: System,
        source_lang: str,
        target_lang: str,
        max_new_tokens: int,
        split_sentences: bool,
        *,
        glossary_index: Index | None = None,
    ) -> TranslationResult:
        """Translate texts, preserving order. One output per input.

        Glossary matching runs on the FULL text before splitting: a term
        straddling a sentence boundary would otherwise be invisible to every
        segment. The spans are then applied to the rejoined translation.
        """
        if not self.is_loaded:
            raise RuntimeError("Gateway is not ready.")
        if system is not self.settings.served_system:
            raise ValueError(
                f"System {system!r} is not what this upstream serves "
                f"({self.settings.served_system})."
            )

        spans_per_text: list[list[Span]] = [
            [] if glossary_index is None else find_spans(glossary_index, text) for text in texts
        ]

        if split_sentences:
            segments_per_text = [self.splitter.split(text, source_lang) for text in texts]
        else:
            segments_per_text = [[text] for text in texts]

        flat_segments = [segment for segments in segments_per_text for segment in segments]
        flat_translations = await self._generate(
            flat_segments, system, source_lang, target_lang, max_new_tokens
        )

        translations: list[str] = []
        reports: list[Report | None] = []
        cursor = 0
        for text_index, segments in enumerate(segments_per_text):
            chunk = flat_translations[cursor : cursor + len(segments)]
            cursor += len(segments)
            joined = " ".join(part for part in chunk if part)
            if glossary_index is None:
                translations.append(joined)
                reports.append(None)
                continue
            rewritten, report = apply_preferred(joined, spans_per_text[text_index])
            translations.append(rewritten)
            reports.append(report)
        return TranslationResult(translations=translations, reports=reports)
```

Add to the imports at the top of `api/translator.py`:

```python
from glossary.apply import apply_preferred
from glossary.matcher import find_spans
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `scripts/test_glossary.sh tests/test_glossary_seam.py -v`
Expected: PASS, 3 tests.

- [ ] **Step 5: Update the response schemas**

In `api/schemas.py`, add above `class TranslationResponse`:

```python
class AppliedTerm(BaseModel):
    source_term: str
    target_term: str
    mode: str
    count: int


class MissedTerm(BaseModel):
    source_term: str
    reason: str


class ForbiddenTerm(BaseModel):
    source_term: str
    forbidden: str


class GlossaryReport(BaseModel):
    """What the termbase did to one translation.

    Omitted entirely when the glossary is disabled, so a caller written against
    a deployment without the feature keeps working when it is switched on.
    """

    status: str
    applied: list[AppliedTerm] = []
    misses: list[MissedTerm] = []
    violations: list[ForbiddenTerm] = []
```

Add to `TranslationOptions`:

```python
    domain: str | None = Field(
        default=None,
        description=(
            "Termbase to layer over the global one. Omitting it is the normal "
            "case and applies the global termbase; an unknown name is a 404. "
            "Ignored when the glossary is disabled."
        ),
    )
    terminology_mode: str | None = Field(
        default=None,
        description="'off' or 'enforce'. Defaults to TG_TERMINOLOGY_MODE.",
    )
```

Add to `TranslationResponse`:

```python
    # Absent, not null, when the glossary is disabled.
    raw_translation: str | None = None
    glossary_version: int | None = None
    glossary: GlossaryReport | None = None
```

Add to `BatchTranslationResponse`:

```python
    raw_translations: list[str] | None = None
    glossary_version: int | None = None
    glossary: list[GlossaryReport] | None = None
```

- [ ] **Step 6: Wire main.py**

In `api/main.py`, add imports:

```python
from glossary.serializers import to_report
from glossary.service import GlossaryService, UnknownDomainError
```

In `lifespan`, after `app.state.engine = engine`:

```python
    glossary = None
    if settings.glossary_enabled:
        glossary = GlossaryService(
            database_url=settings.glossary_db_url,
            default_domain=settings.glossary_default_domain,
            unknown_domain=settings.glossary_unknown_domain,
        )
        await glossary.start()
        logger.info("Glossary enabled: %s", settings.glossary_db_url)
    app.state.glossary = glossary
```

and in the `finally` block, before `await engine.aclose()`:

```python
        if glossary is not None:
            await glossary.aclose()
```

Add after `get_engine`:

```python
def get_glossary() -> GlossaryService | None:
    """None when the feature is off. Callers must not branch on settings instead."""
    return getattr(app.state, "glossary", None)


async def _resolve_index(glossary, prompt, resolved, settings):
    """The snapshot for this request, or None when the glossary does not apply."""
    if glossary is None:
        return None
    mode = prompt.terminology_mode or settings.terminology_mode
    if mode == "off":
        return None
    try:
        return await glossary.resolve(
            resolved.source_lang, resolved.target_lang, prompt.domain
        )
    except UnknownDomainError as error:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"Unknown glossary domain {error.name!r}. Available: {error.available}",
        ) from error
```

Replace `_translate` with:

```python
async def _translate(engine, texts, resolved, glossary_index=None):
    return await engine.translate(
        texts,
        resolved.system,
        resolved.source_lang,
        resolved.target_lang,
        resolved.max_new_tokens,
        resolved.split_sentences,
        glossary_index=glossary_index,
    )
```

In `health_check`, change the translation check to read from the result object:

```python
        result = await _translate(engine, ["Hello."], resolved)
        translations = result.translations
```

Replace the body of `translate` after `resolved = _resolve(prompt, settings)`:

```python
    glossary = get_glossary()
    index = await _resolve_index(glossary, prompt, resolved, settings)
    result = await _translate(engine, [text], resolved, index)
    response = TranslationResponse(
        translation=result.translations[0],
        system=resolved.system,
        source_lang=resolved.source_lang,
        target_lang=resolved.target_lang,
    )
    if index is not None:
        response.raw_translation = result.translations[0]
        response.glossary_version = index.version
        response.glossary = to_report(result.reports[0])
    return response
```

Replace the body of `translate_batch` after `resolved = _resolve(prompt, settings)`:

```python
    glossary = get_glossary()
    index = await _resolve_index(glossary, prompt, resolved, settings)
    result = await _translate(engine, texts, resolved, index)
    response = BatchTranslationResponse(
        translations=result.translations,
        system=resolved.system,
        source_lang=resolved.source_lang,
        target_lang=resolved.target_lang,
    )
    if index is not None:
        response.glossary_version = index.version
        response.glossary = [to_report(report) for report in result.reports]
    return response
```


- [ ] **Step 7: Add the report converter**

Create `api/glossary/serializers.py` (deliberately not `schemas.py`: it imports
the gateway's top-level `schemas` module, and two modules of that name on
`sys.path` is a confusion waiting to happen):

```python
"""Convert the pure-dataclass Report into the API's Pydantic shape.

Kept out of apply.py on purpose: the matching and rewriting code has no
dependency on FastAPI or Pydantic, which is what lets it be tested without the
web stack.
"""

from schemas import AppliedTerm, ForbiddenTerm, GlossaryReport, MissedTerm

from .apply import Report


def to_report(report: Report | None) -> GlossaryReport:
    if report is None or report.is_empty:
        return GlossaryReport(status="no_match")
    if report.misses and not report.applied:
        status = "enforcement_failed"
    elif report.misses:
        status = "partial"
    else:
        status = "applied"
    return GlossaryReport(
        status=status,
        applied=[
            AppliedTerm(
                source_term=item.source_term,
                target_term=item.target_term,
                mode=item.mode,
                count=item.count,
            )
            for item in report.applied
        ],
        misses=[
            MissedTerm(source_term=item.source_term, reason=item.reason)
            for item in report.misses
        ],
        violations=[
            ForbiddenTerm(source_term=item.source_term, forbidden=item.forbidden)
            for item in report.violations
        ],
    )
```

- [ ] **Step 8: Run the whole suite**

Run: `scripts/test_glossary.sh tests/ -v`
Expected: PASS, including the pre-existing tests.

- [ ] **Step 9: Commit**

```bash
git add api/translator.py api/main.py api/schemas.py api/glossary/serializers.py tests/test_glossary_seam.py
git commit -m "feat(api): apply the glossary in the translation path"
```

---

### Task 8: Admin API

**Files:**
- Create: `api/glossary/router.py`
- Modify: `api/main.py` (mount conditionally)
- Test: `tests/test_glossary_admin.py`

**Interfaces:**
- Consumes: `GlossaryService`, `GlossaryStore`.
- Produces: `build_router(service: GlossaryService, api_key: str) -> APIRouter`
  mounted at `/admin/glossary`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_glossary_admin.py`:

```python
"""Admin CRUD, its authorization boundary, and the dry-run endpoint."""

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from glossary.router import build_router
from glossary.service import GlossaryService

KEY = "test-key"
HEADERS = {"X-Admin-Key": KEY}


@pytest.fixture
async def client():
    service = GlossaryService(database_url="sqlite+aiosqlite:///:memory:")
    await service.start()
    app = FastAPI()
    app.include_router(build_router(service, KEY))
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        client.service = service
        yield client
    await service.aclose()


async def test_requests_without_the_key_are_rejected(client):
    response = await client.get("/admin/glossary/entries")
    assert response.status_code == 401


async def test_requests_with_a_wrong_key_are_rejected(client):
    response = await client.get("/admin/glossary/entries", headers={"X-Admin-Key": "wrong"})
    assert response.status_code == 401


async def test_create_and_list_an_entry(client):
    created = await client.post(
        "/admin/glossary/entries",
        headers=HEADERS,
        json={
            "src_lang": "en", "tgt_lang": "fa",
            "source_term": "genome", "target_term": "ژنوم",
        },
    )
    assert created.status_code == 201
    listed = await client.get("/admin/glossary/entries", headers=HEADERS)
    assert [item["source_term"] for item in listed.json()] == ["genome"]


async def test_a_stopword_entry_is_refused(client):
    response = await client.post(
        "/admin/glossary/entries",
        headers=HEADERS,
        json={"src_lang": "en", "tgt_lang": "fa", "source_term": "the", "target_term": "X"},
    )
    assert response.status_code == 422
    assert "stopword" in response.text


async def test_a_multi_word_phrase_containing_a_stopword_is_allowed(client):
    response = await client.post(
        "/admin/glossary/entries",
        headers=HEADERS,
        json={
            "src_lang": "en", "tgt_lang": "fa",
            "source_term": "the cloud", "target_term": "X",
        },
    )
    assert response.status_code == 201


async def test_exact_mode_is_refused_until_phase_two(client):
    response = await client.post(
        "/admin/glossary/entries",
        headers=HEADERS,
        json={
            "src_lang": "en", "tgt_lang": "fa", "source_term": "wordomatic",
            "target_term": "wordomatic", "target_mode": "exact",
        },
    )
    assert response.status_code == 422


async def test_duplicate_entries_conflict(client):
    payload = {
        "src_lang": "en", "tgt_lang": "fa", "source_term": "genome", "target_term": "X",
    }
    assert (await client.post("/admin/glossary/entries", headers=HEADERS, json=payload)).status_code == 201
    duplicate = await client.post("/admin/glossary/entries", headers=HEADERS, json=payload)
    assert duplicate.status_code == 409


async def test_dry_run_reports_matches_without_translating(client):
    await client.post(
        "/admin/glossary/entries",
        headers=HEADERS,
        json={
            "src_lang": "en", "tgt_lang": "fa",
            "source_term": "genome", "target_term": "ژنوم",
        },
    )
    response = await client.post(
        "/admin/glossary/dry-run",
        headers=HEADERS,
        json={"text": "The genome sequence.", "src_lang": "en", "tgt_lang": "fa"},
    )
    assert response.status_code == 200
    matches = response.json()["matches"]
    assert matches[0]["source_term"] == "genome"
    assert matches[0]["start"] == 4


async def test_creating_a_domain_and_listing_it(client):
    created = await client.post(
        "/admin/glossary/domains",
        headers=HEADERS,
        json={"name": "medical", "src_lang": "en", "tgt_lang": "fa"},
    )
    assert created.status_code == 201
    listed = await client.get("/admin/glossary/domains", headers=HEADERS)
    assert [item["name"] for item in listed.json()] == ["medical"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `scripts/test_glossary.sh tests/test_glossary_admin.py -v`
Expected: FAIL — no module `glossary.router`.

- [ ] **Step 3: Implement the router**

Create `api/glossary/router.py`:

```python
"""Admin routes for the termbase.

Mounted only when the glossary is enabled, so on a default deployment these
paths are 404 rather than 401 -- there is nothing there to authorize against.

The static key is the whole authorization boundary for this gateway: there is
no reverse-proxy auth in front of it. CORS is not authorization and plays no
part here.
"""

import secrets

from fastapi import APIRouter, Body, Depends, Header, HTTPException, status
from pydantic import BaseModel, Field

from .service import GlossaryService, UnknownDomainError

# Entries matching one of these exactly are refused. Google ignores such
# entries silently; refusing loudly is better, because an entry that is
# accepted and then never fires looks like a broken feature. Exact match only:
# "the" is refused, "the cloud" is not.
STOPWORDS = frozenset(
    """
    a about an and are as at be by com for from how i in is it of on or that
    the this to was what when where who will with www edu
    """.split()
)

# Widened when sentinel protection ships (phase 2 of the design document).
SUPPORTED_TARGET_MODES = frozenset({"preferred"})


class DomainIn(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    src_lang: str
    tgt_lang: str
    description: str | None = None


class EntryIn(BaseModel):
    src_lang: str
    tgt_lang: str
    source_term: str = Field(min_length=1, max_length=512)
    target_term: str = Field(min_length=1, max_length=512)
    domain: str | None = None
    target_mode: str = "preferred"
    aliases: list[str] = []
    forbidden: list[str] = []
    case_sensitive: bool = False
    whole_word: bool = True
    priority: int = 0
    notes: str | None = None
    created_by: str | None = None


class DryRunIn(BaseModel):
    text: str
    src_lang: str
    tgt_lang: str
    domain: str | None = None


def build_router(service: GlossaryService, api_key: str) -> APIRouter:
    async def require_key(x_admin_key: str | None = Header(default=None)) -> None:
        # compare_digest so a wrong key costs the same time as a right one.
        if x_admin_key is None or not secrets.compare_digest(x_admin_key, api_key):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing admin key.")

    router = APIRouter(
        prefix="/admin/glossary", tags=["glossary-admin"], dependencies=[Depends(require_key)]
    )

    @router.get("/domains")
    async def list_domains():
        return [
            {
                "name": domain.name,
                "src_lang": domain.src_lang,
                "tgt_lang": domain.tgt_lang,
                "description": domain.description,
                "enabled": domain.enabled,
            }
            for domain in await service.store.list_domains()
        ]

    @router.post("/domains", status_code=status.HTTP_201_CREATED)
    async def create_domain(payload: DomainIn):
        try:
            domain = await service.store.create_domain(
                payload.name, payload.src_lang, payload.tgt_lang, payload.description
            )
        except ValueError as error:
            raise HTTPException(status.HTTP_409_CONFLICT, str(error)) from error
        await service.reload()
        return {"name": domain.name}

    @router.delete("/domains/{name}")
    async def delete_domain(name: str):
        if not await service.store.delete_domain(name):
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"No such domain: {name!r}")
        await service.reload()
        return {"deleted": name}

    @router.get("/entries")
    async def list_entries(domain: str | None = None):
        try:
            entries = await service.store.list_entries(domain)
        except ValueError as error:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(error)) from error
        return [
            {
                "id": entry.id,
                "source_term": entry.source_term,
                "target_term": entry.target_term,
                "target_mode": entry.target_mode,
                "aliases": entry.aliases,
                "forbidden": entry.forbidden,
                "enabled": entry.enabled,
            }
            for entry in entries
        ]

    @router.post("/entries", status_code=status.HTTP_201_CREATED)
    async def create_entry(payload: EntryIn):
        if payload.target_mode not in SUPPORTED_TARGET_MODES:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                f"target_mode {payload.target_mode!r} is not supported yet; "
                f"use one of {sorted(SUPPORTED_TARGET_MODES)}.",
            )
        if payload.source_term.strip().lower() in STOPWORDS:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                f"{payload.source_term!r} is a stopword and would never be applied. "
                "Multi-word phrases containing one are fine.",
            )
        try:
            entry = await service.store.create_entry(
                domain_name=payload.domain,
                src_lang=payload.src_lang,
                tgt_lang=payload.tgt_lang,
                source_term=payload.source_term,
                target_term=payload.target_term,
                target_mode=payload.target_mode,
                aliases=payload.aliases,
                forbidden=payload.forbidden,
                case_sensitive=payload.case_sensitive,
                whole_word=payload.whole_word,
                priority=payload.priority,
                notes=payload.notes,
                created_by=payload.created_by,
            )
        except ValueError as error:
            message = str(error)
            code = (
                status.HTTP_404_NOT_FOUND
                if "No such domain" in message
                else status.HTTP_409_CONFLICT
            )
            raise HTTPException(code, message) from error
        await service.reload()
        return {"id": entry.id}

    @router.delete("/entries/{entry_id}")
    async def delete_entry(entry_id: int):
        if not await service.store.delete_entry(entry_id):
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"No entry {entry_id}.")
        await service.reload()
        return {"deleted": entry_id}

    @router.post("/reload")
    async def reload():
        return {"version": await service.reload()}

    @router.post("/dry-run")
    async def dry_run(payload: DryRunIn = Body(...)):
        """Show what would fire on this text. No model call, no side effects."""
        try:
            index = await service.resolve(payload.src_lang, payload.tgt_lang, payload.domain)
        except UnknownDomainError as error:
            # An endpoint whose whole purpose is catching mistakes before they
            # reach traffic must not answer a typo'd domain with a 500.
            raise HTTPException(
                status.HTTP_404_NOT_FOUND,
                f"Unknown glossary domain {error.name!r}. Available: {error.available}",
            ) from error
        spans = service.plan(index, payload.text)
        return {
            "version": index.version,
            "matches": [
                {
                    "source_term": span.term.source_term,
                    "target_term": span.term.target_term,
                    "mode": span.term.target_mode,
                    "start": span.start,
                    "end": span.end,
                    "matched_text": payload.text[span.start : span.end],
                }
                for span in spans
            ],
        }

    return router
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `scripts/test_glossary.sh tests/test_glossary_admin.py -v`
Expected: PASS, 9 tests.

- [ ] **Step 5: Mount it conditionally**

In `api/main.py`, inside `lifespan` after `await glossary.start()`:

```python
        from glossary.router import build_router

        app.include_router(build_router(glossary, settings.admin_api_key))
```

- [ ] **Step 6: Commit**

```bash
git add api/glossary/router.py api/main.py tests/test_glossary_admin.py
git commit -m "feat(api): add glossary admin routes"
```

---

### Task 9: Prove the kill switch

The spec's load-bearing tests. These are what make the feature safe to merge.

**Files:**
- Test: `tests/test_glossary_disabled.py`

**Interfaces:**
- Consumes: everything above. Produces nothing.

- [ ] **Step 1: Write the test**

Create `tests/test_glossary_disabled.py`:

```python
"""With TG_GLOSSARY_ENABLED=false, the feature must be absent, not merely bypassed."""

from pathlib import Path

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from glossary.matcher import build_index
from translator import TranslationEngine


SYSTEM = "adapter"


class FakeSettings:
    max_concurrent_requests = 4
    batch_size = 8
    split_sentences = False
    served_system = SYSTEM


class RecordingEngine(TranslationEngine):
    def __init__(self, outputs):
        super().__init__(FakeSettings())
        self._outputs = outputs
        self.sent: list[list[str]] = []

    @property
    def is_loaded(self):
        return True

    async def _generate(self, segments, system, source_lang, target_lang, max_new_tokens):
        self.sent.append(list(segments))
        return list(self._outputs)


async def test_segments_sent_upstream_are_identical_with_and_without_a_glossary():
    # The guarantee the kill switch rests on: turning the feature off cannot
    # change a single token the model sees.
    texts = ["The genome sequence.", "A second sentence."]

    without = RecordingEngine(["الف", "ب"])
    await without.translate(texts, SYSTEM, "en", "fa", 128, False, glossary_index=None)

    with_empty = RecordingEngine(["الف", "ب"])
    await with_empty.translate(
        texts, SYSTEM, "en", "fa", 128, False, glossary_index=build_index([], version=1)
    )

    assert without.sent == with_empty.sent == [texts]


async def test_no_database_file_is_created_when_disabled(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from config import Settings

    settings = Settings(_env_file=None)
    assert settings.glossary_enabled is False
    # Nothing constructed the service, so nothing opened SQLite.
    assert list(Path(tmp_path).rglob("*.db")) == []


async def test_admin_routes_are_absent_not_unauthorized():
    # 404 rather than 401: with the feature off there is nothing to authorize.
    app = FastAPI()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/admin/glossary/entries")
    assert response.status_code == 404


@pytest.mark.parametrize("field", ["domain", "terminology_mode"])
def test_request_fields_are_accepted_when_the_feature_is_off(field):
    # A caller that learned to send these must not start failing when an
    # operator turns the glossary off mid-incident.
    from schemas import Prompt

    prompt = Prompt(text="hello", **{field: "anything"})
    assert getattr(prompt, field) == "anything"
```

- [ ] **Step 2: Run the test**

Run: `scripts/test_glossary.sh tests/test_glossary_disabled.py -v`
Expected: PASS, 5 tests.

- [ ] **Step 3: Run the entire suite**

Run: `scripts/test_glossary.sh tests/ -v`
Expected: PASS. No pre-existing test regresses — in particular
`tests/test_api_vendored_modules.py`, which fails if `api/prompting.py` was
touched.

- [ ] **Step 4: Commit**

```bash
git add tests/test_glossary_disabled.py
git commit -m "test(api): prove the glossary kill switch is inert"
```

---

### Task 10: Deployment and documentation

**Files:**
- Modify: `api/.env.example`
- Modify: `api/docker-compose.yml`
- Modify: `api/README.md`

**Interfaces:**
- Consumes: settings from Task 1. Produces nothing.

- [ ] **Step 1: Document the settings**

Append to `api/.env.example`:

```bash
# --- Glossary (optional; off by default) -------------------------------------
# Deterministic terminology. When false the feature is absent: no database is
# opened, no admin routes exist, and the tokens sent to vLLM are unchanged.
TG_GLOSSARY_ENABLED=false
# Must live on a mounted volume, never inside the container filesystem.
TG_GLOSSARY_DB_URL=sqlite+aiosqlite:////data/glossary.db
# Required when the glossary is enabled. The only authorization boundary on the
# admin routes; startup fails without it.
TG_ADMIN_API_KEY=
# Applied when a request omits `domain`. Lets a single-field deployment pin its
# termbase without callers knowing domains exist.
TG_GLOSSARY_DEFAULT_DOMAIN=
# reject: an unknown domain name is a 404. fallback: use the global termbase.
TG_GLOSSARY_UNKNOWN_DOMAIN=reject
TG_TERMINOLOGY_MODE=enforce
```

- [ ] **Step 2: Add the volume**

In `api/docker-compose.yml`, add to the API service's `volumes:` list:

```yaml
      # Termbase database. Outside the container filesystem so it survives a
      # rebuild; harmless when TG_GLOSSARY_ENABLED is false, since nothing
      # opens it.
      - glossary-data:/data
```

and add at the file's top level:

```yaml
volumes:
  glossary-data:
```

- [ ] **Step 3: Document the feature**

Add a section to `api/README.md`:

````markdown
## Glossary (deterministic terminology)

Off by default. Enable with `TG_GLOSSARY_ENABLED=true` and a `TG_ADMIN_API_KEY`;
the gateway refuses to start if the key is missing.

Design and measurements: `docs/2026-08-21_glossary_memory_design.md`.

Add a term:

```bash
curl -X POST localhost:8000/admin/glossary/entries \
  -H "X-Admin-Key: $TG_ADMIN_API_KEY" -H 'Content-Type: application/json' \
  -d '{"src_lang":"en","tgt_lang":"fa","source_term":"multi-query attention",
       "target_term":"توجه چندپرسشی","aliases":["توجه چندگانه"]}'
```

Check what would fire before it reaches traffic:

```bash
curl -X POST localhost:8000/admin/glossary/dry-run \
  -H "X-Admin-Key: $TG_ADMIN_API_KEY" -H 'Content-Type: application/json' \
  -d '{"text":"It uses multi-query attention.","src_lang":"en","tgt_lang":"fa"}'
```

`/translate` gains optional `domain` and `terminology_mode` fields, and returns
`glossary_version` plus a `glossary` report naming every term applied, missed,
or found in a forbidden rendering. Curate `aliases` from the misses: a term is
only rewritten when the model's own rendering is one the entry knows about.
````

- [ ] **Step 4: Verify the compose file parses**

Run: `docker compose -f api/docker-compose.yml config >/dev/null && echo OK`
Expected: `OK`.

- [ ] **Step 5: Commit**

```bash
git add api/.env.example api/docker-compose.yml api/README.md
git commit -m "docs(api): document the glossary feature and its deployment"
```

---

## After the plan

Enable against the live gateway and confirm end to end using the SSH-forwarded
vLLM on `127.0.0.1:40001`: add the `multi-query attention` entry with
`توجه چندگانه` as an alias, translate `The model relies on multi-query
attention.`, and check the response rewrites to `توجه چندپرسشی` and reports one
applied term. That sentence and alias are the measured pair from the design
document, so a mismatch is a real failure rather than a bad test fixture.

Then curate from `misses` before considering phase 2.
