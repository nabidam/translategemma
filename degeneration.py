"""Decoding-failure classification for generated translations.

MetricX and COMET are corpus averages and eval_loss is teacher-forced, so none
of them can see a model that translates correctly and then fails to stop. That
failure shipped once already (docs/2026-08-10_adapter_degeneration_analysis.md):
70% of one adapter's rows ended in a newline flood while its eval_loss was the
best of the run.

This module is the shared definition of "a decoding failure", used both by the
evaluation run itself (as a gate) and by scripts/audit_degeneration.py (as an
offline report), so the gate and the report can never disagree.
"""

import collections
import re

# Translator/watermark strings observed leaking from the training corpus.
BOILERPLATE = re.compile(
    r"ترجمه شده توسط|هوش مصنوعی|Translated by|machine[- ]translat|Google Translate",
    re.IGNORECASE,
)
WORD = re.compile(r"\S+")

DEFAULTS = {
    # Whitespace after the last real character. A model that emitted its stop
    # token and was not stopped fills the remaining budget with newlines.
    "trailing_threshold": 20,
    "repeat_ngram": 6,
    "repeat_threshold": 3,
    # Trimmed output longer than this multiple of the reference.
    "length_blowup": 2.0,
    # Character proxy for evaluation.max_new_tokens having been exhausted.
    "near_budget_chars": 1900,
}


def max_ngram_repeat(text, n):
    """Highest occurrence count of any word n-gram in text."""
    words = WORD.findall(text)
    if len(words) < n:
        return 1
    counts = collections.Counter(
        tuple(words[index : index + n]) for index in range(len(words) - n + 1)
    )
    return counts.most_common(1)[0][1]


def classify_output(hypothesis, reference, settings=None):
    """Return (set of failure class names, trailing whitespace) for one row."""
    options = {**DEFAULTS, **(settings or {})}
    text = str(hypothesis or "")
    trimmed = text.rstrip()
    trailing = text[len(trimmed) :]

    if not trimmed.strip():
        return {"empty"}, trailing

    flags = set()
    if len(trailing) > options["trailing_threshold"]:
        flags.add("whitespace_flood")
    if max_ngram_repeat(trimmed, options["repeat_ngram"]) >= options["repeat_threshold"]:
        flags.add("loop")
    if BOILERPLATE.search(trimmed):
        flags.add("boilerplate")
    reference = str(reference or "")
    if reference and len(trimmed) > options["length_blowup"] * len(reference):
        flags.add("length_blowup")
    if len(text) >= options["near_budget_chars"]:
        flags.add("near_budget")
    return flags, trailing


def audit_outputs(hypotheses, references, settings=None, sample_limit=10):
    """Summarize decoding failures across one system's outputs.

    ``clean_rate`` is the headline: the share of rows exhibiting no failure
    class at all. Note that ``loop`` fires on faithful translations of source
    segments that themselves repeat, so a non-zero rate is expected on
    PDF-extracted corpora; compare systems against each other rather than
    against zero.
    """
    hypotheses = list(hypotheses)
    references = list(references) if references is not None else [""] * len(hypotheses)
    counts = collections.Counter()
    trailing_lengths = []
    samples = []
    clean = 0

    for index, (hypothesis, reference) in enumerate(zip(hypotheses, references)):
        flags, trailing = classify_output(hypothesis, reference, settings)
        trailing_lengths.append(len(trailing))
        if not flags:
            clean += 1
            continue
        counts.update(flags)
        if len(samples) < sample_limit:
            samples.append({
                "row": index,
                "flags": sorted(flags),
                "trailing_len": len(trailing),
                "output_head": str(hypothesis)[:200],
            })

    rows = len(hypotheses)
    trimmed_lengths = [len(str(value).rstrip()) for value in hypotheses]
    return {
        "rows": rows,
        "clean_rows": clean,
        "clean_rate": clean / rows if rows else 0.0,
        "failure_rate": (rows - clean) / rows if rows else 0.0,
        "failures": {
            name: {"rows": count, "rate": count / rows}
            for name, count in counts.most_common()
        },
        "mean_chars": sum(len(str(value)) for value in hypotheses) / rows if rows else 0.0,
        "mean_chars_trimmed": sum(trimmed_lengths) / rows if rows else 0.0,
        "mean_trailing_chars": sum(trailing_lengths) / rows if rows else 0.0,
        "max_trailing_chars": max(trailing_lengths, default=0),
        "samples": samples,
    }
