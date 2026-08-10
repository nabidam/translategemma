"""Audit an SFT JSONL corpus for the defects that teach a model not to stop.

The 2026-08-10 adapter evaluation showed the fine-tuned model producing a
correct translation and then failing to terminate: a flood of trailing
newlines (70% of rows), whole-output loops (17%), and leaked boilerplate such
as "ترجمه شده توسط هوش مصنوعی". Every one of those behaviours has a candidate
cause in the *target* column of the training data, because the target column is
the only text the loss is computed on.

This script measures those candidates instead of assuming them. It is pure
text processing: no tokenizer, no model, no GPU, streaming line by line so the
full multi-gigabyte corpus fits in constant memory.

    uv run python scripts/audit_sft_corpus.py --config config.yaml
    uv run python scripts/audit_sft_corpus.py --dataset data/splits/train.jsonl

Checks, in the order they appear in the report:

trailing_whitespace   Targets ending in whitespace. The template appends
                      <end_of_turn> straight after the target, so a target
                      ending in "\n\n\n" trains the model to emit newlines
                      before the stop token. This is the prime suspect for the
                      newline flood.
leading_whitespace    Same problem at the front; shows up as outputs that begin
                      with a stray ". " or newline.
internal_repetition   Targets that already loop (a word n-gram repeated N+
                      times). A model trained on looping targets loops.
boilerplate           Translator/watermark footers that survived extraction.
                      Any hit here is a direct explanation for leaked strings.
copy_source           target == source (untranslated row). Teaches copying.
empty_or_short        Targets that are blank or far shorter than the source.
                      Teaches truncation, and inflates loss variance.
length_ratio          target/source character ratio outliers in both directions.
untranslated_ratio    Share of Latin characters in a Farsi target: a high value
                      means the "translation" is mostly untranslated text.
duplicate_source      Exact duplicate sources (after normalisation), and
                      duplicate source+target pairs. Heavy duplication of short
                      bibliography-style rows is how a citation line becomes a
                      high-probability loop.

Exit code is 0 always; this is a report, not a gate. Use --fail-over to make it
a gate in CI (non-zero when any headline rate exceeds the threshold).
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import logging
import re
import sys
import unicodedata
from pathlib import Path

# Run from anywhere: the project modules live in the repository root, not here.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from logging_utils import load_config, logger  # noqa: E402

# Strings observed leaking verbatim into adapter output, plus the obvious
# family they belong to. Extend this list as new footers are found; a match is
# reported with its row id so the offending document can be traced.
BOILERPLATE_PATTERNS = [
    r"ترجمه شده توسط",
    r"هوش مصنوعی",
    r"ترجمه\s*[:：]\s*$",
    r"Translated by",
    r"machine[- ]translat",
    r"Google Translate",
    r"این متن به صورت خودکار ترجمه شده",
    r"\bDownloaded from\b",
    r"\bAll rights reserved\b",
    r"کلیه حقوق محفوظ است",
]

LATIN = re.compile(r"[A-Za-z]")
ARABIC = re.compile(r"[؀-ۿ]")
WORD = re.compile(r"\S+")


def normalize_for_dedup(text):
    """Collapse whitespace and Unicode form so cosmetic variants dedupe together."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text))).strip()


def max_ngram_repeat(text, n):
    """Return the highest occurrence count of any word n-gram in text."""
    words = WORD.findall(text)
    if len(words) < n:
        return 1
    counts = collections.Counter(
        tuple(words[index : index + n]) for index in range(len(words) - n + 1)
    )
    return counts.most_common(1)[0][1]


def script_ratio(text, pattern):
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for char in letters if pattern.match(char)) / len(letters)


class Audit:
    """Accumulates counts and a bounded sample of offending rows per check."""

    def __init__(self, args):
        self.args = args
        self.rows = 0
        self.counts = collections.Counter()
        self.samples = collections.defaultdict(list)
        self.boilerplate_hits = collections.Counter()
        self.trailing_shapes = collections.Counter()
        self.source_hashes = collections.Counter()
        self.pair_hashes = collections.Counter()
        self.boilerplate = [(pattern, re.compile(pattern, re.IGNORECASE | re.MULTILINE))
                            for pattern in BOILERPLATE_PATTERNS]

    def flag(self, check, row_id, detail):
        self.counts[check] += 1
        bucket = self.samples[check]
        if len(bucket) < self.args.samples_per_check:
            bucket.append({"id": row_id, **detail})

    def visit(self, record):
        self.rows += 1
        args = self.args
        row_id = record.get(args.id_column)
        source = str(record.get(args.source_column) or "")
        target = str(record.get(args.target_column) or "")

        stripped_target = target.strip()

        trailing = target[len(target.rstrip()) :]
        if trailing:
            self.trailing_shapes[repr(trailing[:12])] += 1
            if len(trailing) > 1 or trailing != "\n":
                self.flag("trailing_whitespace", row_id,
                          {"trailing": repr(trailing[:40]), "trailing_len": len(trailing)})

        leading = target[: len(target) - len(target.lstrip())]
        if leading:
            self.flag("leading_whitespace", row_id, {"leading": repr(leading[:40])})

        repeat = max_ngram_repeat(stripped_target, args.repeat_ngram)
        if repeat >= args.repeat_threshold:
            self.flag("internal_repetition", row_id,
                      {"max_ngram_repeat": repeat, "target_head": stripped_target[:200]})

        for pattern, regex in self.boilerplate:
            if regex.search(target):
                self.boilerplate_hits[pattern] += 1
                self.flag("boilerplate", row_id,
                          {"pattern": pattern, "target_head": stripped_target[:200]})
                break

        if stripped_target and stripped_target == source.strip():
            self.flag("copy_source", row_id, {"text_head": stripped_target[:200]})

        if not stripped_target:
            self.flag("empty_target", row_id, {"source_head": source[:200]})
        elif source.strip() and len(stripped_target) < args.short_target_ratio * len(source.strip()):
            self.flag("short_target", row_id,
                      {"source_len": len(source.strip()), "target_len": len(stripped_target),
                       "target": stripped_target[:200]})

        if source.strip():
            ratio = len(stripped_target) / len(source.strip())
            if ratio > args.long_target_ratio:
                self.flag("long_target", row_id,
                          {"ratio": round(ratio, 2), "source_len": len(source.strip()),
                           "target_len": len(stripped_target)})

        # A Farsi target that is mostly Latin letters was never really translated.
        if stripped_target and ARABIC.search(source) is None:
            latin_share = script_ratio(stripped_target, LATIN)
            if latin_share > args.latin_share_threshold:
                self.flag("untranslated_target", row_id,
                          {"latin_share": round(latin_share, 3), "target_head": stripped_target[:200]})

        source_key = normalize_for_dedup(source)
        if source_key:
            self.source_hashes[hashlib.blake2b(source_key.encode(), digest_size=12).digest()] += 1
            pair_key = f"{source_key}␟{normalize_for_dedup(target)}"
            self.pair_hashes[hashlib.blake2b(pair_key.encode(), digest_size=12).digest()] += 1

    def report(self):
        rows = max(1, self.rows)
        duplicate_sources = sum(count - 1 for count in self.source_hashes.values() if count > 1)
        duplicate_pairs = sum(count - 1 for count in self.pair_hashes.values() if count > 1)
        worst_source = max(self.source_hashes.values(), default=0)
        checks = {
            name: {"rows": count, "rate": count / rows}
            for name, count in sorted(self.counts.items(), key=lambda item: -item[1])
        }
        checks["duplicate_source_rows"] = {"rows": duplicate_sources, "rate": duplicate_sources / rows}
        checks["duplicate_pair_rows"] = {"rows": duplicate_pairs, "rate": duplicate_pairs / rows}
        return {
            "dataset_path": str(self.args.dataset),
            "rows": self.rows,
            "checks": checks,
            "max_copies_of_one_source": worst_source,
            "boilerplate_pattern_hits": dict(self.boilerplate_hits),
            "most_common_target_trailing_whitespace": self.trailing_shapes.most_common(15),
            "samples": {name: rows for name, rows in self.samples.items()},
        }


def iter_records(path):
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from error


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--dataset", default=None,
                        help="JSONL to audit. Defaults to data.train_sft_dataset_path.")
    parser.add_argument("--report", default="logs/sft_corpus_audit.json")
    parser.add_argument("--samples-per-check", type=int, default=15)
    parser.add_argument("--repeat-ngram", type=int, default=6)
    parser.add_argument("--repeat-threshold", type=int, default=3,
                        help="Flag a target whose most common word n-gram occurs this many times.")
    parser.add_argument("--short-target-ratio", type=float, default=0.35,
                        help="Flag targets shorter than this fraction of the source length.")
    parser.add_argument("--long-target-ratio", type=float, default=3.0,
                        help="Flag targets longer than this multiple of the source length.")
    parser.add_argument("--latin-share-threshold", type=float, default=0.6)
    parser.add_argument("--fail-over", type=float, default=None,
                        help="Exit 1 when any check's rate exceeds this fraction.")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = load_config(args.config)
    data_cfg = config["data"]

    args.dataset = Path(args.dataset or data_cfg["train_sft_dataset_path"])
    args.source_column = data_cfg["source_column"]
    args.target_column = data_cfg["target_column"]
    args.id_column = data_cfg.get("id_column", "id")

    audit = Audit(args)
    for record in iter_records(args.dataset):
        audit.visit(record)
    report = audit.report()

    logger.info("Audited %s rows from %s", report["rows"], args.dataset)
    for name, stats in report["checks"].items():
        logger.info("  %-24s %8d rows  %6.2f%%", name, stats["rows"], 100 * stats["rate"])
    if report["boilerplate_pattern_hits"]:
        logger.warning("Boilerplate patterns matched: %s", report["boilerplate_pattern_hits"])
    logger.info("Most common target trailing whitespace: %s",
                report["most_common_target_trailing_whitespace"][:5])

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Report written to %s", report_path)

    if args.fail_over is not None:
        breached = {name: stats for name, stats in report["checks"].items() if stats["rate"] > args.fail_over}
        if breached:
            logger.error("Checks over --fail-over=%.4f: %s", args.fail_over, sorted(breached))
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
