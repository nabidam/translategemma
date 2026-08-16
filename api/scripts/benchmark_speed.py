"""Token-generation speed benchmark for the TranslateGemma serving API.

Answers two questions on the machine that actually serves the model:

1. **How fast does this GPU generate?** Decode tokens/second and milliseconds
   per decode step, swept across batch sizes, per loaded system.
2. **How long does a page of a document take?** Measured end to end, both as one
   whole-page request and as a sentence-split request, plus a words-per-minute
   figure that a non-engineer can act on.

Everything is measured through the *serving* path: ``POST /translate/batch`` on
a running API, which renders prompts and forwards them to vLLM. That is the
whole deployment — gateway overhead, JSON, vLLM's scheduler and its continuous
batching — so a number here is a number a caller can actually receive. There is
no in-process transport any more: the API loads no weights, so a benchmark of
"the model without the server" would have to reimplement the serving path
rather than measure it.

What is served comes from the environment, exactly as the API takes it: the
benchmark reads ``Settings`` and measures each system in ``loaded_systems``, and
cross-checks ``/model-info`` so a report cannot silently describe a different
checkpoint from the one that answered.

Token counts are approximate, and labelled as such wherever they are reported:
the response carries text, not token ids, so output tokens are recovered by
re-encoding it with the same tokenizer. Latency and words-per-minute — the
numbers this benchmark exists for — are exact.

Usage (inside the API image, from /app):

    python scripts/benchmark_speed.py --api-url http://localhost:8000
    python scripts/benchmark_speed.py --page-source file --page-file scripts/test.pdf

See --help for the full option list, and README.md for the compose invocation.
"""

import argparse
import json
import math
import os
import platform
import re
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_API_DIR = _SCRIPT_DIR.parent
for _path in (str(_API_DIR), str(_SCRIPT_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import benchmark_corpus  # noqa: E402
from config import Settings, System, get_settings  # noqa: E402

# A page of a translated document, by the convention used for translation
# pricing: 250 words. Overridable, because "a page" is a business decision, not
# a technical one.
DEFAULT_PAGE_WORDS = 250

# Fallback only. Segment counts come from pysbd — the same splitter the server
# uses — so the page table's segment count is the number of generate() rows the
# server would really produce. See split_into_sentences.
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


# --------------------------------------------------------------------------- #
# Measurement records
# --------------------------------------------------------------------------- #


@dataclass
class CallStats:
    """Exact token accounting, summed over the generate() calls of one run."""

    calls: int = 0
    prompt_tokens: int = 0  # Real prompt tokens, padding excluded.
    padded_prompt_tokens: int = 0  # What the GPU actually attended over.
    steps: int = 0  # Decode steps; the batch runs until its slowest row stops.
    output_tokens: int = 0  # Real new tokens returned, padding and stop excluded.
    truncated_rows: int = 0  # Rows that hit max_new_tokens without stopping.

    def add(self, other: "CallStats") -> None:
        self.calls += other.calls
        self.prompt_tokens += other.prompt_tokens
        self.padded_prompt_tokens += other.padded_prompt_tokens
        self.steps += other.steps
        self.output_tokens += other.output_tokens
        self.truncated_rows += other.truncated_rows


@dataclass
class Sample:
    """One timed execution of one configuration."""

    wall_s: float
    stats: CallStats
    peak_vram_gb: float | None = None
    outputs: list[str] = field(default_factory=list)


@dataclass
class Measurement:
    """Aggregate of the repeats of one configuration, plus its derived rates."""

    label: str
    system: str
    transport: str
    batch_size: int
    n_texts: int
    max_new_tokens: int
    split_sentences: bool

    repeats: int = 0
    wall_s: float = 0.0  # Median.
    wall_s_min: float = 0.0
    wall_s_max: float = 0.0
    prefill_s: float | None = None
    decode_s: float | None = None

    prompt_tokens: int = 0
    padded_prompt_tokens: int = 0
    output_tokens: int = 0
    steps: int = 0
    generate_calls: int = 0
    truncated_rows: int = 0

    decode_tok_s: float | None = None  # Output tokens per second of decode time.
    total_tok_s: float | None = None  # Output tokens per second of wall time.
    step_ms: float | None = None  # Milliseconds per decode step (batch-wide).
    latency_per_item_s: float | None = None
    items_per_s: float | None = None
    peak_vram_gb: float | None = None
    sample_output: str = ""
    error: str | None = None

    def finalize(self, samples: list[Sample]) -> "Measurement":
        """Fill the derived fields from the collected samples."""
        walls = sorted(sample.wall_s for sample in samples)
        self.repeats = len(samples)
        self.wall_s = statistics.median(walls)
        self.wall_s_min = walls[0]
        self.wall_s_max = walls[-1]

        # Greedy decoding is deterministic, so token counts are identical across
        # repeats; the median guards the sampling case and costs nothing here.
        self.prompt_tokens = int(statistics.median(s.stats.prompt_tokens for s in samples))
        self.padded_prompt_tokens = int(
            statistics.median(s.stats.padded_prompt_tokens for s in samples)
        )
        self.output_tokens = int(statistics.median(s.stats.output_tokens for s in samples))
        self.steps = int(statistics.median(s.stats.steps for s in samples))
        self.generate_calls = int(statistics.median(s.stats.calls for s in samples))
        self.truncated_rows = max(s.stats.truncated_rows for s in samples)
        peaks = [s.peak_vram_gb for s in samples if s.peak_vram_gb is not None]
        self.peak_vram_gb = max(peaks) if peaks else None
        for sample in samples:
            if sample.outputs and sample.outputs[0].strip():
                self.sample_output = sample.outputs[0].strip()
                break

        if self.prefill_s is not None and self.prefill_s < self.wall_s:
            self.decode_s = self.wall_s - self.prefill_s
        else:
            # No probe, or a probe that cost as much as the full run (a run of
            # one or two decode steps). Attribute everything to decode rather
            # than report a negative interval.
            self.decode_s = self.wall_s
        if self.decode_s > 0:
            self.decode_tok_s = self.output_tokens / self.decode_s
            if self.steps > 0:
                self.step_ms = 1000.0 * self.decode_s / self.steps
        if self.wall_s > 0:
            self.total_tok_s = self.output_tokens / self.wall_s
            self.latency_per_item_s = self.wall_s / max(self.n_texts, 1)
            self.items_per_s = self.n_texts / self.wall_s
        return self


@dataclass
class PageSpec:
    """The document page whose translation time is being measured."""

    name: str
    source: str  # "synthetic" or the file it came from.
    text: str
    words: int
    chars: int
    sentences: int
    tokens: int | None = None


# --------------------------------------------------------------------------- #
# HTTP transport
# --------------------------------------------------------------------------- #


class HttpRunner:
    """Runs translations against a running server, over POST /translate/batch.

    Token counts here are recovered by re-encoding the returned text with the
    tokenizer, since the response carries no token ids. That is within a token
    or two of the exact count and is labelled as approximate wherever it is
    reported. Latency, which is the reason to measure this transport at all, is
    exact.
    """

    transport = "http"

    def __init__(self, settings: Settings, api_url: str, timeout: float, tokenizer=None):
        self.settings = settings
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout
        self._tokenizer = tokenizer
        self._client = None
        self.model_info: dict = {}

    def start(self) -> None:
        try:
            import httpx
        except ImportError as error:  # Shipped by fastapi[standard]; guard anyway.
            raise SystemExit(
                "--mode http needs httpx, which is missing from this environment."
            ) from error

        self._client = httpx.Client(base_url=self.api_url, timeout=self.timeout)
        response = self._client.get("/model-info")
        response.raise_for_status()
        self.model_info = response.json()

    def stop(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    @property
    def tokenizer(self):
        return self._tokenizer

    def set_tokenizer(self, tokenizer) -> None:
        self._tokenizer = tokenizer

    def run(
        self,
        texts: list[str],
        system: System,
        max_new_tokens: int,
        split_sentences: bool,
        batch_size: int,  # Server-side; accepted for signature parity.
    ) -> Sample:
        payload = {
            "texts": list(texts),
            "source_lang": self.settings.source_lang,
            "target_lang": self.settings.target_lang,
            "max_new_tokens": max_new_tokens,
            "split_sentences": split_sentences,
        }
        if len(self.settings.loaded_systems) > 1:
            # Only meaningful in "both" mode; sending it otherwise would 400 on a
            # server whose single loaded system happens to be the other one.
            payload["system"] = str(system)

        started = time.perf_counter()
        response = self._client.post("/translate/batch", json=payload)
        wall_s = time.perf_counter() - started
        response.raise_for_status()
        translations = response.json()["translations"]

        stats = CallStats(calls=1)
        if self._tokenizer is not None:
            stats.prompt_tokens = self._count_tokens(texts)
            stats.padded_prompt_tokens = stats.prompt_tokens
            stats.output_tokens = self._count_tokens(translations)
            stats.steps = stats.output_tokens  # No batch view from out here.
        return Sample(wall_s=wall_s, stats=stats, outputs=translations)

    def _count_tokens(self, texts: list[str]) -> int:
        encoded = self._tokenizer(list(texts), add_special_tokens=False)["input_ids"]
        return sum(len(ids) for ids in encoded)

    def describe(self) -> dict:
        return {"api_url": self.api_url, "model_info": self.model_info}


# --------------------------------------------------------------------------- #
# Driving the measurements
# --------------------------------------------------------------------------- #


def _is_oom(error: BaseException) -> bool:
    """True for an upstream out-of-memory, which arrives here as a 5xx body."""
    return "out of memory" in str(error).lower() or type(error).__name__ == "OutOfMemoryError"


def measure(
    runner,
    *,
    label: str,
    texts: list[str],
    system: System,
    max_new_tokens: int,
    split_sentences: bool,
    batch_size: int,
    repeats: int,
    warmup: int,
    prefill_probe: bool,
    verbose: bool = True,
) -> Measurement:
    """Time one configuration, returning an aggregate with derived rates."""
    measurement = Measurement(
        label=label,
        system=str(system),
        transport=runner.transport,
        batch_size=batch_size,
        n_texts=len(texts),
        max_new_tokens=max_new_tokens,
        split_sentences=split_sentences,
    )
    if verbose:
        print(
            f"  [{runner.transport}] {label:<28} system={system} batch={batch_size} "
            f"texts={len(texts)} ... ",
            end="",
            flush=True,
        )

    def _run(tokens: int) -> Sample:
        return runner.run(texts, system, tokens, split_sentences, batch_size)

    try:
        for _ in range(warmup):
            # The first generate() of a process pays CUDA autotuning, kernel
            # loading and allocator growth. Timing it would mostly measure that.
            _run(max_new_tokens)

        if prefill_probe:
            # A one-token generation is prefill plus a single decode step, which
            # is the cleanest way to separate prompt processing from decoding
            # without instrumenting the decode loop itself.
            probe = _run(1)
            measurement.prefill_s = probe.wall_s

        samples = [_run(max_new_tokens) for _ in range(repeats)]
    except Exception as error:
        measurement.error = f"{type(error).__name__}: {error}"
        if _is_oom(error):
            measurement.error = f"OOM at batch {batch_size} ({type(error).__name__})"
        if verbose:
            print(f"FAILED — {measurement.error}")
        return measurement

    measurement.finalize(samples)
    if verbose:
        rate = measurement.decode_tok_s
        print(
            f"{measurement.wall_s:6.2f}s  "
            + (f"{rate:7.1f} tok/s" if rate is not None else "  n/a tok/s")
        )
    return measurement


def split_into_sentences(text: str, language: str) -> list[str]:
    """Segment text the way the server segments it.

    ``SentenceSplitter`` splits with pysbd, so counting sentences any other way
    would report a segment count the server never produces — and the page table
    divides the measured time by exactly that count. Falls back to a regex only
    if pysbd is missing or has no model for the language, matching
    ``SentenceSplitter``'s own fallback of treating the text as one segment
    rather than failing.
    """
    try:
        import pysbd

        segmenter = pysbd.Segmenter(language=language, clean=False)
    except (ImportError, ValueError):
        return [part.strip() for part in SENTENCE_BOUNDARY.split(text) if part.strip()]
    segments = [segment.strip() for segment in segmenter.segment(text)]
    return [segment for segment in segments if segment] or [text]


def build_texts(count: int) -> list[str]:
    """The first `count` corpus sentences, cycled if more are asked for.

    The corpus order alternates lengths, so any prefix is length-mixed; see
    benchmark_corpus.
    """
    corpus = benchmark_corpus.SENTENCES
    return [corpus[index % len(corpus)] for index in range(count)]


def build_synthetic_page(page_words: int, language: str) -> PageSpec:
    """Assemble continuous prose of roughly `page_words` words."""
    sentences: list[str] = []
    words = 0
    index = 0
    paragraphs = benchmark_corpus.PARAGRAPHS
    while words < page_words:
        paragraph = paragraphs[index % len(paragraphs)]
        index += 1
        for sentence in split_into_sentences(paragraph.strip(), language):
            sentences.append(sentence)
            words += len(sentence.split())
            if words >= page_words:
                break
    text = " ".join(sentences)
    return _page_spec("synthetic-page", "synthetic", text, language)


def load_page_from_file(path: Path, page_number: int, language: str) -> PageSpec:
    """Read a page of source text from .txt/.md, or one page of a PDF."""
    if not path.is_file():
        raise ValueError(f"--page-file does not exist: {path}")
    if path.suffix.lower() == ".pdf":
        text, name = _extract_pdf_page(path, page_number)
    else:
        text = path.read_text(encoding="utf-8")
        name = path.name

    # A PDF's line breaks are typography, not sentence structure: they land
    # mid-sentence and would make pysbd segment the page differently from the
    # same text pasted into a request. Reflow before measuring anything.
    text = re.sub(r"[ \t]*\n[ \t]*", " ", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    if not text:
        raise ValueError(f"No extractable text in {name}.")
    return _page_spec(name, str(path), text, language)


def _extract_pdf_page(path: Path, page_number: int) -> tuple[str, str]:
    """One page of text from a PDF, via PyMuPDF."""
    try:
        import pymupdf
    except ImportError:  # PyMuPDF < 1.24.3 only publishes the `fitz` name.
        try:
            import fitz as pymupdf
        except ImportError as error:
            raise ValueError(
                f"Reading {path.name} needs PyMuPDF, which requirements.txt installs "
                "for this script. This environment does not have it — rebuild the "
                "image, or extract the page to a .txt file and pass that instead."
            ) from error

    with pymupdf.open(str(path)) as document:
        if not 1 <= page_number <= document.page_count:
            raise ValueError(
                f"--page-number {page_number} is outside {path.name} "
                f"(1..{document.page_count})."
            )
        # sort=True reads in reading order rather than in the order the glyphs
        # happen to be stored, which is what multi-column scientific layouts
        # need; without it the sentences arrive interleaved.
        text = document[page_number - 1].get_text("text", sort=True)
    return text, f"{path.name}#p{page_number}"


def _page_spec(name: str, source: str, text: str, language: str) -> PageSpec:
    return PageSpec(
        name=name,
        source=source,
        text=text,
        words=len(text.split()),
        chars=len(text),
        sentences=len(split_into_sentences(text, language)),
    )


# --------------------------------------------------------------------------- #
# Report rendering
# --------------------------------------------------------------------------- #


def _format(value, spec: str = "", dash: str = "—") -> str:
    if value is None:
        return dash
    if isinstance(value, bool):
        return "yes" if value else "no"
    if spec and isinstance(value, (int, float)):
        return format(value, spec)
    return str(value)


def render_table(headers: list[str], rows: list[list[str]]) -> str:
    """A box-drawn table with numeric columns right-aligned."""
    if not rows:
        return "(no rows)\n"
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def is_numeric(index: int) -> bool:
        return all(
            re.fullmatch(r"[-+]?[\d.,]+(\s?\w{1,3})?|—|OOM.*|n/a", row[index] or "")
            for row in rows
        )

    aligns = [">" if is_numeric(index) else "<" for index in range(len(headers))]

    def line(left: str, mid: str, right: str) -> str:
        return left + mid.join("─" * (width + 2) for width in widths) + right

    def render_row(cells: list[str], header: bool = False) -> str:
        parts = []
        for index, cell in enumerate(cells):
            align = "<" if header else aligns[index]
            parts.append(f" {cell:{align}{widths[index]}} ")
        return "│" + "│".join(parts) + "│"

    out = [line("┌", "┬", "┐"), render_row(headers, header=True), line("├", "┼", "┤")]
    out.extend(render_row(row) for row in rows)
    out.append(line("└", "┴", "┘"))
    return "\n".join(out) + "\n"


def render_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_(no rows)_\n"
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out) + "\n"


SWEEP_HEADERS = [
    "system",
    "via",
    "batch",
    "in tok",
    "out tok",
    "steps",
    "prefill s",
    "total s",
    "decode tok/s",
    "ms/step",
    "s/sentence",
    "sentences/s",
    "peak VRAM",
]


def sweep_row(measurement: Measurement) -> list[str]:
    if measurement.error:
        return [
            measurement.system,
            measurement.transport,
            str(measurement.batch_size),
            *(["—"] * 9),
            measurement.error.split("(")[0].strip(),
        ]
    return [
        measurement.system,
        measurement.transport,
        str(measurement.batch_size),
        str(measurement.prompt_tokens),
        str(measurement.output_tokens),
        str(measurement.steps),
        _format(measurement.prefill_s, ".2f"),
        _format(measurement.wall_s, ".2f"),
        _format(measurement.decode_tok_s, ".1f"),
        _format(measurement.step_ms, ".1f"),
        _format(measurement.latency_per_item_s, ".2f"),
        _format(measurement.items_per_s, ".2f"),
        _format(measurement.peak_vram_gb, ".1f"),
    ]


PAGE_HEADERS = [
    "page",
    "system",
    "via",
    "mode",
    "batch",
    "segments",
    "in tok",
    "out tok",
    "total s",
    "tok/s",
    "words/min",
    "predicted s",
]


def page_row(entry: dict) -> list[str]:
    measurement: Measurement = entry["measurement"]
    page: PageSpec = entry["page"]
    if measurement.error:
        return [
            page.name,
            measurement.system,
            measurement.transport,
            entry["mode"],
            *(["—"] * 7),
            measurement.error.split("(")[0].strip(),
        ]
    words_per_minute = page.words / measurement.wall_s * 60 if measurement.wall_s else None
    segments = page.sentences if entry["mode"] == "sentences" else 1
    return [
        page.name,
        measurement.system,
        measurement.transport,
        entry["mode"],
        str(entry["batch_size"]) if entry["mode"] == "sentences" else "—",
        str(segments),
        str(measurement.prompt_tokens),
        str(measurement.output_tokens),
        _format(measurement.wall_s, ".2f"),
        _format(measurement.decode_tok_s, ".1f"),
        _format(words_per_minute, ".0f"),
        _format(entry.get("predicted_s"), ".2f"),
    ]


def _page_summary_row(page: PageSpec) -> list[str]:
    return [
        page.name,
        page.source,
        str(page.words),
        str(page.chars),
        str(page.sentences),
        _format(page.tokens),
    ]


def build_summary(sweeps: list[Measurement], page_entries: list[dict]) -> list[str]:
    """The three or four sentences someone will actually quote."""
    lines: list[str] = []
    http_sweeps = [m for m in sweeps if m.transport == "http" and not m.error]

    for system in sorted({m.system for m in http_sweeps}):
        rows = [m for m in http_sweeps if m.system == system]
        best = max(rows, key=lambda m: m.decode_tok_s or 0)
        single = next((m for m in rows if m.batch_size == 1), None)
        if best.decode_tok_s is not None:
            lines.append(
                f"[{system}] peak throughput {best.decode_tok_s:.1f} tok/s at batch "
                f"{best.batch_size} (approximate: token counts are recovered by "
                "re-encoding the response)."
            )
        if single is not None:
            lines.append(
                f"[{system}] a single sentence takes {single.wall_s:.2f} s end to end "
                f"({single.output_tokens} output tokens)."
            )

    for entry in page_entries:
        measurement: Measurement = entry["measurement"]
        if measurement.error:
            continue
        page: PageSpec = entry["page"]
        minutes = measurement.wall_s / 60
        lines.append(
            f"[{measurement.system}] one page of \"{page.name}\" ({page.words} words, "
            f"{entry['mode']}) takes {measurement.wall_s:.1f} s ({minutes:.2f} min) — "
            f"{page.words / measurement.wall_s * 60:.0f} words/min, "
            f"{3600 / measurement.wall_s:.0f} pages/hour."
        )

    truncated = [m for m in sweeps if m.truncated_rows]
    if truncated:
        lines.append(
            f"WARNING: {len(truncated)} configuration(s) hit max_new_tokens without emitting a "
            "stop token. Those rows measure the generation cap, not the model; raise "
            "--max-new-tokens for an honest number."
        )
    return lines


def render_console(context: dict, sweeps, page_entries, pages, notes) -> str:
    out: list[str] = []

    def rule(title: str) -> None:
        out.append("\n" + "═" * 78)
        out.append(f" {title}")
        out.append("═" * 78 + "\n")

    rule("TranslateGemma — generation speed report")
    env_rows = [[key, _format(value)] for key, value in context["environment"].items()]
    out.append(render_table(["setting", "value"], env_rows))

    rule("Throughput sweep — one batch of sentences per row")
    out.append(render_table(SWEEP_HEADERS, [sweep_row(m) for m in sweeps]))
    out.append(
        "in/out tok are real tokens with padding excluded. 'steps' is decode iterations:\n"
        "a batch runs until its slowest row stops, so decode tok/s below steps x batch\n"
        "is padding waste, and ms/step is the hardware speed independent of it.\n"
    )

    if pages:
        rule("Document page")
        page_rows = [_page_summary_row(p) for p in pages]
        out.append(
            render_table(["page", "source", "words", "chars", "sentences", "tokens"], page_rows)
        )
        out.append(render_table(PAGE_HEADERS, [page_row(entry) for entry in page_entries]))
        out.append(
            "mode=whole sends the page as one segment (one long generation).\n"
            "mode=sentences splits it with pysbd and batches the segments — the same\n"
            "work the server does with TG_SPLIT_SENTENCES=true.\n"
            "'predicted s' extrapolates the sweep; a large gap against 'total s' means\n"
            "the sweep's sentence mix does not represent this page.\n"
        )

    rule("Summary")
    for line in build_summary(sweeps, page_entries):
        out.append(f"  • {line}")
    if notes:
        out.append("\n  Notes:")
        for note in notes:
            out.append(f"  • {note}")
    out.append("")
    return "\n".join(out)


def render_markdown(context: dict, sweeps, page_entries, pages, notes) -> str:
    out = [
        "# TranslateGemma — generation speed report",
        "",
        f"Generated {context['generated_at']} · host `{context['hostname']}`",
        "",
        "## Summary",
        "",
    ]
    out.extend(f"- {line}" for line in build_summary(sweeps, page_entries))
    environment_rows = [[key, _format(value)] for key, value in context["environment"].items()]
    out += ["", "## Environment", "", render_markdown_table(["setting", "value"], environment_rows)]
    out += [
        "## Throughput sweep",
        "",
        render_markdown_table(SWEEP_HEADERS, [sweep_row(m) for m in sweeps]),
        "`in tok` / `out tok` exclude padding. `steps` is decode iterations; a batch runs "
        "until its slowest row stops, so the gap between `decode tok/s` and "
        "`steps x batch` is padding waste. `ms/step` is hardware speed, independent of it.",
        "",
    ]
    if pages:
        out += [
            "## Document page",
            "",
            render_markdown_table(
                ["page", "source", "words", "chars", "sentences", "tokens"],
                [_page_summary_row(p) for p in pages],
            ),
            render_markdown_table(PAGE_HEADERS, [page_row(entry) for entry in page_entries]),
            "`mode=whole` sends the page as a single segment. `mode=sentences` splits with "
            "pysbd and batches the segments, which is what the server does with "
            "`TG_SPLIT_SENTENCES=true`. `predicted s` extrapolates from the sweep.",
            "",
        ]
    if notes:
        out += ["## Notes", ""] + [f"- {note}" for note in notes] + [""]
    out += [
        "## Method",
        "",
        "- Measured through `POST /translate/batch` on the running API, which "
        "renders the prompts and forwards them to vLLM — the served code path, not "
        "a reimplementation.",
        f"- {context['repeats']} timed repeat(s) per configuration after "
        f"{context['warmup']} discarded warmup run(s); the reported time is the median.",
        "- `torch.cuda.synchronize()` brackets every timed region.",
        "- Prefill is measured by a separate `max_new_tokens=1` run; "
        "`decode s = total - prefill`.",
        "- Output tokens are counted per row up to the first stop id "
        f"({context['environment'].get('stop_token_ids')}), so padding is excluded.",
        "",
    ]
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _settings_summary(settings: Settings) -> dict:
    return {
        "base_model_id": settings.base_model_id,
        "vllm_base_url": settings.vllm_base_url,
        "vllm_model": settings.vllm_model,
        "model_mode": str(settings.model_mode),
        "adapter_path": settings.adapter_path,
        "dtype": settings.dtype,
        "attn_implementation": settings.attn_implementation,
        "load_in_4bit": settings.load_in_4bit,
        "do_sample": settings.do_sample,
        "num_beams": settings.num_beams,
        "server_batch_size": settings.batch_size,
        "server_max_new_tokens": settings.max_new_tokens,
        "source_lang": settings.source_lang,
        "target_lang": settings.target_lang,
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Which systems are benchmarked follows TG_MODEL_MODE: base, adapter, or "
            "both. Nothing here overrides what the server is configured to load."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["http"],
        default="http",
        help=(
            "Transport. Only 'http' exists: the API loads no weights, so there is "
            "no in-process engine to benchmark. Kept so existing invocations that "
            "pass --mode http keep working."
        ),
    )
    parser.add_argument(
        "--api-url",
        default=os.environ.get("TG_BENCHMARK_API_URL", "http://localhost:8000"),
        help="Base URL of the running API.",
    )
    parser.add_argument(
        "--http-timeout", type=float, default=1800.0, help="Per-request HTTP timeout, seconds."
    )
    parser.add_argument(
        "--batch-sizes",
        default="1,2,4,8,16",
        help="Comma-separated batch sizes to sweep. Use '1' for a latency-only run.",
    )
    parser.add_argument("--repeats", type=int, default=3, help="Timed repeats per configuration.")
    parser.add_argument("--warmup", type=int, default=1, help="Discarded runs before timing.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Generation cap for the sweep. Default: TG_MAX_NEW_TOKENS.",
    )
    parser.add_argument(
        "--no-prefill-probe",
        action="store_true",
        help="Skip the max_new_tokens=1 probe; prefill and decode are then not separated.",
    )
    parser.add_argument(
        "--systems",
        default=None,
        help=(
            "Comma-separated subset of the loaded systems to benchmark "
            "(base,adapter). Default: every system TG_MODEL_MODE loads."
        ),
    )
    parser.add_argument("--skip-sweep", action="store_true", help="Only run the page benchmark.")
    parser.add_argument("--skip-page", action="store_true", help="Only run the throughput sweep.")
    parser.add_argument(
        "--page-source",
        choices=["synthetic", "file", "both"],
        default="synthetic",
        help="Where the benchmarked page comes from. 'file'/'both' need --page-file.",
    )
    parser.add_argument(
        "--page-file",
        type=Path,
        default=None,
        help="A .txt/.md file, or a .pdf (extracted with PyMuPDF).",
    )
    parser.add_argument(
        "--page-number", type=int, default=1, help="1-based page to extract from a PDF."
    )
    parser.add_argument(
        "--page-words",
        type=int,
        default=DEFAULT_PAGE_WORDS,
        help=f"Words in the synthetic page (default {DEFAULT_PAGE_WORDS}, the usual "
        "translation-industry page).",
    )
    parser.add_argument(
        "--page-modes",
        default="whole,sentences",
        help="Comma-separated: whole (one segment) and/or sentences (pysbd split).",
    )
    parser.add_argument(
        "--page-repeats", type=int, default=1, help="Timed repeats for the page benchmark."
    )
    parser.add_argument(
        "--page-batch-size",
        type=int,
        default=1,
        help=(
            "Accepted for compatibility and reported in the 'batch' column only "
            "when it matches the server. The page request is batched inside the API "
            "by its own TG_BATCH_SIZE, which is what the rows actually measure."
        ),
    )
    parser.add_argument(
        "--page-max-new-tokens",
        type=int,
        default=None,
        help=(
            "Generation cap for mode=whole. Default: sized from the page's own token "
            "count, since a page needs far more than TG_MAX_NEW_TOKENS in one shot."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("TG_BENCHMARK_OUTPUT_DIR", "benchmarks")),
        help="Where the JSON and Markdown reports are written.",
    )
    parser.add_argument("--no-files", action="store_true", help="Print the report, write nothing.")
    parser.add_argument("--tag", default="", help="Suffix for the report filenames.")
    return parser.parse_args(argv)


def resolve_systems(settings: Settings, requested: str | None) -> list[System]:
    loaded = list(settings.loaded_systems)
    if not requested:
        return loaded
    chosen = []
    for name in (part.strip() for part in requested.split(",") if part.strip()):
        try:
            system = System(name)
        except ValueError:
            raise SystemExit(f"--systems: {name!r} is not a system (base|adapter).")
        if system not in loaded:
            raise SystemExit(
                f"--systems: {name!r} is not loaded under TG_MODEL_MODE={settings.model_mode}. "
                f"Loaded: {[str(item) for item in loaded]}."
            )
        chosen.append(system)
    return chosen or loaded


def load_tokenizer_only(settings: Settings):
    """The tokenizer the server renders prompts with, for token counts.

    Loaded through translator.load_processor so this counts tokens with exactly
    the front end the API uses, including its fallback for a model directory
    that ships a bare tokenizer.
    """
    from translator import load_processor

    return load_processor(settings.resolved_tokenizer_path).tokenizer


def main(argv=None) -> int:
    args = parse_args(argv)
    settings = get_settings()
    systems = resolve_systems(settings, args.systems)
    batch_sizes = [int(part) for part in args.batch_sizes.split(",") if part.strip()]
    page_modes = [part.strip() for part in args.page_modes.split(",") if part.strip()]
    sweep_max_new_tokens = args.max_new_tokens or settings.max_new_tokens
    notes: list[str] = []

    if not args.skip_page and args.page_batch_size != settings.batch_size:
        notes.append(
            f"--page-batch-size={args.page_batch_size} has no effect: the page rows were "
            f"batched by the server at TG_BATCH_SIZE={settings.batch_size}, and their "
            "'batch' column reports that."
        )
    if max(batch_sizes, default=0) > settings.max_batch_items:
        notes.append(
            f"Batch sizes above TG_MAX_BATCH_ITEMS={settings.max_batch_items} were dropped "
            "from the HTTP sweep; the server rejects them with 413."
        )

    # --- pages ------------------------------------------------------------
    pages: list[PageSpec] = []
    if not args.skip_page:
        if args.page_source in ("synthetic", "both"):
            pages.append(build_synthetic_page(args.page_words, settings.source_lang))
        if args.page_source in ("file", "both"):
            if args.page_file is None:
                raise SystemExit("--page-source file/both requires --page-file.")
            pages.append(
                load_page_from_file(args.page_file, args.page_number, settings.source_lang)
            )

    # --- runners ----------------------------------------------------------
    # One transport: the API is a gateway, so there is nothing to measure
    # in-process. Kept as a list because everything downstream iterates it, and
    # a second transport (vLLM directly, bypassing the gateway) is the obvious
    # next one to add.
    runners = [HttpRunner(settings, args.api_url, args.http_timeout)]

    sweeps: list[Measurement] = []
    page_entries: list[dict] = []
    environment = {
        "hostname": platform.node(),
        "python": platform.python_version(),
        "systems_benchmarked": ", ".join(str(system) for system in systems),
        **_settings_summary(settings),
    }
    tokenizer = None

    try:
        for runner in runners:
            print(f"\n>>> Starting {runner.transport} transport ...", flush=True)
            runner.start()
            if tokenizer is None:
                try:
                    tokenizer = load_tokenizer_only(settings)
                except Exception as error:
                    notes.append(
                        f"Token counts unavailable: could not load a tokenizer "
                        f"({type(error).__name__}: {error})."
                    )
            runner.set_tokenizer(tokenizer)
            environment.update(runner.describe())
            environment["api_url"] = runner.api_url
            served = runner.model_info.get("base_model_id")
            if served and served != settings.base_model_id:
                notes.append(
                    f"The server reports base_model_id={served!r}, which differs from this "
                    f"process's TG_BASE_MODEL_ID={settings.base_model_id!r}. Token counts "
                    "here were computed with a different tokenizer than the one that "
                    "rendered the prompts."
                )

            for page in pages:
                if page.tokens is None and tokenizer is not None:
                    page.tokens = len(tokenizer(page.text, add_special_tokens=False)["input_ids"])

            # --- throughput sweep ---------------------------------------
            if not args.skip_sweep:
                print(f"--- throughput sweep ({runner.transport}) ---", flush=True)
                for system in systems:
                    for batch_size in batch_sizes:
                        if (
                            runner.transport == "http"
                            and batch_size > settings.max_batch_items
                        ):
                            continue
                        sweeps.append(
                            measure(
                                runner,
                                label=f"sweep batch={batch_size}",
                                texts=build_texts(batch_size),
                                system=system,
                                max_new_tokens=sweep_max_new_tokens,
                                split_sentences=False,
                                batch_size=batch_size,
                                repeats=args.repeats,
                                warmup=args.warmup,
                                prefill_probe=not args.no_prefill_probe,
                            )
                        )

            # --- page ----------------------------------------------------
            for page in pages:
                print(f"--- page: {page.name} ({runner.transport}) ---", flush=True)
                for system in systems:
                    for mode in page_modes:
                        split = mode == "sentences"
                        if split:
                            max_new_tokens = sweep_max_new_tokens
                        else:
                            max_new_tokens = args.page_max_new_tokens or _page_token_budget(
                                page, settings
                            )
                        measurement = measure(
                            runner,
                            label=f"page {mode}",
                            texts=[page.text],
                            system=system,
                            max_new_tokens=max_new_tokens,
                            split_sentences=split,
                            batch_size=args.page_batch_size,
                            repeats=args.page_repeats,
                            warmup=0,  # The sweep already warmed the GPU.
                            prefill_probe=not args.no_prefill_probe,
                        )
                        page_entries.append(
                            {
                                "page": page,
                                "mode": mode,
                                "transport": runner.transport,
                                "batch_size": _page_effective_batch_size(
                                    args, settings, runner.transport
                                ),
                                "measurement": measurement,
                                "predicted_s": _predict_page_seconds(
                                    page,
                                    mode,
                                    system,
                                    _page_effective_batch_size(args, settings, runner.transport),
                                    sweeps,
                                    runner.transport,
                                ),
                            }
                        )

            runner.stop()
    finally:
        for runner in runners:
            try:
                runner.stop()
            except Exception:
                pass

    truncated = [m for m in sweeps + [e["measurement"] for e in page_entries] if m.truncated_rows]
    if truncated:
        notes.append(
            f"{len(truncated)} configuration(s) had rows that never emitted a stop token "
            "and ran to max_new_tokens. Their throughput reflects the cap."
        )

    context = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hostname": platform.node(),
        "repeats": args.repeats,
        "warmup": args.warmup,
        "environment": environment,
    }
    console = render_console(context, sweeps, page_entries, pages, notes)
    print(console)

    if not args.no_files:
        _write_reports(args, context, sweeps, page_entries, pages, notes)
    return 0


def _page_token_budget(page: PageSpec, settings: Settings) -> int:
    """A generation cap large enough for a whole page in one shot.

    TG_MAX_NEW_TOKENS is sized for a sentence. Translating a page unsplit under
    that cap measures truncation, not speed. English-to-Farsi output runs longer
    than the source in tokens, hence the 2x plus slack, capped at the 4096 the
    API schema accepts, which is also the cap a caller can ask for.
    """
    source_tokens = page.tokens or int(page.words * 1.4)
    return min(4096, max(settings.max_new_tokens, int(source_tokens * 2.0) + 128))


def _page_effective_batch_size(args, settings: Settings, transport: str) -> int:
    """The batch size the page run really used.

    The batching happens inside the server, which chunks by its own
    TG_BATCH_SIZE regardless of what this process asked for.
    """
    return settings.batch_size


def _predict_page_seconds(
    page: PageSpec,
    mode: str,
    system: System,
    batch_size: int,
    sweeps: list[Measurement],
    transport: str,
) -> float | None:
    """Extrapolate the page time from the sweep, as a cross-check on the timing.

    Only defined for sentence-split mode, where the page is exactly ceil(n/batch)
    batches of the kind the sweep timed. A large gap against the measured value
    means the sweep's sentence mix is not representative of this page. Returns
    None when the sweep never timed that batch size.
    """
    if mode != "sentences":
        return None
    row = next(
        (
            m
            for m in sweeps
            if m.transport == transport
            and m.system == str(system)
            and m.batch_size == batch_size
            and not m.error
        ),
        None,
    )
    if row is None or page.sentences == 0:
        return None
    return math.ceil(page.sentences / batch_size) * row.wall_s


def _write_reports(args, context, sweeps, page_entries, pages, notes) -> None:
    output_dir: Path = args.output_dir
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        print(f"Could not write reports to {output_dir}: {error}")
        return

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    suffix = f"_{args.tag}" if args.tag else ""
    base = output_dir / f"benchmark_{args.mode}_{stamp}{suffix}"

    payload = {
        "context": context,
        "arguments": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
        "sweep": [asdict(measurement) for measurement in sweeps],
        "pages": [asdict(page) for page in pages],
        "page_runs": [
            {
                "page": entry["page"].name,
                "mode": entry["mode"],
                "transport": entry["transport"],
                "predicted_s": entry["predicted_s"],
                **asdict(entry["measurement"]),
                # Last word: over HTTP the server batches by TG_BATCH_SIZE, not
                # by the batch size this process asked for.
                "batch_size": entry["batch_size"],
            }
            for entry in page_entries
        ],
        "notes": notes,
    }
    base.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    base.with_suffix(".md").write_text(
        render_markdown(context, sweeps, page_entries, pages, notes), encoding="utf-8"
    )
    print(f"Reports written:\n  {base.with_suffix('.json')}\n  {base.with_suffix('.md')}")


if __name__ == "__main__":
    raise SystemExit(main())
