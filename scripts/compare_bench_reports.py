#!/usr/bin/env python3
"""Put two benchmark_speed.py reports side by side and state which arm won.

Written for the vLLM-vs-transformers A/B (docs/SERVING_BENCHMARK_AB.md), but it
compares any two reports the benchmark produced. It reads the JSON payloads
rather than the Markdown, so the numbers here are the measured ones and not a
re-parse of a rendered table.

The comparison is only meaningful when both arms decoded the same text. Greedy
decoding makes that true, and the guard below checks it: if the two arms emitted
materially different output-token counts for the same configuration, their
tokens/second are not comparable and the row is flagged rather than ratioed.

Usage:

    python scripts/compare_bench_reports.py benchmark_ab/
    python scripts/compare_bench_reports.py a.json b.json --baseline hf-lora
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Output-token counts are recovered by re-encoding the returned text, so two
# arms that produced identical translations can still differ by a token or two.
# Beyond this fraction the arms decoded genuinely different text and a
# throughput ratio between them would be comparing unequal work.
TOKEN_DRIFT_TOLERANCE = 0.05


def load_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["_path"] = path
    payload["_label"] = payload.get("arguments", {}).get("tag") or path.stem
    return payload


def newest_per_tag(directory: Path) -> list[dict[str, Any]]:
    """The most recent report for each --tag in a directory.

    Re-running one arm is normal (a failed start, a changed setting), so the
    directory usually holds several reports per tag; only the newest of each is
    the one being compared.
    """
    newest: dict[str, tuple[float, Path]] = {}
    for path in directory.glob("benchmark_*.json"):
        report = load_report(path)
        stamp = path.stat().st_mtime
        label = report["_label"]
        if label not in newest or stamp > newest[label][0]:
            newest[label] = (stamp, path)
    return [load_report(path) for _, path in sorted(newest.values())]


def sweep_index(report: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    """Sweep rows keyed by the configuration they measured."""
    return {
        (row["system"], row["batch_size"]): row
        for row in report.get("sweep", [])
        if not row.get("error")
    }


def page_index(report: dict[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {
        (row["page"], row["mode"], row["system"]): row
        for row in report.get("page_runs", [])
        if not row.get("error")
    }


def fmt(value: float | None, digits: int = 1) -> str:
    return "-" if value is None else f"{value:,.{digits}f}"


def speedup(baseline: float | None, candidate: float | None) -> str:
    """Candidate relative to baseline, as a multiple. Higher is better."""
    if not baseline or not candidate:
        return "-"
    return f"{candidate / baseline:.2f}x"


def ratio_phrase(baseline: float | None, candidate: float | None, higher_is_better: bool) -> str:
    """"3.20x faster" / "7.75x slower", never "0.13x faster".

    A bare multiple below 1 is read as an improvement about as often as not, so
    the direction is stated in words and the number is always >= 1.
    """
    if not baseline or not candidate:
        return "-"
    better = (candidate > baseline) if higher_is_better else (candidate < baseline)
    factor = candidate / baseline if higher_is_better else baseline / candidate
    if factor < 1:
        factor = 1 / factor
    return f"{factor:.2f}x {'faster' if better else 'slower'}"


def compare(baseline: dict[str, Any], candidate: dict[str, Any]) -> int:
    b_label, c_label = baseline["_label"], candidate["_label"]
    print(f"\nBaseline : {b_label}  ({baseline['_path'].name})")
    print(f"Candidate: {c_label}  ({candidate['_path'].name})")

    b_env = baseline.get("context", {}).get("environment", {})
    c_env = candidate.get("context", {}).get("environment", {})
    b_host = baseline.get("context", {}).get("hostname")
    c_host = candidate.get("context", {}).get("hostname")
    if b_host != c_host:
        # Informational, NOT a comparability warning. The benchmark records
        # platform.node(), which inside `docker compose run --rm` is the
        # ephemeral container id -- so two arms benchmarked on one physical
        # machine always report different "hosts". There is nothing in the
        # report that identifies the real host, so this cannot be checked here;
        # keeping the arms on one machine is the runbook's job.
        print(f"\n(reported by containers {b_host} and {c_host}; both arms must be run "
              "on the same physical host -- the report cannot verify that)")

    warnings: list[str] = []
    b_sweep, c_sweep = sweep_index(baseline), sweep_index(candidate)
    shared = sorted(set(b_sweep) & set(c_sweep), key=lambda key: (key[0], key[1]))
    if not shared:
        print("\nNo configuration was measured by both reports.")
        return 1

    print("\n--- throughput sweep -------------------------------------------------")
    header = (
        f"{'system':<8} {'batch':>5} "
        f"{'wall s (' + b_label + ')':>22} {'wall s (' + c_label + ')':>22} "
        f"{'tok/s ' + b_label:>18} {'tok/s ' + c_label:>18} {'speedup':>8}"
    )
    print(header)
    print("-" * len(header))
    for system, batch in shared:
        b_row, c_row = b_sweep[(system, batch)], c_sweep[(system, batch)]
        b_tokens, c_tokens = b_row.get("output_tokens", 0), c_row.get("output_tokens", 0)
        drift_flag = ""
        if b_tokens and c_tokens:
            drift = abs(c_tokens - b_tokens) / max(b_tokens, 1)
            if drift > TOKEN_DRIFT_TOLERANCE:
                drift_flag = " *"
                warnings.append(
                    f"{system}/batch {batch}: output tokens differ by {drift:.0%} "
                    f"({b_tokens} vs {c_tokens}). The arms decoded different text, so "
                    "their tokens/second measure different amounts of work."
                )
        print(
            f"{system:<8} {batch:>5} "
            f"{fmt(b_row.get('wall_s'), 2):>22} {fmt(c_row.get('wall_s'), 2):>22} "
            f"{fmt(b_row.get('total_tok_s')):>18} {fmt(c_row.get('total_tok_s')):>18} "
            f"{speedup(b_row.get('total_tok_s'), c_row.get('total_tok_s')):>8}{drift_flag}"
        )

    # The three numbers that decide a serving strategy, called out because a
    # single headline "speed" figure hides the trade: single-request latency and
    # saturated throughput routinely disagree about which arm is better.
    print("\n--- headline ---------------------------------------------------------")
    for system in sorted({system for system, _ in shared}):
        batches = [batch for sys_, batch in shared if sys_ == system]
        low, high = min(batches), max(batches)
        b_low, c_low = b_sweep[(system, low)], c_sweep[(system, low)]
        b_high, c_high = b_sweep[(system, high)], c_sweep[(system, high)]
        print(f"\n[{system}]")
        print(
            f"  latency  @ batch {low:<3} : "
            f"{fmt(b_low.get('wall_s'), 2)} s -> {fmt(c_low.get('wall_s'), 2)} s "
            f"({ratio_phrase(b_low.get('wall_s'), c_low.get('wall_s'), higher_is_better=False)})"
        )
        print(
            f"  through. @ batch {high:<3} : "
            f"{fmt(b_high.get('total_tok_s'))} -> {fmt(c_high.get('total_tok_s'))} tok/s "
            f"({ratio_phrase(b_high.get('total_tok_s'), c_high.get('total_tok_s'), higher_is_better=True)})"
        )
        step_b, step_c = b_high.get("step_ms"), c_high.get("step_ms")
        print(f"  ms/decode step      : {fmt(step_b, 2)} -> {fmt(step_c, 2)}")

    b_pages, c_pages = page_index(baseline), page_index(candidate)
    shared_pages = sorted(set(b_pages) & set(c_pages))
    if shared_pages:
        print("\n--- page (end to end) ------------------------------------------------")
        row_header = (
            f"{'page':<18} {'mode':<10} {'system':<8} "
            f"{'s (' + b_label + ')':>18} {'s (' + c_label + ')':>18} {'speedup':>8}"
        )
        print(row_header)
        print("-" * len(row_header))
        for key in shared_pages:
            page, mode, system = key
            b_row, c_row = b_pages[key], c_pages[key]
            print(
                f"{page[:18]:<18} {mode:<10} {system:<8} "
                f"{fmt(b_row.get('wall_s'), 2):>18} {fmt(c_row.get('wall_s'), 2):>18} "
                f"{speedup(c_row.get('wall_s'), b_row.get('wall_s')):>8}"
            )

    # The arms serve the same fine-tune and decode greedily, so they should
    # return the same text. This is the only check in the report that looks at
    # WHAT was generated rather than how fast: a serving-side numerical
    # difference (a patched RoPE block, a dtype mismatch, a different prompt
    # rendering) produces fluent output that is quietly not the same model.
    print("\n--- sample output (same segment, both arms) -------------------------")
    shown = 0
    for system, batch in shared:
        if batch != min(b for s, b in shared if s == system):
            continue
        b_text = (b_sweep[(system, batch)].get("sample_output") or "").strip()
        c_text = (c_sweep[(system, batch)].get("sample_output") or "").strip()
        if not b_text and not c_text:
            continue
        verdict = "identical" if b_text == c_text else "DIFFERENT"
        print(f"\n[{system}] {verdict}")
        print(f"  {b_label}: {b_text[:300]}")
        print(f"  {c_label}: {c_text[:300]}")
        shown += 1
    if not shown:
        print("  (no sample outputs recorded)")

    # Settings that change what was measured. A difference here does not
    # invalidate the run, but it does change what the run is evidence of.
    print("\n--- environment ------------------------------------------------------")
    for field in ("api_url", "model_info"):
        b_value, c_value = b_env.get(field), c_env.get(field)
        if field == "model_info" and isinstance(b_value, dict) and isinstance(c_value, dict):
            for key in sorted(set(b_value) | set(c_value)):
                if b_value.get(key) != c_value.get(key):
                    print(f"  {key}: {b_value.get(key)!r} -> {c_value.get(key)!r}")
        elif b_value != c_value:
            print(f"  {field}: {b_value!r} -> {c_value!r}")

    all_notes = baseline.get("notes", []) + candidate.get("notes", [])
    if warnings or all_notes:
        print("\n--- warnings ---------------------------------------------------------")
        for warning in warnings:
            print(f"  * {warning}")
        for note in all_notes:
            print(f"  - {note}")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Two report JSON files, or one directory holding them.",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Tag of the arm to treat as the baseline. Default: the older report.",
    )
    args = parser.parse_args(argv)

    if len(args.paths) == 1 and args.paths[0].is_dir():
        reports = newest_per_tag(args.paths[0])
    else:
        reports = [load_report(path) for path in args.paths]

    if len(reports) < 2:
        print(
            f"Need two reports to compare, found {len(reports)}. "
            "Run the other arm, or pass both files explicitly.",
            file=sys.stderr,
        )
        return 1
    if len(reports) > 2:
        print(
            f"Found {len(reports)} reports ({', '.join(r['_label'] for r in reports)}); "
            "pass two explicitly.",
            file=sys.stderr,
        )
        return 1

    baseline, candidate = reports
    if args.baseline and candidate["_label"] == args.baseline:
        baseline, candidate = candidate, baseline
    return compare(baseline, candidate)


if __name__ == "__main__":
    raise SystemExit(main())
