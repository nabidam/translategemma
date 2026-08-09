#!/usr/bin/env python3
"""Generate, import, score, compare, and report translation model outputs."""

from __future__ import annotations

import argparse
import json

from rich.console import Console

from translation_benchmark.config import load_benchmark_config
from translation_benchmark.pipeline import collect, report, score, validate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="benchmark_config.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate")
    for name in ("collect", "generate", "import"):
        command = subparsers.add_parser(name)
        command.add_argument("--candidates", nargs="+")
        command.add_argument("--force", action="store_true")
    scoring = subparsers.add_parser("score")
    scoring.add_argument("--candidates", nargs="+")
    subparsers.add_parser("report")
    run = subparsers.add_parser("run")
    run.add_argument("--candidates", nargs="+")
    run.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_benchmark_config(args.config)
    console = Console()
    if args.command == "validate":
        console.print_json(json.dumps(validate(config), ensure_ascii=False))
        return
    if args.command in {"collect", "generate", "import"}:
        kind = {"generate": "generated", "import": "imported"}.get(args.command)
        paths = collect(config, args.candidates, kind, args.force)
        console.print(f"Collected {len(paths)} candidate output(s).")
        return
    if args.command == "score":
        paths = score(config, args.candidates)
        console.print(f"Scores written to {config.output_dir}")
        return
    if args.command == "report":
        html_path, _ = report(config)
        console.print(f"HTML report: {html_path}")
        return
    collect(config, args.candidates, None, args.force)
    score(config, args.candidates)
    html_path, _ = report(config)
    console.print(f"Benchmark complete. HTML report: {html_path}")


if __name__ == "__main__":
    main()
