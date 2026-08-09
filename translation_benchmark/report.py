from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import BenchmarkConfig
from .metrics import METRIC_DIRECTIONS, metric_columns


def _table(frame: pd.DataFrame, table_id: str = "") -> str:
    if frame.empty:
        return "<p class='muted'>No data available.</p>"
    display = frame.copy()
    for column in display.select_dtypes(include="number").columns:
        display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    return display.to_html(index=False, escape=True, border=0, table_id=table_id, classes="data-table")


def _example_table(
    scored: pd.DataFrame,
    candidates: list[dict[str, Any]],
    slices: list[str],
    requested_metrics: list[str] | None,
) -> pd.DataFrame:
    base_columns = list(dict.fromkeys(
        column for column in ["example_id", "domain", *slices, "source", "reference"] if column in scored
    ))
    base = scored[base_columns].drop_duplicates("example_id").set_index("example_id")
    labels = scored.drop_duplicates("candidate_id").set_index("candidate_id")["candidate_label"].to_dict()
    translations = scored.pivot(index="example_id", columns="candidate_id", values="translation")
    translations.columns = [f"Translation — {labels.get(column, column)} [{column}]" for column in translations.columns]
    available_metrics = metric_columns(scored)
    metrics = [column for column in (requested_metrics or available_metrics) if column in available_metrics]
    for metric in metrics:
        values = scored.pivot(index="example_id", columns="candidate_id", values=metric)
        values.columns = [f"{metric} — {labels.get(column, column)} [{column}]" for column in values.columns]
        translations = translations.join(values)
    return base.join(translations).reset_index()


def write_reports(
    config: BenchmarkConfig,
    dataset_manifest: dict[str, Any],
    scored: pd.DataFrame,
    summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    slices: pd.DataFrame,
) -> tuple[Path, Path]:
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    configured_slices = config.raw.get("report", {}).get("slices", ["domain"])
    example_metrics = config.raw.get("report", {}).get("example_metrics")
    examples = _example_table(scored, config.candidates, configured_slices, example_metrics)
    metric_notes = ", ".join(
        f"{name}: {direction}" for name, direction in METRIC_DIRECTIONS.items() if name in scored
    )
    title = str(config.benchmark.get("title", "Translation Model Benchmark"))
    summary_html = _table(summary)
    pairwise_html = _table(pairwise)
    slices_html = _table(slices)
    examples_html = _table(examples, "examples")
    report_html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{ --ink:#17212b; --muted:#64748b; --paper:#f7f4ed; --card:#fff; --accent:#075985; --line:#d8dee6; }}
* {{ box-sizing:border-box }} body {{ margin:0; background:var(--paper); color:var(--ink); font:15px/1.5 system-ui,sans-serif }}
main {{ max-width:1500px; margin:auto; padding:40px 24px 80px }} h1 {{ font-size:2.4rem; margin:0 0 8px }} h2 {{ margin-top:42px }}
.lede,.muted {{ color:var(--muted) }} .card {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:18px; overflow:auto; box-shadow:0 5px 20px #17212b0a }}
.data-table {{ border-collapse:collapse; width:100%; font-size:13px }} th,td {{ border-bottom:1px solid var(--line); padding:9px 11px; text-align:left; vertical-align:top }}
th {{ position:sticky; top:0; background:#eaf2f6; color:#0c4a6e; white-space:nowrap }} td {{ min-width:90px }}
#examples td {{ min-width:180px; max-width:440px; white-space:pre-wrap; unicode-bidi:plaintext }} #examples td:first-child {{ min-width:80px }}
input {{ width:100%; max-width:540px; padding:11px 13px; border:1px solid var(--line); border-radius:8px; margin:0 0 14px; background:white }}
code {{ background:#e8edf1; padding:2px 5px; border-radius:4px }}
</style></head><body><main>
<h1>{html.escape(title)}</h1>
<p class="lede">{dataset_manifest['rows']} aligned examples · dataset SHA-256 <code>{dataset_manifest['sha256'][:16]}…</code></p>
<p class="muted">Metric direction: {html.escape(metric_notes)}. Pairwise deltas are candidate B minus candidate A.</p>
<h2>System leaderboard</h2><div class="card">{summary_html}</div>
<h2>Paired comparisons</h2><div class="card">{pairwise_html}</div>
<h2>Dataset slices</h2><div class="card">{slices_html}</div>
<h2>Human review explorer</h2>
<input id="search" type="search" placeholder="Filter by ID, domain, source, reference, or translation…" oninput="filterRows()">
<div class="card">{examples_html}</div>
<script>function filterRows(){{const q=document.getElementById('search').value.toLowerCase();document.querySelectorAll('#examples tbody tr').forEach(r=>r.style.display=r.innerText.toLowerCase().includes(q)?'':'none')}}</script>
</main></body></html>"""
    html_path = output / "report.html"
    html_path.write_text(report_html, encoding="utf-8")

    markdown = [
        f"# {title}", "", f"Dataset rows: {dataset_manifest['rows']}",
        f"Dataset SHA-256: `{dataset_manifest['sha256']}`", "",
        "## System leaderboard", "", summary.to_markdown(index=False), "",
        "## Paired comparisons", "", pairwise.to_markdown(index=False) if not pairwise.empty else "No pairwise results.", "",
        "## Notes", "", f"Metric direction: {metric_notes}.",
        "The HTML report contains the complete aligned human-review explorer.", "",
    ]
    markdown_path = output / "report.md"
    markdown_path.write_text("\n".join(markdown), encoding="utf-8")
    manifest = {
        "dataset": dataset_manifest,
        "candidates": sorted(scored["candidate_id"].unique().tolist()),
        "artifacts": ["scores.csv", "score_manifest.json", "system_summary.csv", "pairwise_comparisons.csv", "slice_summary.csv", "all_model_outputs.csv", "report.html", "report.md"],
    }
    (output / "report_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return html_path, markdown_path
