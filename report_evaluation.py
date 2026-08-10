"""Build a self-contained HTML review report from evaluate_translations.py outputs.

Reads every ``<prefix>_detailed_scores.csv`` in the evaluation output directory
(plus ``summary.json`` when present) and renders one offline HTML file with the
metric summary, per-domain breakdown and a side-by-side sample explorer, so a
reviewer can compare each system's translation of the same source segment.
"""

import argparse
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from logging_utils import load_config, logger, console

HYPOTHESIS_COLUMN = "generated_farsi"
SCORE_COLUMNS = {
    "metricx": {"column": "metricx_score", "label": "MetricX", "lower_is_better": True},
    "comet": {"column": "comet_score", "label": "COMET", "lower_is_better": False},
}


def _clean(value):
    """Turn a pandas cell into a JSON-safe scalar (NaN/NaT become None)."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if value is pd.NaT:
        return None
    return value


def _optional_float(value):
    value = _clean(value)
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def discover_systems(eval_dir, detailed_filename):
    """Return {prefix: csv_path} for every detailed score file in eval_dir."""
    suffix = f"_{detailed_filename}"
    found = {
        path.name[: -len(suffix)]: path
        for path in sorted(eval_dir.glob(f"*{suffix}"))
        if path.name.endswith(suffix) and len(path.name) > len(suffix)
    }
    if not found:
        raise FileNotFoundError(
            f"No '*{suffix}' files in {eval_dir}. Run evaluate_translations.py first, "
            "or point --eval-dir at the directory holding its output."
        )
    return found


def order_systems(systems, baseline_prefix, adapter_prefix):
    """Show baseline first and the adapter second; keep any extras stable after."""
    preferred = [name for name in (baseline_prefix, adapter_prefix) if name in systems]
    return preferred + [name for name in systems if name not in preferred]


def _row_key(row, index, id_column):
    """Join systems on the dataset id when it exists, else on row position."""
    identifier = _clean(row.get(id_column)) if id_column else None
    return str(identifier) if identifier not in (None, "") else f"#row{index}"


def build_samples(frames, system_names, data_cfg):
    """Merge each system's rows into one record per source segment."""
    id_column = data_cfg.get("id_column")
    domain_column = data_cfg["domain_column"]
    source_column = data_cfg["source_column"]
    target_column = data_cfg["target_column"]

    samples = {}
    order = []
    for name in system_names:
        frame = frames[name]
        missing = {source_column, HYPOTHESIS_COLUMN} - set(frame.columns)
        if missing:
            raise ValueError(f"{name} results are missing columns: {sorted(missing)}")
        for index, row in enumerate(frame.to_dict("records")):
            key = _row_key(row, index, id_column)
            sample = samples.get(key)
            if sample is None:
                sample = {
                    "id": _clean(row.get(id_column)) if id_column else None,
                    "key": key,
                    "domain": _clean(row.get(domain_column)) or "—",
                    "source": _clean(row.get(source_column)) or "",
                    "reference": _clean(row.get(target_column)) or "",
                    "outputs": {},
                }
                samples[key] = sample
                order.append(key)
            sample["outputs"][name] = {
                "text": _clean(row.get(HYPOTHESIS_COLUMN)) or "",
                **{
                    metric: _optional_float(row.get(spec["column"]))
                    for metric, spec in SCORE_COLUMNS.items()
                    if spec["column"] in frame.columns
                },
            }
    return [samples[key] for key in order]


def _mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def summarize_systems(samples, system_names, summary_entries):
    """Corpus-level stats per system, merged with evaluate_translations' summary.json."""
    stats = {}
    for name in system_names:
        outputs = [sample["outputs"].get(name) for sample in samples]
        present = [output for output in outputs if output is not None]
        texts = [output["text"] for output in present]
        entry = summary_entries.get(name, {})
        stats[name] = {
            "label": name,
            "adapter_path": entry.get("adapter_path"),
            "examples": len(present),
            "empty_outputs": sum(1 for text in texts if not text.strip()),
            "avg_chars": _mean([float(len(text)) for text in texts]),
            "metrics": {
                metric: _mean([output.get(metric) for output in present])
                for metric in SCORE_COLUMNS
            },
        }
    return stats


def summarize_domains(samples, system_names):
    """Per-domain counts and per-system metric means, sorted by sample count."""
    domains = {}
    for sample in samples:
        bucket = domains.setdefault(
            sample["domain"],
            {"domain": sample["domain"], "count": 0, "systems": {name: {metric: [] for metric in SCORE_COLUMNS} for name in system_names}},
        )
        bucket["count"] += 1
        for name in system_names:
            output = sample["outputs"].get(name)
            if not output:
                continue
            for metric in SCORE_COLUMNS:
                if output.get(metric) is not None:
                    bucket["systems"][name][metric].append(output[metric])
    rows = []
    for bucket in domains.values():
        rows.append({
            "domain": bucket["domain"],
            "count": bucket["count"],
            "systems": {
                name: {metric: _mean(values) for metric, values in metrics.items()}
                for name, metrics in bucket["systems"].items()
            },
        })
    return sorted(rows, key=lambda row: (-row["count"], str(row["domain"])))


def available_metrics(samples, system_names):
    """Metrics that at least one system actually scored."""
    return [
        metric
        for metric in SCORE_COLUMNS
        if any(
            (sample["outputs"].get(name) or {}).get(metric) is not None
            for sample in samples
            for name in system_names
        )
    ]


def build_payload(config, eval_dir, args):
    data_cfg, eval_cfg = config["data"], config["evaluation"]
    system_paths = discover_systems(eval_dir, eval_cfg["detailed_filename"])
    system_names = order_systems(
        list(system_paths), eval_cfg["baseline_prefix"], eval_cfg["adapter_prefix"]
    )
    frames = {name: pd.read_csv(system_paths[name]) for name in system_names}
    for name in system_names:
        logger.info("Loaded %s rows for system '%s' from %s", len(frames[name]), name, system_paths[name])

    samples = build_samples(frames, system_names, data_cfg)
    if args.max_samples is not None and len(samples) > args.max_samples:
        logger.info("Embedding the first %s of %s samples (--max-samples).", args.max_samples, len(samples))
        samples = samples[: args.max_samples]

    summary_path = eval_dir / eval_cfg["summary_filename"]
    summary_entries = {}
    if summary_path.is_file():
        for entry in json.loads(summary_path.read_text(encoding="utf-8")):
            summary_entries[entry.get("label")] = entry

    metrics = available_metrics(samples, system_names)
    return {
        "title": args.title or f"Translation evaluation — {eval_dir.name}",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "eval_dir": str(eval_dir.resolve()),
        "test_dataset_path": data_cfg.get("test_dataset_path"),
        "base_model_id": config.get("model", {}).get("base_model_id"),
        "systems": system_names,
        "metric_specs": {
            metric: {"label": spec["label"], "lower_is_better": spec["lower_is_better"]}
            for metric, spec in SCORE_COLUMNS.items()
            if metric in metrics
        },
        "stats": summarize_systems(samples, system_names, summary_entries),
        "domains": summarize_domains(samples, system_names),
        "samples": samples,
    }


def render_html(payload):
    """Inline the payload into the static template.

    The JSON is embedded in a non-executed <script type="application/json"> tag
    and its '<' characters escaped, so translated text containing markup cannot
    close the tag or inject script.
    """
    data = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    return HTML_TEMPLATE.replace("__PAYLOAD__", data).replace(
        "__TITLE__", re.sub(r"[<>&]", " ", str(payload["title"]))
    )


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root {
    color-scheme: light dark;
    --bg: #f6f7f9; --panel: #ffffff; --panel-2: #fbfbfd; --border: #e3e5ea;
    --text: #14161a; --muted: #666d7a; --accent: #3055d8; --good: #16794a;
    --bad: #b3261e; --chip: #eef1f8; --shadow: 0 1px 2px rgba(16,20,30,.06), 0 8px 24px rgba(16,20,30,.05);
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #0e1014; --panel: #161a21; --panel-2: #1b2028; --border: #262c36;
      --text: #e8eaef; --muted: #98a1b0; --accent: #7d9bff; --good: #4ec98a;
      --bad: #ff8a80; --chip: #222836; --shadow: none;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--text);
    font: 15px/1.55 ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Noto Sans", sans-serif;
  }
  .wrap { max-width: 1280px; margin: 0 auto; padding: 28px 20px 80px; }
  header h1 { margin: 0 0 6px; font-size: 24px; letter-spacing: -0.01em; }
  .sub { color: var(--muted); font-size: 13px; }
  .sub code { background: var(--chip); padding: 1px 6px; border-radius: 5px; }
  section { margin-top: 28px; }
  h2 { font-size: 15px; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); margin: 0 0 12px; }
  .cards { display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); }
  .card, .panel {
    background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
    padding: 16px; box-shadow: var(--shadow);
  }
  .card h3 { margin: 0 0 2px; font-size: 15px; }
  .card .path { color: var(--muted); font-size: 12px; word-break: break-all; margin-bottom: 10px; }
  .metric-row { display: flex; justify-content: space-between; padding: 5px 0; border-top: 1px dashed var(--border); }
  .metric-row span:first-child { color: var(--muted); font-size: 13px; }
  .num { font-variant-numeric: tabular-nums; font-weight: 600; }
  table { width: 100%; border-collapse: collapse; font-size: 14px; }
  th, td { text-align: left; padding: 9px 10px; border-bottom: 1px solid var(--border); }
  th { color: var(--muted); font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: .05em; }
  td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
  .scroll { overflow-x: auto; }
  .good { color: var(--good); } .bad { color: var(--bad); }
  .controls { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin-bottom: 14px; }
  input[type="search"], select, button {
    font: inherit; color: var(--text); background: var(--panel); border: 1px solid var(--border);
    border-radius: 9px; padding: 8px 11px;
  }
  input[type="search"] { flex: 1 1 260px; }
  button { cursor: pointer; }
  button:hover, button:focus-visible { border-color: var(--accent); }
  button.toggle[aria-pressed="true"] { background: var(--accent); border-color: var(--accent); color: #fff; }
  .count { color: var(--muted); font-size: 13px; margin-inline-start: auto; }
  .sample { margin-bottom: 14px; }
  .sample-head { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-bottom: 10px; }
  .chip { background: var(--chip); border-radius: 999px; padding: 2px 10px; font-size: 12px; color: var(--muted); }
  .grid { display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }
  .cell { background: var(--panel-2); border: 1px solid var(--border); border-radius: 10px; padding: 12px; }
  .cell h4 { margin: 0 0 8px; font-size: 12px; text-transform: uppercase; letter-spacing: .06em; color: var(--muted);
             display: flex; justify-content: space-between; gap: 8px; align-items: baseline; }
  .text { white-space: pre-wrap; overflow-wrap: anywhere; }
  .text[dir="rtl"] { font-size: 16px; line-height: 1.9; }
  .empty { color: var(--bad); font-style: italic; }
  ins { background: rgba(78,201,138,.22); text-decoration: none; border-radius: 3px; }
  del { background: rgba(255,138,128,.22); text-decoration: line-through; border-radius: 3px; }
  .pager { display: flex; gap: 10px; align-items: center; justify-content: center; margin-top: 18px; }
  footer { margin-top: 36px; color: var(--muted); font-size: 12px; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1 id="title"></h1>
    <div class="sub" id="meta"></div>
  </header>

  <section>
    <h2>Systems</h2>
    <div class="cards" id="cards"></div>
  </section>

  <section id="compare-section">
    <h2>Head-to-head</h2>
    <div class="panel scroll"><table id="compare"></table></div>
  </section>

  <section>
    <h2>By domain</h2>
    <div class="panel scroll"><table id="domains"></table></div>
  </section>

  <section>
    <h2>Samples</h2>
    <div class="controls">
      <input type="search" id="q" placeholder="Search source, reference or translations…">
      <select id="domain"></select>
      <select id="sort"></select>
      <button class="toggle" id="diff" aria-pressed="false">Diff vs reference</button>
      <span class="count" id="count"></span>
    </div>
    <div id="samples"></div>
    <div class="pager">
      <button id="prev">← Prev</button><span id="page" class="sub"></span><button id="next">Next →</button>
    </div>
  </section>

  <footer id="footer"></footer>
</div>

<script type="application/json" id="payload">__PAYLOAD__</script>
<script>
const DATA = JSON.parse(document.getElementById("payload").textContent);
const METRICS = Object.entries(DATA.metric_specs);
const PAGE_SIZE = 25;
const RTL = /[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]/;

const fmt = (v, digits = 4) => (v === null || v === undefined ? "—" : Number(v).toFixed(digits));
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const dirOf = (s) => (RTL.test(String(s || "")) ? "rtl" : "ltr");

// Better of two metric means, honouring each metric's direction.
function improvement(metric, base, other) {
  if (base === null || other === null || base === undefined || other === undefined) return null;
  const spec = DATA.metric_specs[metric];
  return spec.lower_is_better ? base - other : other - base;
}

document.getElementById("title").textContent = DATA.title;
document.getElementById("meta").innerHTML = [
  `Generated ${esc(DATA.generated_at)}`,
  DATA.base_model_id ? `base model <code>${esc(DATA.base_model_id)}</code>` : null,
  DATA.test_dataset_path ? `test set <code>${esc(DATA.test_dataset_path)}</code>` : null,
  `${DATA.samples.length} samples · ${DATA.systems.length} system(s)`,
].filter(Boolean).join(" · ");
document.getElementById("footer").textContent = `Source: ${DATA.eval_dir}`;

/* Summary cards */
document.getElementById("cards").innerHTML = DATA.systems.map((name) => {
  const s = DATA.stats[name];
  const rows = METRICS.map(([metric, spec]) =>
    `<div class="metric-row"><span>${esc(spec.label)} ${spec.lower_is_better ? "↓" : "↑"}</span><span class="num">${fmt(s.metrics[metric])}</span></div>`
  ).join("");
  return `<div class="card">
    <h3>${esc(name)}</h3>
    <div class="path">${esc(s.adapter_path || "base model (no adapter)")}</div>
    <div class="metric-row"><span>Examples</span><span class="num">${s.examples}</span></div>
    <div class="metric-row"><span>Avg. characters</span><span class="num">${fmt(s.avg_chars, 1)}</span></div>
    <div class="metric-row"><span>Empty outputs</span><span class="num ${s.empty_outputs ? "bad" : ""}">${s.empty_outputs}</span></div>
    ${rows}
  </div>`;
}).join("");

/* Head-to-head: every system against the first one */
const baseName = DATA.systems[0];
const compareSection = document.getElementById("compare-section");
if (DATA.systems.length < 2 || METRICS.length === 0) {
  compareSection.style.display = "none";
} else {
  const head = `<tr><th>System</th>${METRICS.map(([, s]) => `<th class="num">${esc(s.label)}</th>`).join("")}` +
    `${METRICS.map(([, s]) => `<th class="num">Δ ${esc(s.label)} vs ${esc(baseName)}</th>`).join("")}` +
    `${METRICS.map(([, s]) => `<th class="num">Wins (${esc(s.label)})</th>`).join("")}</tr>`;
  const body = DATA.systems.map((name) => {
    const cells = METRICS.map(([metric]) => `<td class="num">${fmt(DATA.stats[name].metrics[metric])}</td>`).join("");
    const deltas = METRICS.map(([metric]) => {
      const d = name === baseName ? null : improvement(metric, DATA.stats[baseName].metrics[metric], DATA.stats[name].metrics[metric]);
      const cls = d === null ? "" : d > 0 ? "good" : d < 0 ? "bad" : "";
      return `<td class="num ${cls}">${d === null ? "—" : (d > 0 ? "+" : "") + fmt(d)}</td>`;
    }).join("");
    const wins = METRICS.map(([metric]) => {
      if (name === baseName) return `<td class="num">—</td>`;
      let win = 0, total = 0;
      for (const sample of DATA.samples) {
        const a = (sample.outputs[baseName] || {})[metric], b = (sample.outputs[name] || {})[metric];
        const d = improvement(metric, a, b);
        if (d === null) continue;
        total++; if (d > 0) win++;
      }
      return `<td class="num">${total ? `${win}/${total} (${((100 * win) / total).toFixed(1)}%)` : "—"}</td>`;
    }).join("");
    return `<tr><td>${esc(name)}</td>${cells}${deltas}${wins}</tr>`;
  }).join("");
  document.getElementById("compare").innerHTML = head + body;
}

/* Domain table */
document.getElementById("domains").innerHTML =
  `<tr><th>Domain</th><th class="num">Samples</th>` +
  DATA.systems.map((n) => METRICS.map(([, s]) => `<th class="num">${esc(n)} · ${esc(s.label)}</th>`).join("")).join("") +
  `</tr>` +
  DATA.domains.map((row) =>
    `<tr><td>${esc(row.domain)}</td><td class="num">${row.count}</td>` +
    DATA.systems.map((n) => METRICS.map(([metric]) => `<td class="num">${fmt(row.systems[n][metric])}</td>`).join("")).join("") +
    `</tr>`
  ).join("");

/* Word-level diff (LCS) used to highlight a translation against the reference */
function diffWords(reference, candidate) {
  const a = String(reference).split(/(\s+)/), b = String(candidate).split(/(\s+)/);
  const n = a.length, m = b.length;
  const lcs = Array.from({ length: n + 1 }, () => new Uint32Array(m + 1));
  for (let i = n - 1; i >= 0; i--)
    for (let j = m - 1; j >= 0; j--)
      lcs[i][j] = a[i] === b[j] ? lcs[i + 1][j + 1] + 1 : Math.max(lcs[i + 1][j], lcs[i][j + 1]);
  let out = "", i = 0, j = 0;
  while (i < n && j < m) {
    if (a[i] === b[j]) { out += esc(b[j]); i++; j++; }
    else if (lcs[i + 1][j] >= lcs[i][j + 1]) { i++; }
    else { out += `<ins>${esc(b[j])}</ins>`; j++; }
  }
  while (j < m) { out += `<ins>${esc(b[j])}</ins>`; j++; }
  return out;
}

/* Sample explorer */
const state = { query: "", domain: "", sort: "index", diff: false, page: 0 };

const domainSelect = document.getElementById("domain");
domainSelect.innerHTML = `<option value="">All domains</option>` +
  DATA.domains.map((d) => `<option value="${esc(d.domain)}">${esc(d.domain)} (${d.count})</option>`).join("");

const sortOptions = [{ value: "index", label: "Dataset order" }];
if (DATA.systems.length > 1) {
  for (const [metric, spec] of METRICS) {
    for (const name of DATA.systems.slice(1)) {
      sortOptions.push({ value: `gain:${metric}:${name}`, label: `Biggest ${spec.label} gain (${name} vs ${baseName})` });
      sortOptions.push({ value: `loss:${metric}:${name}`, label: `Biggest ${spec.label} regression (${name} vs ${baseName})` });
    }
  }
}
for (const [metric, spec] of METRICS) {
  sortOptions.push({ value: `worst:${metric}`, label: `Worst ${spec.label} (any system)` });
}
document.getElementById("sort").innerHTML = sortOptions.map((o) => `<option value="${o.value}">${esc(o.label)}</option>`).join("");

function scoreOf(sample, name, metric) {
  const output = sample.outputs[name];
  return output ? output[metric] ?? null : null;
}

function sortKey(sample) {
  const [kind, metric, name] = state.sort.split(":");
  if (kind === "index") return 0;
  if (kind === "worst") {
    const spec = DATA.metric_specs[metric];
    const values = DATA.systems.map((n) => scoreOf(sample, n, metric)).filter((v) => v !== null);
    if (!values.length) return Infinity;
    return spec.lower_is_better ? -Math.max(...values) : Math.min(...values);
  }
  const delta = improvement(metric, scoreOf(sample, baseName, metric), scoreOf(sample, name, metric));
  if (delta === null) return Infinity;
  return kind === "gain" ? -delta : delta;
}

function filtered() {
  const q = state.query.trim().toLowerCase();
  let rows = DATA.samples.filter((sample) => {
    if (state.domain && String(sample.domain) !== state.domain) return false;
    if (!q) return true;
    const haystack = [sample.source, sample.reference, sample.id, ...DATA.systems.map((n) => (sample.outputs[n] || {}).text)];
    return haystack.some((value) => String(value ?? "").toLowerCase().includes(q));
  });
  if (state.sort !== "index") rows = rows.map((s) => [sortKey(s), s]).sort((x, y) => x[0] - y[0]).map(([, s]) => s);
  return rows;
}

function cellHtml(sample, name) {
  const output = sample.outputs[name];
  if (!output) return `<div class="cell"><h4><span>${esc(name)}</span></h4><div class="empty">no output</div></div>`;
  const scores = METRICS.map(([metric, spec]) => {
    const value = output[metric];
    if (value === null || value === undefined) return null;
    const delta = name === baseName ? null : improvement(metric, scoreOf(sample, baseName, metric), value);
    const cls = delta === null ? "" : delta > 0 ? "good" : delta < 0 ? "bad" : "";
    const suffix = delta === null ? "" : ` <span class="${cls}">(${delta > 0 ? "+" : ""}${fmt(delta, 3)})</span>`;
    return `${esc(spec.label)} ${fmt(value, 3)}${suffix}`;
  }).filter(Boolean).join(" · ");
  const text = output.text.trim();
  const body = !text
    ? `<div class="empty">empty output</div>`
    : `<div class="text" dir="${dirOf(text)}">${state.diff && sample.reference ? diffWords(sample.reference, text) : esc(text)}</div>`;
  return `<div class="cell"><h4><span>${esc(name)}</span><span class="num">${scores}</span></h4>${body}</div>`;
}

function render() {
  const rows = filtered();
  const pages = Math.max(1, Math.ceil(rows.length / PAGE_SIZE));
  state.page = Math.min(state.page, pages - 1);
  const slice = rows.slice(state.page * PAGE_SIZE, (state.page + 1) * PAGE_SIZE);

  document.getElementById("count").textContent = `${rows.length} of ${DATA.samples.length} samples`;
  document.getElementById("page").textContent = `Page ${state.page + 1} / ${pages}`;
  document.getElementById("prev").disabled = state.page === 0;
  document.getElementById("next").disabled = state.page >= pages - 1;

  document.getElementById("samples").innerHTML = slice.map((sample) => `
    <div class="panel sample">
      <div class="sample-head">
        <span class="chip">${esc(sample.id ?? sample.key)}</span>
        <span class="chip">${esc(sample.domain)}</span>
      </div>
      <div class="grid">
        <div class="cell"><h4><span>Source</span></h4><div class="text" dir="${dirOf(sample.source)}">${esc(sample.source)}</div></div>
        <div class="cell"><h4><span>Reference</span></h4><div class="text" dir="${dirOf(sample.reference)}">${esc(sample.reference)}</div></div>
      </div>
      <div class="grid" style="margin-top:12px">${DATA.systems.map((name) => cellHtml(sample, name)).join("")}</div>
    </div>`).join("") || `<div class="panel sub">No samples match the current filters.</div>`;
}

document.getElementById("q").addEventListener("input", (e) => { state.query = e.target.value; state.page = 0; render(); });
domainSelect.addEventListener("change", (e) => { state.domain = e.target.value; state.page = 0; render(); });
document.getElementById("sort").addEventListener("change", (e) => { state.sort = e.target.value; state.page = 0; render(); });
document.getElementById("diff").addEventListener("click", (e) => {
  state.diff = !state.diff;
  e.currentTarget.setAttribute("aria-pressed", String(state.diff));
  render();
});
document.getElementById("prev").addEventListener("click", () => { state.page--; render(); window.scrollTo({ top: document.getElementById("samples").offsetTop - 80 }); });
document.getElementById("next").addEventListener("click", () => { state.page++; render(); window.scrollTo({ top: document.getElementById("samples").offsetTop - 80 }); });

render();
</script>
</body>
</html>
"""


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--eval-dir", default=None, help="Defaults to evaluation.output_dir from the config.")
    parser.add_argument("--output", default=None, help="Report path. Defaults to <eval-dir>/report.html.")
    parser.add_argument("--max-samples", type=int, default=None, help="Embed at most this many samples.")
    parser.add_argument("--title", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    eval_dir = Path(args.eval_dir or config["evaluation"]["output_dir"])
    if not eval_dir.is_dir():
        raise FileNotFoundError(f"Evaluation directory does not exist: {eval_dir.resolve()}")
    payload = build_payload(config, eval_dir, args)
    output_path = Path(args.output) if args.output else eval_dir / "report.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_html(payload), encoding="utf-8")
    console.print(
        f"[bold green]Report written:[/bold green] {output_path.resolve()} "
        f"({len(payload['samples'])} samples, systems: {', '.join(payload['systems'])})"
    )


if __name__ == "__main__":
    main()
