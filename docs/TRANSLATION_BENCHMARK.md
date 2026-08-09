# Multi-model translation benchmark

`benchmark_translations.py` compares aligned outputs from any number of base,
fine-tuned, and externally-run translation models. Its primary deliverable is a
self-contained HTML report for human review; it also preserves CSV artifacts
for analysis and reproducibility.

For the concrete Docker procedure comparing two imported base outputs with
fine-tuned NLLB and TranslateGemma generation, use
`docs/TRANSLATION_BENCHMARK_RUNBOOK.md`.

The benchmark deliberately separates translation generation from metric
scoring and presentation. A saved model output can be rescored with a new
metric or rendered into a new report without loading that translation model
again.

## What it compares

A **candidate** is one translation system and configuration. For example:

- TranslateGemma 4B, 12B, or 27B base;
- the same TranslateGemma checkpoint with a LoRA adapter;
- base or fine-tuned NLLB variants;
- translations produced earlier by NLLB, an API, or another system.

Generated and imported candidates have the same canonical output contract, so
they appear together in every leaderboard, paired comparison, slice, and
human-review table.

## Architecture

```text
versioned evaluation dataset
        │
        ├── generated candidate (TranslateGemma/NLLB runner)
        └── imported candidate (CSV/TSV/JSONL/Parquet)
                         │
                         ▼
         candidates/<candidate-id>/translations.csv
                         │
                         ▼
          transparent + learned metric scoring
                         │
                         ▼
        paired statistics, slices, CSV and HTML report
```

The stages have independent commands:

- `validate`: validate configuration, candidate IDs, and the dataset contract;
- `generate`: run only candidates whose type is `generated`;
- `import`: canonicalize only candidates whose type is `imported`;
- `collect`: generate and import selected candidates;
- `score`: calculate metrics from already-collected translations;
- `report`: rebuild HTML/Markdown from already-calculated scores;
- `run`: collect, score, and report in one invocation.

## Dataset contract

CSV, TSV, JSON, JSONL, and Parquet inputs are supported. Configure the column
mapping under `benchmark.dataset.columns`. Three values are required:

| Canonical value | Purpose |
| --- | --- |
| `id` | Stable unique example identifier |
| `source` | Source-language text |
| `reference` | Trusted reference translation |

`domain`, `document_id`, and any configured slice columns are optional. IDs,
not row positions, align every candidate. Missing, additional, null, or
duplicate IDs cause a hard failure.

The dataset's SHA-256 and selected ID hash are written to each candidate
manifest. An existing output is reused only when both the dataset hash and
candidate configuration hash match. Use `--force` only when replacement is
intentional.

Keep the final 500-row dataset out of training and prompt/decoding tuning. Use a
separate development set for those choices. For statistically meaningful
domain reporting, ensure each important domain has enough examples; the paired
confidence intervals show when 500 total rows do not support a strong claim.

## Candidate configuration

Start with `benchmark_config.yaml`. Every candidate needs a unique filesystem-
safe `id`, a readable `label`, and a `type`.

### TranslateGemma

```yaml
- id: translategemma-12b-lora
  label: TranslateGemma 12B — LoRA
  type: generated
  runner: translategemma
  model: google/translategemma-12b-it
  adapter: translategemma-farsi-science/sft_final
  source_lang: en
  target_lang: fa
  dtype: bfloat16
  generation_profile: deterministic
```

Remove `adapter` for the base candidate. Pin a local snapshot or immutable
model revision in controlled experiments. The manifest records the complete
candidate configuration.

### NLLB

```yaml
- id: nllb-600m-base
  label: NLLB 600M — base
  type: generated
  runner: nllb
  model: facebook/nllb-200-distilled-600M
  source_lang: eng_Latn
  target_lang: pes_Arab
  dtype: bfloat16
  generation_profile: deterministic
```

NLLB requires its own language tags. A fine-tuned/LoRA NLLB candidate adds an
`adapter` exactly like TranslateGemma.

For a Hub-hosted adapter, use `adapter_repo` instead of a local `adapter` path;
the offline staging script will download it and the runner will load the same
repository ID. `adapter_revision` pins either form when supported by PEFT.

### Existing translations

```yaml
- id: nllb-existing
  label: Historical NLLB run
  type: imported
  path: existing_translations/nllb.csv
  columns:
    id: row_id
    translation: generated_text
```

The imported file must contain exactly one translation for every evaluation
ID. Additional IDs are rejected unless `allow_extra_ids: true`; with that
setting, the benchmark selects only the frozen evaluation IDs. The original
file hash is retained in the candidate manifest.

Set `enabled: false` to retain a candidate definition without including it in
default commands. An explicit `--candidates` selection overrides `enabled`,
which is useful for running one disabled candidate without editing the file:

```bash
docker compose run --rm trainer python benchmark_translations.py \
  --config benchmark_config.yaml generate \
  --candidates translategemma-12b-base translategemma-12b-lora
```

## Recommended workflow

The supported production runtime is the repository's Docker Compose `trainer`
service. The project root is mounted at `/workspace`, model snapshots are
mounted at `/models`, and benchmark outputs written below `/workspace` are
immediately visible on the host. Host-side `python` or `uv run` examples are
intended only for developer environments.

Validate before allocating a GPU:

```bash
docker compose run --rm trainer python benchmark_translations.py \
  --config benchmark_config.yaml validate
```

Import historical outputs independently:

```bash
docker compose run --rm trainer python benchmark_translations.py \
  --config benchmark_config.yaml import
```

Generate new model outputs. Running candidates separately is often more
practical when model sizes require different machines:

```bash
docker compose run --rm trainer python benchmark_translations.py \
  --config benchmark_config.yaml generate \
  --candidates translategemma-12b-base
docker compose run --rm trainer python benchmark_translations.py \
  --config benchmark_config.yaml generate \
  --candidates translategemma-12b-lora
```

Completed candidates are reused. A changed dataset or changed candidate
configuration is rejected instead of silently mixing runs. To intentionally
replace an output:

```bash
docker compose run --rm trainer python benchmark_translations.py \
  --config benchmark_config.yaml generate \
  --candidates translategemma-12b-lora --force
```

Score all enabled candidates and build the report:

```bash
docker compose run --rm trainer python benchmark_translations.py \
  --config benchmark_config.yaml score
docker compose run --rm trainer python benchmark_translations.py \
  --config benchmark_config.yaml report
```

For a small benchmark that fits on one host, run all stages together:

```bash
docker compose run --rm trainer python benchmark_translations.py \
  --config benchmark_config.yaml run
```

Separate `generate`, `score`, and `report` containers are recommended for the
real benchmark. Each process exits after its stage, guaranteeing that model
VRAM is returned before COMET or MetricX is loaded. `run` is a convenience for
small smoke runs and high-memory hosts.

Long candidate generation can be detached from SSH:

```bash
docker compose run --rm -d --name tg-benchmark-12b trainer \
  python benchmark_translations.py --config benchmark_config.yaml generate \
  --candidates translategemma-12b-lora
docker logs -f tg-benchmark-12b
```

## Metrics

The transparent metrics run by default:

| Metric | Direction | Interpretation |
| --- | --- | --- |
| sentence BLEU | higher | Per-example lexical overlap, mainly for auditability |
| sentence chrF++ | higher | Character/word n-gram overlap; useful for Persian morphology |
| number preservation | higher | Fraction of source numbers retained, with Persian/Arabic digits normalized |
| acronym preservation | higher | Fraction of uppercase Latin acronyms retained |
| formula preservation | higher | Fraction of detected formula-like strings retained |
| empty output | lower | Candidate returned no text |
| source copy | lower | Candidate copied the complete source unchanged |

Preservation scores are blank when the source contains no applicable item; the
summary mean therefore describes only relevant examples.

COMET and MetricX are optional because they load large evaluator models. They
operate on saved candidate outputs and can be enabled later without repeating
translation inference. MetricX is lower-is-better; the report labels metric
directions explicitly.

Do not interpret any single metric as ground truth. Use learned metrics for
semantic quality, chrF++ for a transparent second view, preservation checks for
scientific content, and the aligned HTML examples for direct inspection.

## Paired model comparisons

All systems translate the same examples, so the benchmark calculates paired
comparisons rather than comparing unrelated averages. For each model pair and
metric it reports:

- mean for candidate A and candidate B;
- raw delta `B - A`;
- paired bootstrap 95% confidence interval;
- A win, tie, and B win rates;
- whether the raw confidence interval excludes zero.

For lower-is-better metrics, win rates are reversed appropriately while the
reported delta remains the auditable raw `B - A`. Statistical significance
does not replace effect size or human judgment, particularly when many model
pairs are examined.

## Output layout

```text
benchmark_output/
├── candidates/
│   └── <candidate-id>/
│       ├── translations.csv
│       └── manifest.json
├── scores.csv
├── score_manifest.json
├── system_summary.csv
├── pairwise_comparisons.csv
├── slice_summary.csv
├── all_model_outputs.csv
├── report.html
├── report.md
└── report_manifest.json
```

`all_model_outputs.csv` is the simplest complete human-review export: one row
per evaluation example and one translation column per model. `scores.csv` is
long-form and includes each model's per-example metrics. Both are suitable for
spreadsheet analysis.

`score_manifest.json` records the exact dataset, candidate-output hashes,
resolved benchmark configuration, metric settings, and statistical settings
used to derive the comparison. Report generation refuses to combine saved
scores with a subsequently changed dataset.

`report.html` is the preferred review artifact. It is standalone, requires no
server, includes aggregate and paired tables, and has a searchable aligned
table containing source, reference, every model output, and all per-example
scores. Open it locally in any browser. Set `report.example_metrics` to a list
such as `[sentence_chrf, comet, metricx]` if a narrower review table is easier
to work with.

## Human evaluation protocol

For the final decision, use the report to choose a stratified subset of roughly
100–150 rows across domains and hard phenomena. Export those rows with model
names replaced by randomized A/B labels. Reviewers should judge:

- adequacy and omissions/additions;
- fluency and natural Persian phrasing;
- scientific terminology;
- numbers, units, acronyms, and formula preservation;
- overall pairwise preference.

Candidate order should be randomized and reviewers must not see automatic
scores. Use two reviewers on at least an overlapping subset to estimate
agreement. Automatic metrics identify patterns and regressions; blinded human
review determines whether those differences matter in the actual product.

## Fair-comparison checklist

- Use the same frozen dataset and references for all candidates.
- Confirm the test rows and near-duplicates were excluded from fine-tuning.
- Pin model, adapter, tokenizer, and evaluator revisions where possible.
- Use deterministic generation for the primary comparison.
- Select prompts, beams, and other decoding settings on a development set.
- Preserve each model family's required native prompt or language-tag format.
- Record truncations and empty outputs rather than dropping rows.
- Compare quality and latency separately; do not hide them in one composite score.
- Keep candidate manifests and the resolved benchmark configuration with published results.

The existing `evaluate_translations.py` remains suitable for a quick base-versus-
adapter check during training. Use this benchmark for the final cross-family,
cross-size comparison and human-facing review.

## Docker image and offline assets

The standard image contains every Python runtime dependency used here:
Transformers and PEFT for both runners, Pandas and SacreBLEU for transparent
metrics, and COMET. The default `INSTALL_METRICX=1` image also vendors MetricX
under `/opt/metricx`. No NLLB-specific Python package is required.

Model weights are not baked into the image. Before exporting the offline model
archive, enable every generated candidate that must be runnable offline and use
the benchmark-aware staging command:

```bash
uv run --no-project --with huggingface_hub --with pyyaml \
  python scripts/fetch_offline_assets.py \
  --config config.yaml \
  --testset-config testset_config.yaml \
  --benchmark-config benchmark_config.yaml \
  --dest offline_assets/models
```

The fetcher stages each enabled generated candidate's model, explicit processor
or tokenizer, enabled benchmark evaluator checkpoints, MetricX's tokenizer,
and COMET's indirect encoder files. Candidate `revision` and
`adapter_revision` pins are passed to snapshot staging, so the offline cache
matches the revisions requested at runtime. Disabled candidates are not staged unless
their repository is supplied with a repeated `--repo` argument. Local adapter
directories and imported translation files are project/run artifacts; transfer
them beside the source and ensure their configured paths exist under the
`/workspace` bind mount.

An image built from the previous dependency set normally already contains
SacreBLEU through COMET and can run the transparent benchmark after the updated
source and model assets are mounted. Rebuilding and re-exporting from the
current `pyproject.toml` and `uv.lock` is nevertheless recommended so the image
manifest records SacreBLEU as a direct benchmark dependency.

## Production verification

This repository's implementation host may not contain the GPU/runtime packages.
On the production machine, after synchronizing the locked environment, run:

```bash
docker compose run --rm trainer \
  python benchmark_translations.py --config benchmark_config.yaml validate
```

The lean production image does not install pytest as a runtime dependency. Run
`tests/test_translation_benchmark.py` in a test-enabled image when validating a
new build; the operational production checks are configuration validation and
a small real generation pass.

Then perform a small generation smoke run by temporarily setting
`benchmark.dataset.max_examples` to 2–4 and selecting one candidate. Remove the
limit before collecting the final frozen 500-row outputs.
