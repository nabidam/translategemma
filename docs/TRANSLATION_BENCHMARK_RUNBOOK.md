# Translation benchmark production runbook

This runbook produces the final human-readable comparison for four translation
systems evaluated on the same frozen 500-row dataset:

1. NLLB base translations that already exist in a CSV file;
2. TranslateGemma base translations that already exist in a CSV file;
3. newly generated translations from a fine-tuned NLLB revision;
4. newly generated translations from a fine-tuned TranslateGemma revision.

The supported production runtime is the Docker Compose `trainer` service. All
paths below are relative to the repository root, which is mounted at
`/workspace` inside the container. Generated CSV and HTML files therefore
remain visible on the host after each container exits.

For benchmark architecture, metric interpretation, and the complete artifact
contract, see `docs/TRANSLATION_BENCHMARK.md`. For image building, transfer, and
air-gapped installation, see `docs/OFFLINE_DEPLOYMENT.md`.

## 1. Prepare the frozen evaluation dataset

Place the evaluation dataset inside the repository, for example:

```text
data/benchmark/evaluation.csv
```

The required columns are a stable ID, source text, and trusted reference
translation. A domain column is strongly recommended:

```csv
id,source_text,target_text,domain
doc1:1,English source text,ترجمه مرجع,physics
doc1:2,Another source text,ترجمه مرجع دوم,chemistry
```

Requirements:

- `id` must be non-null and unique;
- the same ID must identify the same source in every candidate file;
- `target_text` must be the human reference, not another model output;
- the 500 rows and their near-duplicates must not have been used for training;
- prompt, decoding, and checkpoint choices must be made on a separate
  development set, not these final rows.

## 2. Place the existing base translations

Use separate files such as:

```text
existing_translations/
├── nllb_base.csv
└── translategemma_base.csv
```

Each file needs an ID and translation column:

```csv
id,translation
doc1:1,ترجمه مدل
doc1:2,ترجمه مدل دوم
```

Row order is irrelevant because alignment is strictly ID-based. Missing or
duplicate IDs fail validation. If a historical output file contains a superset
of the 500 evaluation IDs, set `allow_extra_ids: true` for that candidate.

## 3. Place the fine-tuned revisions

For local LoRA adapters, a typical layout is:

```text
models/
├── nllb_finetuned/
└── translategemma_finetuned/
```

These become `/workspace/models/...` in the container. If a candidate is a
complete fine-tuned checkpoint rather than a LoRA adapter, configure that
directory as `model` and omit `adapter`.

For a Hub-hosted adapter, configure `adapter_repo` and optionally
`adapter_revision`; the offline staging script will include it. Local adapters
are not downloaded and must be transferred with the source/data artifacts.

## 4. Configure the four candidates

Edit `benchmark_config.yaml` as follows, adjusting model sizes, paths, and batch
sizes to the actual revisions:

```yaml
benchmark:
  title: "NLLB vs TranslateGemma — Base and Fine-tuned"
  output_dir: "benchmark_output"
  dataset:
    path: "data/benchmark/evaluation.csv"
    source_lang: "en"
    target_lang: "fa"
    columns:
      id: "id"
      source: "source_text"
      reference: "target_text"
      domain: "domain"

candidates:
  - id: "nllb-base"
    label: "NLLB — Base"
    family: "nllb"
    size: "600m"
    type: "imported"
    path: "existing_translations/nllb_base.csv"
    columns:
      id: "id"
      translation: "translation"
    allow_extra_ids: false
    enabled: true

  - id: "translategemma-base"
    label: "TranslateGemma 12B — Base"
    family: "translategemma"
    size: "12b"
    type: "imported"
    path: "existing_translations/translategemma_base.csv"
    columns:
      id: "id"
      translation: "translation"
    allow_extra_ids: false
    enabled: true

  - id: "nllb-finetuned"
    label: "NLLB 600M — Fine-tuned"
    family: "nllb"
    size: "600m"
    type: "generated"
    runner: "nllb"
    model: "facebook/nllb-200-distilled-600M"
    adapter: "models/nllb_finetuned"
    source_lang: "eng_Latn"
    target_lang: "pes_Arab"
    dtype: "bfloat16"
    generation_profile: "deterministic"
    generation:
      batch_size: 16
    enabled: true

  - id: "translategemma-finetuned"
    label: "TranslateGemma 12B — Fine-tuned"
    family: "translategemma"
    size: "12b"
    type: "generated"
    runner: "translategemma"
    model: "google/translategemma-12b-it"
    adapter: "models/translategemma_finetuned"
    source_lang: "en"
    target_lang: "fa"
    dtype: "bfloat16"
    attn_implementation: "sdpa"
    generation_profile: "deterministic"
    generation:
      batch_size: 4
    enabled: true

generation_profiles:
  deterministic:
    max_new_tokens: 1024
    do_sample: false
    num_beams: 1

metrics:
  transparent:
    enabled: true
    chrf_word_order: 2
  comet:
    enabled: true
    model: "Unbabel/wmt22-comet-da"
    batch_size: 8
    gpus: 1
  metricx:
    enabled: true
    model: "google/metricx-24-hybrid-large-v2p6"
    tokenizer: "google/mt5-xl"
    max_length: 1536

statistics:
  bootstrap_samples: 2000
  seed: 42

report:
  slices:
    - "domain"
    - "has_math"
    - "has_numbers_units"
    - "has_acronyms"
    - "has_mixed_script"
  example_metrics:
    - "sentence_chrf"
    - "number_preservation"
    - "acronym_preservation"
    - "formula_preservation"
    - "comet"
    - "metricx"
```

Slice columns absent from the evaluation file are ignored.

### Full fine-tuned NLLB checkpoint

If NLLB is a complete checkpoint, replace its model/adapter fields with:

```yaml
model: "models/nllb_finetuned"
```

Do not configure `adapter` in that case.

### Immutable Hub revisions

For a pinned model revision:

```yaml
model: "facebook/nllb-200-distilled-600M"
revision: "<immutable-commit-sha>"
```

For a pinned Hub adapter:

```yaml
adapter_repo: "your-organization/nllb-farsi-adapter"
adapter_revision: "<immutable-commit-sha>"
```

The staging script and runtime use the same revisions.

## 5. Stage every remote model for offline use

Perform this step on the online staging machine after the final candidate and
metric configuration is saved. All generated candidates that must run offline
must have `enabled: true` before staging.

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxx

uv run --no-project --with huggingface_hub --with pyyaml \
  python scripts/fetch_offline_assets.py \
  --config config.yaml \
  --testset-config testset_config.yaml \
  --benchmark-config benchmark_config.yaml \
  --dest offline_assets/models
```

This stages enabled TranslateGemma/NLLB models, Hub adapters, explicit
processors/tokenizers, COMET and its indirect XLM-R files, and MetricX plus its
mT5 tokenizer. Then rebuild the portable model archive:

```bash
tar -I 'zstd -10 -T0' \
  -cf translategemma-models.tar.zst \
  offline_assets/models
```

Transfer the image, source, model archive, evaluation dataset, existing base
translation CSVs, and local adapter directories as described in the offline
deployment guide.

## 6. Verify benchmark imports in Docker

On the production machine, run the dependency preflight before loading any
weights:

```bash
docker compose run --rm trainer python -c \
  "import pandas, sacrebleu, torch, yaml; from comet import download_model, load_from_checkpoint; from metricx24.models import MT5ForRegression; from peft import PeftModel; from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoProcessor, AutoTokenizer; import translation_benchmark.config, translation_benchmark.generation, translation_benchmark.io, translation_benchmark.metrics, translation_benchmark.pipeline, translation_benchmark.report; print('translation benchmark dependency preflight: OK')"
```

Expected output:

```text
translation benchmark dependency preflight: OK
```

This command assumes an `INSTALL_METRICX=1` image. For an image intentionally
built without MetricX, remove only its import and keep MetricX disabled in the
benchmark configuration.

## 7. Validate the dataset and configuration

```bash
docker compose run --rm trainer python benchmark_translations.py \
  --config benchmark_config.yaml validate
```

This checks the dataset path, required columns, unique IDs, candidate IDs,
runner types, generation profiles, and NLLB language tags without loading
translation or evaluator weights.

## 8. Import the existing base translations

```bash
docker compose run --rm trainer python benchmark_translations.py \
  --config benchmark_config.yaml import \
  --candidates nllb-base translategemma-base
```

Expected artifacts:

```text
benchmark_output/candidates/nllb-base/translations.csv
benchmark_output/candidates/nllb-base/manifest.json
benchmark_output/candidates/translategemma-base/translations.csv
benchmark_output/candidates/translategemma-base/manifest.json
```

The command fails instead of silently dropping or reordering rows when IDs do
not align.

## 9. Generate the fine-tuned NLLB output

Run each generated candidate in a separate container so its GPU memory is
returned completely afterward:

```bash
docker compose run --rm trainer python benchmark_translations.py \
  --config benchmark_config.yaml generate \
  --candidates nllb-finetuned
```

Expected output:

```text
benchmark_output/candidates/nllb-finetuned/translations.csv
benchmark_output/candidates/nllb-finetuned/manifest.json
```

On an out-of-memory error, lower only this candidate's `generation.batch_size`
and rerun. If a completed candidate configuration changed, replacement requires
an explicit `--force`.

## 10. Generate the fine-tuned TranslateGemma output

```bash
docker compose run --rm trainer python benchmark_translations.py \
  --config benchmark_config.yaml generate \
  --candidates translategemma-finetuned
```

Expected output:

```text
benchmark_output/candidates/translategemma-finetuned/translations.csv
benchmark_output/candidates/translategemma-finetuned/manifest.json
```

Lower its candidate-specific `generation.batch_size` if necessary. Keep
`attn_implementation: sdpa` unless the selected image contains FlashAttention 3
and the host is a supported Hopper GPU.

For a long generation detached from SSH:

```bash
docker compose run --rm -d --name tg-benchmark-finetuned trainer \
  python benchmark_translations.py --config benchmark_config.yaml generate \
  --candidates translategemma-finetuned
docker logs -f tg-benchmark-finetuned
```

## 11. Confirm all candidate outputs

From the host:

```bash
find benchmark_output/candidates -maxdepth 2 \
  -name translations.csv -print
```

The four expected files are:

```text
benchmark_output/candidates/nllb-base/translations.csv
benchmark_output/candidates/translategemma-base/translations.csv
benchmark_output/candidates/nllb-finetuned/translations.csv
benchmark_output/candidates/translategemma-finetuned/translations.csv
```

## 12. Score all four candidates

Use a new container so no translation model occupies GPU memory:

```bash
docker compose run --rm trainer python benchmark_translations.py \
  --config benchmark_config.yaml score
```

This calculates transparent metrics, optional COMET and MetricX scores,
preservation/failure metrics, paired bootstrap confidence intervals,
win/tie/loss rates, and configured slices. Metric settings can be changed and
rescored without regenerating translations.

If an evaluator was not staged, disable it before scoring. Do not delete the
candidate translation artifacts.

## 13. Build the final report

```bash
docker compose run --rm trainer python benchmark_translations.py \
  --config benchmark_config.yaml report
```

The primary review artifact is available directly on the host:

```text
benchmark_output/report.html
```

Open it in a browser to search and compare the aligned source, reference, every
model translation, and selected per-example scores. The complete outputs are:

```text
benchmark_output/report.html
benchmark_output/report.md
benchmark_output/all_model_outputs.csv
benchmark_output/scores.csv
benchmark_output/system_summary.csv
benchmark_output/pairwise_comparisons.csv
benchmark_output/slice_summary.csv
benchmark_output/score_manifest.json
```

Use `all_model_outputs.csv` for spreadsheet review,
`pairwise_comparisons.csv` for base-versus-fine-tuned deltas, and
`slice_summary.csv` to find domain-specific regressions.

## 14. Two fine-tuned NLLB revisions

If there are two fine-tuned NLLB revisions, define both as separate candidates:

```yaml
- id: "nllb-finetuned-r1"
  label: "NLLB Fine-tuned — Revision 1"
  family: "nllb"
  type: "generated"
  runner: "nllb"
  model: "facebook/nllb-200-distilled-600M"
  adapter: "models/nllb_finetuned_r1"
  source_lang: "eng_Latn"
  target_lang: "pes_Arab"
  dtype: "bfloat16"
  generation_profile: "deterministic"
  enabled: true

- id: "nllb-finetuned-r2"
  label: "NLLB Fine-tuned — Revision 2"
  family: "nllb"
  type: "generated"
  runner: "nllb"
  model: "facebook/nllb-200-distilled-600M"
  adapter: "models/nllb_finetuned_r2"
  source_lang: "eng_Latn"
  target_lang: "pes_Arab"
  dtype: "bfloat16"
  generation_profile: "deterministic"
  enabled: true
```

Generate them independently:

```bash
docker compose run --rm trainer python benchmark_translations.py \
  --config benchmark_config.yaml generate --candidates nllb-finetuned-r1

docker compose run --rm trainer python benchmark_translations.py \
  --config benchmark_config.yaml generate --candidates nllb-finetuned-r2
```

Then run the ordinary `score` and `report` commands. Both revisions appear
automatically in the leaderboard, pairwise tables, CSV export, and HTML review
explorer.

## 15. Rerun and replacement rules

- Matching candidate outputs are reused automatically.
- A changed dataset or changed candidate configuration causes a hard failure.
- Use `--force` only to intentionally replace a generated/imported candidate.
- A metric or report-layout change requires only `score` and `report`.
- A translation-model, adapter, prompt, or decoding change requires generation
  again for that candidate.
- Keep `benchmark_output/candidates/*/manifest.json` and
  `benchmark_output/score_manifest.json` with any published result.
