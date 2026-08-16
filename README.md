# TranslateGemma: multilingual scientific translation fine-tuning

This directory contains data preparation, QLoRA SFT, optional DPO, and evaluation
tools for a scientific TranslateGemma adapter. `config.yaml` contains the run
settings; save it with every released adapter.

## Setup

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e .

# MetricX is optional, but required when evaluation.metricx_enabled is true.
# It has no packaging metadata, so it is used as a source checkout, not installed.
git clone https://github.com/google-research/metricx.git ../metricx
export PYTHONPATH="$PWD/../metricx:$PYTHONPATH"
```

For an air-gapped machine, use the Docker environment instead — see
[`docs/OFFLINE_DEPLOYMENT.md`](docs/OFFLINE_DEPLOYMENT.md).

## Data contract

SFT JSONL records must contain the text columns configured in `data`: by default
`id`, `domain`, `source_text`, and `target_text`. IDs must group neighbouring
segments from the same document, for example `42:17`. The splitter uses the part
before `:` so a document can never appear in more than one split.

Each record may also contain `src_lang` and `tgt_lang`. These values
override the config-level `data.source_lang` and `data.target_lang` for that record,
so one dataset can mix directions such as English→Persian and Russian→Persian:

```json
{"id":"1:1","domain":"science","source_text":"Hello","target_text":"سلام","src_lang":"en","tgt_lang":"fa"}
{"id":"2:1","domain":"science","source_text":"Привет","target_text":"سلام","src_lang":"ru","tgt_lang":"fa"}
```

Old records without these fields continue to use the config-level pair. The column
names can be changed with `data.source_lang_column` and `data.target_lang_column`.
Codes must be supported by TranslateGemma's
[chat template](https://huggingface.co/google/translategemma-27b-it/blob/main/chat_template.jinja),
which accepts its listed codes and supported regional variants such as `en-US`.

## Reproducible workflow

```bash
# 1. Validate, de-duplicate, and create document-level train/validation/test splits.
python split_dataset.py --config config.yaml

# 2. Measure the tokenized length distribution before committing to a
#    training.max_length. Writes length_analysis.report_path.
python scripts/analyze_token_lengths.py --config config.yaml

# 3. Train using only the configured train and validation paths. Select the
#    Accelerate profile matching the hardware/process count.
accelerate launch --config_file accelerate_configs/h200_1gpu.yaml \
  train.py --config config.yaml

# Optional: validate enabled inputs/template/arguments without model weights or outputs.
python train.py --config config.yaml --dry-run

# Optional: run enabled stages with ≤10 rows per split and one train step in temporary outputs.
accelerate launch --config_file accelerate_configs/h200_1gpu.yaml \
  train.py --config config.yaml --smoke-test

# Optional: run the actual configured training loop on the smaller canary subset.
accelerate launch --config_file accelerate_configs/h200_1gpu.yaml \
  train.py --config config.yaml --canary

# Run the configured model-size and benchmark-dimension matrix. Built-in H200F
# profiles select 4B=12/96, 12B=6/48, and 27B=2/16 micro/global batches.
# The run ends with a Rich summary and writes JSON, CSV, Markdown, HTML, per-job
# logs, and sampled GPU telemetry. See docs/TRAINING_BENCHMARKS.md.
python scripts/benchmark_training.py --config config.yaml

# Benchmark only GPU scaling for one model size.
python scripts/benchmark_training.py --config config.yaml \
  --benchmark-types gpu_count --model-sizes 12b \
  --gpu-counts 1 2 4 --max-examples 20000 --max-steps 200

# Compare per-device batches on 4 GPUs while keeping the effective global batch
# fixed. Accumulation is derived and non-divisible combinations fail early.
python scripts/benchmark_training.py --config config.yaml \
  --benchmark-types batch_size --model-sizes 4b \
  --batch-gpu-count 4 --batch-sizes 3 6 12

# Compare all configured checkpointing/packing combinations at a fixed GPU count.
python scripts/benchmark_training.py --config config.yaml \
  --benchmark-types training_options --model-sizes 12b \
  --training-options-gpu-count 4

# 4. Evaluate an existing adapter (or inspect final outputs again).
# Single-GPU with GPU batching (eval_batch_size configured in config.yaml):
python evaluate_translations.py --config config.yaml --adapter-path path/to/adapter

# Multi-GPU data-parallel evaluation using Accelerate:
accelerate launch --config_file accelerate_configs/h200_4gpu.yaml \
  evaluate_translations.py --config config.yaml --adapter-path path/to/adapter

# Optional: Force a clean re-evaluation from scratch (bypassing cached progress).
python evaluate_translations.py --config config.yaml --adapter-path path/to/adapter --force

# 5. Turn the evaluation CSVs into one offline HTML review page.
python report_evaluation.py --config config.yaml
```

Enabling MetricX and COMET needs a MetricX checkout, a licence acceptance for
gated XCOMET checkpoints, and staged model files. That setup, the multi-GPU and
single-GPU execution model, and the cache semantics are in
[`docs/EVALUATION_RUNBOOK.md`](docs/EVALUATION_RUNBOOK.md). Which metric to trust
for English → Persian is argued in
[`docs/EVALUATION_BACKLOG.md`](docs/EVALUATION_BACKLOG.md).

### Reviewing an evaluation run

`report_evaluation.py` renders `evaluation/report.html` from whatever
`<prefix>_detailed_scores.csv` files the evaluation wrote (`base` and `adapter`
by default), joined on `data.id_column`. It is a single self-contained file with
no network dependency, so it can be copied off the offline host and opened
directly.

The page contains the per-system metric cards, a head-to-head table with the
MetricX/COMET deltas and the per-sample win rate, a per-domain breakdown, and a
sample explorer that shows the source, the reference, and every system's
translation side by side. The explorer supports full-text search, domain
filtering, sorting by biggest gain or regression, word-level diffing against the
reference, and renders Farsi right-to-left automatically.

```bash
# Report a run kept outside the configured evaluation directory.
python report_evaluation.py --config config.yaml \
  --eval-dir evaluation-canary --output evaluation-canary/report.html

# Keep the page small for a very large test set.
python report_evaluation.py --config config.yaml --max-samples 2000
```

## Compare multiple translation models

Use `benchmark_translations.py` for the final comparison of TranslateGemma
sizes, base and LoRA checkpoints, NLLB variants, and translations generated by
other systems. Generation/import, metric scoring, and reporting are separate,
so expensive model outputs are reused.

```bash
# Configure the frozen dataset and candidate systems first.
docker compose run --rm trainer \
  python benchmark_translations.py --config benchmark_config.yaml validate

# Generate enabled models and import enabled historical outputs.
docker compose run --rm trainer \
  python benchmark_translations.py --config benchmark_config.yaml collect

# Score the aligned outputs, then create CSV, Markdown, and standalone HTML.
docker compose run --rm trainer \
  python benchmark_translations.py --config benchmark_config.yaml score
docker compose run --rm trainer \
  python benchmark_translations.py --config benchmark_config.yaml report
```

The preferred human-review artifact is `benchmark_output/report.html`.
`benchmark_output/all_model_outputs.csv` contains the source, reference, and a
side-by-side translation column for every evaluated model. See
[`docs/TRANSLATION_BENCHMARK.md`](docs/TRANSLATION_BENCHMARK.md) for the full
configuration, metrics, paired statistics, artifact layout, fairness rules, and
production verification procedure. A concrete four-candidate Docker walkthrough
is in
[`docs/TRANSLATION_BENCHMARK_RUNBOOK.md`](docs/TRANSLATION_BENCHMARK_RUNBOOK.md).

Generated candidates support data-parallel inference with the existing
Accelerate profiles. For example, use four GPUs with:

```bash
docker compose run --rm trainer accelerate launch \
  --config_file accelerate_configs/h200_4gpu.yaml \
  benchmark_translations.py --config benchmark_config.yaml generate \
  --candidates translategemma-12b-lora
```

Each GPU holds one full model replica. Run scoring and report generation later
as ordinary single-process Compose commands.

## Finding the evaluation batch size

`scripts/benchmark_eval_batch.py` performs one tiny SFT run using the normal
configuration, then evaluates the resulting model on the configured validation
split with each candidate `per-device` evaluation batch size. It does not load
or evaluate the test split. The default search trains on one example for one
optimizer step and tests the values in `eval_batch_search.eval_batch_sizes`.

Run it with the Accelerate profile matching the hardware:

```bash
accelerate launch --config_file accelerate_configs/h200_1gpu.yaml \
  scripts/benchmark_eval_batch.py --config config.yaml
```

The script prints a Rich results table and summary, stops at the first
out-of-memory candidate in the ascending sweep, and writes the machine-readable
report to `logs/eval_batch_search/eval_batch_results.json`. To test a different
candidate set:

```bash
accelerate launch --config_file accelerate_configs/h200_1gpu.yaml \
  scripts/benchmark_eval_batch.py --config config.yaml \
  --eval-batch-sizes 2 4 6 8 12 16
```

Throughput settings (`model.dtype`, `model.attn_implementation`,
`model.use_4bit`, `training.use_liger_kernel`, `training.max_length`) and the
measurements they still need are documented in
[docs/2026-08-03_training_speed_tier1_tier2_applied.md](docs/2026-08-03_training_speed_tier1_tier2_applied.md).
Benchmark profiles, comparison rules, and output fields are documented in
[docs/TRAINING_BENCHMARKS.md](docs/TRAINING_BENCHMARKS.md).
The production DDP recipe keeps an effective packed-block batch of 48 by deriving
gradient accumulation from the active GPU count. Rank zero tokenizes and BFD-packs
the SFT split into `data.prepared_cache_dir` before model weights are loaded; other
ranks wait and then load that immutable cache. Packing requires FlashAttention 3:
TRL's resetting position IDs preserve attention boundaries between examples.
`group_by_length` remains disabled because sampler construction stalled startup on
the full corpus. Shared metadata and processor files are written only by rank zero,
while every rank receives a distinct file log.

TranslateGemma includes a vision tower, but this pipeline prepares text-only
batches. `lora.exclude_modules` therefore excludes `vision_tower` from PEFT's
suffix-based target matching, preventing `q_proj`, `k_proj`, and `v_proj` adapters
from being added to unused vision layers. The name follows Transformers' upstream
[Gemma 3 implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma3/modular_gemma3.py),
which exposes the image encoder as `vision_tower`. Normal, smoke, canary, DPO-only,
and benchmark runs all use the same model setup and exclusion. Remove the exclusion
only after adding image inputs and `pixel_values` handling to the data pipeline.

The canary run reads its limits and isolated output paths from the `canary` section
of `config.yaml`. It uses the normal Trainer loop, including all configured enabled
stages; `canary.max_examples` limits each loaded train/validation split. Set
`canary.max_steps` to cap optimizer updates, or leave it `null` to use the configured
epoch count. Canary runs start without a checkpoint unless a canary-specific resume
path is configured.

The benchmark runner first performs a discarded one-GPU warm-up to populate
the Liger/FA3 JIT cache, then launches the real SFT path sequentially
for each requested local GPU count. `benchmark.accelerate_config_pattern` maps each
count to its profile in `accelerate_configs/` and the runner verifies the profile's
`num_processes` before launching. Model, data, LoRA, precision, batch size, gradient
accumulation, collator, and optimizer settings all come from the ordinary sections
of `config.yaml`, except for the benchmark's deliberate batch-math overrides. For the
12B H200 recipe, `benchmark.per_device_batch_size: 6` and
`benchmark.effective_batch_size: 48` derive gradient accumulation of `8, 4, 2, 1`
for `1, 2, 4, 8` GPUs. Every run therefore performs the same-size optimizer update
and, with a fixed `max_steps`, processes the same sample workload. Invalid GPU/batch
combinations fail before launch rather than rounding accumulation. Dataset/step
bounds, the GPU matrix, Accelerate profile pattern, optional validation pass, and
report location also come from `benchmark` and may be overridden on the command line.
The report includes samples/second, padded and non-padding tokens/second, padding
efficiency, peak per-GPU memory, loss, speedup, scaling efficiency, and the derived
batch math. Results are written as JSON and CSV below `benchmark.output_dir`; each
result also records the selected Accelerate profile and hash.

Locking is GPU- and toolkit-independent because `pyproject.toml` supplies FA3's
static dependency metadata. Building the optional package requires CUDA >=12.3
(12.8 recommended), but the build machine itself does not need a Hopper GPU:

```bash
uv lock
# Run this one only in the CUDA 12.8 toolkit/devel build environment:
MAX_JOBS=4 uv sync --extra speed
```

The checked-in production configuration uses `flash_attention_3` because
boundary-safe BFD packing depends on its padding-free attention path. CUDA 12.8
is recommended. Use an unpacked configuration before switching back to `sdpa`.

To convert a held-out CSV with `id`, `domain`, `en`, and `fa` columns directly to the
configured test location, without creating DPO data:

```bash
python prepare_data.py --input_csv path/to/test.csv \
  --sft_output data/splits/test.jsonl --skip_dpo
```

For train-only runs set both split ratios to `0`, set
`data.validation_sft_dataset_path: null`, and set
`evaluation.run_after_training: false`.

The trainer evaluates validation loss and checkpoints every configured interval. It
restores the best validation checkpoint before writing the final adapter. The held-out
test split is never used during training; after training, the evaluator optionally
compares the base model and adapter with deterministic decoding, COMET, MetricX, and a
per-domain human-review CSV sample. Evaluation includes live Rich progress indicators
and writes line-flushed cache files (`.cache_*_hypotheses.jsonl`) to make generation and
metric scoring crash-proof and instantly resumable.

## Checkpointing and resume

SFT checkpoints are written below
`<model.output_dir>/<model.sft_checkpoint_subdir>/checkpoint-<step>`; DPO checkpoints
use `model.dpo_checkpoint_subdir`. A Trainer checkpoint contains the adapter weights
and training state needed to continue, including optimizer, scheduler, global-step,
and RNG state. The `sft_final` and `dpo_final` directories contain final adapters and
are not resumable Trainer checkpoints.

The default uses matching step-based evaluation and checkpointing every 500
optimizer updates:

```yaml
training:
  evaluation_strategy: "steps"
  eval_steps: 500
  save_strategy: "steps"
  save_steps: 500
```

An optimizer update happens after `gradient_accumulation_steps` micro-batches. When
`load_best_model_at_end` is enabled, the save strategy must match the evaluation
strategy and `save_steps` must be a multiple of `eval_steps`. `save_total_limit`
controls how many checkpoints are retained. `gradient_checkpointing` is a separate
memory-saving feature and does not create resumable checkpoints.

To resume the latest SFT checkpoint automatically:

```yaml
training:
  run_sft: true
  run_dpo: false
  resume_from_checkpoint: true
```

To select a checkpoint explicitly, use its directory (relative paths are resolved
from the directory where `train.py` is launched):

```yaml
training:
  run_sft: true
  run_dpo: false
  resume_from_checkpoint: "./translategemma-farsi-science/sft/checkpoint-12500"
```

Run the ordinary training command after changing the configuration. Keep the dataset,
base model, LoRA parameters, batch/accumulation settings, optimizer, scheduler, and
seed consistent with the interrupted run. The configured epoch count remains the
total target, rather than a number of additional epochs.

There is one shared `resume_from_checkpoint` setting for both stages. When using an
explicit path, enable only the stage that owns that checkpoint. For example, resume
DPO with `run_sft: false`, `run_dpo: true`, and a path below the DPO checkpoint
directory. Passing an SFT path while both stages are enabled would also pass that path
to the DPO trainer. With `resume_from_checkpoint: true`, each enabled stage searches
its own checkpoint directory, but a stage with no checkpoint will fail rather than
start fresh. Resume stages in separate invocations to avoid either ambiguity.

## Notes

- Review `split_dataset.py`'s manifest before training. It records row counts,
  duplicate removals, and document IDs for each split.
- The SFT objective masks the source prompt, so loss is calculated only on the
  target-language response.
- `max_length` and `max_new_tokens` must be large enough for scientific passages.
  The pipeline logs any training examples truncated to `max_length`.
- Keep DPO disabled until preference pairs have genuine curated rejected translations.
  It is a second-stage experiment, not a substitute for a clean SFT baseline.

## Serving

[`api/`](api/README.md) is a FastAPI **gateway**: vLLM holds the weights, and
the gateway owns the prompt rendering, the stop set, sentence splitting and the
endpoint shapes (the same ones as the existing NLLB service, plus a batch
endpoint). It is a self-contained deployment unit: copy the directory to the
serving host and build from inside it, with no part of this repository present.

```bash
cd api && docker build -t translategemma-api .
docker compose up -d          # vLLM, then the gateway once vLLM is healthy
```

vLLM serves the **merged** checkpoint that `scripts/merge_lora_adapter.py`
writes, so no `--enable-lora` and no adapter loading at serve time:

```bash
docker compose run --rm trainer python scripts/merge_lora_adapter.py \
  outputs/sft_final /models/translategemma-12b-merged
```

A 12B merge is ~24 GB in bf16. When it has to share a GPU with another model,
quantise it to FP8 **after** the merge — merging into quantised weights rounds
the adapter delta away:

```bash
IMAGE_TAG=cu128-quant-py312 INSTALL_LLMCOMPRESSOR=1 docker compose build trainer
IMAGE_TAG=cu128-quant-py312 docker compose run --rm trainer \
  python scripts/quantize_fp8.py \
  /models/translategemma-12b-merged /models/translategemma-12b-merged-fp8
```

FP8_DYNAMIC needs no calibration corpus, which is why it is preferred here over
4-bit AWQ/GPTQ: a general-purpose calibration set biases a domain-fine-tuned
model away from its domain. The script refuses an adapter directory or an
already-quantised input, and fails rather than write an output whose stop set
has lost `<end_of_turn>` (106). llm-compressor is a build-arg opt-in so it stays
out of the training image by default.

The gateway reproduces the generation contract of `evaluate_translations.py`
exactly — it renders prompts locally and sends **token ids** to vLLM's
`/v1/completions`, with the resolved stop set on every request — so a served
translation is the translation the harness scored. That requires a copy of
`prompting.py` inside `api/`, kept byte-identical from here:

```bash
uv run python scripts/sync_api_vendored.py          # after editing the root module
uv run python scripts/sync_api_vendored.py --check  # exit 1 on drift; also a test
```

`TG_MODEL_MODE` names which system the upstream answers as, so `/model-info` and
the per-request `system` field keep attributing a translation to a checkpoint.
