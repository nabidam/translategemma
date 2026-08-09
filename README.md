# TranslateGemma: scientific English → Farsi fine-tuning

This directory contains the data preparation, QLoRA SFT, optional DPO, and evaluation
tools for a scientific English-to-Farsi TranslateGemma adapter. All run choices are in
`config.yaml`; save that file with every released adapter.

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

SFT JSONL records must contain the columns configured in `data`: by default `id`,
`domain`, `english`, and `farsi`. IDs must group neighbouring segments from the same
document, for example `42:17`. The splitter uses the part before `:` so a document can
never appear in more than one split.

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
python evaluate_translations.py --config config.yaml --adapter-path path/to/adapter
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
per-domain human-review CSV sample.

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
- The SFT objective masks the source prompt, so loss is calculated only on the Farsi
  response.
- `max_length` and `max_new_tokens` must be large enough for scientific passages.
  The pipeline logs any training examples truncated to `max_length`.
- Keep DPO disabled until preference pairs have genuine curated rejected translations.
  It is a second-stage experiment, not a substitute for a clean SFT baseline.

## vLLM serving

```bash
python -m vllm.entrypoints.openai.api_server \
  --model google/translategemma-12b-it \
  --enable-lora \
  --lora-modules farsi-science=path/to/final_adapter \
  --max-lora-rank 64 \
  --host 0.0.0.0 \
  --port 8000
```
