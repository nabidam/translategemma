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

# 3. Train using only the configured train and validation paths.
python train.py --config config.yaml

# Optional: validate enabled inputs/template/arguments without model weights or outputs.
python train.py --config config.yaml --dry-run

# Optional: run enabled stages with ≤10 rows per split and one train step in temporary outputs.
python train.py --config config.yaml --smoke-test

# 4. Evaluate an existing adapter (or inspect final outputs again).
python evaluate_translations.py --config config.yaml --adapter-path path/to/adapter
```

Throughput settings (`model.dtype`, `model.attn_implementation`,
`model.use_4bit`, `training.use_liger_kernel`, `training.max_length`) and the
measurements they still need are documented in
[docs/2026-08-03_training_speed_tier1_tier2_applied.md](docs/2026-08-03_training_speed_tier1_tier2_applied.md).

Locking is GPU- and toolkit-independent because `pyproject.toml` supplies FA3's
static dependency metadata. Building the optional package requires CUDA >=12.3
(12.8 recommended), but the build machine itself does not need a Hopper GPU:

```bash
uv lock
# Run this one only in the CUDA 12.8 toolkit/devel build environment:
MAX_JOBS=4 uv sync --extra speed
```

Then set `model.attn_implementation: "flash_attention_3"` in `config.yaml`.
CUDA 12.8 is recommended. The checked-in configuration remains on `sdpa` so
the default slim Docker image and non-Hopper machines continue to work.

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

The default `save_strategy: "epoch"` writes a checkpoint after each completed epoch,
so an interruption loses work since the previous epoch boundary. To checkpoint and
evaluate by optimizer update step instead, configure matching step strategies:

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
