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
uv pip install "git+https://github.com/google-research/metricx.git"
```

## Data contract

SFT JSONL records must contain the columns configured in `data`: by default `id`,
`domain`, `english`, and `farsi`. IDs must group neighbouring segments from the same
document, for example `42:17`. The splitter uses the part before `:` so a document can
never appear in more than one split.

## Reproducible workflow

```bash
# 1. Validate, de-duplicate, and create document-level train/validation/test splits.
python split_dataset.py --config config.yaml

# 2. Train using only the configured train and validation paths.
python train.py --config config.yaml

# 3. Evaluate an existing adapter (or inspect final outputs again).
python evaluate_translations.py --config config.yaml --adapter-path path/to/adapter
```

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
