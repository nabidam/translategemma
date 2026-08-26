# Fine-tuning data-volume benchmark: methodology and first-run findings

**Date**: 2026-08-27
**Sweep**: `scripts/finetune_benchmark/`, run `fixed_epochs_e1`
**Question**: what does more fine-tuning data buy, for TranslateGemma 12B versus
NLLB-200 3.3B, on in-domain / NTREX / FLORES text?

The matrix is `{TranslateGemma 12B, NLLB-200 3.3B} x {base, 5k, 10k, 50k, 100k}`
evaluated on three frozen test sets. This document records why the training
budget is defined the way it is, what the first run measured, and which of its
cells can carry an interpretation.

## The budget is fixed epochs, deliberately

`sweep.budget.mode` offers two contracts:

| Mode | Every cell gets | Answers |
|---|---|---|
| `fixed_epochs` | the same number of passes over its own data | "what does more fine-tuning data buy me" |
| `fixed_steps` | the same number of optimizer updates | "at equal compute, does more *diverse* data help" |

**`fixed_epochs` is the primary contract for this project**, because the
question is the practical one: a team with 100k rows trains on 100k rows, and
the extra compute that implies is part of the answer rather than a confound to
be removed. The report prices every cell in GPU-hours, peak VRAM and energy, so
the compute that came with the data is visible rather than hidden.

`fixed_steps` is a **secondary** run, not a replacement. Two reasons it cannot
be the headline:

* It answers a different question. Holding updates constant removes exactly the
  effect the project is trying to measure.
* At a shared cap the small cells overfit rather than generalise. A 400-step cap
  is roughly 4 epochs of 5k rows for the packed TranslateGemma arm and about 20
  for NLLB, so a flat curve would largely be reporting overfitting.

Run it second, once the fixed-epoch story is in, if the "more data or more
compute?" split is worth separating. Both budgets share one data stage and write
to their own `run_id`, so neither can overwrite the other.

## Sequence packing makes one epoch mean different things per arm

Measured on the production corpus (`fixed_epochs_e1`, one epoch):

| Cell | train rows | units the trainer saw | rows / optimizer step | steps run |
|---|---:|---:|---:|---:|
| translategemma-5k | 4,750 | 861 packed blocks | 264 | **18** |
| translategemma-10k | 9,500 | 1,697 | 264 | **36** |
| translategemma-50k | 47,500 | 8,398 | 264 | 175 |
| translategemma-100k | 95,000 | 16,825 | 264 | 351 |
| nllb-5k | 4,750 | 4,791 rows | 48 | 100 |
| nllb-10k | 9,500 | 9,473 | 48 | 198 |
| nllb-50k | 47,500 | 47,309 | 48 | 986 |
| nllb-100k | 95,000 | 94,657 | 48 | 1,973 |

TranslateGemma trains with BFD packing (`training.packing: true`, required for
the throughput contract in `docs/MULTI_GPU_TRAINING.md`), which packs roughly
5.9 rows of this corpus into each 2048-token block. NLLB is an encoder-decoder
trained unpacked, one row per example. At an identical effective batch of 48,
one epoch of the same data is therefore **5.5x fewer optimizer updates for the
packed arm**.

This is a property of the training recipes, not a bug, and it is not worth
"fixing" by disabling packing: packing is what makes the 12B arm affordable.
What it requires is honesty about the low-volume cells.

### Consequence: a floor on usable cells

An 18-update LoRA run is not a converged fine-tune. With
`warmup_ratio: 0.03` it barely leaves warmup before the cosine schedule decays.
The first run shows the signature directly:

| Cell | steps | train_loss | eval_loss |
|---|---:|---:|---:|
| translategemma-5k | 18 | 0.773 | 0.269 |
| translategemma-10k | 36 | **1.059** | 0.239 |
| translategemma-50k | 175 | 0.225 | 0.176 |
| translategemma-100k | 351 | 0.162 | 0.162 |

`train_loss` rising from 5k to 10k while `eval_loss` falls monotonically is a
step-count artefact, not a data effect: the 10k cell's average training loss is
dominated by its early, high-loss updates because it has so few of them.

**Rule of thumb: keep every cell above ~150 optimizer updates.** Estimate before
committing GPU time:

```text
updates ≈ (rows x epochs) / (effective_batch x packing_factor)
```

with `packing_factor ≈ 6` for this corpus at `max_length: 2048` and 1 unpacked.
For the smallest cell (4,750 rows, effective batch 48, packed) that means at
least 3 epochs, and even then 54 updates is thin — the fine-tune stage logs its
projected upper bound per cell so this is visible before training starts, and
the report states the realised per-arm step counts next to the conclusion.

## What the first run established

Valid and reusable:

* **The data stage.** 500-row in-domain test set carved out first with
  document-level holdout and near-duplicate purge; nested subsets
  (5k ⊂ 10k ⊂ 50k ⊂ 100k) drawn at row level to an exact 80/15/5 domain
  composition; `general` withheld from the test set so it stays trainable.
* **The whole NLLB arm** (100 to 1,973 updates), with monotone eval_loss
  1.269 → 1.080 → 0.751 → 0.694.
* **TranslateGemma 50k and 100k** (175 and 351 updates).
* **All 30 generation passes and the decoding audit.**

Not interpretable as data-volume effects: **TranslateGemma 5k and 10k**. Treat
them as a floor, not a measurement.

### Decoding audit (before any neural metric)

Worst failure rate across all 30 system x test-set pairs is 3.8%, against the
`max_degeneration_rate: 0.15` gate. Fine-tuning *reduced* decoding failures on
both arms:

| System | in-domain clean | NTREX clean | mean output chars (in-domain) |
|---|---:|---:|---:|
| translategemma-base | 96.2% | 98.1% | 214 |
| translategemma-tuned (any volume) | 98.4-98.8% | 99.0% | 188 |
| nllb-base | 96.4% | 99.0% | 189 |
| nllb-tuned (any volume) | 97.0-97.8% | 99.0% | 188-193 |

Every residual failure is the loop class, and the base model shares it, which
matches `docs/2026-08-10_adapter_degeneration_analysis.md`: the repeated span is
in the source segment. The base model's 214 versus 188 mean characters is the
documented base verbosity, now measured on this corpus. No trailing-whitespace
flood anywhere — the `prompting.py` stop-token contract is holding.

## Reading the cost table

Three columns exist because two of them used to be conflated:

| Column | Meaning |
|---|---|
| `train_rows` | rows in the cell's train split, from `split_dataset.py`'s manifest |
| `train_units_seen` | what the Trainer iterated: packed blocks for TranslateGemma, rows for NLLB |
| `rows_per_optimizer_step` | their ratio over the realised step count — the packing factor, made explicit |

`trainable_parameters` is counted from the adapter's own safetensors header
(65.5M for r=16 on the 12B, 34.6M for the 3.3B), because `train.py`'s
`run_metadata.json` does not record it.

## Cross-arm comparison: what it can and cannot say

* Compare cells **within** an arm freely: same recipe, same tokenizer, same step
  arithmetic.
* Across arms, the comparison is of **two systems as they would actually be
  built** — 12B with 65.5M LoRA parameters and packing, against 3.3B with 34.6M
  and no packing. It is not a controlled study of architecture, and should not be
  reported as one.
* COMET and MetricX correlations were established on WMT MQM language pairs,
  which exclude Persian (`docs/EVALUATION_BACKLOG.md`). Scores are ordinal
  **within one test set**. Never compare a score on FLORES against a score on
  the in-domain set, and do not read an absolute quality threshold into either.
  The only absolute gate is the structural decoding audit.

## Decisions taken

1. Score `fixed_epochs_e1` as it stands. It answers the question for the NLLB
   curve and the two large TranslateGemma cells, at no extra GPU cost.
2. Re-run fine-tuning and evaluation at `budget.epochs: 3` into its own
   `run_id`, so the low-volume TranslateGemma cells clear the update floor
   (18 → 54 at 5k). Estimated 23.5 GPU-hours; the 100k cells set a ~7.5 h wall
   floor on four GPUs. All four cells per arm, never a subset: mixing 3 epochs at
   5k with 1 epoch at 100k would destroy the curve.
3. Raise generation batch size on that run (32 for the 12B, 96 for the 3.3B):
   batch 8 peaked at 33 of 140 GiB, and `docs/2026-08-17_serving_ab_vllm_vs_transformers.md`
   measures the transformers path at 35 tok/s at batch 8 against 129 at batch 32.
4. Keep `fixed_steps` for a later, optional run.
