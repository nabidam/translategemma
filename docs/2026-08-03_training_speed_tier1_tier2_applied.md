# Training Speed: What Was Changed and Why

Companion to `2026-08-03_translategemma_training_speed_optimization.md`. That
document lists candidate optimizations; this one records what was actually
applied to the code, why, and what still has to be measured on the GPU host.

Baseline complaint: SFT on one H100 NVL was far slower than the hardware should
allow. A review of `train.py` found that two of the largest costs were not in
the candidate list at all — they were defects in how the model was loaded and
how the loss was computed.

Every value introduced here is a key in `config.yaml`. Nothing is hardcoded.

## Tier 1 — defects and one-line wins

Applied. These change throughput, not training semantics.

### 1. The model was loading in float32

`AutoModelForCausalLM.from_pretrained` was called with no `dtype`, so
transformers used its default: float32 for every parameter that bitsandbytes
did not quantize. For Gemma 3 12B that is the embedding table, every norm, and
the 262k-row `lm_head`. `prepare_model_for_kbit_training` then upcast more.

`model.dtype` (default `bfloat16`) is now passed at load time and reused as
`bnb_4bit_compute_dtype`. `training.bf16` and `model.dtype` are cross-checked at
startup, because they describe the same decision — parameter storage and
autocast — and disagreement means a cast on every matmul.

### 2. Cross-entropy was materializing a 262k-wide logit tensor

Gemma 3's vocabulary is ~262k. An unfused loss allocates
`batch x seq x 262144` logits in the forward pass and again for the backward.
At batch 4 x 2048 that is billions of elements per step. On this model it is a
larger cost than attention, and it is what forces the micro-batch to stay small.

`training.use_liger_kernel: true` turns on Liger's fused linear+cross-entropy
(plus fused RMSNorm/RoPE/SwiGLU), which never materializes the full logit
tensor. `liger-kernel` is a normal dependency in `pyproject.toml`: it is
Triton-only, so it adds no compiler requirement to the offline image.

If the fused kernels ever fail to bind to this model, `--smoke-test` is where
that surfaces — the smoke config deliberately leaves `use_liger_kernel` alone.

### 3. `device_map: "auto"` on a single GPU

`auto` exists to spread a model that does not fit on one device. Given one
H100 it is still free to place layers on CPU, converting a training step into a
PCIe round trip. Now `{"": 0}`.

### 4. Attention implementation was never specified

`model.attn_implementation` is now explicit. The default is `sdpa`, which
already dispatches to fused kernels on Hopper.

`flash_attention_3` is available but **not** the default: `flash-attn-3` is
built from the official repository's `hopper/` source and compiles against `nvcc`, which the
`python:3.12-slim` base image does not carry. It is declared as an optional
extra (`uv sync --extra speed`) for hosts with the CUDA toolkit. Against SDPA
the gain is modest today; it becomes significant only once sequences are packed
or padding-free batching is used, which is future work.

The extra needs `[tool.uv.extra-build-dependencies]` to build at all —
flash-attn-3 imports its build modules before declaring dependencies, so an
isolated build cannot see them. Its dynamic package metadata also prevents uv's
`match-runtime` feature from working, so the build environment repeats the
project's exact `torch==2.8.0` pin. Those two pins must change together; a
mismatch can compile cleanly and then fail at import with an undefined symbol.
`[tool.uv.dependency-metadata]` separately supplies the pinned source metadata,
so dependency locking does not execute the CUDA-sensitive setup script. Full
procedure is in §6.7 of the deployment guide. None of the validation steps
require it.

### 5. Ragged batch widths

`TranslationDataCollator` padded to the exact longest example in the batch, so
the matmul dimensions were arbitrary. `training.pad_to_multiple_of: 8` keeps
them tensor-core aligned.

The collator now takes its label width from the padded batch rather than from
the raw feature lengths. `pad_to_multiple_of` can round the batch past the
longest example, and labels must match `input_ids` exactly.

### 6. Paged optimizer for a LoRA-sized state

`paged_adamw_32bit` keeps optimizer state in pageable host memory and pages it
over PCIe. That is the right trade for full fine-tuning. LoRA r=16 trains a
fraction of a percent of the parameters, so the state is small and paging only
adds synchronization stalls. Now `adamw_torch_fused`.

### 7. Single-process tokenization

`dataset.map` ran on one core, calling `apply_chat_template` twice and the
tokenizer twice per row. At 3M rows that is hours of wall clock before the
first training step. `training.tokenize_num_proc` (default 16) parallelizes
both the map and the filter. It is clamped to the split size, so dry runs and
smoke tests stay cheap.

Dataloader feeding was also left at defaults: `dataloader_pin_memory` and
`dataloader_persistent_workers` are now configurable.
`dataloader_persistent_workers` is forced off when `dataloader_num_workers` is
0, a combination `TrainingArguments` rejects.

### Also fixed while in here

- `BitsAndBytesConfig` was constructed and passed even when `use_4bit: false`,
  and `prepare_model_for_kbit_training` ran unconditionally. Both are now
  gated on `use_4bit`. On the bf16 path, `model.enable_input_require_grads()`
  is called instead when gradient checkpointing is on — otherwise the
  checkpointed blocks have no graph to recompute and the adapters get no
  gradient.
- `training.gradient_checkpointing_kwargs` defaults to
  `{"use_reentrant": false}`. Non-reentrant checkpointing is the supported
  path for LoRA; the reentrant implementation drops the graph for inputs that
  do not require grad.
- `evaluate_translations.py` now loads with the same `model.dtype` and
  `model.attn_implementation`, so evaluation cannot silently measure a
  different numeric setup than the adapter was trained under. It also stops
  using the `torch_dtype` argument, which is deprecated in transformers 4.57.

## Tier 2 — configured, but you must measure

These are exposed as configuration with a chosen default. The defaults are
reasoned, not measured; confirm them on the GPU host.

### 8. Sequence length — measured, and it moved the plan

`scripts/analyze_token_lengths.py` renders the *same* chat template `train.py`
uses with the same tokenizer, so its numbers are the lengths actually trained
on, not a character-count proxy.

```bash
uv run python scripts/analyze_token_lengths.py --config config.yaml
```

Result on `data/splits/train.jsonl`, 2,723,638 examples:

| Statistic | Tokens |
| --- | ---: |
| min | 80 |
| mean | 335.8 |
| p50 | 219 |
| p75 | 378 |
| p90 | 704 |
| p95 | 1072 |
| p99 | 1921 |
| p99.9 | 2915 |
| max | 6509 |

| `max_length` | Examples truncated | Tokens retained |
| ---: | ---: | ---: |
| 512 | 15.82% | 77.43% |
| 1024 | 5.46% | 91.71% |
| 1536 | 2.11% | 97.06% |
| 2048 | 0.69% | 99.12% |

**`training.max_length` stays at 2048.** Two conclusions follow from the shape
of this distribution, and both contradict the review document.

First, the distribution is unimodal and right-skewed, not bimodal. There is no
second cluster of long documents to route into a separate stage, so the
two-stage split proposed in that document has nothing to split on.

Second — and this is the part worth internalising — **`max_length` is not the
throughput lever it was assumed to be.** `TranslationDataCollator` pads each
batch to that batch's own longest member, not to `max_length`. The cap therefore
decides only who gets truncated and how wide the worst-case batch can be. It
does not set the cost of a typical step. Lowering it to 1024, as the review
document urged, would truncate roughly 149,000 examples for no gain in average
step time.

Truncation is not free here. `input_ids = full_ids[:max_length]` cuts the *end
of the Farsi target*, so a truncated example teaches the model to stop
mid-translation. Trading 0.69% of examples for 5.46% to buy a memory ceiling
that a 94 GB card does not need is a bad trade.

The corpus is also 2.72M examples, not the 3M the review document assumed.

### 8b. Length grouping — where the padding waste actually was

Batches are padded to their own maximum, and for a distribution with this tail
the expected batch maximum climbs steeply with batch size. Estimating each at
the `B/(B+1)` quantile of the measured distribution:

| `batch_size` | Expected batch max | Real tokens / padded slots |
| ---: | ---: | ---: |
| 4 | ~430 | ~78% |
| 8 | ~690 | ~49% |
| 16 | ~1000 | ~34% |
| 32 | ~1400 | ~24% |

At the old `batch_size: 4` the waste was mild, which is why it was never the
obvious suspect. But batch 4 badly underuses an H100 once items 1, 2 and 9 have
freed the memory — and the moment the batch size is raised to use the card,
roughly two-thirds of the compute goes into padding.

`training.group_by_length: true` puts similar-length examples in the same batch,
holding efficiency near 90% at any batch size. That is what makes "raise
`batch_size`" a win rather than a trap: roughly **3x fewer padded token slots
per epoch at batch 32** compared with ungrouped batching.

`tokenize_sft_dataset` now records a post-truncation `length` column
(`training.length_column_name`) for Trainer's `LengthGroupedSampler` to read.
It is enabled for SFT only: DPO examples are not tokenized when the sampler is
built, so Trainer would fall back to deriving lengths from raw features and
fail. `make_training_arguments` takes the flag as an explicit argument rather
than reading it from config, so that distinction cannot be lost by accident.

This was listed as future work in the first draft of this document, on the
grounds that a corpus might not be skewed enough to justify it. It is.

Sequence packing would reach ~100% instead of ~90%, but at much higher risk —
see "Not done" below. Grouping captures most of the benefit for about ten lines.

### 9. QLoRA vs bf16 LoRA

`model.use_4bit` now defaults to **false**.

4-bit NF4 buys VRAM at the cost of a dequantization step on every matmul. On a
94 GB H100 NVL, a 12B model in bf16 with LoRA adapters is roughly 24 GB of
weights — the VRAM that QLoRA was saving is VRAM this machine does not need.

This is the one change here that alters numerics, so treat the default as a
hypothesis. Benchmark both on the same subset and compare tokens/second at
equal loss. Reverting is one key:

```yaml
model:
  use_4bit: true
```

### 10. Batch size and gradient checkpointing

`training.batch_size` is still 4 and `training.gradient_checkpointing` is still
`true`. Both are now under-set for this hardware, but the right values depend on
measurements only the GPU host can supply.

Checkpointing trades roughly 30-40% throughput for memory. Items 1, 2, 8b and 9
all free memory, so re-test it *after* those land, not before:

```yaml
training:
  gradient_checkpointing: false
  batch_size: <raise until VRAM is near, but not at, the limit>
```

Raise `batch_size` deliberately. With grouping on, a typical batch is roughly
six times narrower than the 2048 cap implies — the mean example is 336 tokens —
so the memory headroom is much larger than the cap suggests. Size it against
the *longest* batches, not the typical ones: with `group_by_length` the long
examples end up batched together, which is exactly the case that OOMs.

An OOM twenty hours into an epoch costs more than the speedup is worth.

## How to validate on the GPU host

In order. Do not skip to the full run.

```bash
uv run python train.py --config config.yaml --dry-run     # data + tokenization only
uv run python train.py --config config.yaml --smoke-test  # 10 rows, 1 step, temp dir
uv run python scripts/analyze_token_lengths.py --config config.yaml
```

The smoke test is what proves Liger binds, the bf16 path builds a valid
gradient graph, and the collator produces matching `input_ids`/`labels` widths.

Then benchmark on a fixed subset — the review document suggests 20k-50k rows
and at least 200 optimizer steps per configuration — and record for each run:
real tokens/second, non-padding tokens/second, samples/second, peak VRAM, and
train/validation loss. Samples/second alone is misleading once `max_length` or
`group_by_length` changes between runs.

Benchmark at the batch size you intend to train at, not at the current 4. The
padding-efficiency table in §8b shows the two settings interact: `batch_size`
and `group_by_length` measured together say something that neither says alone.

## Not done

Deliberately out of scope, in rough order of expected value:

- **Sequence packing.** The largest remaining win, and the most invasive. It
  needs either a port from `Trainer` to `SFTTrainer` or a packing collator that
  emits correct `position_ids` and masks labels across example boundaries.
  Getting it subtly wrong trains on cross-document attention without failing.
  Do it only after Tier 1+2 are measured, and only with FlashAttention 3
  working. Note that `group_by_length` has already taken most of what packing
  was going to deliver — roughly 90% padding efficiency against packing's
  ~100% — so the remaining headroom is around 10%, not the 1.5-4x the review
  document estimated against an ungrouped baseline.
- **Pre-tokenizing to disk.** `datasets` already caches the map output, so this
  mainly helps repeated experiments across machines.
- **Multi-GPU.** Do not size a cluster from the old numbers. Re-measure
  single-GPU throughput after these changes first.

## Changed files

| File | Change |
|---|---|
| `config.yaml` | New `model.dtype`, `model.attn_implementation`, `model.bnb_4bit_*`; `model.device_map` pinned; `model.use_4bit` now false; new `training.pad_to_multiple_of`, `gradient_checkpointing_kwargs`, `use_liger_kernel`, `dataloader_pin_memory`, `dataloader_persistent_workers`, `tokenize_num_proc`, `group_by_length`, `length_column_name`; `training.optimizer` changed; new `length_analysis` section |
| `train.py` | dtype/attn/quantization gating in `setup_model_and_processor`; `resolve_dtype` and `map_workers` helpers; parallel tokenization; `length` column for the grouped sampler; collator padding alignment and width fix; new `TrainingArguments` passthroughs; smoke-test overrides |
| `evaluate_translations.py` | Loads with the configured dtype and attention implementation; drops deprecated `torch_dtype` |
| `scripts/analyze_token_lengths.py` | New — token-length percentiles and `max_length` candidate report |
| `pyproject.toml` | `liger-kernel` dependency; optional `speed` extra for official `flash-attn-3` Hopper source; `extra-build-dependencies` so that extra can build |
| `docker-compose.yml` | `HF_DATASETS_CACHE` moved off the model mount; allocator comment covers Hopper |
| `Dockerfile` | Target GPU list includes sm_90; notes Triton is now load-bearing |
| `docs/OFFLINE_DEPLOYMENT.md` | §3.2 lock/rebuild requirement, §4 cache sizing, §5.3 length-analysis step, §6.1 rewritten for bf16, new §6.7 on FlashAttention |
| `docs/DEPLOYMENT_BACKLOG.md` | Three new items: commit `uv.lock`, toolkit image for FA3, record measured throughput |

`uv.lock` must be regenerated (`uv lock`) before the offline image is rebuilt —
the Dockerfile's cu128 path uses `uv sync --locked` and will fail against a
stale lock. It is also currently untracked; see `DEPLOYMENT_BACKLOG.md`.
