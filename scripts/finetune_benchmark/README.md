# Fine-tuning data-volume benchmark

Answers one question with numbers: **what does more fine-tuning data buy, for
which model, on which kind of text, and at what cost?**

The matrix is `{TranslateGemma 12B, NLLB-200 3.3B} x {base, 5k, 10k, 50k, 100k}`
evaluated on `{in-domain 500, NTREX, FLORES}` — 10 systems, 30 evaluation cells,
8 fine-tunes. Everything is configured in [`sweep_config.yaml`](sweep_config.yaml).

Nothing in the repository root is modified. This directory *drives* `train.py`,
`build_test_set.py`, `split_dataset.py` and `translation_benchmark/` by
generating configs for them, and adds only what does not exist yet: an NLLB
seq2seq LoRA trainer, a GPU-pool scheduler, and the cross-test-set report.

## Run it

```bash
# See the matrix, the job counts and every path before spending GPU time.
docker compose run --rm trainer \
  python -m scripts.finetune_benchmark.run_sweep plan

# Stages are independent and resumable; `all` runs them in order.
docker compose run --rm trainer \
  python -m scripts.finetune_benchmark.run_sweep all
```

### Training budget

`sweep.budget.mode` decides what a volume-to-volume delta means:

| Mode | Every cell gets | Reads as | Watch for |
| --- | --- | --- | --- |
| `fixed_epochs` (default, 1 epoch) | the same number of passes over its own data | "what does more data buy me" | volume and compute move together; the cost table prices it |
| `fixed_steps` | the same number of optimizer steps | "what does more *diverse* data buy me at equal compute" | small volumes repeat their data and will overfit — read `eval_loss` |

**`fixed_epochs` is the primary contract for this project.** The question is the
practical one — a team with 100k rows trains on 100k rows, and the extra compute
that implies is part of the answer, not a confound to remove. `fixed_steps`
answers a different question and makes a poor headline: holding updates constant
removes the effect being measured, and at a shared cap the small cells overfit
rather than generalise. Run it second, if at all.

### Keep every cell above ~150 optimizer updates

Packing is why this needs saying. TranslateGemma trains packed (~5.9 rows per
2048-token block on this corpus); NLLB trains unpacked. At the same effective
batch of 48, one epoch of the same data is **5.5x fewer updates** for the packed
arm — 18 updates for a 5k cell against NLLB's 100. An 18-update LoRA run is not
a converged fine-tune, and its rising `train_loss` is a step-count artefact
rather than a data effect.

```text
updates ≈ (rows x epochs) / (effective_batch x packing_factor)
```

`packing_factor` is ~6 for this corpus at `max_length: 2048`, and 1 unpacked. The
fine-tune stage logs its projected upper bound per cell before training starts,
and the report states the realised per-arm step counts beside the conclusion.
Full analysis, with the measured numbers:
`docs/2026-08-27_finetune_volume_benchmark_methodology.md`.

In `fixed_steps` mode, `max_steps: "auto"` derives the cap from the smallest
volume at `epochs` epochs; an integer applies to both arms and a mapping
(`{ translategemma: 120, nllb: 700 }`) sets each arm separately. Prefer explicit
values for the TranslateGemma arm: packing turns rows into fewer, longer blocks,
so a row-derived cap covers more epochs than the arithmetic suggests. Evaluation
and checkpointing switch to a step cadence (`max_steps / evals_per_run`) in that
mode, because a capped run can stop before the first epoch ends.

Run both to separate the two effects. Artefacts live under
`<output_dir>/<run_id>`, and `run_id` defaults to the budget's own name, so the
second run neither overwrites nor silently reuses the first one's cells:

```bash
docker compose run --rm trainer python -m scripts.finetune_benchmark.run_sweep all   # fixed_epochs_e1
# edit sweep.budget.mode: fixed_steps
docker compose run --rm trainer python -m scripts.finetune_benchmark.run_sweep all   # fixed_steps_auto
```

The data stage is shared between them by design — same subsets, same test sets,
so the two budgets are comparable.

| Stage | What it does | Parallelism |
| --- | --- | --- |
| `data` | Normalizes the corpus, carves out the 500-row in-domain test set with `build_test_set.py`, builds nested subsets, splits each into train/validation with `split_dataset.py` | CPU, sequential |
| `finetune` | One LoRA job per (model, volume) cell | one job per GPU, 4-wide |
| `evaluate` | `generate` per candidate, then `score` (through this directory's MetricX shim) and `report` per test set | generation 4-wide; scoring one job per test set |
| `report` | Cross-test-set tables, paired-bootstrap deltas, cost, HTML conclusion | CPU |

Three more commands: `plan` prints the matrix and every path without running
anything, `preflight` verifies the run's inputs, and `assets` (on the **online**
machine) writes the configs `scripts/fetch_offline_assets.py` needs for this
sweep and prints the staging command.

### Preflight

```bash
docker compose run --rm trainer python -m scripts.finetune_benchmark.run_sweep preflight
```

Fully offline, seconds, loads no weights. It checks, per staged checkpoint, that
every file the shard index names exists, is not a dangling symlink, is not a
git-lfs pointer, starts with the right magic bytes, and sums to what the index
declares — plus unfinished `*.incomplete` downloads in the cache. It also
verifies each model arm's tokenizer/processor beside its weights (vocabulary
file, `tokenizer_config.json`, `preprocessor_config.json` for the multimodal
TranslateGemma path, and that the configured NLLB language tags are actually in
the vocabulary — a converted checkpoint missing them translates into the wrong
language instead of failing), the COMET encoder repository that `hparams.yaml`
names (its absence surfaces as an unrelated `AttributeError` deep inside
transformers), the MetricX tokenizer, the corpus and test-set paths and columns,
the subsets, the adapters, the GPU ids, and free disk.

This runs automatically at the start of the `finetune` and `evaluate` stages, so
a half-transferred shard fails in seconds instead of after the queue has spent
GPU minutes on other cells. Set `sweep.preflight: false` to skip it.

Useful flags: `--force` (redo completed jobs), `--systems translategemma-5k …`,
`--test-sets flores`, and `--config` for an alternative sweep file.

## When something fails

Nothing is lost and nothing silently disappears:

- Each job writes `jobs/<stage>/<name>/result.json` with its exit code, command,
  duration, telemetry, and log path. A later run reuses `status: "ok"` results
  and re-runs everything else, so `all` can simply be run again.
- A failed job's last 25 log lines are printed at the time of failure, and the
  full log stays at `jobs/<stage>/<name>/<stage>.log`.
- `sweep.fail_fast: false` (default) keeps the other GPUs working; the report
  marks the failed cells and covers the rest.
- A trainer that exits 0 but writes no adapter is rewritten as `failed` — a zero
  exit code alone is not accepted as success.
- A failed generation job no longer takes its test set down: the sweep scores
  the candidates that did produce output and logs which ones are missing. Fix
  the cell, re-run `evaluate`, and it scores the full set (scoring is redone
  because its own job result is absent, while completed generations are reused).

### Stopping a run

Do **not** Ctrl+C a running sweep: the trainers are children of the same process
group, so they take the signal too, and with an epoch save cadence a long cell
loses everything since its last save.

To end one cell cleanly, touch its stop file — Trainer then evaluates, saves, and
runs its normal end-of-training path, so `sft_final` is written:

```bash
touch logs/finetune_benchmark/<run_id>/finetune/translategemma-100k/STOP
```

A cell that was killed or crashed anyway is resumed from its newest checkpoint on
the next run, rather than restarting at step 0. For a long single-epoch cell,
give it intermediate checkpoints to resume from:

```yaml
models:
  translategemma:
    overrides:
      training:
        evaluation_strategy: "steps"
        save_strategy: "steps"
        eval_steps: 200
        save_steps: 200
```

To redo exactly one cell:

```bash
# One fine-tune, keeping everything else
docker compose run --rm trainer python -m scripts.finetune_benchmark.run_sweep \
  finetune --systems nllb-50k --force

# One test set's evaluation
docker compose run --rm trainer python -m scripts.finetune_benchmark.run_sweep \
  evaluate --test-sets flores --force
```

Every run is reproducible from its own artefacts: `derived_configs/` holds the
exact config each subprocess was given, `result.json` holds the exact command,
and the data stage's manifests record the corpus hash, the selected test ids and
each subset's realized composition.

## Before the first run

1. **Corpus.** Point `corpus.csv_path` at the domain CSV (`id, en, fa, domain`).
   Document identity is the part of `id` before `corpus.id_separator`; the whole
   design depends on it, because test rows are held out by document.
2. **Test sets.** Put NTREX and FLORES anywhere and name them under
   `test_sets`. Any of CSV/TSV/JSONL/Parquet works, and a missing `id` or
   `domain` column is filled in. Set `enabled: false` to drop one.
3. **Domain composition.** `data.composition.mode` is `pool_proportional`
   (mirror the pool), `random` (ignore domains), or `domain_shares` with explicit
   targets, e.g.

   ```yaml
   data:
     composition:
       mode: "domain_shares"
       domain_shares: { Computer: 0.80, Mathematics: 0.15, General: 0.05 }
       on_shortfall: "error"   # or redistribute
   ```

   `unit` decides what a quota is filled from: `row` (default) shuffles the
   domain's rows and takes a prefix, so the composition is exact and the subset
   is as document-diverse as the pool allows; `document` takes whole documents,
   keeping a subset's rows contiguous inside their sources. With few, very large
   documents the difference is large — on a 12-document pool a 5k subset touches
   all 12 documents under `row` and 4 under `document`. Both are exact on shares
   and both stay nested; the test set's documents are already out of the pool, so
   this is a diversity choice, not a contamination one.

   Shares must sum to 1 and name domains that exist in the **train pool**. A
   domain small enough that the document-level holdout consumes all of it has no
   pool rows left, and the data stage says so; withhold it from the test set
   instead so it stays trainable:

   ```yaml
   data:
     in_domain_test:
       exclude_domains: ["general"]
   ```

   Its rows are appended back to the train pool (documents that contributed a
   test row are still excluded), and its quality is then read from NTREX/FLORES
   rather than from in-domain rows of its own. Alternatives: drop it from
   `domain_shares`, or `on_shortfall: redistribute` to fill its quota from the
   other domains. The realized
   shares of every volume land in
   `data/finetune_benchmark/subsets/subset_manifest.json`. Composition is shared
   by all volumes so the subsets stay nested; `per_volume` overrides one volume
   and the data stage logs that nesting no longer holds.
4. **Models.** `models.<key>.enabled`, the checkpoint ids, and the LoRA/optimizer
   settings for each arm. TranslateGemma settings are merged over `config.yaml`;
   NLLB settings are read directly by `train_nllb_lora.py`. Both arms evaluate
   and checkpoint on **epoch** boundaries: a 5k cell is only tens of optimizer
   steps long, so a step interval tuned for the full corpus would never fire and
   `load_best_model_at_end` would have nothing to load.
5. **Offline assets.** On the online machine:

   ```bash
   docker compose run --rm trainer python -m scripts.finetune_benchmark.run_sweep assets
   # then run the fetch_offline_assets.py command it prints, with HF_TOKEN set
   ```

   That stages both base models, `Unbabel/XCOMET-XL` (plus the
   `facebook/xlm-roberta-xl` tokenizer it loads indirectly),
   `google/metricx-24-hybrid-xl-v2p6-bfloat16`, `google/mt5-xl` and the test-set
   builder's `sentence-transformers/LaBSE`. See `docs/OFFLINE_DEPLOYMENT.md`
   §3.4 and §6.4. XCOMET-XL is gated: accept its licence once with the account
   whose token is used.
6. **GPUs.** `sweep.gpus: [4, 5, 6, 7]` are *physical* nvidia-smi ids, and each
   job receives one as `CUDA_VISIBLE_DEVICES`. Run the container with **all**
   GPUs visible (`GPUS=all docker compose run …`) and let the sweep select: a
   partial `NVIDIA_VISIBLE_DEVICES` renumbers the devices to 0..3 and the
   preflight check refuses to start.
7. **Image.** The TranslateGemma arm trains with packing, which requires
   FlashAttention 3, so use the FA3 image
   (`IMAGE_TAG=cu128-fa3-py312 INSTALL_FLASH_ATTN3=1 docker compose build trainer`),
   or set `models.translategemma.overrides.training.packing: false` and
   `model.attn_implementation: sdpa`. Run the dependency preflight from
   `docs/TRANSLATION_BENCHMARK_RUNBOOK.md` §6 once per new image.

## Two workarounds this directory carries

Both live here rather than as edits to the repository's own scripts.

**MetricX must run with `use_cache=False`.** MetricX builds its decoder with
`is_encoder_decoder=False`, so with caching on, `MT5Stack` allocates a plain
`DynamicCache` instead of an `EncoderDecoderCache`; the cross-attention keys are
appended to the self-attention cache and the forward pass dies in
`position_bias + causal_mask` with `size of tensor a (266) must match ... b
(265)`. `evaluate_translations.py` passes the flag;
`translation_benchmark.metrics` does not. `score.py` replaces that one function
on the imported module and then runs the benchmark's ordinary scoring stage, so
no repository file changes and every other metric is computed by the shared
implementation.

**Scoring stages are cached per row.** `score_candidates` computes transparent
metrics, then COMET, then MetricX inside one function, and `pipeline.score`
writes nothing until all three return — so a MetricX crash discarded a completed
XCOMET pass over every candidate. `score.py` wraps the COMET and MetricX stages
in a per-row cache (`.cache_comet_scores.csv`, `.cache_metricx_scores.csv` beside
the run's output), and MetricX writes its cache per batch. A re-run recomputes
only what is genuinely missing. The cache key includes a digest of the scored
translation, so a regenerated candidate cannot inherit scores for text that no
longer exists.

The replacement also **batches** MetricX, which the benchmark's version does not
(`padding=False`, one row per forward — a known gap in
`docs/EVALUATION_RUNBOOK.md`). Rows are tokenized individually so the trailing
EOS is dropped before padding, length-sorted to keep padding small, and restored
to the caller's order; `metrics.metricx.batch_size: 1` reproduces the original
behaviour exactly.

### Tuning generation

Batch 8 peaked at ~33 GiB on a 140 GiB H200, so there is a lot of headroom.
Overrides go in two places, and the difference matters for reuse:

```yaml
evaluation:
  generation: { batch_size: 8, max_new_tokens: 512 }   # shared baseline
test_sets:
  - id: "flores"
    generation: { max_new_tokens: 256 }                # into the profile: safe
models:
  nllb:
    evaluation:
      generation: { batch_size: 96 }                   # into the candidate: forces regeneration
```

A per-test-set override lands in the shared generation profile, which is not part
of a candidate's identity hash — existing outputs stay valid. A per-arm override
lands in the candidate itself, so that candidate regenerates. Starting points:
TranslateGemma 12B at 32 (~24 GiB weights plus ~0.6 GiB KV per sequence), NLLB
3.3B at 96. Check `hit_max_new_tokens` in the report before lowering
`max_new_tokens` for a test set.

**NLLB-200's `.bin` checkpoints may not load on a current torch.** transformers
wraps any `torch.load` failure in an unrelated-sounding *"Unable to load weights
from pytorch checkpoint file ... If you tried to load a PyTorch model from a TF
2.0 checkpoint"*. With `weights_only=True` the default in torch 2.6+, a
2022-era pickle can trip the safe unpickler on the production host while loading
fine on an older staging machine. Convert once, on the machine that can load it:

```bash
uv run --no-project --with transformers --with torch --with safetensors python -c "
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
AutoModelForSeq2SeqLM.from_pretrained('facebook/nllb-200-3.3B').save_pretrained(
    'nllb-200-3.3B-st', safe_serialization=True)
AutoTokenizer.from_pretrained('facebook/nllb-200-3.3B').save_pretrained('nllb-200-3.3B-st')"
```

Transfer the directory to the offline host and point the arm at the path —
local directories work everywhere in this pipeline (trainer, benchmark runner,
preflight):

```yaml
models:
  nllb:
    base_model_id: "/models/nllb-200-3.3B-st"
```

safetensors also loads faster and `preflight` can verify it exactly (every
tensor's declared byte range must end on the end of the file). Run `preflight`
afterwards to confirm the tokenizer travelled with it, including the
`eng_Latn`/`pes_Arab` language tags.

## What the design guarantees

- **No contamination.** The test set is selected first; every document it touches
  and every near-duplicate of its rows leaves the training pool before a single
  subset is drawn.
- **Nested volumes.** 5k ⊂ 10k ⊂ 50k ⊂ 100k, from one fixed per-domain draw
  order. A step along the curve is data *added*, so the marginal-gain table
  means what it says.
- **Exact composition.** Each domain's quota is filled only from that domain's
  rows, so the realized shares equal the configured ones even when one document
  carries several domains — which is the normal case here.
- **A stated budget.** `fixed_epochs` mixes volume with compute on purpose and
  the report prices each cell in GPU-hours; `fixed_steps` holds compute roughly
  constant instead. Whichever ran, the report's notes say which and how to read
  the curve, and the cost table carries the steps actually run.
- **Shared metric implementations.** Scoring goes through
  `translation_benchmark`, so a chrF++ or MetricX number here is the same
  quantity as in any other benchmark run in this repository.
- **Comparable within an arm; honest across arms.** Two cells of one arm share a
  recipe, a tokenizer and the same step arithmetic. Across arms the comparison is
  of two systems as they would actually be built — 12B with 65.5M LoRA parameters
  and packing against 3.3B with 34.6M and none — not a controlled study of
  architecture, and the report says so.
- **Paired statistics.** Deltas are paired bootstrap estimates over identical
  examples with 95% intervals, plus win/tie rates. An interval spanning zero is
  reported as not resolvable rather than dressed up as a win.
- **Structural gate, not a quality threshold.** Every candidate's output is run
  through `degeneration.py`'s audit (empty output, unstopped whitespace, loops,
  leaked boilerplate, length blowup) and a system above
  `report.max_degeneration_rate` is called out next to its scores — that failure
  swamps every corpus average, as `docs/2026-08-10_adapter_degeneration_analysis.md`
  documents. No absolute neural-metric gate is applied, deliberately:
  COMET/MetricX correlations were never established on Persian, so scores are
  ordinal within one test set only (`docs/EVALUATION_BACKLOG.md`).

## Outputs

```
logs/finetune_benchmark/<run_id>/     e.g. fixed_epochs_e1, fixed_steps_auto
├── derived_configs/          generated train/testset/benchmark configs (reproducibility)
├── jobs/<stage>/<name>/      result.json, log, telemetry.json per job
├── finetune/<system>/        adapter, checkpoints, trainer metrics
├── evaluation/<test_set>/    translation_benchmark output + its own HTML report
└── report/
    ├── master_summary.csv    system x test set x metric, plus eval throughput
    ├── training_cost.csv     wall clock, GPU-hours, peak VRAM, energy
    ├── deltas.csv            each cell vs its family's base, with 95% CIs
    ├── marginal_gains.csv    per-step gain and GPU-hours per metric point
    ├── degeneration_audit.csv decoding-failure classes per system and test set
    ├── human_review_blind.csv randomized-label side-by-side rows, no scores
    ├── human_review_key.csv   the mapping that unblinds them
    ├── conclusion.json       the machine-readable verdict
    └── finetune_benchmark_report.html
```

## Sizing note

The default matrix is ~8 fine-tunes and 30 generation passes. The 100k
TranslateGemma cell dominates the training cost, and XCOMET-XL plus MetricX-XL
dominate scoring. Start with `sweep.volumes: [5000]` and one test set to validate
the whole path end to end, then widen the config — every stage is resumable, so
the smoke run's artefacts are reused.
