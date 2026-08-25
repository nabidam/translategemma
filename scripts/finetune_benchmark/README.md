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

| Stage | What it does | Parallelism |
| --- | --- | --- |
| `data` | Normalizes the corpus, carves out the 500-row in-domain test set with `build_test_set.py`, builds nested subsets, splits each into train/validation with `split_dataset.py` | CPU, sequential |
| `finetune` | One LoRA job per (model, volume) cell | one job per GPU, 4-wide |
| `evaluate` | `generate` per candidate, then `score` and `report` per test set | generation 4-wide; scoring one job per test set |
| `report` | Cross-test-set tables, paired-bootstrap deltas, cost, HTML conclusion | CPU |

Two more commands: `plan` prints the matrix and every path without running
anything, and `assets` (on the **online** machine) writes the configs
`scripts/fetch_offline_assets.py` needs for this sweep and prints the staging
command.

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

   Shares must sum to 1 and name domains that exist in the corpus. The realized
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

## What the design guarantees

- **No contamination.** The test set is selected first; every document it touches
  and every near-duplicate of its rows leaves the training pool before a single
  subset is drawn.
- **Nested volumes.** 5k ⊂ 10k ⊂ 50k ⊂ 100k, from one domain-balanced document
  order. A step along the curve is data *added*, so the marginal-gain table
  means what it says.
- **Fixed epochs.** Every cell trains for `sweep.epochs`, so larger volumes also
  get more optimizer steps. Volume and compute are intentionally not separated —
  the report prices each cell in GPU-hours so the confound stays visible. For the
  compute-matched variant instead, cap steps in `models.<key>` and re-run.
- **Shared metric implementations.** Scoring goes through
  `translation_benchmark`, so a chrF++ or MetricX number here is the same
  quantity as in any other benchmark run in this repository.
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
logs/finetune_benchmark/
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
    ├── conclusion.json       the machine-readable verdict
    └── finetune_benchmark_report.html
```

## Sizing note

The default matrix is ~8 fine-tunes and 30 generation passes. The 100k
TranslateGemma cell dominates the training cost, and XCOMET-XL plus MetricX-XL
dominate scoring. Start with `sweep.volumes: [5000]` and one test set to validate
the whole path end to end, then widen the config — every stage is resumable, so
the smoke run's artefacts are reused.
