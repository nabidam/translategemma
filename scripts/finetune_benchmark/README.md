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

Useful flags: `--force` (redo completed jobs), `--systems translategemma-5k …`,
`--test-sets flores`, and `--config` for an alternative sweep file.

A completed job writes `result.json`; the next run reads it back instead of
repeating the work, so an interrupted sweep resumes where it stopped. Delete a
job's directory (or pass `--force`) to redo just that cell.

## Before the first run

1. **Corpus.** Point `corpus.csv_path` at the domain CSV (`id, en, fa, domain`).
   Document identity is the part of `id` before `corpus.id_separator`; the whole
   design depends on it, because test rows are held out by document.
2. **Test sets.** Put NTREX and FLORES anywhere and name them under
   `test_sets`. Any of CSV/TSV/JSONL/Parquet works, and a missing `id` or
   `domain` column is filled in. Set `enabled: false` to drop one.
3. **Models.** `models.<key>.enabled`, the checkpoint ids, and the LoRA/optimizer
   settings for each arm. TranslateGemma settings are merged over `config.yaml`;
   NLLB settings are read directly by `train_nllb_lora.py`.
4. **Offline assets.** Stage every checkpoint the sweep touches with
   `scripts/fetch_offline_assets.py` before starting: both base models, plus
   `Unbabel/XCOMET-XL` (and the `facebook/xlm-roberta-xl` tokenizer it loads),
   `google/metricx-24-hybrid-xl-v2p6-bfloat16` and `google/mt5-xl`. See
   `docs/OFFLINE_DEPLOYMENT.md` §3.4 and §6.4.
5. **GPUs.** `sweep.gpus: [4, 5, 6, 7]` are *physical* ids. The container must be
   able to see them (`GPUS=4,5,6,7 docker compose run …` or `GPUS=all`).

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
    ├── conclusion.json       the machine-readable verdict
    └── finetune_benchmark_report.html
```

## Sizing note

The default matrix is ~8 fine-tunes and 30 generation passes. The 100k
TranslateGemma cell dominates the training cost, and XCOMET-XL plus MetricX-XL
dominate scoring. Start with `sweep.volumes: [5000]` and one test set to validate
the whole path end to end, then widen the config — every stage is resumable, so
the smoke run's artefacts are reused.
