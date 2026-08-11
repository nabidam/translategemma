# Evaluation runbook

How to run `evaluate_translations.py` with MetricX and COMET enabled, what it
does across GPUs, and what it reports. Written 2026-08-11, after the scoring
stages were made rank-sharded.

For metric *selection* — which metric to trust and why — see
`docs/EVALUATION_BACKLOG.md`. For cross-model comparison of several systems, use
the separate benchmark described in `docs/TRANSLATION_BENCHMARK.md`; this
runbook covers only the base-versus-adapter evaluation that training triggers.

## 1. Install MetricX

MetricX is not on PyPI. It is source-only, and `evaluate_metricx` fails with a
pointed ImportError if it is missing:

```bash
git clone https://github.com/google-research/metricx.git ../metricx
export PYTHONPATH="$PWD/../metricx:$PYTHONPATH"
uv run python -c "from metricx24.models import MT5ForRegression; print('metricx ok')"
```

Put the `export` in whatever shell or unit file launches evaluation. Without it
on the launching environment, the run reaches the MetricX stage — after
generation has already completed — and only then fails.

COMET needs no such step; `unbabel-comet` is a declared dependency.

## 2. Accept the XCOMET licence and authenticate

`Unbabel/XCOMET-XL` is a gated repository. Accept the licence in a browser with
the same account whose token is on the host, then authenticate:

```bash
hf auth login
```

## 3. Stage the checkpoints before the GPU job

```bash
uv run python scripts/fetch_offline_assets.py --config config.yaml
```

Equivalent manual downloads:

```bash
uv run python -c "from comet import download_model; download_model('Unbabel/XCOMET-XL')"
uv run hf download google/metricx-24-hybrid-large-v2p6
uv run hf download google/mt5-xl --include 'tokenizer*' 'spiece.model' 'special_tokens_map.json'
```

Approximate sizes: XCOMET-XL 14 GB, MetricX-24-hybrid-large 4.9 GB, mT5-XL
tokenizer 20 MB. MetricX checkpoints contain no tokenizer of their own, which is
why `metricx_tokenizer_id` points at mT5 separately.

## 4. Smoke test first

Set `evaluation.smoke_test_max_examples: 32` and run on two GPUs. This exercises
sharding, both scoring models, and the cache-completeness check in about a
minute, and it surfaces a licence or gating problem before a full generation
pass has been spent.

```bash
uv run accelerate launch --config_file accelerate_configs/h200_2gpu.yaml \
  evaluate_translations.py --config config.yaml \
  --adapter-path translategemma-farsi-science/sft_final
```

Remove `smoke_test_max_examples` afterwards, and remember that the smoke run's
caches cover only those 32 rows; the full run resumes from them rather than
conflicting with them.

## 5. Full run

```bash
uv run accelerate launch --config_file accelerate_configs/h200_8gpu.yaml \
  evaluate_translations.py --config config.yaml \
  --adapter-path translategemma-farsi-science/sft_final
```

## Execution model

Three stages run per system, in this order, and each frees its model before the
next loads. Peak VRAM is therefore the largest single stage, not their sum.

| Stage | Parallelism | Cache file |
| --- | --- | --- |
| Generation | Sharded round-robin across ranks | `.cache_<prefix>_rank<i>.jsonl` |
| MetricX | Sharded round-robin across ranks | `.cache_<prefix>_metricx_rank<i>.jsonl` |
| COMET | Sharded round-robin across ranks | `.cache_<prefix>_comet_rank<i>.jsonl` |

The transparent metrics (BLEU, chrF++, preservation rates) run on the main
process only. They are CPU-bound and cheap next to generation, and only the main
process writes the CSV they land in.

Single-process runs write the same files without the `_rank<i>` suffix.

### Why COMET is sharded

COMET originally ran on the main process only. With a large checkpoint that is
not merely slow, it is dangerous: the other ranks block at the next barrier for
the whole COMET pass, and if that exceeds the process group's timeout the NCCL
watchdog aborts a run whose generation had already succeeded. Sharding removes
the long one-sided wait entirely.

Two mechanisms make a per-rank COMET Trainer safe under `accelerate launch`:

- `predict(devices=[state.local_process_index])` pins each rank's Lightning
  Trainer to the GPU accelerate already assigned it. Without it every rank
  defaults to `cuda:0` and stacks every copy of the model onto one device.
- `_distributed_env_hidden()` removes `WORLD_SIZE`, `RANK`, `LOCAL_RANK`,
  `TORCHELASTIC_RUN_ID` and friends for the duration of `predict`, so Lightning
  cannot auto-detect this script's process group and attempt to join it.
  `torch.distributed` is already bootstrapped, so the real group is unaffected;
  the variables are restored in a `finally`.

`_download_comet_checkpoint` resolves the checkpoint on the main process first
and then barriers, because concurrent `download_model` calls from every rank
race on one cache directory. Every rank calls it, so the barrier stays
symmetric.

The reported COMET system score is recomputed as the mean of the segment scores
rather than taken from `output.system_score`, since each rank sees only its own
shard. For these checkpoints those are the same quantity.

### Single GPU

Both single-GPU launch styles work with no configuration change:

- `python evaluate_translations.py` — no process group exists, so every
  `wait_for_everyone()` is a no-op and `num_processes` is 1.
- `accelerate launch --num_processes 1` — same values; the environment scrubbing
  pops whatever accelerate set and restores it.

`devices=[state.local_process_index]` resolves to `[0]`, the same device
`.to(state.device)` uses. Under `CUDA_VISIBLE_DEVICES=3` torch renumbers that
card to index 0, so `[0]` still selects the intended physical GPU.

With no CUDA at all, COMET is invoked with `gpus=0` and MetricX lands on CPU.
This runs, slowly, and is only useful for wiring checks.

On a card smaller than the production host, note that `load_from_checkpoint`
loads COMET in **fp32**: XCOMET-XL is roughly 14 GB of weights before
activations. Lower `comet_batch_size` to 4 or 2 on a 24 GB card. `comet_gpus` is
per rank and stays at 1 under every profile.

### Caches interoperate across launch modes

`_gather_cached_scores` globs `.cache_<prefix>_<stage>*.jsonl`, matching both the
per-rank files and the single-process file. Consequences worth using
deliberately:

- A killed 8-GPU run can be resumed on one GPU, or the reverse.
- Generation can run multi-GPU and scoring single-process, since the completed
  stages are read from cache and skipped. This is the fallback if sharded COMET
  ever misbehaves: run the full job with `comet_enabled: false`, then re-run
  single-process with it true.
- `--force` deletes every `.cache_<prefix>_*` file, so it re-runs generation as
  well as scoring. There is no per-stage force flag.

### Missing rows fail the run

No stage substitutes a default for a row that is absent from the caches after
its barrier. Generation and both scorers raise, naming the count and the first
ten missing indices.

This is deliberate. The previous behaviour filled missing MetricX scores with
`0.0`, and MetricX is lower-is-better, so a rank that died made the corpus mean
*improve*: a run could degrade and report success. The generation equivalent
filled with `""`, which then entered the degeneration audit as a genuine empty
translation and moved the failure rate.

## What the run reports

Rich renders four blocks on the main process after both systems finish:

1. **Systems** — one row per system, carrying only the headline metrics (clean
   decoding rate, MetricX, COMET). Fixed width regardless of how many metrics
   are enabled.
2. **`<adapter> vs <base>`** — one row *per metric*, with base value, adapter
   value, and a delta coloured by that metric's own direction. MetricX falling
   is green; `empty_output` rising is red. Enabling more metrics grows this
   table downward, never wider, which is why it exists as a separate table.
3. **Decoding failures** — per-class breakdown from the degeneration audit,
   previously visible only in the log.
4. **Panel** — the degeneration gate verdict and the output paths.

The `Scored rows` column exists because preservation metrics are NaN for rows
whose source contains no number, acronym, or formula. Their mean is over that
subset, and the column makes the denominator explicit.

Files written to `evaluation/`: `summary.json`,
`<prefix>_detailed_scores.csv`, `<prefix>_human_review_sample.csv`. Render the
HTML review page from them with `report_evaluation.py`.

## Disabling the metrics

`evaluation.metricx_enabled`, `evaluation.comet_enabled`, and
`evaluation.transparent_metrics_enabled` can each be set false independently,
and the degeneration audit has its own `degeneration_audit_enabled`. With all
four off, only generation and the report run.

Branch symmetry is what makes this safe under `accelerate`: every
`wait_for_everyone()` sits outside any main-process guard, and every flag that
gates a barrier-containing stage is read from the same config on every rank. A
flag that was true on one rank and false on another would hang the job, which is
why these are configuration values and never derived from local state.

The report degrades cleanly. With every metric disabled the systems table prints
its identity columns, the delta and failure tables are skipped rather than
printed empty, and the panel still reports paths and the gate verdict.

## Known gaps

- **Non-coherent filesystems.** In `_score_sharded`, a rank that sees a complete
  cache returns after one barrier while a rank that sees a partial cache
  proceeds to a second. On local disk this cannot happen; across nodes on NFS
  with attribute caching it could, and would hang. Single-node runs are
  unaffected.
- **MetricX scores one example per forward pass.** `padding=False` and a
  single-row loop mean there is no batching to amortize a large checkpoint.
  Sharding hides this at 8 ranks; it is the reason XL rather than XXL is
  recommended in `docs/EVALUATION_BACKLOG.md`.
- **XCOMET error spans are discarded.** `evaluate_comet` keeps only the scalar
  scores. See the backlog for why persisting them is worth doing.
