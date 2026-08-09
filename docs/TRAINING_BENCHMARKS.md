# Training benchmarks

`scripts/benchmark_training.py` runs bounded SFT jobs through the same model,
dataset, collator, and `Trainer` path used for normal training. It measures the
training loop after dataset preparation and model loading. Optional validation
runs after the timed section.

The parent process expands the configured matrix and launches each entry as an
isolated Accelerate job. Run `--dry-run` before renting GPU time. The dry run
validates batch math and Accelerate profiles, then prints every launch command.

```bash
python scripts/benchmark_training.py --config config.yaml --dry-run
```

## Model-size profiles

The script includes H200 profiles for the three TranslateGemma sizes:

| Profile | Model | Per-device batch | Effective global batch | Batch sweep |
|---|---|---:|---:|---|
| `4b` | `google/translategemma-4b-it` | 12 | 96 | 3, 6, 12 |
| `12b` | `google/translategemma-12b-it` | 6 | 48 | 2, 3, 6 |
| `27b` | `google/translategemma-27b-it` | 2 | 16 | 1, 2 |

Each profile also sets `training.max_length: 2048`. Entries under
`benchmark.model_profiles` merge over these built-in values. A profile can
override keys in the `model`, `data`, `lora`, and `training` sections as well as
its benchmark batch values.

```yaml
benchmark:
  model_sizes: [12b]
  model_profiles:
    12b:
      model:
        base_model_id: "/models/translategemma-12b-it"
      data:
        prepared_cache_dir: "/scratch/benchmark-cache-12b"
      training:
        max_length: 1536
      per_device_batch_size: 6
      effective_batch_size: 48
      batch_sizes: [2, 3, 6]
```

You can add another named profile by defining all required values and including
its name in `model_sizes`.

## Benchmark types

### GPU count

`gpu_count` varies the number of Accelerate processes. It keeps the profile's
per-device and effective global batches fixed, then derives gradient
accumulation for each GPU count.

```text
gradient accumulation = effective global batch / (per-device batch × GPUs)
```

For the 12B profile, the 1/2/4/8-GPU entries use accumulation 8/4/2/1. Every
entry therefore processes an effective global batch of 48 per optimizer step.

```bash
python scripts/benchmark_training.py --config config.yaml \
  --benchmark-types gpu_count --model-sizes 12b \
  --gpu-counts 1 2 4 8
```

The JSON report includes `throughput_vs_baseline` and `scaling_efficiency`.
The run with the smallest GPU count acts as the baseline.

### Per-device batch size

`batch_size` uses one configured GPU count and varies the per-device micro-batch.
The script changes gradient accumulation so every entry retains the model
profile's effective global batch.

```bash
python scripts/benchmark_training.py --config config.yaml \
  --benchmark-types batch_size --model-sizes 4b \
  --batch-gpu-count 4 --batch-sizes 3 6 12
```

The effective global batch must divide cleanly by `per-device batch × GPUs`.
The script rejects invalid combinations before it launches Accelerate. Keep the
GPU count, maximum steps, model profile, and dataset limit unchanged when you
compare these results.

### Gradient checkpointing and packing

`training_options` runs the profiles listed in
`benchmark.training_option_profiles` at one fixed GPU count. The checked-in
configuration uses the full two-by-two matrix:

| Variant | Gradient checkpointing | Packing |
|---|---|---|
| `checkpointing_packing` | on | on |
| `checkpointing_only` | on | off |
| `packing_only` | off | on |
| `neither` | off | off |

```bash
python scripts/benchmark_training.py --config config.yaml \
  --benchmark-types training_options --model-sizes 12b \
  --training-options-gpu-count 4
```

Packing requires the repository's Flash Attention 3 configuration. Runs with
packing disabled continue to use the configured attention implementation, so
the comparison isolates packing instead of changing two settings at once.

You can reduce the option matrix in YAML when a model does not fit without
checkpointing:

```yaml
benchmark:
  training_option_profiles:
    - name: checkpointing_packing
      gradient_checkpointing: true
      packing: true
    - name: checkpointing_only
      gradient_checkpointing: true
      packing: false
```

## Selecting the matrix

The checked-in configuration selects all three benchmark types and model sizes.
That expands to 32 measured jobs:

- 12 GPU-count jobs: three model sizes × four GPU counts
- 8 batch-size jobs: three 4B, three 12B, and two 27B batches
- 12 training-option jobs: three model sizes × four option profiles

The script also runs one discarded warm-up per model size to populate the
model-specific Triton and Flash Attention caches. Use CLI selectors for smaller
experiments:

```bash
python scripts/benchmark_training.py --config config.yaml \
  --benchmark-types gpu_count batch_size \
  --model-sizes 4b 12b \
  --max-examples 20000 --max-steps 200 \
  --no-run-evaluation
```

CLI values override the matching `benchmark` keys. `--per-device-batch-size`
and `--effective-batch-size` replace the baseline values for every selected
model profile. `--batch-sizes` replaces each selected profile's batch sweep.

## Reports

The script prints a Rich report after the matrix finishes. It includes system
and software details, GPU inventory, dataset artifacts, measured runs, device
telemetry, comparison winners, scaling efficiency, failures, and run totals.

Each measured job writes its result, console log, and raw GPU samples below the
configured output directory:

```text
logs/training_benchmark/
├── warmup/<model-size>/result.json
├── gpu_count/<model-size>/<variant>/
│   ├── result.json
│   ├── run.log
│   └── gpu_telemetry.json
├── batch_size/<model-size>/<variant>/...
├── training_options/<model-size>/<variant>/...
├── benchmark_results.json
├── benchmark_results.csv
├── benchmark_results_summary.md
└── benchmark_results_summary.html
```

`benchmark_results.json` is the complete machine-readable report. The CSV keeps
one row per matrix entry. Markdown provides a compact review artifact, and HTML
captures the full Rich console report.

### Training measurements

Each successful entry records:

- training-loop wall time, job wall time, optimizer steps, and seconds per step;
- samples, padded tokens, non-padding tokens, and each throughput rate;
- non-padding tokens per GPU-second and total GPU-seconds;
- train loss, optional validation loss, and the raw Trainer metric mappings;
- model ID and parameter counts, sequence length, precision, attention backend,
  optimizer, learning rate, quantization, Liger, checkpointing, and packing;
- per-device batch, accumulation, GPU count, and effective global batch;
- relative throughput and GPU scaling efficiency within its comparison group.

`non_padding_tokens_per_second` is the main throughput measure.
`padding_efficiency` shows how much work unpacked runs spend on real tokens.
The timer starts immediately before `trainer.train()` and stops after CUDA
synchronization. Dataset preparation, model loading, and optional validation
stay outside the training-loop timer. `job_wall_seconds` covers the complete
Accelerate process.

### VRAM and GPU telemetry

The report keeps two memory measurements because they answer different
questions:

- `per_rank_pytorch_memory` records peak allocated and reserved CUDA memory for
  every distributed rank. The summary table shows the largest rank value.
- `gpu_telemetry.json` samples device memory through `nvidia-smi`. This includes
  the CUDA context, NCCL, kernels, and allocations outside PyTorch's allocator.

Device samples also contain GPU and memory utilization, power draw and limit,
temperature, SM and memory clocks, device UUID, and total VRAM. The report
calculates average/minimum/maximum values per GPU and estimates watt-hours from
average sampled power and job duration. Treat energy as an estimate because
`nvidia-smi` sampling is not a hardware power meter.

The same sampling loop records host CPU utilization, RAM use, and 1/5/15-minute
load averages. These host-wide values help identify tokenization, dataloader, or
memory-pressure bottlenecks during a job.

```yaml
benchmark:
  collect_gpu_telemetry: true
  telemetry_interval_seconds: 1.0
```

Use `--no-collect-gpu-telemetry` when another profiler owns NVML. You can change
the interval with `--telemetry-interval-seconds`.

### Reproducibility inventory

The combined JSON records hostname, OS, kernel, CPU model, socket/core/thread
counts, system RAM, workspace disk capacity, GPU inventory, VRAM, power limits,
topology, driver, CUDA, cuDNN, Python, training-package versions, selected
environment variables, Git revision, dirty files, the resolved configuration,
and configuration hashes.

Source dataset files include resolved paths, sizes, modification times, and
SHA-256 hashes. Directory-backed datasets record the resolved directory but do
not hash its contents.

### Failed entries

The default `fail_fast: false` keeps the matrix running after an entry fails.
The reports retain its exit code, command, log path, job duration, and any GPU
telemetry collected before failure. Set `fail_fast: true` when later entries
cannot provide useful results after the first failure.

Use a new `benchmark.output_dir` for each hardware or software environment.
Otherwise a later matrix can replace raw result files that share the same model,
type, and variant path.

## Comparison checklist

Before comparing two results, confirm that they share:

- the same model profile, dataset limit, maximum steps, and effective batch;
- the same top-level config SHA and Accelerate config SHA;
- the same GPU model, CUDA/PyTorch/Flash Attention stack, and source dataset.

Change only the dimension named by the benchmark type. Run long enough that
startup effects form a small fraction of the measured training loop. If a
variant runs out of memory, record the failure rather than lowering its sequence
length or effective batch inside the same comparison.
