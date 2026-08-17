# Serving A/B: vLLM vs in-process transformers — measured

**Date**: 2026-08-17
**Host**: production GPU host, single device, nothing else resident
**Verdict**: vLLM is ~7x faster at every batch size measured. Keep it.

Method and reproduction: `docs/SERVING_BENCHMARK_AB.md`, `./scripts/bench_ab.sh`.
Raw reports: `benchmark_ab/benchmark_http_20260817-172557_vllm-bf16.{json,md}`
and `benchmark_ab/benchmark_http_20260817-173232_hf-lora.{json,md}`.

## What was compared

| | Arm A (`vllm-bf16`) | Arm B (`hf-lora`) |
|---|---|---|
| Runtime | vLLM 0.13.0 | transformers 4.57 `generate()` |
| Weights | merged checkpoint, bf16 | base bf16 + LoRA applied at run time |
| Batching | continuous, scheduler-owned | fixed chunks behind a GPU lock |
| API process | CPU-only gateway | holds the weights itself |

Same fine-tune (`sft/checkpoint-23500`), same greedy decoding, same prompt
rendering, `TG_BATCH_SIZE=32` on both so the sweep's large batches are real
batches rather than sequential chunks. One arm resident at a time.

## Results

| Measure | Arm B (transformers) | Arm A (vLLM) | vLLM advantage |
|---|---|---|---|
| Latency @ batch 1 | 0.93 s | 0.12 s | **7.8x faster** |
| Throughput @ batch 32 | 129.6 tok/s | 915.9 tok/s | **7.1x** |
| ms / decode step | 7.32 | 0.98 | 7.5x |
| Page (250 words), sentence-split | 8.03 s | 1.14 s | 7.0x |
| Page (250 words), whole | 33.91 s | 5.11 s | 6.6x |

Full sweep, wall-clock seconds and output tokens/second:

| batch | transformers wall | vLLM wall | transformers tok/s | vLLM tok/s |
|---:|---:|---:|---:|---:|
| 1 | 0.93 | 0.12 | 8.6 | 68.6 |
| 2 | 2.29 | 0.24 | 11.4 | 106.8 |
| 4 | 6.77 | 0.89 | 16.5 | 125.0 |
| 8 | 7.29 | 0.98 | 35.1 | 257.2 |
| 16 | 8.77 | 1.19 | 67.1 | 491.3 |
| 32 | 9.08 | 1.29 | 129.6 | 915.9 |

## The win is per-token efficiency, not scheduling

The obvious hypothesis — vLLM's continuous batching beats a serialised
`generate()` loop under load — is **not** what the data shows. The ratio is
flat at ~7x from batch 1 to batch 32, and arm B batches perfectly respectably
(32x the work in 9.8x the time).

vLLM is simply ~7x cheaper per decode step: CUDA graphs, paged attention, fused
kernels, and merged weights that skip the per-layer LoRA matmul entirely. It
wins at concurrency 1, on a single short segment, with nothing else in flight.

Report it that way. "vLLM wins under load" would be the wrong story and would
predict the wrong thing about a low-traffic deployment.

## Correctness

Both arms returned **byte-identical** translations for the same segments
(`نتایج قابل تکرار بودند.`), and no configuration tripped the comparison's 5%
output-token drift check. Two things follow:

1. The timing comparison measures equal work, so tokens/second are commensurable.
2. The RoPE overlay that arm A needs to load at all
   (`scripts/vllm_rope_shim.py`, see `docs/DEPLOYMENT_BACKLOG.md`) does not
   change generation. Arm B served the unshimmed base and produced the same
   text, which is the only evidence available that the flattened rope block is
   numerically equivalent.

## Confounds

- **Merged vs adapter is not purely a runtime difference.** Arm B pays a real
  per-layer `B@A` matmul on every generated token that arm A does not. Some
  unknown share of the 7x is that, not vLLM. Isolating pure runtime would need a
  third arm: vLLM with `--enable-lora` over the unmerged base.
- **Arm A splits prompt rendering across a container hop**; arm B renders in the
  same process that generates. Small, and it counts against arm A, so it does
  not inflate the result.
- **Concurrency was never swept.** The benchmark drives both arms through one
  client. A many-simultaneous-callers test would likely widen the gap, since
  that is where continuous batching earns its keep — but it was not measured and
  should not be claimed.
- **bf16 both sides.** The fp8 quant was deliberately excluded so this answers a
  runtime question. fp8-vs-bf16 on vLLM is a separate experiment.
