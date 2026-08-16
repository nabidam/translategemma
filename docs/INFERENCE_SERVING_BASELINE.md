# TranslateGemma Inference Serving Baseline

Status: Active Reference
Date: 2026-08-16

## 1. Purpose

This document establishes the performance, quality, and correctness baseline for serving TranslateGemma adapter weights before migrating from the single-process PyTorch/PEFT API to the merged-checkpoint vLLM + FastAPI gateway stack.

## 2. Invariant Specifications

The serving system must strictly preserve the conditioning and stopping invariants established during adapter training:

| Invariant | Specification | Verification Method |
|---|---|---|
| **Prompt Template** | Exact SFT training rendering (`render_training_prompt`) with `\n\n        ` assistant prefix | `prompting.py` unit tests & byte-level prompt inspection |
| **Stop Tokens** | `<eos>` (`id=1`) and `<end_of_turn>` (`id=106`) | `resolve_stop_token_ids()` & generation config validation |
| **Decoding Policy** | Deterministic greedy decoding (`temperature=0.0` or `do_sample=False`, `top_p=1.0`) | Repeated identical query equivalence |
| **Response Contract** | `/translate` -> `{"translation": "...", "system": "adapter", "source_lang": "en", "target_lang": "fa"}` | Gateway schema parity tests |
| **Batch Response Contract** | `/translate/batch` -> `{"translations": [...], ...}` preserving input order | Gateway batch ordering tests |
| **Stop Behavior** | Output finishes at `<end_of_turn>` boundary without runaway text or trailing loop | Inspection of `finish_reason="stop"` and token lengths |

## 3. Workload Classes

The serving workload is divided into three representative profiles:

### 3.1 Interactive (Latency-Sensitive)
- **Characteristics**: Single segment, 10 to 60 words (15 - 90 tokens).
- **Target SLA**: Time to first token (TTFT) < 150ms, End-to-end P95 < 600ms at concurrency = 16.
- **Primary Route**: Direct continuous batching submission.

### 3.2 Document (Coherence & Chunking)
- **Characteristics**: Multi-sentence paragraphs or structured documents (100 - 1,000 words).
- **Processing**: Optional sentence-aware splitting (`pysbd`), bounded parallel generation, ordered reassembly.
- **Target SLA**: End-to-end P95 < 2,500ms at concurrency = 4.

### 3.3 Bulk (Throughput-Sensitive)
- **Characteristics**: Batch requests containing 10 to 100 independent sentences.
- **Processing**: Concurrent submission into vLLM scheduler with length bucketing.
- **Target SLA**: Aggregate throughput > 250 output tokens/second on a single H200 GPU.

## 4. Evaluation and Parity Suite

When validating the merged checkpoint and vLLM serving stack against the reference Transformers implementation:

1. **Exact Match Ratio**: Greedy outputs should achieve >= 99.5% token/string equality against the unmerged PEFT reference. Any deviation must be verified to be floating-point BF16 accumulation noise rather than structural drift.
2. **Quality Metrics**:
   - COMET score delta: $\Delta \le 0.05$
   - chrF++ score delta: $\Delta \le 0.1$
   - Zero degeneration / runaway repetition rate (0 instances across 1,000 test sequences).
3. **Stop Invariant Compliance**: 100% of completed generations terminate on token `106` or `1` prior to reaching `max_new_tokens`.

## 5. Metrics Recording Schema

All benchmark runs must record the following parameters:
- Model release ID and Git commit SHA
- Hardware configuration (GPU model, count, driver version, CUDA version)
- Engine configuration (vLLM version, max-model-len, max-num-batched-tokens, max-num-seqs)
- Concurrency levels (1, 2, 4, 8, 16, 32, 64)
- Metrics: TTFT (p50, p95), ITL (inter-token latency), E2E latency (p50, p95, p99), Request throughput (req/s), Token throughput (prompt tokens/s, completion tokens/s), Finish reason distribution (`stop` vs `length`).
