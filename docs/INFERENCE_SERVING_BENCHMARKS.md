# TranslateGemma Serving Benchmarks & Performance Guide

Status: Active Reference
Date: 2026-08-16

## 1. Benchmarking Objectives

Evaluate and compare the serving performance between:
1. **Legacy Reference**: Single-process Transformers + PEFT API with serialized lock.
2. **Production Stack**: Merged BF16 checkpoint on vLLM with continuous batching + FastAPI Gateway.

## 2. Metrics to Measure

- **Throughput**: Requests per second (RPS), generated output tokens per second.
- **Latency Distribution**: Time to first token (TTFT), P50, P95, and P99 end-to-end request latencies.
- **Queue Time**: Duration requests spend waiting in the gateway queue under concurrency.
- **Resource Utilization**: Peak GPU VRAM (GB), GPU core utilization (%).
- **Correctness & Reliability**: Finish reasons (`stop` vs `length`), error rates (429, 502, 504).

## 3. Benchmark Execution

Run the serving benchmark tool against the gateway:
```bash
python serving/benchmarks/benchmark_serving.py \
    --gateway-url http://localhost:8080 \
    --concurrencies 1,2,4,8,16,32,64 \
    --num-requests 100 \
    --output-file reports/serving_benchmarks_20260816.json
```

## 4. Acceptance SLA Matrix

| Concurrency | Metric | Legacy Transformers API | Target vLLM + Gateway |
|---|---|---|---|
| 1 (Single) | P50 Latency | ~450ms | < 300ms |
| 16 (Interactive) | P95 Latency | ~5,800ms (queue wait) | < 800ms |
| 32 (High load) | RPS | ~2.5 req/s | > 25 req/s |
| 64 (Saturation) | Error rate | Timeout / OOM | 0% (Graceful 429 if queue full) |
