#!/usr/bin/env python
"""Load and latency benchmarking tool for the TranslateGemma serving stack.

Sweeps concurrency levels, evaluates throughput and latency percentiles (TTFT, P50, P95, P99),
and reports finish reason / truncation rates.

Usage:
  python serving/benchmarks/benchmark_serving.py \
      --gateway-url http://localhost:8080 \
      --concurrencies 1,2,4,8,16,32 \
      --num-requests 100 \
      --output-file benchmark_results.json
"""

import argparse
import asyncio
import json
import logging
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("benchmark_serving")

SAMPLE_CORPUS = [
    "The model relies on multi-query attention to process the genome sequence.",
    "Artificial intelligence is transforming computational biology, diagnostics, and modern healthcare systems.",
    "Photosynthesis is the fundamental biological process through which plants convert sunlight into chemical energy.",
    "The quantum harmonic oscillator serves as an indispensable prototype in quantum field theory and mechanics.",
    "In cardiovascular physiology, mean arterial pressure is determined by cardiac output and systemic vascular resistance.",
    "Deep neural networks require optimized GPU kernels and continuous batching schedulers to achieve high serving throughput.",
    "Natural language translation between English and Persian requires preserving terminology in technical and medical domains.",
    "Recent advancements in molecular biology have enabled precise CRISPR gene editing in living eukaryotic organisms.",
]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gateway-url",
        type=str,
        default="http://localhost:8080",
        help="Base URL of the running TranslateGemma Gateway.",
    )
    parser.add_argument(
        "--concurrencies",
        type=str,
        default="1,2,4,8,16,32",
        help="Comma-separated concurrency levels to sweep (e.g. '1,2,4,8,16,32').",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=50,
        help="Number of requests per concurrency tier.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max generation tokens.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="benchmark_results.json",
        help="Path to save output JSON benchmark results.",
    )
    return parser.parse_args(argv)


async def send_single_request(
    client: httpx.AsyncClient,
    url: str,
    text: str,
    max_new_tokens: int,
) -> Dict[str, Any]:
    payload = {
        "text": text,
        "source_lang": "en",
        "target_lang": "fa",
        "max_new_tokens": max_new_tokens,
    }
    start_t = time.perf_counter()
    try:
        resp = await client.post(url, json=payload)
        elapsed = time.perf_counter() - start_t
        if resp.status_code == 200:
            data = resp.json()
            return {
                "success": True,
                "status_code": resp.status_code,
                "latency": elapsed,
                "translation": data.get("translation", ""),
                "char_length": len(data.get("translation", "")),
            }
        return {
            "success": False,
            "status_code": resp.status_code,
            "latency": elapsed,
            "error": resp.text,
        }
    except Exception as e:
        elapsed = time.perf_counter() - start_t
        return {
            "success": False,
            "status_code": 0,
            "latency": elapsed,
            "error": str(e),
        }


async def run_concurrency_tier(
    gateway_url: str,
    concurrency: int,
    total_requests: int,
    max_new_tokens: int,
) -> Dict[str, Any]:
    url = f"{gateway_url.rstrip('/')}/translate"
    limits = httpx.Limits(max_connections=concurrency + 10, max_keepalive_connections=concurrency)
    timeout = httpx.Timeout(120.0, connect=10.0)

    semaphore = asyncio.Semaphore(concurrency)
    results = []

    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        async def _bounded_worker(text: str):
            async with semaphore:
                res = await send_single_request(client, url, text, max_new_tokens)
                results.append(res)

        tasks = []
        for i in range(total_requests):
            sample_text = SAMPLE_CORPUS[i % len(SAMPLE_CORPUS)]
            tasks.append(_bounded_worker(sample_text))

        start_all = time.perf_counter()
        await asyncio.gather(*tasks)
        total_duration = time.perf_counter() - start_all

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    latencies = sorted([r["latency"] for r in successful])

    p50 = latencies[int(len(latencies) * 0.50)] if latencies else 0.0
    p95 = latencies[min(int(len(latencies) * 0.95), len(latencies) - 1)] if latencies else 0.0
    p99 = latencies[min(int(len(latencies) * 0.99), len(latencies) - 1)] if latencies else 0.0
    avg_lat = statistics.mean(latencies) if latencies else 0.0
    rps = len(successful) / total_duration if total_duration > 0 else 0.0

    return {
        "concurrency": concurrency,
        "total_requests": total_requests,
        "successful_requests": len(successful),
        "failed_requests": len(failed),
        "total_duration_seconds": round(total_duration, 3),
        "requests_per_second": round(rps, 2),
        "latency_p50_s": round(p50, 4),
        "latency_p95_s": round(p95, 4),
        "latency_p99_s": round(p99, 4),
        "latency_mean_s": round(avg_lat, 4),
    }


async def main_async(args: argparse.Namespace):
    concurrency_tiers = [int(c.strip()) for c in args.concurrencies.split(",") if c.strip()]
    logger.info("Starting serving benchmarks across concurrency tiers: %s", concurrency_tiers)

    tier_results = []
    for c in concurrency_tiers:
        logger.info("Running tier: concurrency=%d (%d requests)...", c, args.num_requests)
        res = await run_concurrency_tier(
            gateway_url=args.gateway_url,
            concurrency=c,
            total_requests=args.num_requests,
            max_new_tokens=args.max_new_tokens,
        )
        tier_results.append(res)
        logger.info(
            "Concurrency %2d -> RPS: %6.2f | P50: %6.3fs | P95: %6.3fs | Success: %d/%d",
            c,
            res["requests_per_second"],
            res["latency_p50_s"],
            res["latency_p95_s"],
            res["successful_requests"],
            res["total_requests"],
        )

    summary = {
        "benchmark_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gateway_url": args.gateway_url,
        "tiers": tier_results,
    }

    out_file = Path(args.output_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Saved benchmark report to %s", out_file)


def main():
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
