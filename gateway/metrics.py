"""Production observability and structured metrics collection for the Gateway."""

import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class MetricsCollector:
    """In-memory structured metrics collector."""

    request_counts: Counter = field(default_factory=Counter)
    error_counts: Counter = field(default_factory=Counter)
    finish_reason_counts: Counter = field(default_factory=Counter)
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_truncations: int = 0

    # Latency tracking (sliding or lifetime)
    latencies: List[float] = field(default_factory=list)
    queue_wait_times: List[float] = field(default_factory=list)

    def record_request(self, endpoint: str, workload_class: str):
        self.request_counts[f"{endpoint}:{workload_class}"] += 1

    def record_error(self, endpoint: str, status_code: int):
        self.error_counts[f"{endpoint}:{status_code}"] += 1

    def record_completion(
        self,
        endpoint: str,
        workload_class: str,
        latency: float,
        queue_wait: float,
        prompt_tokens: int,
        completion_tokens: int,
        finish_reason: str,
    ):
        self.finish_reason_counts[finish_reason] += 1
        if finish_reason == "length":
            self.total_truncations += 1

        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens

        # Keep a bounded sample of recent latencies for percentiles
        if len(self.latencies) > 2000:
            self.latencies = self.latencies[-1000:]
        self.latencies.append(latency)

        if len(self.queue_wait_times) > 2000:
            self.queue_wait_times = self.queue_wait_times[-1000:]
        self.queue_wait_times.append(queue_wait)

    def get_percentiles(self, data: List[float]) -> Dict[str, float]:
        if not data:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        s = sorted(data)
        n = len(s)
        return {
            "p50": round(s[int(n * 0.50)], 4),
            "p95": round(s[min(int(n * 0.95), n - 1)], 4),
            "p99": round(s[min(int(n * 0.99), n - 1)], 4),
        }

    def get_summary(self) -> Dict[str, Any]:
        return {
            "requests": dict(self.request_counts),
            "errors": dict(self.error_counts),
            "finish_reasons": dict(self.finish_reason_counts),
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_truncations": self.total_truncations,
            "latency_seconds": self.get_percentiles(self.latencies),
            "queue_wait_seconds": self.get_percentiles(self.queue_wait_times),
            "recorded_samples": len(self.latencies),
        }


_metrics: MetricsCollector | None = None


def get_metrics() -> MetricsCollector:
    global _metrics
    if _metrics is None:
        _metrics = MetricsCollector()
    return _metrics
