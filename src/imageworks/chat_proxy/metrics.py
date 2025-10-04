from __future__ import annotations

import time
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Deque, Dict


@dataclass
class MetricSample:
    ts: float
    model: str
    backend: str
    ttft_ms: float
    tokens_out: int
    duration_ms: float
    tokens_per_second: float
    stream: bool
    estimated_counts: bool


class MetricsAggregator:
    def __init__(self, capacity: int = 500):
        self.capacity = capacity
        self.samples: Deque[MetricSample] = deque(maxlen=capacity)
        self.start_ts = time.time()
        self.backend_counters: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"total_requests": 0, "streaming_requests": 0}
        )
        self.request_index = 0

    def add(self, sample: MetricSample):
        self.samples.append(sample)
        counters = self.backend_counters[sample.backend]
        counters["total_requests"] += 1
        if sample.stream:
            counters["streaming_requests"] += 1
        self.request_index += 1

    def summary(self) -> dict:
        if not self.samples:
            return {
                "uptime_seconds": time.time() - self.start_ts,
                "rolling": {"count": 0},
                "requests_by_backend": self.backend_counters,
                "schema_version": 1,
            }
        ttfts = [s.ttft_ms for s in self.samples]
        tps = [s.tokens_per_second for s in self.samples if s.tokens_per_second > 0]
        ttfts_sorted = sorted(ttfts)
        p95 = (
            ttfts_sorted[int(0.95 * (len(ttfts_sorted) - 1))] if ttfts_sorted else None
        )
        return {
            "uptime_seconds": time.time() - self.start_ts,
            "rolling": {
                "count": len(self.samples),
                "avg_ttft_ms": sum(ttfts) / len(ttfts),
                "p95_ttft_ms": p95,
                "avg_tokens_per_second": (sum(tps) / len(tps)) if tps else None,
            },
            "requests_by_backend": self.backend_counters,
            "schema_version": 1,
        }
