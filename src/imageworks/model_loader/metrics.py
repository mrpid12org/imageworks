"""Performance metrics utilities (Phase 1)."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional


@dataclass
class PerformanceSample:
    ttft_ms: Optional[float]
    tokens_generated: int
    duration_ms: float


class RollingMetrics:
    def __init__(self, maxlen: int = 50) -> None:
        self._samples: Deque[PerformanceSample] = deque(maxlen=maxlen)

    def add(self, sample: PerformanceSample) -> None:
        self._samples.append(sample)

    def summary(self) -> Dict[str, Optional[float]]:
        if not self._samples:
            return {
                "rolling_samples": 0,
                "ttft_ms_avg": None,
                "throughput_toks_per_s_avg": None,
                "last_sample": None,
            }
        ttfts = [s.ttft_ms for s in self._samples if s.ttft_ms is not None]
        ttft_avg = sum(ttfts) / len(ttfts) if ttfts else None
        throughputs: List[float] = []
        for s in self._samples:
            if s.ttft_ms is None:
                continue
            active_ms = s.duration_ms - s.ttft_ms
            if active_ms <= 0:
                continue
            throughputs.append((max(s.tokens_generated - 1, 1)) / (active_ms / 1000))
        throughput_avg = sum(throughputs) / len(throughputs) if throughputs else None
        last = self._samples[-1]
        return {
            "rolling_samples": len(self._samples),
            "ttft_ms_avg": ttft_avg,
            "throughput_toks_per_s_avg": throughput_avg,
            "last_sample": {
                "ttft_ms": last.ttft_ms,
                "tokens_generated": last.tokens_generated,
                "duration_ms": last.duration_ms,
            },
        }


# ---------------- Batch / Stage Timing Helpers -----------------
@dataclass
class StageTiming:
    name: str
    start: float
    end: Optional[float] = None

    def close(self) -> None:
        if self.end is None:
            self.end = time.perf_counter()

    @property
    def duration(self) -> Optional[float]:
        return None if self.end is None else self.end - self.start


@dataclass
class BatchRunMetrics:
    model_name: str
    backend: str
    batch_start: float = field(default_factory=time.perf_counter)
    batch_end: Optional[float] = None
    model_load_seconds: Optional[float] = None
    stages: Dict[str, List[StageTiming]] = field(default_factory=dict)

    def record_model_load(self, seconds: float) -> None:
        self.model_load_seconds = seconds

    def start_stage(self, stage: str) -> StageTiming:
        timing = StageTiming(name=stage, start=time.perf_counter())
        self.stages.setdefault(stage, []).append(timing)
        return timing

    def end_stage(self, timing: StageTiming) -> None:
        timing.close()

    def close_batch(self) -> None:
        if self.batch_end is None:
            self.batch_end = time.perf_counter()

    def summary(self) -> Dict[str, object]:
        total = None if self.batch_end is None else self.batch_end - self.batch_start
        stage_aggregates: Dict[str, Dict[str, float]] = {}
        for stage, timings in self.stages.items():
            durations = [t.duration for t in timings if t.duration is not None]
            if not durations:
                continue
            stage_aggregates[stage] = {
                "count": len(durations),
                "total_seconds": sum(durations),
                "avg_seconds": sum(durations) / len(durations),
                "min_seconds": min(durations),
                "max_seconds": max(durations),
            }
        return {
            "model_name": self.model_name,
            "backend": self.backend,
            "model_load_seconds": self.model_load_seconds,
            "batch_total_seconds": total,
            "stages": stage_aggregates,
        }


__all__ = [
    "PerformanceSample",
    "RollingMetrics",
    "StageTiming",
    "BatchRunMetrics",
]
