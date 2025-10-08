"""Data models for image similarity analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional


class SimilarityVerdict(str, Enum):
    """Final status assigned to a candidate image."""

    PASS = "pass"
    QUERY = "query"
    FAIL = "fail"


@dataclass(frozen=True)
class StrategyMatch:
    """Similarity evidence returned by an individual strategy."""

    candidate: Path
    reference: Path
    score: float
    strategy: str
    reason: str = ""
    extra: Dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {
            "candidate": str(self.candidate),
            "reference": str(self.reference),
            "score": float(self.score),
            "strategy": self.strategy,
            "reason": self.reason,
            "extra": dict(self.extra),
        }


@dataclass
class CandidateSimilarity:
    """Aggregated similarity result for a candidate image."""

    candidate: Path
    verdict: SimilarityVerdict
    top_score: float
    matches: List[StrategyMatch] = field(default_factory=list)
    thresholds: Dict[str, float] = field(default_factory=dict)
    metric: str = "cosine"
    notes: List[str] = field(default_factory=list)

    def to_json(self) -> Dict[str, object]:
        return {
            "candidate": str(self.candidate),
            "verdict": self.verdict.value,
            "top_score": float(self.top_score),
            "metric": self.metric,
            "thresholds": dict(self.thresholds),
            "matches": [match.as_dict() for match in self.matches],
            "notes": list(self.notes),
        }

    def best_match(self) -> Optional[StrategyMatch]:
        return self.matches[0] if self.matches else None
