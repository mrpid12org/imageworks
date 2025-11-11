"""Dataclasses for Judge Vision outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from imageworks.apps.judge_vision import (
    PairwiseReport,
    ComplianceReport,
    RubricScores,
    TechnicalSignals,
)


@dataclass
class JudgeVisionRecord:
    image: Path
    critique: str = ""
    critique_title: Optional[str] = None
    critique_category: Optional[str] = None
    competition_category: str = "Colour"
    style_inference: Optional[str] = None
    image_title: Optional[str] = None
    critique_subscores: RubricScores = field(default_factory=RubricScores)
    critique_total_initial: Optional[float] = None
    critique_total: Optional[float] = None
    critique_award: Optional[str] = None
    critique_compliance_flag: Optional[str] = None
    compliance: Optional[ComplianceReport] = None
    technical_signals: TechnicalSignals = field(default_factory=TechnicalSignals)
    backend: str = ""
    model: str = ""
    duration_seconds: float = 0.0
    pairwise: Optional[PairwiseReport] = None
    pairwise_score_initial: Optional[float] = None
    pairwise_wins: Optional[int] = None
    pairwise_comparisons: Optional[int] = None
    pairwise_win_ratio: Optional[float] = None

    def to_json(self, schema_version: str) -> Dict[str, object]:
        return {
            "image": str(self.image),
            "critique": self.critique,
            "critique_title": self.critique_title,
            "critique_category": self.critique_category,
            "competition_category": self.competition_category,
            "style_inference": self.style_inference,
            "image_title": self.image_title,
            "critique_subscores": self.critique_subscores.as_dict(),
            "critique_total_initial": self.critique_total_initial,
            "critique_total": self.critique_total,
            "critique_award": self.critique_award,
            "critique_compliance_flag": self.critique_compliance_flag,
            "technical_signals": self.technical_signals.to_dict(),
            "compliance": self.compliance.to_dict() if self.compliance else None,
            "backend": self.backend,
            "model": self.model,
            "duration_seconds": self.duration_seconds,
            "schema_version": schema_version,
            "pairwise": self.pairwise.to_dict() if self.pairwise else None,
            "pairwise_score_initial": self.pairwise_score_initial,
            "pairwise_wins": self.pairwise_wins,
            "pairwise_comparisons": self.pairwise_comparisons,
            "pairwise_win_ratio": self.pairwise_win_ratio,
        }


__all__ = ["JudgeVisionRecord"]
