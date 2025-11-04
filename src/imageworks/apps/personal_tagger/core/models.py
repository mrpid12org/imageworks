"""Dataclasses and shared models for the personal tagger pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .judge_types import ComplianceReport, PairwiseReport, RubricScores, TechnicalSignals


@dataclass(frozen=True)
class KeywordPrediction:
    """Single keyword prediction with associated confidence."""

    keyword: str
    score: float

    def to_json(self) -> Dict[str, float]:
        return {"keyword": self.keyword, "score": self.score}


@dataclass(frozen=True)
class GenerationModels:
    """Track which models were used for each stage."""

    caption: str
    keywords: str
    description: str
    critique: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "caption": self.caption,
            "keywords": self.keywords,
            "description": self.description,
            "critique": self.critique,
        }


@dataclass
class PersonalTaggerRecord:
    """Aggregated output for a processed image."""

    image: Path
    keywords: List[KeywordPrediction] = field(default_factory=list)
    caption: str = ""
    description: str = ""
    critique: str = ""
    critique_score: Optional[int] = None
    critique_title: Optional[str] = None
    critique_category: Optional[str] = None
    critique_subscores: RubricScores = field(default_factory=RubricScores)
    critique_total: Optional[float] = None
    critique_award: Optional[str] = None
    critique_compliance_flag: Optional[str] = None
    technical_signals: TechnicalSignals = field(default_factory=TechnicalSignals)
    compliance: Optional[ComplianceReport] = None
    pairwise: Optional[PairwiseReport] = None
    duration_seconds: float = 0.0
    backend: str = ""
    models: GenerationModels = field(
        default_factory=lambda: GenerationModels("", "", "", "")
    )
    metadata_written: bool = False
    notes: str = ""

    def keywords_as_list(self) -> List[str]:
        return [prediction.keyword for prediction in self.keywords]

    def to_json(self, schema_version: str) -> Dict[str, object]:
        return {
            "image": str(self.image),
            "keywords": [kp.to_json() for kp in self.keywords],
            "caption": self.caption,
            "description": self.description,
            "critique": self.critique,
            "critique_score": self.critique_score,
            "critique_title": self.critique_title,
            "critique_category": self.critique_category,
            "critique_subscores": self.critique_subscores.as_dict(),
            "critique_total": self.critique_total,
            "critique_award": self.critique_award,
            "critique_compliance_flag": self.critique_compliance_flag,
            "technical_signals": self.technical_signals.to_dict(),
            "compliance": self.compliance.to_dict() if self.compliance else None,
            "pairwise": self.pairwise.to_dict() if self.pairwise else None,
            "metadata_written": self.metadata_written,
            "backend": self.backend,
            "models": self.models.to_dict(),
            "notes": self.notes,
            "duration_seconds": self.duration_seconds,
            "schema_version": schema_version,
        }

    def summary_keywords(self, limit: int = 8) -> str:
        keywords = [kw.title() for kw in self.keywords_as_list()]
        if not keywords:
            return "<none>"
        if len(keywords) <= limit:
            return ", ".join(keywords)
        return ", ".join(keywords[:limit]) + f" … (+{len(keywords) - limit})"


def ensure_unique_keywords(
    predictions: Sequence[KeywordPrediction],
) -> List[KeywordPrediction]:
    """Return ordered unique keywords preserving highest confidence."""

    seen: Dict[str, KeywordPrediction] = {}
    for prediction in predictions:
        key = prediction.keyword.lower()
        if key not in seen or prediction.score > seen[key].score:
            seen[key] = prediction
    return list(seen.values())
