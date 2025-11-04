"""Typed helpers for the Judge Vision workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


def _clamp(value: Optional[float], lower: float, upper: float) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):  # noqa: BLE001
        return None
    if numeric < lower:
        return lower
    if numeric > upper:
        return upper
    return numeric


@dataclass
class RubricScores:
    """Container for rubric subscores returned by the VLM critique."""

    impact: Optional[float] = None
    composition: Optional[float] = None
    technical: Optional[float] = None
    category_fit: Optional[float] = None

    def as_dict(self) -> Dict[str, Optional[float]]:
        return {
            "impact": _clamp(self.impact, 0.0, 5.0),
            "composition": _clamp(self.composition, 0.0, 5.0),
            "technical": _clamp(self.technical, 0.0, 5.0),
            "category_fit": _clamp(self.category_fit, 0.0, 5.0),
        }

    def summary(self) -> str:
        parts: List[str] = []
        mapping = {
            "impact": self.impact,
            "composition": self.composition,
            "technical": self.technical,
            "category": self.category_fit,
        }
        for key, value in mapping.items():
            if value is None:
                continue
            parts.append(f"{key}={value:.1f}")
        return ", ".join(parts) if parts else "<no scores>"


@dataclass
class ComplianceReport:
    """Outcome of Stage 0 compliance checks."""

    passed: bool
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    checks: Dict[str, object] = field(default_factory=dict)

    def to_prompt(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        details: List[str] = [status]
        if self.issues:
            details.append("Issues: " + "; ".join(self.issues))
        if self.warnings:
            details.append("Warnings: " + "; ".join(self.warnings))
        if not self.issues and not self.warnings:
            details.append("No rule infractions detected")
        return " | ".join(details)

    def to_dict(self) -> Dict[str, object]:
        return {
            "passed": bool(self.passed),
            "issues": list(self.issues),
            "warnings": list(self.warnings),
            "checks": dict(self.checks),
        }


@dataclass
class TechnicalSignals:
    """Stage 1 technical signal outputs used to brief the VLM."""

    metrics: Dict[str, float] = field(default_factory=dict)
    notes: str = ""

    def to_prompt(self) -> str:
        if self.notes:
            return self.notes
        if not self.metrics:
            return "No technical priors"
        formatted = ", ".join(
            f"{name}={value:.2f}" for name, value in sorted(self.metrics.items())
        )
        return formatted

    def to_dict(self) -> Dict[str, object]:
        return {
            "metrics": {str(k): float(v) for k, v in self.metrics.items()},
            "notes": self.notes,
        }


@dataclass
class PairwiseMatch:
    """Single match-up inside the pairwise tournament."""

    round_index: int
    left_image: str
    right_image: str
    winner: str
    reason: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "round": self.round_index,
            "left": self.left_image,
            "right": self.right_image,
            "winner": self.winner,
            "reason": self.reason,
        }


@dataclass
class PairwiseReport:
    """Aggregated results of the tournament stage."""

    rounds: List[PairwiseMatch] = field(default_factory=list)
    final_rankings: List[Dict[str, object]] = field(default_factory=list)
    stability_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "rounds": [match.to_dict() for match in self.rounds],
            "final_rankings": list(self.final_rankings),
            "stability_metrics": dict(self.stability_metrics),
        }

