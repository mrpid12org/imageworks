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

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, object]]) -> "RubricScores":
        payload = payload or {}
        return cls(
            impact=_clamp(payload.get("impact"), 0.0, 5.0),
            composition=_clamp(payload.get("composition"), 0.0, 5.0),
            technical=_clamp(payload.get("technical"), 0.0, 5.0),
            category_fit=_clamp(payload.get("category_fit"), 0.0, 5.0),
        )


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

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, object]]) -> "ComplianceReport|None":
        if not payload:
            return None
        return cls(
            passed=bool(payload.get("passed", True)),
            issues=list(payload.get("issues") or []),
            warnings=list(payload.get("warnings") or []),
            checks=dict(payload.get("checks") or {}),
        )


@dataclass
class TechnicalSignals:
    """Stage 1 technical signal outputs used to brief the VLM."""

    metrics: Dict[str, float] = field(default_factory=dict)
    notes: str = ""
    tonal_summary: Optional[str] = None

    def to_prompt(self) -> str:
        parts: List[str] = []
        if self.notes:
            parts.append(self.notes)
        if self.tonal_summary:
            parts.append(f"Tonal analysis: {self.tonal_summary}")
        if not self.metrics:
            if parts:
                return " | ".join(parts)
            return "No technical priors"
        formatted = ", ".join(
            f"{name}={value:.2f}" for name, value in sorted(self.metrics.items())
        )
        if parts:
            parts.append(formatted)
            return " | ".join(parts)
        return formatted

    def to_dict(self) -> Dict[str, object]:
        return {
            "metrics": {str(k): float(v) for k, v in self.metrics.items()},
            "notes": self.notes,
            "tonal_summary": self.tonal_summary,
        }

    def deterministic_summary(self) -> str:
        metrics = self.metrics or {}
        parts: List[str] = []
        if (mos := metrics.get("musiq_mos")) is not None:
            parts.append(f"MUSIQ MOS {mos:.1f}/100")
        if (nima_a := metrics.get("nima_aesthetic_mean")) is not None:
            parts.append(f"NIMA aesthetic {nima_a:.2f}/10")
        if (nima_t := metrics.get("nima_technical_mean")) is not None:
            parts.append(f"NIMA technical {nima_t:.2f}/10")
        if self.tonal_summary:
            parts.append(self.tonal_summary)
        return "; ".join(parts) if parts else "No deterministic scores recorded"

    def technical_analysis_block(self) -> str:
        """Render a TECHNICAL ANALYSIS block for VLM prompting."""

        def _fmt(value: Optional[float], scale: str) -> str:
            if value is None:
                return "n/a"
            if scale == "100":
                return f"{value:.2f} / 100"
            if scale == "10":
                return f"{value:.2f} / 10"
            return f"{value:.3f}"

        musiq = self.metrics.get("musiq_mos")
        nima_a = self.metrics.get("nima_aesthetic_mean")
        nima_t = self.metrics.get("nima_technical_mean")

        lines = [
            "TECHNICAL ANALYSIS:",
            f"- MUSIQ MOS: {_fmt(musiq, '100')} (higher = better)",
            f"- NIMA Aesthetic: {_fmt(nima_a, '10')}",
            f"- NIMA Technical: {_fmt(nima_t, '10')}",
        ]

        tonal_keys = [
            "mean_luminance",
            "dynamic_range",
            "clip_low_percent",
            "clip_high_percent",
            "local_contrast",
        ]
        tonal_lines: List[str] = []
        for key in tonal_keys:
            value = self.metrics.get(key)
            if value is None:
                continue
            tonal_lines.append(f"    • {key}: {value:.3f}")

        if tonal_lines:
            lines.append("- Tonal metrics:")
            lines.extend(tonal_lines)
        else:
            lines.append("- Tonal metrics: unavailable")

        summary = self.tonal_summary or "Not available."
        lines.append("- Tonal summary:")
        lines.append(f'    "{summary}"')
        return "\n".join(lines)

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, object]]) -> "TechnicalSignals":
        if not payload:
            return cls()
        metrics = payload.get("metrics") or {}
        return cls(
            metrics={
                str(k): float(v)
                for k, v in metrics.items()
                if isinstance(v, (int, float))
            },
            notes=str(payload.get("notes") or ""),
            tonal_summary=payload.get("tonal_summary"),
        )


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
