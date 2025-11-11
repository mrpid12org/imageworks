"""Competition configuration support for Judge Vision."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore


@dataclass(frozen=True)
class CompetitionRules:
    """Basic compliance rules applied during Stage 0."""

    max_width: Optional[int] = None
    max_height: Optional[int] = None
    min_width: Optional[int] = None
    min_height: Optional[int] = None
    borders: Optional[str] = None
    manipulation: Optional[str] = None
    watermark_allowed: bool = False
    colour_space: Optional[str] = None
    competition_class: Optional[str] = None
    entries_per_round: Optional[str] = None
    extra: Dict[str, str] = field(default_factory=dict)

    def describe(self) -> str:
        parts: List[str] = []
        if self.max_width and self.max_height:
            parts.append(f"max {self.max_width}x{self.max_height} pixels")
        elif self.max_width:
            parts.append(f"max width {self.max_width}")
        elif self.max_height:
            parts.append(f"max height {self.max_height}")
        if self.min_width and self.min_height:
            parts.append(f"min {self.min_width}x{self.min_height}")
        if self.borders:
            parts.append(f"borders: {self.borders}")
        if self.manipulation:
            parts.append(f"manipulation: {self.manipulation}")
        if self.colour_space:
            parts.append(f"colour space: {self.colour_space}")
        if self.competition_class:
            parts.append(f"class: {self.competition_class}")
        if self.entries_per_round:
            parts.append(f"entries/round: {self.entries_per_round}")
        if self.watermark_allowed:
            parts.append("watermarks permitted")
        else:
            parts.append("watermarks disallowed")
        return "; ".join(parts)


@dataclass(frozen=True)
class ScoreBands:
    """Mapping of award labels to eligible score bands."""

    bands: Dict[str, List[int]] = field(default_factory=dict)

    def award_for(self, score: Optional[float]) -> Optional[str]:
        if score is None:
            return None
        try:
            numeric = float(score)
        except (TypeError, ValueError):  # noqa: BLE001
            return None
        rounded = int(round(numeric))
        for award, values in self.bands.items():
            if rounded in values:
                return award
        return None


@dataclass(frozen=True)
class CompetitionConfig:
    """Resolved competition entry within a registry file."""

    identifier: str
    categories: List[str]
    rules: CompetitionRules
    awards: List[str] = field(default_factory=list)
    score_bands: ScoreBands = field(default_factory=ScoreBands)
    pairwise_rounds: int = 0
    anchors: List[str] = field(default_factory=list)
    notes: str = ""
    display_name: Optional[str] = None

    def to_prompt_brief(self) -> str:
        return (
            f"Competition '{self.identifier}' categories: {', '.join(self.categories)}."
            f" Rules: {self.rules.describe()}"
        )


@dataclass(frozen=True)
class CompetitionRegistry:
    """Collection of competitions parsed from a TOML file."""

    competitions: Dict[str, CompetitionConfig]
    source: Path

    def get(self, identifier: Optional[str]) -> Optional[CompetitionConfig]:
        if identifier is None:
            return None
        return self.competitions.get(identifier)


def _normalise_list(value: Optional[Iterable[str]]) -> List[str]:
    if not value:
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _normalise_bands(value: object) -> ScoreBands:
    if not isinstance(value, dict):
        return ScoreBands()
    bands: Dict[str, List[int]] = {}
    for award, raw_scores in value.items():
        scores: List[int] = []
        if isinstance(raw_scores, Iterable) and not isinstance(
            raw_scores, (str, bytes)
        ):
            for entry in raw_scores:
                try:
                    scores.append(int(entry))
                except (TypeError, ValueError):  # noqa: BLE001
                    continue
        bands[str(award)] = sorted(set(scores))
    return ScoreBands(bands=bands)


def load_competition_registry(path: Path) -> CompetitionRegistry:
    """Load a TOML registry containing competition definitions."""

    if not path.exists():
        raise FileNotFoundError(path)
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    raw_competitions = data.get("competition", {})
    competitions: Dict[str, CompetitionConfig] = {}

    for identifier, payload in raw_competitions.items():
        if not isinstance(payload, dict):
            continue
        categories = _normalise_list(payload.get("categories")) or ["Open"]
        rules_data = payload.get("rules", {})
        if not isinstance(rules_data, dict):
            rules_data = {}
        extra_fields: Dict[str, str] = {}
        known_rule_keys = {
            "max_width",
            "max_height",
            "min_width",
            "min_height",
            "borders",
            "manipulation",
            "watermark_allowed",
            "colour_space",
            "competition_class",
            "entries_per_round",
        }
        extra_fields = {
            str(key): str(value)
            for key, value in rules_data.items()
            if key not in known_rule_keys
        }
        rules = CompetitionRules(
            max_width=_as_int(rules_data.get("max_width")),
            max_height=_as_int(rules_data.get("max_height")),
            min_width=_as_int(rules_data.get("min_width")),
            min_height=_as_int(rules_data.get("min_height")),
            borders=str(rules_data.get("borders", "")).strip() or None,
            manipulation=str(rules_data.get("manipulation", "")).strip() or None,
            watermark_allowed=bool(rules_data.get("watermark_allowed", False)),
            colour_space=str(rules_data.get("colour_space", "")).strip() or None,
            competition_class=str(rules_data.get("competition_class", "")).strip()
            or None,
            entries_per_round=str(rules_data.get("entries_per_round", "")).strip()
            or None,
            extra=extra_fields,
        )
        score_bands = _normalise_bands(payload.get("score_bands"))
        awards = _normalise_list(payload.get("awards"))
        pairwise_rounds = _as_int(payload.get("pairwise_rounds"), default=0)
        anchors = _normalise_list(payload.get("anchors"))
        notes = str(payload.get("notes", "")).strip()
        display_name = str(payload.get("name", "")).strip() or None

        competitions[str(identifier)] = CompetitionConfig(
            identifier=str(identifier),
            categories=categories,
            rules=rules,
            awards=awards,
            score_bands=score_bands,
            pairwise_rounds=pairwise_rounds,
            anchors=anchors,
            notes=notes,
            display_name=display_name,
        )

    return CompetitionRegistry(competitions=competitions, source=path)


def _as_int(value: object, *, default: Optional[int] = None) -> Optional[int]:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):  # noqa: BLE001
        return default
