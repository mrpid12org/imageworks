"""Judge Vision domain modules shared across GUI and CLI entry points."""

from .competition import (
    CompetitionConfig,
    CompetitionRegistry,
    CompetitionRules,
    ScoreBands,
    load_competition_registry,
)
from .compliance import evaluate_compliance
from .judge_types import (
    ComplianceReport,
    PairwiseMatch,
    PairwiseReport,
    RubricScores,
    TechnicalSignals,
)
from .pairwise import JudgeVisionEntry, run_pairwise_tournament
from .technical_signals import TechnicalSignalExtractor
from .runner import JudgeVisionRunner
from .models import JudgeVisionRecord

__all__ = [
    "CompetitionConfig",
    "CompetitionRegistry",
    "CompetitionRules",
    "ScoreBands",
    "load_competition_registry",
    "evaluate_compliance",
    "ComplianceReport",
    "PairwiseMatch",
    "PairwiseReport",
    "RubricScores",
    "TechnicalSignals",
    "JudgeVisionEntry",
    "run_pairwise_tournament",
    "TechnicalSignalExtractor",
    "JudgeVisionRunner",
    "JudgeVisionRecord",
]
