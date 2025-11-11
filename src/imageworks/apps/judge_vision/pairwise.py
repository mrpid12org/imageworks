"""Swiss-style pairwise tournament helpers."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from .judge_types import PairwiseMatch, PairwiseReport, RubricScores


@dataclass
class JudgeVisionEntry:
    """Minimal view of a judged image for tournament orchestration."""

    image: str
    total: float
    rubric: RubricScores


def run_pairwise_tournament(
    entries: Sequence[JudgeVisionEntry], *, rounds: int, seed: int = 42
) -> PairwiseReport:
    if rounds <= 0 or len(entries) < 2:
        rankings = _rank_by_total(entries)
        return PairwiseReport(rounds=[], final_rankings=rankings, stability_metrics={})

    rng = random.Random(seed)
    working = list(entries)
    rounds_buffer: List[PairwiseMatch] = []

    for round_index in range(1, rounds + 1):
        rng.shuffle(working)
        pairs = _pairwise(working)
        for left, right in pairs:
            winner = _decide_winner(left, right)
            loser = right if winner is left else left
            reason = _render_reason(winner, loser)
            rounds_buffer.append(
                PairwiseMatch(
                    round_index=round_index,
                    left_image=left.image,
                    right_image=right.image,
                    winner=winner.image,
                    reason=reason,
                )
            )
        working = _rank_entries(working)

    rankings = _rank_by_total(working)
    stability = {"seed": float(seed), "rounds": float(rounds)}
    return PairwiseReport(
        rounds=rounds_buffer, final_rankings=rankings, stability_metrics=stability
    )


def _pairwise(
    entries: Iterable[JudgeVisionEntry],
) -> List[tuple[JudgeVisionEntry, JudgeVisionEntry]]:
    queue = list(entries)
    pairs: List[tuple[JudgeVisionEntry, JudgeVisionEntry]] = []
    while len(queue) >= 2:
        left = queue.pop()
        right = queue.pop()
        pairs.append((left, right))
    return pairs


def _decide_winner(left: JudgeVisionEntry, right: JudgeVisionEntry) -> JudgeVisionEntry:
    if left.total > right.total:
        return left
    if right.total > left.total:
        return right
    left_score = left.rubric.as_dict()
    right_score = right.rubric.as_dict()
    for key in ("impact", "composition", "technical", "category_fit"):
        if left_score.get(key, 0.0) > right_score.get(key, 0.0):
            return left
        if right_score.get(key, 0.0) > left_score.get(key, 0.0):
            return right
    return left if left.image < right.image else right


def _render_reason(winner: JudgeVisionEntry, loser: JudgeVisionEntry) -> str:
    winner_scores = winner.rubric.as_dict()
    loser_scores = loser.rubric.as_dict()
    advantages = []
    for key, label in (
        ("impact", "impact"),
        ("composition", "composition"),
        ("technical", "technical"),
        ("category_fit", "category fit"),
    ):
        if winner_scores.get(key, 0.0) > loser_scores.get(key, 0.0):
            advantages.append(label)
    if not advantages:
        return "Marginal edge on tie-break"
    joined = ", ".join(advantages[:2])
    return f"Stronger {joined}" if len(advantages) == 1 else f"Better {joined}"


def _rank_entries(entries: Sequence[JudgeVisionEntry]) -> List[JudgeVisionEntry]:
    return sorted(entries, key=lambda entry: (entry.total, entry.image), reverse=True)


def _rank_by_total(entries: Sequence[JudgeVisionEntry]) -> List[dict]:
    ordered = _rank_entries(entries)
    return [{"image": entry.image, "score": float(entry.total)} for entry in ordered]
