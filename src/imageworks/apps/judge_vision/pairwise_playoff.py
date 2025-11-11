"""LLM-driven pairwise playoff helpers for Judge Vision."""

from __future__ import annotations

import base64
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence

from imageworks.apps.judge_vision.inference import OpenAIChatClient
from imageworks.apps.judge_vision.models import JudgeVisionRecord


def recommended_pairwise_rounds(count: int) -> int:
    if count <= 10:
        return 3
    if count <= 20:
        return 4
    if count <= 35:
        return 5
    return 6


PAIRWISE_SYSTEM_PROMPT = (
    "You are the same experienced UK camera-club competition judge who already critiqued "
    "these entries individually.\n"
    "Two shortlisted images (Entry A and Entry B) are shown for a head-to-head playoff.\n"
    "Compare them directly for overall strength in this competition, weighing impact, design, "
    "technical execution, and presentation while grounding comments in the supplied technical "
    "analysis and critique summaries.\n"
    "Ignore any previous numeric scores – base your choice purely on the qualitative evidence.\n"
    "Return valid JSON only:\n"
    '{ "winner": "A" | "B", "reason": "<1-2 sentence justification>" }'
)


@dataclass
class _PairwiseRecord:
    record: JudgeVisionRecord
    wins: int = 0
    comparisons: int = 0

    @property
    def win_ratio(self) -> float:
        if self.comparisons <= 0:
            return 0.0
        return self.wins / self.comparisons


class PairwisePlayoffRunner:
    """Run balanced pairwise comparisons for Colour/Mono groups."""

    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        api_key: str,
        timeout: int,
        comparisons_per_image: int,
        threshold: int,
        seed: int | None = None,
    ) -> None:
        self.comparisons_per_image = max(0, comparisons_per_image)
        self.seed = seed if seed is not None else int(time.time())
        self.rng = random.Random(self.seed)
        self.threshold = threshold
        self.client = OpenAIChatClient(
            base_url=base_url,
            model_name=model_name,
            api_key=api_key,
            timeout=timeout,
        )

    def close(self) -> None:
        self.client.close()

    def apply(
        self,
        records: Sequence[JudgeVisionRecord],
        *,
        on_schedule: Callable[[int], None] | None = None,
        on_progress: (
            Callable[[int, int, str, JudgeVisionRecord, JudgeVisionRecord], None] | None
        ) = None,
    ) -> int:
        if self.comparisons_per_image <= 0:
            return 0

        groups: Dict[str, List[JudgeVisionRecord]] = {}
        for record in records:
            cat = (
                record.competition_category or "Uncategorised"
            ).strip() or "Uncategorised"
            groups.setdefault(cat, []).append(record)

        schedules: List[
            tuple[
                str,
                List[_PairwiseRecord],
                List[tuple[_PairwiseRecord, _PairwiseRecord]],
            ]
        ] = []
        total_matches = 0
        for category, group_records in groups.items():
            eligible = [
                rec
                for rec in group_records
                if (rec.critique_total or 0) >= self.threshold and rec.image.exists()
            ]
            if len(eligible) < 2:
                continue
            roster = [_PairwiseRecord(record=rec) for rec in eligible]
            matches = self._build_schedule(roster)
            if not matches:
                continue
            schedules.append((category, roster, matches))
            total_matches += len(matches)

        if total_matches == 0:
            return 0

        if on_schedule:
            on_schedule(total_matches)

        completed = 0
        for category, roster, matches in schedules:
            for entry_a, entry_b in matches:
                winner = self._run_match(entry_a.record, entry_b.record)
                entry_a.comparisons += 1
                entry_b.comparisons += 1
                if winner == "A":
                    entry_a.wins += 1
                elif winner == "B":
                    entry_b.wins += 1
                completed += 1
                if on_progress:
                    on_progress(
                        completed,
                        total_matches,
                        category,
                        entry_a.record,
                        entry_b.record,
                    )
            self._apply_adjustments(category, roster)
        return total_matches

    # ------------------------------------------------------------------
    def _build_schedule(
        self, entries: List[_PairwiseRecord]
    ) -> List[tuple[_PairwiseRecord, _PairwiseRecord]]:
        """Generate approximately balanced pairings."""

        used_pairs: set[tuple[Path, Path]] = set()
        matches: List[tuple[_PairwiseRecord, _PairwiseRecord]] = []

        def _pair_key(a: _PairwiseRecord, b: _PairwiseRecord) -> tuple[Path, Path]:
            left, right = sorted([a.record.image, b.record.image])
            return (left, right)

        for round_idx in range(self.comparisons_per_image):
            shuffled = entries[:]
            self.rng.shuffle(shuffled)
            while len(shuffled) >= 2:
                first = shuffled.pop()
                second = shuffled.pop()
                key = _pair_key(first, second)
                if first is second or key in used_pairs:
                    swapped = False
                    for idx, candidate in enumerate(shuffled):
                        alt_key = _pair_key(first, candidate)
                        if candidate is not first and alt_key not in used_pairs:
                            shuffled[idx] = second
                            second = candidate
                            key = alt_key
                            swapped = True
                            break
                    if not swapped:
                        continue
                used_pairs.add(key)
                matches.append((first, second))
        return matches

    def _run_match(
        self, record_a: JudgeVisionRecord, record_b: JudgeVisionRecord
    ) -> str:
        messages = [
            {"role": "system", "content": PAIRWISE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": self._render_pairwise_prompt(record_a, record_b),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{self._encode_image(record_a.image)}"
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{self._encode_image(record_b.image)}"
                        },
                    },
                ],
            },
        ]
        response = self.client.chat(
            messages,
            max_tokens=128,
            temperature=0.2,
            top_p=0.9,
        )
        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        try:
            payload = json.loads(content)
        except Exception:
            return "A"
        winner = (payload.get("winner") or "A").strip().upper()
        return "A" if winner == "A" else "B"

    def _render_pairwise_prompt(
        self, entry_a: JudgeVisionRecord, entry_b: JudgeVisionRecord
    ) -> str:
        def _summarise_text(text: str | None, limit: int = 280) -> str:
            if not text:
                return "n/a"
            cleaned = " ".join(text.strip().split())
            if len(cleaned) <= limit:
                return cleaned
            return cleaned[: limit - 1].rstrip() + "…"

        def _summarise(entry: JudgeVisionRecord, label: str) -> str:
            critique = _summarise_text(entry.critique)
            analysis = entry.technical_signals.technical_analysis_block()
            title = entry.image_title or entry.critique_title or entry.image.name
            summary = [
                f"{label}:",
                f"- Title: {title}",
                f"- Category: {entry.competition_category or 'n/a'}",
                f"- Style: {entry.style_inference or 'n/a'}",
                f"- Critique summary: {critique or 'No prior notes'}",
                analysis,
                "",
            ]
            return "\n".join(summary)

        instruction = (
            "Compare these two shortlisted images directly and pick the stronger one overall.\n"
            "Explain the deciding factor in 1–2 sentences when you return the JSON response."
        )
        return "\n".join(
            [
                instruction,
                _summarise(entry_a, "Entry A"),
                _summarise(entry_b, "Entry B"),
                'Respond with JSON: {"winner": "A" | "B", "reason": "<1-2 sentence justification>"}',
            ]
        )

    @staticmethod
    def _encode_image(path: Path) -> str:
        data = path.read_bytes()
        return base64.b64encode(data).decode("utf-8")

    def _apply_adjustments(self, category: str, stats: List[_PairwiseRecord]) -> None:
        stats.sort(key=lambda item: item.win_ratio, reverse=True)
        total = len(stats)
        if total <= 2:
            for entry in stats:
                record = entry.record
                record.pairwise_score_initial = record.critique_total
                record.pairwise_wins = entry.wins
                record.pairwise_comparisons = entry.comparisons
                record.pairwise_win_ratio = round(entry.win_ratio, 4)
            return

        k = max(1, round(total * 0.15))
        winners = stats[:k]
        losers = stats[-k:]
        middle = stats[k:-k] if total > 2 * k else []

        for entry in stats:
            record = entry.record
            record.pairwise_score_initial = record.critique_total
            record.pairwise_wins = entry.wins
            record.pairwise_comparisons = entry.comparisons
            record.pairwise_win_ratio = round(entry.win_ratio, 4)

        def _as_int(value: float | None) -> int:
            try:
                return int(round(float(value)))
            except (TypeError, ValueError):
                return 0

        for entry in winners:
            record = entry.record
            base_score = _as_int(record.pairwise_score_initial)
            record.critique_total = float(min(20, base_score + 1))

        for entry in losers:
            record = entry.record
            base_score = _as_int(record.pairwise_score_initial)
            record.critique_total = float(max(16, base_score - 1))

        for entry in middle:
            record = entry.record
            record.critique_total = float(_as_int(record.pairwise_score_initial))

        # Ensure the top-ranked entry receives a 20
        top_entry = stats[0]
        top_entry.record.critique_total = 20.0
