"""Report generation for the image similarity checker."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from .models import CandidateSimilarity


def write_jsonl(results: Sequence[CandidateSimilarity], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result.to_json(), ensure_ascii=False))
            handle.write("\n")


def write_markdown(results: Sequence[CandidateSimilarity], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = ["# Image Similarity Checker Report", ""]
    if not results:
        lines.append("No candidate images were processed.")
    else:
        lines.extend(
            [
                "| Candidate | Verdict | Top score | Best match | Strategies |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for result in results:
            best = result.best_match()
            best_path = best.reference.name if best else "—"
            strategies = ", ".join(sorted(best.extra.get("strategies", {}).keys())) if best else "—"
            lines.append(
                "| {candidate} | {verdict} | {score:.3f} | {best_match} | {strategies} |".format(
                    candidate=result.candidate.name,
                    verdict=result.verdict.value.upper(),
                    score=result.top_score,
                    best_match=best_path,
                    strategies=strategies or "—",
                )
            )
        lines.append("")

        for result in results:
            lines.append(f"## {result.candidate.name}")
            lines.append("")
            lines.append(f"- Verdict: **{result.verdict.value.upper()}**")
            lines.append(f"- Top score: {result.top_score:.3f} ({result.metric})")
            if result.matches:
                lines.append("- Matches:")
                for match in result.matches:
                    strategies = match.extra.get("strategies", {})
                    strategy_summary = ", ".join(
                        f"{name}={score:.3f}" for name, score in strategies.items()
                    )
                    lines.append(
                        f"  - {match.reference}: {match.score:.3f} [{strategy_summary or match.strategy}]"
                    )
            if result.notes:
                lines.append("- Notes:")
                for note in result.notes:
                    lines.append(f"  - {note}")
            lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
