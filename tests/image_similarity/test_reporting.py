import json
from pathlib import Path

from imageworks.apps.image_similarity_checker.core.models import (
    CandidateSimilarity,
    SimilarityVerdict,
    StrategyMatch,
)
from imageworks.apps.image_similarity_checker.core.reporting import (
    write_jsonl,
    write_markdown,
)


def _sample_result(tmp_path: Path) -> CandidateSimilarity:
    candidate = tmp_path / "candidate.jpg"
    reference = tmp_path / "match.jpg"
    match = StrategyMatch(
        candidate=candidate,
        reference=reference,
        score=0.95,
        strategy="perceptual_hash",
        reason="perceptual hash similarity 0.950",
        extra={"strategies": {"perceptual_hash": 0.95}},
    )
    result = CandidateSimilarity(
        candidate=candidate,
        verdict=SimilarityVerdict.FAIL,
        top_score=0.95,
        matches=[match],
        thresholds={"fail": 0.9, "query": 0.8},
        metric="cosine",
        notes=["Similarity score exceeds fail threshold; duplicate likely"],
    )
    return result


def test_write_jsonl_and_markdown(tmp_path: Path) -> None:
    result = _sample_result(tmp_path)
    jsonl_path = tmp_path / "out.jsonl"
    md_path = tmp_path / "summary.md"

    write_jsonl([result], jsonl_path)
    write_markdown([result], md_path)

    data = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(data) == 1
    record = json.loads(data[0])
    assert record["verdict"] == "fail"
    assert record["matches"][0]["reference"].endswith("match.jpg")

    markdown = md_path.read_text(encoding="utf-8")
    assert "Image Similarity Checker Report" in markdown
    assert "candidate.jpg" in markdown
    assert "FAIL" in markdown
