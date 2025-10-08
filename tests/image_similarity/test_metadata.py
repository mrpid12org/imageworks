from pathlib import Path
from typing import List

import pytest

from imageworks.apps.image_similarity_checker.core.metadata import (
    SimilarityMetadataWriter,
)
from imageworks.apps.image_similarity_checker.core.models import (
    CandidateSimilarity,
    SimilarityVerdict,
    StrategyMatch,
)


def _result(tmp_path: Path) -> CandidateSimilarity:
    candidate = tmp_path / "candidate.jpg"
    reference = tmp_path / "match.jpg"
    match = StrategyMatch(
        candidate=candidate,
        reference=reference,
        score=0.92,
        strategy="perceptual_hash",
        extra={"strategies": {"perceptual_hash": 0.92}},
    )
    return CandidateSimilarity(
        candidate=candidate,
        verdict=SimilarityVerdict.QUERY,
        top_score=0.92,
        matches=[match],
        thresholds={"fail": 0.95, "query": 0.85},
        metric="cosine",
        notes=["Similarity score within query band; manual review recommended"],
    )


def test_metadata_writer_keywords(tmp_path: Path, monkeypatch) -> None:
    candidate = tmp_path / "candidate.jpg"
    candidate.write_bytes(b"fake")
    result = _result(tmp_path)

    writer = SimilarityMetadataWriter(backup_originals=False, overwrite_existing=False)

    commands: List[List[str]] = []

    monkeypatch.setattr("shutil.which", lambda name: str(tmp_path / "exiftool"))

    def fake_run(cmd, **kwargs):  # noqa: ANN001 - mimic subprocess.run signature
        commands.append(cmd)
        class Completed:
            returncode = 0
            stdout = b""
            stderr = b""

        return Completed()

    monkeypatch.setattr("subprocess.run", fake_run)

    wrote = writer.write(candidate, result)
    assert wrote is True
    assert commands
    # Ensure custom keywords are appended
    joined = " ".join(commands[0])
    assert "similarity:verdict=query" in joined
    assert "similarity:perceptual_hash=0.920" in joined


def test_metadata_writer_missing_exiftool(tmp_path: Path, monkeypatch) -> None:
    candidate = tmp_path / "candidate.jpg"
    candidate.write_bytes(b"fake")
    result = _result(tmp_path)

    writer = SimilarityMetadataWriter(backup_originals=False, overwrite_existing=False)

    monkeypatch.setattr("shutil.which", lambda name: None)

    with pytest.raises(RuntimeError):
        writer.write(candidate, result)
