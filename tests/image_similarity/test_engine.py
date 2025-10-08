from dataclasses import replace
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("PIL")
from PIL import Image

from imageworks.apps.image_similarity_checker.core.config import (
    SimilaritySettings,
    build_runtime_config,
)
from imageworks.apps.image_similarity_checker.core.engine import SimilarityEngine
from imageworks.apps.image_similarity_checker.core.models import SimilarityVerdict


def _make_image(path: Path, colour: tuple[int, int, int]) -> None:
    Image.new("RGB", (32, 32), colour).save(path)


def test_engine_perceptual_hash(tmp_path: Path) -> None:
    candidate_dir = tmp_path / "candidates"
    candidate_dir.mkdir()
    library_dir = tmp_path / "library"
    library_dir.mkdir()

    candidate = candidate_dir / "candidate.jpg"
    duplicate = library_dir / "duplicate.jpg"
    different = library_dir / "different.jpg"

    _make_image(candidate, (200, 50, 50))
    _make_image(duplicate, (200, 50, 50))
    _make_image(different, (20, 200, 200))

    settings = SimilaritySettings(
        default_candidates=(candidate,),
        default_library_root=library_dir,
        default_output_jsonl=tmp_path / "results.jsonl",
        default_summary_path=tmp_path / "summary.md",
        default_cache_dir=tmp_path / "cache",
        default_strategies=("perceptual_hash",),
        default_fail_threshold=0.9,
        default_query_threshold=0.8,
        default_write_metadata=False,
        default_generate_explanations=False,
    )

    config = build_runtime_config(
        settings=settings,
        candidates=[candidate],
        library_root=library_dir,
        strategies=["perceptual_hash"],
        fail_threshold=0.9,
        query_threshold=0.8,
        write_metadata=False,
        generate_explanations=False,
    )

    engine = SimilarityEngine(config)
    try:
        results = engine.run()
    finally:
        engine.close()

    assert len(results) == 1
    result = results[0]
    assert result.verdict == SimilarityVerdict.FAIL
    assert result.matches
    assert result.matches[0].reference == duplicate

    # Dry-run mode returns placeholder data
    dry_config = replace(config, dry_run=True)
    engine = SimilarityEngine(dry_config)
    try:
        dry_results = engine.run()
    finally:
        engine.close()

    assert dry_results[0].verdict == SimilarityVerdict.PASS
    assert not dry_results[0].matches
