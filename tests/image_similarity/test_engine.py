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
from imageworks.model_loader.models import SelectedModel


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


def test_engine_loader_resolution(monkeypatch, tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.jpg"
    candidate.touch()
    library = tmp_path / "library"
    library.mkdir()

    settings = SimilaritySettings(
        default_candidates=(candidate,),
        default_library_root=library,
        default_output_jsonl=tmp_path / "results.jsonl",
        default_summary_path=tmp_path / "summary.md",
        default_cache_dir=tmp_path / "cache",
        default_strategies=("perceptual_hash",),
        default_write_metadata=False,
        default_generate_explanations=False,
    )

    config = build_runtime_config(
        settings=settings,
        candidates=[candidate],
        library_root=library,
        strategies=["perceptual_hash"],
        use_loader=True,
        registry_model="similarity-vlm",
        registry_capabilities=["vision", "embedding"],
    )

    captured: dict[str, object] = {}

    def fake_select_model(name: str, require_capabilities=None):  # type: ignore[override]
        captured["name"] = name
        captured["capabilities"] = tuple(require_capabilities or [])
        return SelectedModel(
            logical_name=name,
            endpoint_url="http://example:8000/v1",
            internal_model_id="resolved-model",
            backend="lmdeploy",
            capabilities={"vision": True, "embedding": True},
        )

    monkeypatch.setattr(
        "imageworks.apps.image_similarity_checker.core.engine.select_model",
        fake_select_model,
    )

    engine = SimilarityEngine(config, strategies=[])
    engine.close()

    assert captured["name"] == "similarity-vlm"
    assert captured["capabilities"] == ("vision", "embedding")
    assert engine.config.model == "resolved-model"
    assert engine.config.base_url == "http://example:8000/v1"
    assert engine.config.backend == "lmdeploy"
