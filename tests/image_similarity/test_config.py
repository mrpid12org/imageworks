from pathlib import Path

import pytest

from imageworks.apps.image_similarity_checker.core.config import (
    SimilarityConfig,
    SimilaritySettings,
    build_runtime_config,
)


def test_build_runtime_config_validates_thresholds():
    settings = SimilaritySettings(default_candidates=(Path("candidate.jpg"),))
    with pytest.raises(ValueError):
        build_runtime_config(
            settings=settings,
            candidates=[Path("candidate.jpg")],
            library_root=Path("library"),
            fail_threshold=0.5,
            query_threshold=0.6,
        )


def test_build_runtime_config_overrides(tmp_path: Path):
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
        fail_threshold=0.95,
        query_threshold=0.85,
        embedding_backend="simple",
        generate_explanations=True,
    )

    assert isinstance(config, SimilarityConfig)
    assert config.library_root == library
    assert config.fail_threshold == pytest.approx(0.95)
    assert config.query_threshold == pytest.approx(0.85)
    assert config.strategies == ("perceptual_hash",)
    assert config.embedding_backend == "simple"
    assert config.generate_explanations is True
    assert config.registry_capabilities == ("vision",)
