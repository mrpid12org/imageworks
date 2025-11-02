from pathlib import Path

from imageworks.apps.personal_tagger.core import build_runtime_config, load_config


def test_build_runtime_config_defaults(tmp_path):
    sample_dir = tmp_path / "images"
    sample_dir.mkdir()

    settings = load_config(Path.cwd())
    config = build_runtime_config(
        settings=settings,
        input_dirs=[sample_dir],
        output_jsonl=tmp_path / "results.jsonl",
        summary_path=tmp_path / "summary.md",
    )

    assert config.input_paths == (sample_dir,)
    assert config.output_jsonl == tmp_path / "results.jsonl"
    assert config.summary_path == tmp_path / "summary.md"
    assert config.backend == settings.default_backend
    if settings.default_use_registry:
        assert config.caption_model
        assert config.keyword_model
        assert config.description_model
    else:
        assert config.caption_model == settings.caption_model
        assert config.keyword_model == settings.keyword_model
        assert config.description_model == settings.description_model
    assert config.max_keywords == settings.max_keywords
    assert config.api_key == settings.default_api_key
    assert config.critique_title_template == settings.critique_title_template
    assert config.critique_category == settings.critique_default_category
    assert config.critique_notes == settings.critique_default_notes


def test_build_runtime_config_critique_overrides(tmp_path):
    sample_dir = tmp_path / "images"
    sample_dir.mkdir()

    settings = load_config(Path.cwd())
    config = build_runtime_config(
        settings=settings,
        input_dirs=[sample_dir],
        output_jsonl=tmp_path / "results.jsonl",
        summary_path=tmp_path / "summary.md",
        critique_title_template="{caption}",
        critique_category="Open",
        critique_notes="Theme: motion",
    )

    assert config.critique_title_template == "{caption}"
    assert config.critique_category == "Open"
    assert config.critique_notes == "Theme: motion"
