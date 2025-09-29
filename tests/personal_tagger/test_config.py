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
    assert config.caption_model == settings.caption_model
    assert config.keyword_model == settings.keyword_model
    assert config.description_model == settings.description_model
    assert config.max_keywords == settings.max_keywords
    assert config.api_key == settings.default_api_key
