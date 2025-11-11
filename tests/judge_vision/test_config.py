from imageworks.apps.judge_vision.config import (
    JudgeVisionSettings,
    build_runtime_config,
)


def test_build_runtime_config_uses_description_model(tmp_path):
    settings = JudgeVisionSettings()
    config = build_runtime_config(
        settings=settings,
        input_dirs=[tmp_path],
        output_jsonl=tmp_path / "judge.jsonl",
        summary_path=tmp_path / "judge.md",
        progress_path=tmp_path / "progress.json",
    )

    assert config.model == settings.description_model
    assert config.image_extensions == settings.image_extensions
    assert config.iqa_cache_path.name == "judge_vision_iqa.jsonl"
