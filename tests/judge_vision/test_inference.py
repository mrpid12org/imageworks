from __future__ import annotations

import json
from pathlib import Path

from imageworks.apps.judge_vision.inference import JudgeVisionInferenceEngine


def test_infer_competition_category_detects_mono_and_colour():
    assert (
        JudgeVisionInferenceEngine._infer_competition_category(Path("01_Test.JPG"))
        == "Mono"
    )
    assert (
        JudgeVisionInferenceEngine._infer_competition_category(Path("holiday.png"))
        == "Colour"
    )


def test_read_image_title_prefers_known_fields(monkeypatch, tmp_path):
    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(b"\x00")

    def fake_run(*args, **kwargs):
        class Result:
            stdout = json.dumps([{"Title": "Evening Mist"}])
            stderr = ""

        return Result()

    monkeypatch.setattr(
        "imageworks.apps.judge_vision.inference.subprocess.run",
        fake_run,
    )

    assert JudgeVisionInferenceEngine._read_image_title(image_path) == "Evening Mist"


def test_read_image_title_handles_missing_exiftool(monkeypatch, tmp_path):
    def fake_run(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(
        "imageworks.apps.judge_vision.inference.subprocess.run",
        fake_run,
    )

    assert (
        JudgeVisionInferenceEngine._read_image_title(tmp_path / "missing.jpg") is None
    )
