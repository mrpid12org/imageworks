from __future__ import annotations

import json
from pathlib import Path
from typing import List

from PIL import Image

from imageworks.apps.personal_tagger.core import (
    PersonalTaggerRunner,
    build_runtime_config,
    load_config,
)
from imageworks.apps.personal_tagger.core.inference import BaseInferenceEngine
from imageworks.apps.personal_tagger.core.metadata_writer import PersonalMetadataWriter
from imageworks.apps.personal_tagger.core.models import (
    GenerationModels,
    KeywordPrediction,
    PersonalTaggerRecord,
)


class FakeInferenceEngine(BaseInferenceEngine):
    def __init__(self, config):
        super().__init__(config)
        self._models = GenerationModels(
            caption="fake/caption",
            keywords="fake/keywords",
            description="fake/description",
        )

    @property
    def models_used(self) -> GenerationModels:  # pragma: no cover - not used directly
        return self._models

    def process(self, image_path: Path) -> PersonalTaggerRecord:
        return PersonalTaggerRecord(
            image=image_path,
            keywords=[
                KeywordPrediction(keyword="sunset", score=0.9),
                KeywordPrediction(keyword="ocean", score=0.82),
            ],
            caption="A calm ocean horizon at sunset.",
            description="A warm orange glow falls across the calm ocean while a silhouetted shoreline frames the scene.",
            duration_seconds=0.01,
            backend="test-backend",
            models=self._models,
        )


class TrackingMetadataWriter(PersonalMetadataWriter):
    def __init__(self):
        super().__init__(backup_originals=False, overwrite_existing=True)
        self.called_with: List[Path] = []

    def write(self, image_path: Path, record: PersonalTaggerRecord) -> bool:  # type: ignore[override]
        self.called_with.append(image_path)
        return True


def create_test_image(path: Path) -> None:
    image = Image.new("RGB", (64, 64), color=(255, 128, 0))
    image.save(path, format="JPEG")


def test_runner_produces_outputs(tmp_path):
    image_dir = tmp_path / "library"
    image_dir.mkdir()
    image_path = image_dir / "sample.jpg"
    create_test_image(image_path)

    settings = load_config(Path.cwd())
    config = build_runtime_config(
        settings=settings,
        input_dirs=[image_dir],
        output_jsonl=tmp_path / "results.jsonl",
        summary_path=tmp_path / "summary.md",
        dry_run=True,
        preflight=False,  # disable network preflight for isolated test
    )

    inference = FakeInferenceEngine(config)
    metadata_writer = TrackingMetadataWriter()
    runner = PersonalTaggerRunner(
        config,
        inference_engine=inference,
        metadata_writer=metadata_writer,
    )

    records = runner.run()

    assert len(records) == 1
    record = records[0]
    assert record.caption
    assert record.description
    assert record.metadata_written is False
    assert metadata_writer.called_with == []  # dry-run skips metadata writes

    json_path = config.output_jsonl
    summary_path = config.summary_path

    assert json_path.exists()
    assert summary_path.exists()

    entries = [
        json.loads(line)
        for line in json_path.read_text(encoding="utf-8").strip().splitlines()
    ]
    assert entries[0]["caption"] == record.caption
    assert entries[0]["keywords"][0]["keyword"]
    assert "dry-run" in records[0].notes

    summary_text = summary_path.read_text(encoding="utf-8")
    assert "Personal Tagger Summary" in summary_text
    assert "sample.jpg" in summary_text
