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
from imageworks.apps.personal_tagger.core.config import PersonalTaggerConfig
from imageworks.apps.personal_tagger.core.inference import (
    BaseInferenceEngine,
    ChatResult,
    OpenAIInferenceEngine,
)
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
            critique="fake/critique",
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
            critique="Judge notes the serene mood but suggests adding foreground interest for impact.",
            critique_score=17,
            critique_title="Sunset Serenity",
            critique_category="Open",
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
    assert record.critique
    assert record.critique_score == 17
    assert record.critique_title == "Sunset Serenity"
    assert record.critique_category == "Open"
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
    assert entries[0]["critique"] == record.critique
    assert entries[0]["critique_score"] == 17
    assert entries[0]["critique_title"] == "Sunset Serenity"
    assert entries[0]["critique_category"] == "Open"
    assert entries[0]["keywords"][0]["keyword"]
    assert "dry-run" in records[0].notes

    summary_text = summary_path.read_text(encoding="utf-8")
    assert "Personal Tagger Summary" in summary_text
    assert "sample.jpg" in summary_text
    assert "Critique score: 17/20" in summary_text
    assert "Critique title: Sunset Serenity" in summary_text


class StubCritiqueEngine(OpenAIInferenceEngine):
    def __init__(self, config, response_text: str) -> None:
        super().__init__(config)
        self._response_text = response_text
        self._resolved_critique_model = config.description_model

    def _chat(self, *, model, messages, max_tokens):  # type: ignore[override]
        return ChatResult(content=self._response_text, raw_response={})


def _build_test_config(tmp_path: Path) -> PersonalTaggerConfig:
    images_dir = tmp_path / "critique"
    images_dir.mkdir()
    settings = load_config(Path.cwd())
    return build_runtime_config(
        settings=settings,
        input_dirs=[images_dir],
        output_jsonl=tmp_path / "out.jsonl",
        summary_path=tmp_path / "out.md",
        prompt_profile="club_judge_json",
    )


def test_critique_json_clamps_score(tmp_path):
    config = _build_test_config(tmp_path)
    payload = """```json\n{\n  \"title\": \"Moorland\",\n  \"category\": \"Open\",\n  \"critique\": \"Strong mood with minor distractions.\",\n  \"score\": \"24\"\n}\n```"""
    engine = StubCritiqueEngine(config, payload)
    critique, score, title, category, error = engine._run_critique_stage(
        image_b64="dGVzdA==",
        image_path=config.input_paths[0] / "example.jpg",
        caption="A windswept ridge",
        keywords=["ridge", "storm"],
    )

    assert error is None
    assert score == 20
    assert title == "Moorland"
    assert category == "Open"
    assert "Strong mood" in critique


def test_critique_json_failure_returns_text(tmp_path):
    config = _build_test_config(tmp_path)
    engine = StubCritiqueEngine(config, "not-json response")
    critique, score, title, category, error = engine._run_critique_stage(
        image_b64="dGVzdA==",
        image_path=config.input_paths[0] / "example.jpg",
        caption="Evening pier",
        keywords=["pier"],
    )

    assert score is None
    assert title is None
    assert category is None
    assert error == "critique_json_error"
    assert critique == "not-json response"
