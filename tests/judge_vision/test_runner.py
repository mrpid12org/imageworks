from __future__ import annotations

import json
from pathlib import Path

from imageworks.apps.judge_vision import RubricScores
from imageworks.apps.judge_vision.config import JudgeVisionConfig
from imageworks.apps.judge_vision.models import JudgeVisionRecord
from imageworks.apps.judge_vision.runner import JudgeVisionRunner
from imageworks.apps.judge_vision.judge_types import TechnicalSignals


def test_runner_writes_outputs_and_progress(monkeypatch, tmp_path):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    image_path = input_dir / "sample.jpg"
    image_path.write_bytes(b"\x00\x01")

    output_jsonl = tmp_path / "results.jsonl"
    summary_path = tmp_path / "summary.md"
    progress_path = tmp_path / "progress.json"
    iqa_cache = tmp_path / "iqa.jsonl"

    config = JudgeVisionConfig(
        input_paths=[input_dir],
        recursive=False,
        image_extensions=(".jpg", ".jpeg", ".png"),
        backend="stub-backend",
        base_url="http://localhost:8000",
        api_key="",
        timeout=30,
        max_new_tokens=64,
        temperature=0.2,
        top_p=0.9,
        model="stub-model",
        use_registry=False,
        critique_role=None,
        skip_preflight=True,
        dry_run=True,
        competition_id=None,
        competition_config=None,
        pairwise_rounds=None,
        pairwise_enabled=False,
        pairwise_threshold=17,
        critique_title_template="{stem}",
        critique_category=None,
        critique_notes="",
        output_jsonl=output_jsonl,
        summary_path=summary_path,
        progress_path=progress_path,
        iqa_cache_path=iqa_cache,
        enable_musiq=False,
        enable_nima=False,
        stage="full",
        iqa_device="cpu",
    )

    stub_record = JudgeVisionRecord(
        image=image_path,
        critique="Energetic framing with confident lighting.",
        critique_total=18.0,
        critique_award="Gold",
        critique_subscores=RubricScores(
            impact=4.5,
            composition=4.4,
            technical=4.3,
            category_fit=4.0,
        ),
        backend="stub-backend",
        model="stub-model",
    )

    class StubEngine:
        def __init__(self, config, competition=None, precomputed_signals=None):
            self.config = config

        def process(self, image_path: Path) -> JudgeVisionRecord:
            assert image_path == stub_record.image
            return stub_record

        def close(self) -> None:
            pass

    monkeypatch.setattr(
        "imageworks.apps.judge_vision.runner.JudgeVisionInferenceEngine",
        StubEngine,
    )

    runner = JudgeVisionRunner(config)
    records = runner.run()

    assert len(records) == 1
    assert records[0].critique_award == "Gold"

    assert output_jsonl.exists()
    payloads = [
        json.loads(line)
        for line in output_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert payloads[0]["critique_total"] == 18.0

    summary_text = summary_path.read_text(encoding="utf-8")
    assert "## Colour Entries" in summary_text
    assert "### sample.jpg" in summary_text
    assert "- Image title: n/a" in summary_text
    assert "- Score: 18.0" in summary_text

    progress_data = json.loads(progress_path.read_text(encoding="utf-8"))
    assert progress_data["total"] == 1
    assert progress_data["processed"] == 1
    assert progress_data["current_image"] == str(image_path)

    cache_lines = [
        json.loads(line)
        for line in iqa_cache.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(cache_lines) == 1
    assert cache_lines[0]["image"] == str(image_path)


def test_runner_iqa_stage_only(monkeypatch, tmp_path):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    image_path = input_dir / "sample.jpg"
    image_path.write_bytes(b"\x00\x01")

    output_jsonl = tmp_path / "results.jsonl"
    summary_path = tmp_path / "summary.md"
    progress_path = tmp_path / "progress.json"
    iqa_cache = tmp_path / "iqa.jsonl"

    config = JudgeVisionConfig(
        input_paths=[input_dir],
        recursive=False,
        image_extensions=(".jpg",),
        backend="stub-backend",
        base_url="http://localhost:8000",
        api_key="",
        timeout=30,
        max_new_tokens=64,
        temperature=0.2,
        top_p=0.9,
        model="stub-model",
        use_registry=False,
        critique_role=None,
        skip_preflight=True,
        dry_run=True,
        competition_id=None,
        competition_config=None,
        pairwise_rounds=None,
        pairwise_enabled=False,
        pairwise_threshold=17,
        critique_title_template="{stem}",
        critique_category=None,
        critique_notes="",
        output_jsonl=output_jsonl,
        summary_path=summary_path,
        progress_path=progress_path,
        enable_musiq=False,
        enable_nima=False,
        iqa_cache_path=iqa_cache,
        stage="iqa",
        iqa_device="cpu",
    )

    class StubExtractor:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, path):
            assert path == image_path
            return TechnicalSignals(metrics={"mean_luma": 0.5}, notes="nominal")

    monkeypatch.setattr(
        "imageworks.apps.judge_vision.runner.TechnicalSignalExtractor",
        StubExtractor,
    )

    runner = JudgeVisionRunner(config)
    records = runner.run()

    assert records == []
    cache_lines = [
        json.loads(line)
        for line in iqa_cache.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert cache_lines[0]["technical_signals"]["metrics"]["mean_luma"] == 0.5


def test_pairwise_plans_split_by_category(tmp_path):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()

    output_jsonl = tmp_path / "results.jsonl"
    summary_path = tmp_path / "summary.md"
    progress_path = tmp_path / "progress.json"
    iqa_cache = tmp_path / "iqa.jsonl"

    config = JudgeVisionConfig(
        input_paths=[input_dir],
        recursive=False,
        image_extensions=(".jpg",),
        backend="stub-backend",
        base_url="http://localhost:8000",
        api_key="",
        timeout=30,
        max_new_tokens=64,
        temperature=0.2,
        top_p=0.9,
        model="stub-model",
        use_registry=False,
        critique_role=None,
        skip_preflight=True,
        dry_run=True,
        competition_id=None,
        competition_config=None,
        pairwise_rounds=None,
        pairwise_enabled=True,
        pairwise_threshold=17,
        critique_title_template="{stem}",
        critique_category=None,
        critique_notes="",
        output_jsonl=output_jsonl,
        summary_path=summary_path,
        progress_path=progress_path,
        enable_musiq=False,
        enable_nima=False,
        iqa_cache_path=iqa_cache,
        stage="full",
        iqa_device="cpu",
    )

    def _record(name: str, category: str, total: float) -> JudgeVisionRecord:
        path = tmp_path / f"{name}.jpg"
        path.write_bytes(b"\x00")
        return JudgeVisionRecord(
            image=path,
            competition_category=category,
            critique_total=total,
        )

    records = [
        _record("colour_a", "Colour", 18.0),
        _record("colour_b", "Colour", 19.0),
        _record("colour_c", "Colour", 17.5),
        _record("colour_low", "Colour", 15.0),
        _record("mono_a", "Mono", 18.0),
        _record("mono_b", "Mono", 17.2),
        _record("mono_low", "Mono", 16.5),
    ]

    runner = JudgeVisionRunner(config)
    plans = runner._build_pairwise_plans(records)
    assert len(plans) == 2
    plan_map = {plan.category: plan for plan in plans}
    assert plan_map["Colour"].eligible_count == 3
    assert plan_map["Mono"].eligible_count == 2
    assert plan_map["Colour"].comparisons == 3  # recommended for <=10 entrants
    assert plan_map["Mono"].comparisons == 3
