from __future__ import annotations

from pathlib import Path

from PIL import Image

from imageworks.apps.judge_vision import (
    CompetitionConfig,
    CompetitionRules,
    JudgeVisionEntry,
    RubricScores,
    ScoreBands,
    TechnicalSignalExtractor,
    evaluate_compliance,
    load_competition_registry,
    run_pairwise_tournament,
)


def _write_test_image(path: Path, color: int = 120) -> None:
    image = Image.new("RGB", (64, 64), color=(color, color, color))
    image.save(path)


def test_load_competition_registry(tmp_path: Path) -> None:
    config_toml = """
[competition.club_open]
categories = ["Open", "Nature"]
rules = { max_width = 1920, max_height = 1200, borders = "disallowed", watermark_allowed = false }
awards = ["Gold", "Silver", "Bronze"]
score_bands = { Gold = [19,20], Silver = [18], Bronze = [17] }
pairwise_rounds = 4
"""
    path = tmp_path / "competition.toml"
    path.write_text(config_toml, encoding="utf-8")

    registry = load_competition_registry(path)
    competition = registry.get("club_open")

    assert competition is not None
    assert competition.rules.max_width == 1920
    assert competition.score_bands.award_for(19) == "Gold"
    assert competition.pairwise_rounds == 4


def test_evaluate_compliance_flags_dimensions(tmp_path: Path) -> None:
    image_path = tmp_path / "test.png"
    _write_test_image(image_path, color=180)

    competition = CompetitionConfig(
        identifier="test",
        categories=["Open"],
        rules=CompetitionRules(max_width=32, max_height=32, watermark_allowed=False),
        awards=[],
        score_bands=ScoreBands(),
        pairwise_rounds=0,
        anchors=[],
    )

    report = evaluate_compliance(image_path, competition)
    assert not report.passed
    assert any("Width" in issue for issue in report.issues)
    assert "Watermarks disallowed" in report.warnings


def test_technical_signal_extractor_outputs_metrics(tmp_path: Path) -> None:
    image_path = tmp_path / "signals.png"
    _write_test_image(image_path, color=90)

    extractor = TechnicalSignalExtractor(enable_nima=False, enable_musiq=False)
    signals = extractor.run(image_path)

    assert "mean_luma" in signals.metrics
    assert "contrast" in signals.metrics
    assert signals.notes


def test_pairwise_tournament_orders_by_total() -> None:
    entries = [
        JudgeVisionEntry(
            image="A",
            total=19.0,
            rubric=RubricScores(
                impact=4.5, composition=4.2, technical=4.0, category_fit=4.0
            ),
        ),
        JudgeVisionEntry(
            image="B",
            total=17.0,
            rubric=RubricScores(
                impact=3.5, composition=3.8, technical=3.6, category_fit=3.2
            ),
        ),
        JudgeVisionEntry(
            image="C",
            total=18.5,
            rubric=RubricScores(
                impact=4.2, composition=4.0, technical=3.9, category_fit=3.8
            ),
        ),
    ]

    report = run_pairwise_tournament(entries, rounds=2, seed=7)
    assert len(report.final_rankings) == 3
    assert report.final_rankings[0]["image"] == "A"
    assert report.final_rankings[1]["image"] in {"C", "B"}
