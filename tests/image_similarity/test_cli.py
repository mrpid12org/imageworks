from pathlib import Path

import pytest

typer_testing = pytest.importorskip("typer.testing")
pytest.importorskip("PIL")
from PIL import Image

from imageworks.apps.image_similarity_checker.cli.main import app

CliRunner = typer_testing.CliRunner

runner = CliRunner()


def _make_image(path: Path, colour: tuple[int, int, int]) -> None:
    Image.new("RGB", (32, 32), colour).save(path)


def test_cli_runs_similarity(tmp_path: Path) -> None:
    candidate_dir = tmp_path / "candidates"
    library_dir = tmp_path / "library"
    candidate_dir.mkdir()
    library_dir.mkdir()

    candidate = candidate_dir / "candidate.jpg"
    duplicate = library_dir / "duplicate.jpg"
    _make_image(candidate, (100, 120, 130))
    _make_image(duplicate, (100, 120, 130))

    jsonl_path = tmp_path / "results.jsonl"
    md_path = tmp_path / "summary.md"

    result = runner.invoke(
        app,
        [
            "check",
            str(candidate),
            "--library-root",
            str(library_dir),
            "--strategy",
            "perceptual_hash",
            "--output-jsonl",
            str(jsonl_path),
            "--summary",
            str(md_path),
            "--fail-threshold",
            "0.9",
            "--query-threshold",
            "0.8",
            "--no-write-metadata",
            "--no-explain",
        ],
    )

    assert result.exit_code == 0, result.output
    assert jsonl_path.exists()
    assert md_path.exists()
    content = jsonl_path.read_text(encoding="utf-8")
    assert "fail" in content.lower()
