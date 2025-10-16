import json
import pytest
from pathlib import Path

from typer.testing import CliRunner

from imageworks.tools.model_downloader.cli import app as downloader_app

runner = CliRunner()


def _load(snapshot: Path):
    return json.loads(snapshot.read_text(encoding="utf-8"))


def test_backfill_ollama_paths_dry_run_and_apply(isolated_configs_dir, monkeypatch):
    monkeypatch.chdir(isolated_configs_dir.parent)
    snapshot = isolated_configs_dir / "model_registry.json"
    data = _load(snapshot)
    logical_name = "synthetic-test-ollama-gguf"
    has_candidate = any(
        e for e in data if e.get("backend") == "ollama" and not e.get("download_path")
    )
    if not has_candidate:
        if not data:
            pytest.skip("registry snapshot empty")
        base = data[0].copy()
        base.update(
            {
                "name": logical_name,
                "backend": "ollama",
                "download_path": None,
                "download_format": None,
                "download_location": None,
                "served_model_id": logical_name.replace("-", ":"),
            }
        )
        data.append(base)
        snapshot.write_text(json.dumps(data, indent=2), encoding="utf-8")

    before = snapshot.read_text(encoding="utf-8")
    dry = runner.invoke(
        downloader_app,
        ["backfill-ollama-paths", "--dry-run", "--verbose"],
    )
    assert dry.exit_code == 0, dry.stdout
    assert snapshot.read_text(encoding="utf-8") == before

    result = runner.invoke(downloader_app, ["backfill-ollama-paths"])
    assert result.exit_code == 0, result.stdout
    updated = _load(snapshot)
    matches = [
        e
        for e in updated
        if e.get("backend") == "ollama" and e.get("name") == logical_name
    ]
    assert matches, "Synthetic logical Ollama entry missing after backfill"
    for entry in matches:
        assert entry.get("download_path"), "download_path should be populated"
        assert entry.get("download_format") == "gguf"
