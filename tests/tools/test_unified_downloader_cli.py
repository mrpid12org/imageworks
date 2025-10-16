import json

from typer.testing import CliRunner

from imageworks.tools.model_downloader.cli import app as downloader_app

runner = CliRunner()


def test_list_models_json_smoke(isolated_configs_dir, monkeypatch):
    monkeypatch.chdir(isolated_configs_dir.parent)
    result = runner.invoke(downloader_app, ["list", "--json"])
    assert result.exit_code == 0, result.stdout
    data = json.loads(result.stdout or "[]")
    assert isinstance(data, list)


def test_remove_nonexistent_variant(isolated_configs_dir, monkeypatch):
    monkeypatch.chdir(isolated_configs_dir.parent)
    result = runner.invoke(
        downloader_app, ["remove", "nonexistent-variant-name-xyz", "--force"]
    )
    assert result.exit_code != 0
    assert "Variant not found" in (result.stdout or "")


def test_verify_no_entries(isolated_configs_dir, monkeypatch):
    monkeypatch.chdir(isolated_configs_dir.parent)
    result = runner.invoke(downloader_app, ["verify"])
    assert result.exit_code == 0, result.stdout
    assert "Verifying" in (result.stdout or "")
