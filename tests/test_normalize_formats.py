from typer.testing import CliRunner

from imageworks.tools.model_downloader.cli import app as downloader_app

runner = CliRunner()


def test_normalize_formats_dry_run(isolated_configs_dir, monkeypatch):
    monkeypatch.chdir(isolated_configs_dir.parent)
    reg_path = isolated_configs_dir / "model_registry.json"
    original = reg_path.read_text(encoding="utf-8")
    result = runner.invoke(downloader_app, ["normalize-formats", "--dry-run"])
    assert result.exit_code == 0, result.stdout
    after = reg_path.read_text(encoding="utf-8")
    assert original == after, "Dry run should not modify registry"
