import json
from pathlib import Path
from typer.testing import CliRunner

from imageworks.model_loader.cli import app as models_app
from imageworks.model_loader import registry

runner = CliRunner()


def _write_registry(tmp_path: Path):
    sample = [
        {
            "name": "cli-model",
            "backend": "vllm",
            "backend_config": {"port": 9009, "model_path": str(tmp_path / "m")},
            "capabilities": {"vision": True},
            "artifacts": {"aggregate_sha256": "", "files": []},
            "chat_template": {"source": "embedded"},
            "version_lock": {"locked": False},
            "performance": {"rolling_samples": 0},
            "probes": {},
        }
    ]
    reg_path = tmp_path / "registry.json"
    reg_path.write_text(json.dumps(sample))
    registry.load_registry(reg_path, force=True)
    return reg_path


def test_cli_list_and_select(tmp_path: Path, monkeypatch):
    _write_registry(tmp_path)
    res = runner.invoke(models_app, ["list"])  # noqa: S603 - test invocation
    assert res.exit_code == 0
    assert "cli-model" in res.stdout

    res2 = runner.invoke(models_app, ["select", "cli-model"])  # noqa: S603
    assert res2.exit_code == 0
    assert "endpoint" in res2.stdout


def test_cli_verify(tmp_path: Path):
    _write_registry(tmp_path)
    res = runner.invoke(models_app, ["verify", "cli-model"])  # noqa: S603
    # verify may fail if no files present but should still exit 0 (best-effort); adjust if logic changes
    assert res.exit_code == 0
