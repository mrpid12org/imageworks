import json
from pathlib import Path

from typer.testing import CliRunner

from imageworks.tools.model_downloader.cli import app as downloader_app

runner = CliRunner()


def _ensure_logical_entry(configs_dir: Path, logical_name: str) -> None:
    registry_snapshot = configs_dir / "model_registry.json"
    discovered_path = configs_dir / "model_registry.discovered.json"

    data = json.loads(registry_snapshot.read_text(encoding="utf-8"))
    if any(e for e in data if e.get("name") == logical_name):
        return
    template = (
        data[0].copy()
        if data
        else {
            "backend": "ollama",
            "backend_config": {"port": 0, "model_path": "", "extra_args": []},
            "capabilities": {"text": True},
            "artifacts": {"aggregate_sha256": "", "files": []},
            "chat_template": {"source": "embedded", "path": None, "sha256": None},
            "version_lock": {
                "locked": False,
                "expected_aggregate_sha256": None,
                "last_verified": None,
            },
            "performance": {
                "rolling_samples": 0,
                "ttft_ms_avg": None,
                "throughput_toks_per_s_avg": None,
                "last_sample": None,
            },
            "probes": {"vision": None},
            "profiles_placeholder": None,
            "metadata": {},
            "model_aliases": [],
            "roles": [],
            "license": None,
            "source": None,
            "deprecated": False,
        }
    )
    template.update(
        {
            "name": logical_name,
            "backend": "ollama",
            "download_path": None,
            "download_format": "gguf",
            "download_location": None,
            "family": "logical-only-test",
            "quantization": None,
            "served_model_id": "logical-only-test:base",
        }
    )
    data.append(template)
    registry_snapshot.write_text(json.dumps(data, indent=2), encoding="utf-8")

    try:
        discovered = json.loads(discovered_path.read_text(encoding="utf-8"))
    except Exception:
        discovered = []
    discovered.append(template)
    discovered_path.write_text(json.dumps(discovered, indent=2), encoding="utf-8")


def test_list_include_logical_shows_entry_without_download_path(
    isolated_configs_dir, monkeypatch
):
    monkeypatch.chdir(isolated_configs_dir.parent)
    logical_name = "logical-only-testcase-ollama-gguf"
    _ensure_logical_entry(isolated_configs_dir, logical_name)

    # Without flag the logical-only entry should stay hidden
    result = runner.invoke(downloader_app, ["list", "--json"])
    assert result.exit_code == 0, result.stdout
    names = {e["name"] for e in json.loads(result.stdout or "[]")}
    assert logical_name not in names

    # With --include-logical the synthetic entry should be surfaced
    result_with = runner.invoke(downloader_app, ["list", "--json", "--include-logical"])
    assert result_with.exit_code == 0, result_with.stdout
    names_with = {e["name"] for e in json.loads(result_with.stdout or "[]")}
    assert logical_name in names_with
