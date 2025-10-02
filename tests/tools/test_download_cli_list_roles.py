from typer.testing import CliRunner
from pathlib import Path
import json

from imageworks.tools.model_downloader.cli import app as download_app

runner = CliRunner()

REG_SKELETON = '{"name":"vlm-caption","display_name":"Caption Model","backend":"vllm","backend_config":{"port":0,"model_path":"/tmp/model","extra_args":[]},"capabilities":{"text":true,"vision":true,"audio":false,"embedding":false},"artifacts":{"aggregate_sha256":"","files":[]},"chat_template":{"source":"embedded","path":null,"sha256":null},"version_lock":{"locked":false,"expected_aggregate_sha256":null,"last_verified":null},"performance":{"rolling_samples":0,"ttft_ms_avg":null,"throughput_toks_per_s_avg":null,"last_sample":null},"probes":{"vision":null},"profiles_placeholder":null,"metadata":{},"served_model_id":null,"model_aliases":[],"roles":["caption"],"license":null,"source":null,"deprecated":false,"download_format":null,"download_location":null,"download_path":null,"download_size_bytes":null,"download_files":[],"download_directory_checksum":null,"downloaded_at":null,"last_accessed":null}'


def write_registry(path: Path, entries_json: str):
    path.write_text(f"[{entries_json}]")


def test_list_roles_text(tmp_path: Path):
    reg_path = tmp_path / "model_registry.json"
    write_registry(reg_path, REG_SKELETON)
    res = runner.invoke(download_app, ["list-roles", "--registry-path", str(reg_path)])
    assert res.exit_code == 0, res.stdout
    assert "Role-capable Models" in res.stdout
    assert "caption" in res.stdout
    assert "vlm-caption" in res.stdout


def test_list_roles_json(tmp_path: Path):
    reg_path = tmp_path / "model_registry.json"
    write_registry(reg_path, REG_SKELETON)
    res = runner.invoke(
        download_app, ["list-roles", "--registry-path", str(reg_path), "--json-output"]
    )
    assert res.exit_code == 0, res.stdout
    data = json.loads(res.stdout)
    assert any(r["role"] == "caption" for r in data)
    assert data[0]["name"] == "vlm-caption"
