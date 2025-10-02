import json
from pathlib import Path
from typer.testing import CliRunner

from imageworks.model_loader.cli_sync import app as sync_app

runner = CliRunner()


def test_list_roles_text_output(tmp_path: Path):
    # create minimal registry with roles
    reg_path = tmp_path / "registry.json"
    reg_path.write_text(
        '[{"name":"model_a","display_name":"Model A","backend":"vllm","backend_config":{"port":0,"model_path":"/models/a","extra_args":[]},"capabilities":{"text":true,"vision":false,"audio":false,"embedding":false},"artifacts":{"aggregate_sha256":"","files":[]},"chat_template":{"source":"embedded","path":null,"sha256":null},"version_lock":{"locked":false,"expected_aggregate_sha256":null,"last_verified":null},"performance":{"rolling_samples":0,"ttft_ms_avg":null,"throughput_toks_per_s_avg":null,"last_sample":null},"probes":{"vision":null},"profiles_placeholder":null,"metadata":{},"served_model_id":null,"model_aliases":[],"roles":["caption"],"license":null,"source":null,"deprecated":false,"download_format":null,"download_location":null,"download_path":null,"download_size_bytes":null,"download_files":[],"download_directory_checksum":null,"downloaded_at":null,"last_accessed":null}]'
    )
    res = runner.invoke(sync_app, ["list-roles", "--registry-path", str(reg_path)])
    assert res.exit_code == 0, res.stdout
    assert "ROLE" in res.stdout
    assert "caption" in res.stdout
    assert "model_a" in res.stdout


def test_list_roles_json_output(tmp_path: Path):
    reg_path = tmp_path / "registry.json"
    reg_path.write_text(
        '[{"name":"emb_model","display_name":null,"backend":"vllm","backend_config":{"port":0,"model_path":"/models/e","extra_args":[]},"capabilities":{"text":false,"vision":false,"audio":false,"embedding":true},"artifacts":{"aggregate_sha256":"","files":[]},"chat_template":{"source":"embedded","path":null,"sha256":null},"version_lock":{"locked":false,"expected_aggregate_sha256":null,"last_verified":null},"performance":{"rolling_samples":0,"ttft_ms_avg":null,"throughput_toks_per_s_avg":null,"last_sample":null},"probes":{"vision":null},"profiles_placeholder":null,"metadata":{},"served_model_id":null,"model_aliases":[],"roles":["embedding"],"license":null,"source":null,"deprecated":false,"download_format":null,"download_location":null,"download_path":null,"download_size_bytes":null,"download_files":[],"download_directory_checksum":null,"downloaded_at":null,"last_accessed":null}]'
    )
    res = runner.invoke(
        sync_app, ["list-roles", "--registry-path", str(reg_path), "--json-output"]
    )
    assert res.exit_code == 0, res.stdout
    data = json.loads(res.stdout)
    assert isinstance(data, list)
    assert any(r["role"] == "embedding" for r in data)
    assert data[0]["name"] == "emb_model"
