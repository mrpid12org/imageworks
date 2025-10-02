import json
import pathlib
import subprocess

REG = pathlib.Path("configs/model_registry.json")


def test_list_include_logical_shows_entry_without_download_path():
    data = json.loads(REG.read_text())
    # Inject logical-only entry if not present
    logical_name = "logical-only-test-ollama-gguf"
    if not any(e for e in data if e.get("name") == logical_name):
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
        REG.write_text(json.dumps(data, indent=2))
    # Run list without flag (should not include logical-only)
    proc = subprocess.run(
        ["uv", "run", "imageworks-download", "list", "--json"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    names = {e["name"] for e in json.loads(proc.stdout)}
    assert (
        logical_name not in names
    ), "Logical-only entry must be hidden without --include-logical"
    # Run list with flag
    proc2 = subprocess.run(
        ["uv", "run", "imageworks-download", "list", "--json", "--include-logical"],
        capture_output=True,
        text=True,
    )
    assert proc2.returncode == 0, proc2.stderr
    names2 = {e["name"] for e in json.loads(proc2.stdout)}
    assert (
        logical_name in names2
    ), "Logical-only entry should appear with --include-logical"
