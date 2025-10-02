import subprocess
import json
import pathlib

REG = pathlib.Path("configs/model_registry.json")


def test_backend_filter_lists_only_ollama():
    proc = subprocess.run(
        ["uv", "run", "imageworks-download", "list", "--backend", "ollama", "--json"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    # Test resilience if no ollama entries present (pass empty) else ensure all are ollama
    if data:
        assert all(e["backend"] == "ollama" for e in data)
