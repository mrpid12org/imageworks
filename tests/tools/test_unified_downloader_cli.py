import json
import subprocess
from pathlib import Path

REGISTRY_PATH = Path("configs/model_registry.json")


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["uv", "run", "imageworks-download", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_list_models_json_smoke():
    # Just ensure list runs and returns JSON array (may be empty)
    proc = run_cli("list", "--json")
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert isinstance(data, list)


def test_remove_nonexistent_variant():
    proc = run_cli("remove", "nonexistent-variant-name-xyz", "--force")
    assert proc.returncode != 0  # expect failure
    assert "Variant not found" in proc.stdout or "Variant not found" in proc.stderr


def test_verify_no_entries():
    # verify should succeed even if no downloads
    proc = run_cli("verify")
    assert proc.returncode == 0, proc.stderr
    assert "Verifying" in proc.stdout
