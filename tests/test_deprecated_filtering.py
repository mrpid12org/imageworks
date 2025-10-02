import subprocess
import json
import pathlib

REG = pathlib.Path("configs/model_registry.json")


def run_list(extra=None, json_mode=False):
    cmd = ["uv", "run", "imageworks-download", "list"]
    if extra:
        cmd += extra
    if json_mode:
        cmd += ["--json"]
    return subprocess.run(cmd, capture_output=True, text=True)


def test_list_hides_deprecated_by_default():
    # Ensure at least one deprecated entry exists (if none, skip)
    data = json.loads(REG.read_text())
    deprecated = [e for e in data if e.get("deprecated")]
    if not deprecated:
        return  # nothing to test
    proc = run_list(["--format", "gguf", "--location", "linux_wsl"], json_mode=True)
    assert proc.returncode == 0, proc.stderr
    entries = json.loads(proc.stdout)
    names = {e["name"] for e in entries}
    for d in deprecated:
        assert d["name"] not in names, "Deprecated entry should be hidden"


def test_list_includes_deprecated_with_flag():
    data = json.loads(REG.read_text())
    deprecated = [e for e in data if e.get("deprecated")]
    if not deprecated:
        return
    # Don't apply a format filter here; some deprecated entries use other formats (fp16, awq, etc.)
    # Include --json in args so Typer sets json_output=True; we treat output as text for substring matching.
    proc = run_list(
        ["--location", "linux_wsl", "--show-deprecated", "--json"], json_mode=False
    )
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    for d in deprecated:
        assert d["name"] in out, f"Deprecated entry {d['name']} should appear in output"
