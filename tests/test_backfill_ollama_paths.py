import json
import pathlib
import subprocess

REG = pathlib.Path("configs/model_registry.json")


def _load():
    return json.loads(REG.read_text())


def test_backfill_ollama_paths_dry_run_and_apply():
    data = _load()
    # Synthesize a logical-only ollama entry (if none exists) by cloning first non-ollama entry
    has_candidate = any(
        e for e in data if e.get("backend") == "ollama" and not e.get("download_path")
    )
    if not has_candidate:
        # Duplicate first entry as synthetic logical ollama entry
        if not data:
            return  # nothing to do
        base = data[0].copy()
        base["name"] = "synthetic-test-ollama-gguf"
        base["backend"] = "ollama"
        base["download_path"] = None
        base["download_format"] = None
        base["download_location"] = None
        data.append(base)
        REG.write_text(json.dumps(data, indent=2))
    # Snapshot before dry run
    before = REG.read_text()
    proc = subprocess.run(
        [
            "uv",
            "run",
            "imageworks-download",
            "backfill-ollama-paths",
            "--dry-run",
            "-v",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    after_dry = REG.read_text()
    assert before == after_dry, "Dry run must not modify registry"
    # Apply
    proc2 = subprocess.run(
        ["uv", "run", "imageworks-download", "backfill-ollama-paths"],
        capture_output=True,
        text=True,
    )
    assert proc2.returncode == 0, proc2.stderr
    updated = _load()
    matches = [
        e
        for e in updated
        if e.get("backend") == "ollama"
        and e.get("name").startswith("synthetic-test-ollama-gguf")
    ]
    if matches:
        for m in matches:
            assert m.get("download_path"), "download_path should be populated"
            assert m.get("download_format") == "gguf", "download_format should be gguf"
