from pathlib import Path
import subprocess

REG_PATH = Path("configs/model_registry.json")


def test_normalize_formats_dry_run():
    # Run dry-run normalize (should not modify file hash)
    original = REG_PATH.read_text(encoding="utf-8")
    proc = subprocess.run(
        ["uv", "run", "imageworks-download", "normalize-formats", "--dry-run"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    after = REG_PATH.read_text(encoding="utf-8")
    assert original == after, "Dry run should not modify registry"
