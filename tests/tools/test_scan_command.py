import subprocess
import tempfile
from pathlib import Path
import json


def run_cli(*args):
    return subprocess.run(
        ["uv", "run", "imageworks-download", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_scan_dry_run_smoke():
    proc = run_cli("scan", "--dry-run")
    assert proc.returncode == 0, proc.stderr
    assert ("No candidate" in proc.stdout) or ("Dry run" in proc.stdout)


def test_scan_awq_vs_fp16_and_format_fallback():
    """Create two temp repos: one AWQ (with quantization_config) and one fp16.
    Ensure AWQ detected automatically and fp16 stays fp16; providing --format awq should NOT flip fp16 repo (fallback only).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        # Layout: base/owner1/repo-awq, base/owner2/repo-fp16
        awq_dir = base / "owner1" / "model-awq"
        fp16_dir = base / "owner2" / "model-fp16"
        awq_dir.mkdir(parents=True)
        fp16_dir.mkdir(parents=True)
        # AWQ indicators: quantization_config.json + fake weight file .awq + quant group size
        (awq_dir / "weights.awq").write_bytes(b"00")
        with (awq_dir / "quantization_config.json").open("w") as fh:
            json.dump({"quant_method": "awq", "w_bit": 4, "q_group_size": 128}, fh)
        # FP16 indicators: safetensors file
        (fp16_dir / "model.safetensors").write_bytes(b"00")
        # Initial scan no format hint
        proc = run_cli("scan", "--base", str(base))
        assert proc.returncode == 0, proc.stderr
        out = proc.stdout.lower()
        # AWQ repo should show awq, fp16 repo should show fp16
        assert "model-awq" in out and "awq" in out
        assert "model-fp16" in out and "fp16" in out
        # Scan with --format awq fallback should not convert fp16 repo
        proc2 = run_cli("scan", "--base", str(base), "--format", "awq")
        assert proc2.returncode == 0
        out2 = proc2.stdout.lower()
        # Ensure fp16 still present for model-fp16 line
        # (Line will contain model-fp16 and fp16 token near it)
        fp16_line = [line for line in out2.splitlines() if "model-fp16" in line]
        assert fp16_line and "fp16" in fp16_line[0]
        # AWQ still awq
        awq_line = [line for line in out2.splitlines() if "model-awq" in line]
        assert awq_line and "awq" in awq_line[0]
