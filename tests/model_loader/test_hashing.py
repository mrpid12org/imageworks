import json
from pathlib import Path
import pytest

from imageworks.model_loader import registry
from imageworks.model_loader.hashing import verify_model, VersionLockViolation


def _make_registry(
    tmp_path: Path, model_path: Path, locked: bool = False, expected: str | None = None
):
    sample = [
        {
            "name": "hash-model",
            "backend": "vllm",
            "backend_config": {"port": 8100, "model_path": str(model_path)},
            "capabilities": {"vision": True},
            "artifacts": {"aggregate_sha256": "", "files": []},
            "chat_template": {"source": "embedded"},
            "version_lock": {"locked": locked, "expected_aggregate_sha256": expected},
            "performance": {"rolling_samples": 0},
            "probes": {},
        }
    ]
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(sample))
    registry.load_registry(path, force=True)
    return registry.get_entry("hash-model")


def test_verify_model_unlocked(tmp_path: Path, monkeypatch):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    # create dummy artifact
    (model_dir / "config.json").write_text("{}")

    entry = _make_registry(tmp_path, model_dir, locked=False)
    verify_model(entry)
    assert entry.artifacts.aggregate_sha256
    assert entry.version_lock.last_verified is not None


def test_verify_model_lock_violation(tmp_path: Path, monkeypatch):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    art = model_dir / "config.json"
    art.write_text("original")

    entry = _make_registry(tmp_path, model_dir, locked=True, expected="deadbeef")
    with pytest.raises(VersionLockViolation):
        verify_model(entry)


def test_verify_model_lock_sets_expected_on_first_lock(tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")

    entry = _make_registry(tmp_path, model_dir, locked=True, expected=None)
    verify_model(entry)
    assert (
        entry.version_lock.expected_aggregate_sha256 == entry.artifacts.aggregate_sha256
    )
