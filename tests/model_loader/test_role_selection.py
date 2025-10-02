import json
from pathlib import Path
from imageworks.model_loader.role_selection import (
    select_by_role,
    list_models_for_role,
)
from imageworks.model_loader import registry


def _write_registry(tmp_path: Path, entries: list[dict]):
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(entries))
    registry.load_registry(path, force=True)


def test_select_by_role_prefers_preferred(tmp_path: Path):
    entries = [
        {
            "name": "model-a",
            "backend": "vllm",
            "backend_config": {"port": 8000, "model_path": "/m/a"},
            "capabilities": {"vision": True},
            "artifacts": {"aggregate_sha256": "", "files": []},
            "chat_template": {"source": "embedded"},
            "version_lock": {"locked": False},
            "performance": {"rolling_samples": 0},
            "probes": {},
            "roles": ["caption"],
        },
        {
            "name": "model-b",
            "backend": "vllm",
            "backend_config": {"port": 8001, "model_path": "/m/b"},
            "capabilities": {"vision": True},
            "artifacts": {"aggregate_sha256": "", "files": []},
            "chat_template": {"source": "embedded"},
            "version_lock": {"locked": False},
            "performance": {"rolling_samples": 0},
            "probes": {},
            "roles": ["caption"],
        },
    ]
    _write_registry(tmp_path, entries)
    # preferred list should pick model-b if provided
    chosen = select_by_role("caption", preferred=["non-existent", "model-b"])  # type: ignore[arg-type]
    assert chosen == "model-b"


def test_list_models_for_role_filters(tmp_path: Path):
    entries = [
        {
            "name": "x",
            "backend": "vllm",
            "backend_config": {"port": 8000, "model_path": "/m/x"},
            "capabilities": {"vision": True},
            "artifacts": {"aggregate_sha256": "", "files": []},
            "chat_template": {"source": "embedded"},
            "version_lock": {"locked": False},
            "performance": {"rolling_samples": 0},
            "probes": {},
            "roles": ["caption"],
        },
        {
            "name": "y",
            "backend": "vllm",
            "backend_config": {"port": 8001, "model_path": "/m/y"},
            "capabilities": {"vision": True},
            "artifacts": {"aggregate_sha256": "", "files": []},
            "chat_template": {"source": "embedded"},
            "version_lock": {"locked": False},
            "performance": {"rolling_samples": 0},
            "probes": {},
            "roles": ["description"],
        },
    ]
    _write_registry(tmp_path, entries)
    listed = list_models_for_role("caption")
    assert listed == ["x"]


def test_select_by_role_capability_fail(tmp_path: Path):
    entries = [
        {
            "name": "bad",
            "backend": "vllm",
            "backend_config": {"port": 8000, "model_path": "/m/bad"},
            "capabilities": {"vision": False},
            "artifacts": {"aggregate_sha256": "", "files": []},
            "chat_template": {"source": "embedded"},
            "version_lock": {"locked": False},
            "performance": {"rolling_samples": 0},
            "probes": {},
            "roles": ["caption"],
        },
    ]
    _write_registry(tmp_path, entries)
    import pytest

    with pytest.raises(Exception):
        select_by_role("caption")
