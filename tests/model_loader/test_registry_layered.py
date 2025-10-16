import json
from dataclasses import replace
from pathlib import Path

import pytest

from imageworks.model_loader import registry
from imageworks.model_loader.models import RegistryEntry
from imageworks.model_loader.registry import RegistryLoadError


def _write(path: Path, payload):
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _base_entry(name: str, backend: str = "vllm") -> dict:
    return {
        "name": name,
        "display_name": name,
        "backend": backend,
        "backend_config": {"port": 8000, "model_path": "/weights", "extra_args": []},
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


def test_save_registry_promotes_dynamic_entries(tmp_path, monkeypatch):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    curated_path = config_dir / "model_registry.curated.json"
    discovered_path = config_dir / "model_registry.discovered.json"
    merged_path = config_dir / "model_registry.json"

    curated_entry = _base_entry("curated-model")
    curated_entry["download_path"] = None
    discovered_entry = _base_entry("dynamic-model")
    discovered_entry.update(
        {"download_path": "/models/dynamic", "download_format": "gguf"}
    )

    _write(curated_path, [curated_entry])
    _write(discovered_path, [discovered_entry])
    _write(merged_path, [curated_entry, discovered_entry])

    monkeypatch.setenv("IMAGEWORKS_REGISTRY_DIR", str(config_dir))
    registry._REGISTRY_CACHE = None  # type: ignore[attr-defined]
    loaded = registry.load_registry(force=True)

    entry = loaded["curated-model"]
    # Introduce dynamic state; save should write overlay copy, not touch curated file
    entry.download_path = "/models/curated"
    entry.download_location = "linux_wsl"
    registry.update_entries([entry], save=True)

    curated_raw = json.loads(curated_path.read_text(encoding="utf-8"))
    assert curated_raw[0]["download_path"] is None

    discovered_raw = json.loads(discovered_path.read_text(encoding="utf-8"))
    names = {item["name"] for item in discovered_raw}
    assert "curated-model" in names
    updated_overlay = next(
        item for item in discovered_raw if item["name"] == "curated-model"
    )
    assert updated_overlay["download_path"] == "/models/curated"
    assert updated_overlay["download_location"] == "linux_wsl"


def test_duplicate_download_identity_raises(tmp_path, monkeypatch):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    curated_path = config_dir / "model_registry.curated.json"
    discovered_path = config_dir / "model_registry.discovered.json"
    merged_path = config_dir / "model_registry.json"

    curated_entry = _base_entry("base-model")
    curated_entry.update({"download_path": "/models/shared", "download_format": "gguf"})

    _write(curated_path, [curated_entry])
    _write(discovered_path, [])
    _write(merged_path, [curated_entry])

    monkeypatch.setenv("IMAGEWORKS_REGISTRY_DIR", str(config_dir))
    registry._REGISTRY_CACHE = None  # type: ignore[attr-defined]
    loaded = registry.load_registry(force=True)

    existing = loaded["base-model"]
    duplicate: RegistryEntry = replace(existing, name="duplicate-model")

    with pytest.raises(RegistryLoadError):
        registry.update_entries([duplicate], save=False)
