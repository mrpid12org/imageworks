import json
from pathlib import Path
import pytest

from imageworks.model_loader import registry
from imageworks.model_loader.registry import RegistryLoadError
from imageworks.model_loader.service import select_model, CapabilityError


def test_load_registry_success(tmp_path: Path):
    sample = [
        {
            "name": "demo-model",
            "backend": "vllm",
            "backend_config": {"port": 8000, "model_path": "/models/demo"},
            "capabilities": {"vision": True},
            "artifacts": {"aggregate_sha256": "", "files": []},
            "chat_template": {"source": "embedded"},
            "version_lock": {"locked": False},
            "performance": {"rolling_samples": 0},
            "probes": {},
        }
    ]
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(sample))
    # Reset global cache to ensure isolation
    registry._REGISTRY_CACHE = None  # type: ignore[attr-defined]
    loaded = registry.load_registry(path, force=True)
    assert "demo-model" in loaded
    entry = loaded["demo-model"]
    assert entry.backend == "vllm"


def test_load_registry_duplicate_name(tmp_path: Path):
    sample = [
        {"name": "dup", "backend": "vllm"},
        {"name": "dup", "backend": "lmdeploy"},
    ]
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(sample))
    with pytest.raises(RegistryLoadError):
        registry.load_registry(path, force=True)


def test_get_entry_missing(monkeypatch):
    # ensure cache is cleared
    monkeypatch.setattr(registry, "_REGISTRY_CACHE", {})
    with pytest.raises(KeyError):
        registry.get_entry("missing")


def test_select_model_and_capabilities(tmp_path: Path, monkeypatch):
    sample = [
        {
            "name": "vision-ok",
            "backend": "vllm",
            "backend_config": {"port": 8123, "model_path": "/m"},
            "capabilities": {"vision": True, "function_calling": True},
            "artifacts": {"aggregate_sha256": "", "files": []},
            "chat_template": {"source": "embedded"},
            "version_lock": {"locked": False},
            "performance": {"rolling_samples": 0},
            "probes": {},
        },
        {
            "name": "text-only",
            "backend": "vllm",
            "backend_config": {"port": 8124, "model_path": "/m"},
            "capabilities": {"vision": False},
            "artifacts": {"aggregate_sha256": "", "files": []},
            "chat_template": {"source": "embedded"},
            "version_lock": {"locked": False},
            "performance": {"rolling_samples": 0},
            "probes": {},
        },
        {
            "name": "reasoner",
            "backend": "vllm",
            "backend_config": {"port": 8125, "model_path": "/m"},
            "capabilities": {"thinking": True},
            "artifacts": {"aggregate_sha256": "", "files": []},
            "chat_template": {"source": "embedded"},
            "version_lock": {"locked": False},
            "performance": {"rolling_samples": 0},
            "probes": {},
        },
    ]
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(sample))
    monkeypatch.setattr(registry, "_REGISTRY_CACHE", None)
    registry.load_registry(path, force=True)

    sel = select_model("vision-ok", require_capabilities=["vision"])
    assert sel.endpoint_url.endswith(":8123/v1")
    assert sel.capabilities["vision"] is True
    assert sel.capabilities["tools"] is True
    assert sel.capabilities["tool_calls"] is True

    with pytest.raises(CapabilityError):
        select_model("text-only", require_capabilities=["vision"])

    # Synonym handling
    sel_reason = select_model("reasoner", require_capabilities=["reasoning"])
    assert sel_reason.capabilities["reasoning"] is True
    assert sel_reason.capabilities["thinking"] is True

    with pytest.raises(CapabilityError):
        select_model("text-only", require_capabilities=["reasoning"])


def test_select_ollama_backend(tmp_path: Path, monkeypatch):
    sample = [
        {
            "name": "qwen2.5-vl-7b-gguf-q4",
            "backend": "ollama",
            "backend_config": {"port": 11434, "model_path": "/unused"},
            "capabilities": {"vision": True, "text": True},
            "artifacts": {"aggregate_sha256": "", "files": []},
            "chat_template": {"source": "embedded"},
            "version_lock": {"locked": False},
            "performance": {"rolling_samples": 0},
            "probes": {},
        }
    ]
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(sample))
    monkeypatch.setattr(registry, "_REGISTRY_CACHE", None)
    registry.load_registry(path, force=True)
    sel = select_model("qwen2.5-vl-7b-gguf-q4", require_capabilities=["vision"])
    assert sel.backend == "ollama"
    assert sel.endpoint_url.endswith(":11434/v1")


def test_select_model_respects_host_override(tmp_path: Path, monkeypatch):
    sample = [
        {
            "name": "custom-host",
            "backend": "vllm",
            "backend_config": {
                "port": 8126,
                "model_path": "/m",
                "host": "imageworks-ollama",
            },
            "capabilities": {"vision": True},
            "artifacts": {"aggregate_sha256": "", "files": []},
            "chat_template": {"source": "embedded"},
            "version_lock": {"locked": False},
            "performance": {"rolling_samples": 0},
            "probes": {},
        }
    ]
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(sample))
    monkeypatch.setattr(registry, "_REGISTRY_CACHE", None)
    registry.load_registry(path, force=True)

    sel = select_model("custom-host")
    assert sel.endpoint_url.startswith("http://imageworks-ollama:8126")


def test_select_model_respects_base_url_override(tmp_path: Path, monkeypatch):
    sample = [
        {
            "name": "custom-base",
            "backend": "vllm",
            "backend_config": {
                "port": 0,
                "model_path": "/m",
                "base_url": "https://example.local/v1",
            },
            "capabilities": {"vision": True},
            "artifacts": {"aggregate_sha256": "", "files": []},
            "chat_template": {"source": "embedded"},
            "version_lock": {"locked": False},
            "performance": {"rolling_samples": 0},
            "probes": {},
        }
    ]
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(sample))
    monkeypatch.setattr(registry, "_REGISTRY_CACHE", None)
    registry.load_registry(path, force=True)

    sel = select_model("custom-base")
    assert sel.endpoint_url == "https://example.local/v1"


def test_tokenizer_chat_template_extraction(tmp_path: Path, monkeypatch):
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    tokenizer_cfg = {
        "chat_template": "<s>{% for m in messages %}{{ m.content }}{% endfor %}</s>"
    }
    (weights_dir / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_cfg), encoding="utf-8"
    )

    sample = [
        {
            "name": "qwen-demo",
            "backend": "vllm",
            "backend_config": {"port": 8000, "model_path": str(weights_dir)},
            "capabilities": {"text": True},
            "artifacts": {"aggregate_sha256": "", "files": []},
            "chat_template": {"source": "embedded"},
            "version_lock": {"locked": False},
            "performance": {"rolling_samples": 0},
            "probes": {},
            "download_path": str(weights_dir),
            "download_files": ["tokenizer_config.json"],
        }
    ]
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(sample), encoding="utf-8")

    cache_dir = tmp_path / "templates"
    monkeypatch.setenv("IMAGEWORKS_TEMPLATE_CACHE_DIR", str(cache_dir))
    monkeypatch.setattr(registry, "_REGISTRY_CACHE", None)

    loaded = registry.load_registry(path, force=True)
    entry = loaded["qwen-demo"]
    assert entry.chat_template.path
    tpl_file = Path(entry.chat_template.path)
    assert tpl_file.exists()
    assert tpl_file.read_text(encoding="utf-8") == tokenizer_cfg["chat_template"]
