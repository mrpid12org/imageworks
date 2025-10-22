from fastapi.testclient import TestClient

from imageworks.chat_proxy import app as app_module
from imageworks.chat_proxy.app import app

app_module._cfg.vllm_single_port = False
app_module._forwarder.cfg.vllm_single_port = False
app_module._forwarder.vllm_manager = None


def test_models_endpoint(monkeypatch):
    # monkeypatch registry list_models & get_entry

    class DummyVision:
        vision_ok = True

    class DummyProbes:
        vision = DummyVision()

    class DummyTemplate:
        path = "templates/chat.jinja"

    class DummyEntry:
        display_name = "llava"
        name = "llava"
        quantization = "q4"
        backend = "vllm"
        # Ensure it's treated as installed by the API filter
        download_path = "."
        probes = DummyProbes()
        capabilities = {"vision": True}
        chat_template = DummyTemplate()
        backend_config = type("cfg", (), {"port": 12345, "model_path": "."})
        served_model_id = None

    # Patch the symbols actually used inside the app module (imported at module load)
    monkeypatch.setattr(app_module, "list_models", lambda: ["llava"], raising=True)
    monkeypatch.setattr(
        app_module, "get_entry", lambda name: DummyEntry(), raising=True
    )

    client = TestClient(app)
    r = client.get("/v1/models")
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "list"
    ids = [m["id"] for m in data["data"]]
    # IDs favour human-readable display names; duplicates fall back to logical ids
    assert "llava" in ids
    assert "vision" in data["data"][0]["extensions"]["modalities"]


def test_models_endpoint_uses_declared_capabilities(monkeypatch):
    class DummyTemplate:
        path = "templates/chat.jinja"

    class DummyEntry:
        display_name = "qwen"
        name = "qwen"
        quantization = None
        backend = "vllm"
        download_path = "."
        probes = type("P", (), {"vision": None})
        capabilities = {"vision": True}
        chat_template = DummyTemplate()
        backend_config = type("cfg", (), {"port": 12346, "model_path": "."})
        served_model_id = None

    monkeypatch.setattr(app_module, "list_models", lambda: ["qwen"], raising=True)
    monkeypatch.setattr(
        app_module, "get_entry", lambda name: DummyEntry(), raising=True
    )

    client = TestClient(app)
    r = client.get("/v1/models")
    assert r.status_code == 200
    data = r.json()
    assert data["data"][0]["extensions"]["modalities"] == ["text", "vision"]


def test_models_endpoint_includes_served_without_download(monkeypatch):
    class DummyTemplate:
        path = "templates/chat.jinja"

    class DummyEntry:
        display_name = "qwen"
        name = "qwen"
        quantization = None
        backend = "vllm"
        download_path = None
        probes = type("P", (), {"vision": None})
        capabilities = {"vision": True}
        chat_template = DummyTemplate()
        backend_config = type("cfg", (), {"port": 8001, "model_path": None})
        served_model_id = "qwen"

    monkeypatch.setattr(app_module, "list_models", lambda: ["qwen"], raising=True)
    monkeypatch.setattr(
        app_module, "get_entry", lambda name: DummyEntry(), raising=True
    )

    client = TestClient(app)
    r = client.get("/v1/models")
    assert r.status_code == 200
    data = r.json()
    assert data["data"][0]["id"] == "qwen"


def test_models_endpoint_skips_entries_without_assets(monkeypatch):
    class DummyTemplate:
        path = "templates/chat.jinja"

    class DummyEntry:
        display_name = "stub"
        name = "stub"
        quantization = None
        backend = "vllm"
        download_path = None
        probes = type("P", (), {"vision": None})
        capabilities = {"vision": True}
        chat_template = DummyTemplate()
        backend_config = type("cfg", (), {"port": 0, "model_path": None})
        served_model_id = None

    monkeypatch.setattr(app_module, "list_models", lambda: ["stub"], raising=True)
    monkeypatch.setattr(
        app_module, "get_entry", lambda name: DummyEntry(), raising=True
    )

    client = TestClient(app)
    r = client.get("/v1/models")
    assert r.status_code == 200
    assert r.json()["data"] == []
