from fastapi.testclient import TestClient

from imageworks.chat_proxy import app as app_module
from imageworks.chat_proxy.app import app


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
        chat_template = DummyTemplate()
        backend_config = type("cfg", (), {"port": 12345})
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
    # IDs now include quantization when present to avoid duplicates
    assert "llava-q4" in ids
    assert "vision" in data["data"][0]["extensions"]["modalities"]
