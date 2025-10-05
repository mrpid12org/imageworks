from fastapi.testclient import TestClient
from imageworks.chat_proxy.app import app
from imageworks.chat_proxy import app as app_module, forwarder as forwarder_module
import base64


def test_capability_mismatch(monkeypatch):
    class DummyEntry:
        name = "textmodel"
        display_name = "textmodel"
        quantization = None
        backend = "vllm"
        probes = type("P", (), {"vision": None})  # No vision
        capabilities = {}
        chat_template = type("T", (), {"path": "templates/chat.jinja"})
        backend_config = type("cfg", (), {"port": 12352})
        served_model_id = None

    monkeypatch.setattr(app_module, "get_entry", lambda name: DummyEntry())
    monkeypatch.setattr(app_module, "list_models", lambda: ["textmodel"])
    monkeypatch.setattr(forwarder_module, "get_entry", lambda name: DummyEntry())

    client = TestClient(app)
    img_b64 = base64.b64encode(b"fakeimage").decode()
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "textmodel",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                        },
                    ],
                }
            ],
        },
    )
    assert r.status_code == 409
    assert r.json()["error"]["type"] == "capability_mismatch"


def test_oversize(monkeypatch):
    class DummyEntry:
        name = "visionmodel"
        display_name = "visionmodel"
        quantization = None
        backend = "vllm"
        probes = type("P", (), {"vision": type("V", (), {"vision_ok": True})()})
        capabilities = {"vision": True}
        chat_template = type("T", (), {"path": "templates/chat.jinja"})
        backend_config = type("cfg", (), {"port": 12353})
        served_model_id = None

    monkeypatch.setattr(app_module, "get_entry", lambda name: DummyEntry())
    monkeypatch.setattr(app_module, "list_models", lambda: ["visionmodel"])
    monkeypatch.setattr(forwarder_module, "get_entry", lambda name: DummyEntry())

    # Craft a base64 string exceeding limit (approx 0.75 factor) so > 6MB -> need > 8MB b64
    big = b"x" * (8_100_000)
    import base64 as b64

    payload_b64 = b64.b64encode(big).decode()

    client = TestClient(app)
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "visionmodel",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{payload_b64}"
                            },
                        },
                    ],
                }
            ],
        },
    )
    assert r.status_code == 413
    assert r.json()["error"]["type"] == "payload_too_large"
