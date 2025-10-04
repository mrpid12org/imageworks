from fastapi.testclient import TestClient
from imageworks.chat_proxy.app import app
from imageworks.chat_proxy import app as app_module
from imageworks.chat_proxy import forwarder as forwarder_module


def test_chat_basic(monkeypatch):
    from imageworks.chat_proxy import forwarder

    class DummyEntry:
        name = "llava"
        display_name = "llava"
        quantization = None
        backend = "vllm"
        probes = type("P", (), {"vision": None})
        chat_template = type("T", (), {"path": "templates/chat.jinja"})
        backend_config = type("cfg", (), {"port": 12346})
        served_model_id = None

    monkeypatch.setattr(
        app_module, "get_entry", lambda name: DummyEntry(), raising=True
    )
    monkeypatch.setattr(app_module, "list_models", lambda: ["llava"], raising=True)
    # Forwarder module also imports registry.get_entry indirectly; patch there too
    monkeypatch.setattr(
        forwarder_module, "get_entry", lambda name: DummyEntry(), raising=True
    )

    async def fake_probe(self, base_url):
        return True

    async def fake_post(self, url, json=None):  # noqa: A002
        class Resp:
            status_code = 200

            def json(self_inner):
                return {
                    "id": "resp1",
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "Hello"},
                        }
                    ],
                    "created": 1,
                    "model": "llava",
                }

        return Resp()

    monkeypatch.setattr(forwarder.ChatForwarder, "_probe", fake_probe)
    monkeypatch.setattr(
        forwarder.httpx.AsyncClient, "post", fake_post
    )  # monkeypatch network

    client = TestClient(app)
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "llava",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "Hello"
