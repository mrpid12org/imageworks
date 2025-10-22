from fastapi.testclient import TestClient
from imageworks.chat_proxy.app import app
from imageworks.chat_proxy import app as app_module, forwarder as forwarder_module

app_module._cfg.vllm_single_port = False
app_module._forwarder.cfg.vllm_single_port = False
app_module._forwarder.vllm_manager = None


def test_chat_streaming(monkeypatch):
    class DummyEntry:
        name = "llava"
        display_name = "llava"
        quantization = None
        backend = "vllm"
        probes = type("P", (), {"vision": None})
        chat_template = type("T", (), {"path": "templates/chat.jinja"})
        backend_config = type("cfg", (), {"port": 12350})
        served_model_id = None

    monkeypatch.setattr(app_module, "get_entry", lambda name: DummyEntry())
    monkeypatch.setattr(app_module, "list_models", lambda: ["llava"])
    monkeypatch.setattr(forwarder_module, "get_entry", lambda name: DummyEntry())

    async def fake_probe(self, base_url):
        return True

    def fake_stream(self, method, url, json=None):  # noqa: A002
        class Resp:
            status_code = 200

            async def __aenter__(self_inner):
                return self_inner

            async def __aexit__(self_inner, exc_type, exc, tb):
                return False

            async def aiter_lines(self_inner):
                # Simulate two tokens then DONE
                yield 'data: {"id": "c1", "object": "chat.completion.chunk", "choices": [{"delta": {"content": "Hel"}}]}'
                yield 'data: {"id": "c1", "object": "chat.completion.chunk", "choices": [{"delta": {"content": "lo"}}]}'
                yield "data: [DONE]"

        return Resp()

    monkeypatch.setattr(forwarder_module.httpx.AsyncClient, "stream", fake_stream)
    monkeypatch.setattr(forwarder_module.ChatForwarder, "_probe", fake_probe)

    client = TestClient(app)
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "llava",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        },
    )
    assert r.status_code == 200
    body = r.text
    assert "Hel" in body and "lo" in body and "[DONE]" in body
