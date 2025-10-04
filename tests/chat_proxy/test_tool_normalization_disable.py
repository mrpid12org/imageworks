from fastapi.testclient import TestClient
from imageworks.chat_proxy.app import app
from imageworks.chat_proxy import app as app_module, forwarder as forwarder_module


def test_tool_normalization_disable(monkeypatch):
    class DummyEntry:
        name = "funcmodel"
        display_name = "funcmodel"
        quantization = None
        backend = "vllm"
        probes = type("P", (), {"vision": None})
        chat_template = type("T", (), {"path": "templates/chat.jinja"})
        backend_config = type("cfg", (), {"port": 12351})
        served_model_id = None

    monkeypatch.setattr(app_module, "get_entry", lambda name: DummyEntry())
    monkeypatch.setattr(app_module, "list_models", lambda: ["funcmodel"])
    monkeypatch.setattr(forwarder_module, "get_entry", lambda name: DummyEntry())

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
                            "message": {
                                "role": "assistant",
                                "function_call": {"name": "tool_a", "arguments": "{}"},
                            },
                        }
                    ],
                    "created": 1,
                    "model": "funcmodel",
                }

        return Resp()

    # Disable normalization by patching the forwarder's config directly
    import imageworks.chat_proxy.app as app_mod

    app_mod._cfg.disable_tool_normalization = True
    app_mod._forwarder.cfg.disable_tool_normalization = True

    monkeypatch.setattr(forwarder_module.httpx.AsyncClient, "post", fake_post)
    monkeypatch.setattr(forwarder_module.ChatForwarder, "_probe", fake_probe)

    client = TestClient(app)
    r = client.post(
        "/v1/chat/completions",
        json={"model": "funcmodel", "messages": [{"role": "user", "content": "Hi"}]},
    )
    assert r.status_code == 200
    body = r.json()
    msg = body["choices"][0]["message"]
    # function_call should remain (not normalized to tool_calls)
    assert "function_call" in msg and "tool_calls" not in msg
