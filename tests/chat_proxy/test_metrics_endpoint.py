from fastapi.testclient import TestClient
from imageworks.chat_proxy.app import app
from imageworks.chat_proxy import app as app_module, forwarder as forwarder_module

app_module._cfg.vllm_single_port = False
app_module._forwarder.cfg.vllm_single_port = False
app_module._forwarder.vllm_manager = None


def test_metrics_after_requests(monkeypatch):
    class DummyEntry:
        name = "m1"
        display_name = "m1"
        quantization = None
        backend = "vllm"
        probes = type("P", (), {"vision": None})
        chat_template = type("T", (), {"path": "templates/chat.jinja"})
        backend_config = type("cfg", (), {"port": 12351})
        served_model_id = None

    monkeypatch.setattr(app_module, "get_entry", lambda name: DummyEntry())
    monkeypatch.setattr(
        app_module, "list_models", lambda: ["m1"]
    )  # for models endpoint
    monkeypatch.setattr(forwarder_module, "get_entry", lambda name: DummyEntry())

    async def fake_probe(self, base_url):
        return True

    class FakeResp:
        def __init__(self):
            self.status_code = 200

        def json(self):
            return {
                "id": "cmpl",
                "object": "chat.completion",
                "choices": [{"message": {"role": "assistant", "content": "hello"}}],
                "usage": {"completion_tokens": 4},
            }

    async def fake_post(self, url, json):  # noqa: A002
        return FakeResp()

    monkeypatch.setattr(forwarder_module.ChatForwarder, "_probe", fake_probe)
    monkeypatch.setattr(forwarder_module.httpx.AsyncClient, "post", fake_post)
    # Enable metrics in runtime config
    import imageworks.chat_proxy.app as app_mod

    app_mod._cfg.enable_metrics = True
    app_mod._forwarder.cfg.enable_metrics = True

    client = TestClient(app)
    # make a few non-stream requests
    for _ in range(3):
        r = client.post(
            "/v1/chat/completions",
            json={"model": "m1", "messages": [{"role": "user", "content": "Hi"}]},
        )
        assert r.status_code == 200

    # hit metrics endpoint
    m = client.get("/v1/metrics")
    assert m.status_code == 200
    body = m.json()
    assert body["rolling"]["count"] >= 3
    # backend counters should reflect requests
    reqs = body["requests_by_backend"].get("vllm")
    assert reqs and reqs["total_requests"] >= 3
    # ttft average should be numeric and positive
    assert isinstance(body["rolling"]["avg_ttft_ms"], (int, float))
    assert body["rolling"]["avg_ttft_ms"] >= 0
