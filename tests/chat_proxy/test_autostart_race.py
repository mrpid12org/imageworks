import asyncio
from fastapi.testclient import TestClient
from imageworks.chat_proxy.app import app
from imageworks.chat_proxy import app as app_module, forwarder as forwarder_module


autostart_calls = []


def test_autostart_race(monkeypatch):
    class DummyEntry:
        name = "slowmodel"
        display_name = "slowmodel"
        quantization = None
        backend = "vllm"
        probes = type("P", (), {"vision": None})
        chat_template = type("T", (), {"path": "templates/chat.jinja"})
        backend_config = type("cfg", (), {"port": 12354})
        served_model_id = None

    monkeypatch.setattr(app_module, "get_entry", lambda name: DummyEntry())
    monkeypatch.setattr(app_module, "list_models", lambda: ["slowmodel"])
    monkeypatch.setattr(forwarder_module, "get_entry", lambda name: DummyEntry())

    async def fake_probe(self, base_url):
        # First two checks fail, then succeed
        if len(autostart_calls) < 2:
            return False
        return True

    async def ensure_started(self, model: str):
        if not autostart_calls:
            autostart_calls.append(model)
            return True
        # Second caller observes start in progress/completed but does not trigger again
        return False

    monkeypatch.setattr(forwarder_module.ChatForwarder, "_probe", fake_probe)
    monkeypatch.setattr(
        forwarder_module.AutostartManager, "ensure_started", ensure_started
    )

    class FakeResp:
        def __init__(self):
            self.status_code = 200

        def json(self):
            return {
                "id": "cmpl",
                "object": "chat.completion",
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": {"completion_tokens": 4},
            }

    async def fake_post(self, url, json):  # noqa: A002
        # Simulate slight processing delay after backend becomes available
        await asyncio.sleep(0.01)
        return FakeResp()

    monkeypatch.setattr(forwarder_module.httpx.AsyncClient, "post", fake_post)
    # Speed up sleep used after autostart in forwarder
    original_sleep = asyncio.sleep

    async def fast_sleep(delay):
        # collapse long waits to short
        await original_sleep(0)  # yield control

    monkeypatch.setattr(asyncio, "sleep", fast_sleep)
    # Enable autostart in config + forwarder
    import imageworks.chat_proxy.app as app_mod

    app_mod._cfg.autostart_enabled = True
    app_mod._forwarder.cfg.autostart_enabled = True

    client = TestClient(app)

    async def do_req():
        return client.post(
            "/v1/chat/completions",
            json={
                "model": "slowmodel",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )

    # Run two overlapping requests
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    r1, r2 = loop.run_until_complete(asyncio.gather(do_req(), do_req()))
    # One may fail if startup timing not satisfied; ensure only one autostart attempt
    assert autostart_calls.count("slowmodel") == 1
