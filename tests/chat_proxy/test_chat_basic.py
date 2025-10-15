from fastapi.testclient import TestClient
from imageworks.chat_proxy.app import app
from imageworks.chat_proxy import app as app_module
from imageworks.chat_proxy import forwarder as forwarder_module
import imageworks.model_loader.registry as registry_module


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


def test_chat_resolves_simplified_display(monkeypatch):
    class DummyEntry:
        name = "qwen2.5vl_7b_(Q4_K_M)"
        display_name = "qwen2.5vl 7b (Q4 K M)"
        quantization = "q4_k_m"
        backend = "vllm"
        probes = type("P", (), {"vision": None})
        chat_template = type("T", (), {"path": "templates/chat.jinja"})
        backend_config = type("cfg", (), {"port": 12355})
        served_model_id = "qwen2.5vl_7b_(Q4_K_M)-served"

    entry = DummyEntry()

    def fake_get_entry(name):
        if name == entry.name:
            return entry
        raise KeyError(name)

    monkeypatch.setattr(app_module, "get_entry", fake_get_entry, raising=True)
    monkeypatch.setattr(forwarder_module, "get_entry", fake_get_entry, raising=True)
    monkeypatch.setattr(registry_module, "get_entry", fake_get_entry, raising=True)
    monkeypatch.setattr(
        registry_module,
        "load_registry",
        lambda: {entry.name: entry},
    )

    async def fake_probe(self, base_url):
        return True

    monkeypatch.setattr(forwarder_module.ChatForwarder, "_probe", fake_probe)

    captured = {}

    async def fake_post(self, url, json=None):  # noqa: A002
        captured["payload"] = json

        class Resp:
            status_code = 200

            def json(self_inner):
                return {
                    "id": "resp2",
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "Resolved"},
                        }
                    ],
                }

        return Resp()

    monkeypatch.setattr(forwarder_module.httpx.AsyncClient, "post", fake_post)

    client = TestClient(app)
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen2.5vl 7b (Q4 K M)",
            "messages": [{"role": "user", "content": "Hi"}],
        },
    )

    assert r.status_code == 200
    assert captured["payload"]["model"] == entry.served_model_id


def test_chat_autostart_uses_resolved_id(monkeypatch):
    class DummyEntry:
        name = "deepseek-coder-6.7b-instruct_(AWQ)"
        display_name = "deepseek coder 6.7b instruct (AWQ)"
        quantization = "awq"
        backend = "vllm"
        probes = type("P", (), {"vision": None})
        chat_template = type("T", (), {"path": "templates/chat.jinja"})
        backend_config = type("cfg", (), {"port": 18000})
        served_model_id = "deepseek-coder-6.7b-instruct_(AWQ)"

    entry = DummyEntry()

    def fake_get_entry(name):
        if name == entry.name:
            return entry
        raise KeyError(name)

    monkeypatch.setattr(app_module, "get_entry", fake_get_entry, raising=True)
    monkeypatch.setattr(forwarder_module, "get_entry", fake_get_entry, raising=True)
    monkeypatch.setattr(registry_module, "get_entry", fake_get_entry, raising=True)
    monkeypatch.setattr(
        registry_module,
        "load_registry",
        lambda: {entry.name: entry},
    )

    probe_calls = {"count": 0}

    async def fake_probe(self, base_url):
        probe_calls["count"] += 1
        return probe_calls["count"] >= 2

    monkeypatch.setattr(forwarder_module.ChatForwarder, "_probe", fake_probe)

    async def fake_sleep(delay):
        return None

    monkeypatch.setattr(forwarder_module.asyncio, "sleep", fake_sleep)

    monkeypatch.setattr(app_module._cfg, "autostart_enabled", True)
    monkeypatch.setattr(app_module._forwarder.cfg, "autostart_enabled", True)
    monkeypatch.setattr(app_module._forwarder.cfg, "autostart_grace_period_s", 0)

    autostart_calls = []

    async def fake_ensure_started(model):
        autostart_calls.append(model)
        return True

    monkeypatch.setattr(
        app_module._forwarder.autostart,
        "ensure_started",
        fake_ensure_started,
    )

    async def fake_post(self, url, json=None):  # noqa: A002
        class Resp:
            status_code = 200

            def json(self_inner):
                return {
                    "id": "resp3",
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "Started"},
                        }
                    ],
                }

        return Resp()

    monkeypatch.setattr(forwarder_module.httpx.AsyncClient, "post", fake_post)

    client = TestClient(app)
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "deepseek coder 6.7b instruct (AWQ)",
            "messages": [{"role": "user", "content": "Hi"}],
        },
    )

    assert r.status_code == 200
    assert autostart_calls == [entry.name]
    assert probe_calls["count"] >= 2


def test_chat_forwarder_uses_backend_host_override(monkeypatch):
    from imageworks.chat_proxy import forwarder

    class DummyEntry:
        name = "llava"
        display_name = "llava"
        quantization = None
        backend = "vllm"
        probes = type("P", (), {"vision": None})
        chat_template = type("T", (), {"path": "templates/chat.jinja"})
        backend_config = type(
            "cfg", (), {"port": 12346, "host": "host.docker.internal"}
        )
        served_model_id = None

    entry = DummyEntry()

    monkeypatch.setattr(app_module, "get_entry", lambda name: entry, raising=True)
    monkeypatch.setattr(forwarder_module, "get_entry", lambda name: entry, raising=True)

    captured: dict[str, str] = {}

    async def fake_probe(self, base_url):
        captured["probe_base"] = base_url
        return True

    async def fake_post(self, url, json=None):  # noqa: A002
        captured["post_url"] = url

        class Resp:
            status_code = 200

            def json(self_inner):
                return {
                    "id": "resp-host",
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "Host"},
                        }
                    ],
                    "created": 1,
                    "model": "llava",
                }

        return Resp()

    monkeypatch.setattr(forwarder.ChatForwarder, "_probe", fake_probe)
    monkeypatch.setattr(forwarder.httpx.AsyncClient, "post", fake_post)

    client = TestClient(app)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "llava",
            "messages": [{"role": "user", "content": "Hi"}],
        },
    )

    assert resp.status_code == 200
    assert captured["probe_base"].startswith("http://host.docker.internal:12346")
    assert captured["post_url"].startswith(
        "http://host.docker.internal:12346/v1/chat/completions"
    )


def test_chat_forwarder_uses_base_url_override(monkeypatch):
    from imageworks.chat_proxy import forwarder

    class DummyEntry:
        name = "llava"
        display_name = "llava"
        quantization = None
        backend = "vllm"
        probes = type("P", (), {"vision": None})
        chat_template = type("T", (), {"path": "templates/chat.jinja"})
        backend_config = type(
            "cfg", (), {"port": 0, "base_url": "https://example.local/v1"}
        )
        served_model_id = None

    entry = DummyEntry()

    monkeypatch.setattr(app_module, "get_entry", lambda name: entry, raising=True)
    monkeypatch.setattr(forwarder_module, "get_entry", lambda name: entry, raising=True)

    captured: dict[str, str] = {}

    async def fake_probe(self, base_url):
        captured["probe_base"] = base_url
        return True

    async def fake_post(self, url, json=None):  # noqa: A002
        captured["post_url"] = url

        class Resp:
            status_code = 200

            def json(self_inner):
                return {
                    "id": "resp-base",
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "Base"},
                        }
                    ],
                    "created": 1,
                    "model": "llava",
                }

        return Resp()

    monkeypatch.setattr(forwarder.ChatForwarder, "_probe", fake_probe)
    monkeypatch.setattr(forwarder.httpx.AsyncClient, "post", fake_post)

    client = TestClient(app)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "llava",
            "messages": [{"role": "user", "content": "Hi"}],
        },
    )

    assert resp.status_code == 200
    assert captured["probe_base"] == "https://example.local/v1"
    assert captured["post_url"].startswith(
        "https://example.local/v1/chat/completions"
    )
