import asyncio
from types import SimpleNamespace

import imageworks.chat_proxy.forwarder as forwarder_module
from imageworks.chat_proxy.config import ProxyConfig
from imageworks.chat_proxy.autostart import AutostartManager
from imageworks.chat_proxy.metrics import MetricsAggregator
from imageworks.chat_proxy.forwarder import ChatForwarder


class DummyLogger:
    def __init__(self):
        self.records = []

    def log(self, record):
        self.records.append(record)


class DummyClient:
    def __init__(self, response_json):
        self.response_json = response_json
        self.posts = []

    async def post(self, url, json=None):  # noqa: A003
        self.posts.append((url, json))
        return SimpleNamespace(status_code=200, json=lambda: self.response_json)


class OllamaStub:
    def __init__(self):
        self.calls = []

    async def ensure_available(self):
        self.calls.append("ensure")

    async def unload_all(self):
        self.calls.append("unload")


class VllmStub:
    def __init__(self):
        self.calls = []

    async def deactivate(self):
        self.calls.append("deactivate")


def test_forwarder_unloads_ollama_for_non_ollama_backend(monkeypatch):
    cfg = ProxyConfig()
    cfg.autostart_enabled = False  # avoid probe/autostart work
    metrics = MetricsAggregator()
    autostart = AutostartManager(None, cfg, None)
    ollama_stub = OllamaStub()
    forwarder = ChatForwarder(cfg, metrics, autostart, DummyLogger(), None, ollama_stub)

    class DummyEntry:
        name = "dummy-model"
        display_name = "Dummy Model"
        backend = "vllm"
        probes = SimpleNamespace(vision=None)
        chat_template = SimpleNamespace(path="templates/dummy.jinja")
        backend_config = SimpleNamespace(port=12345)
        served_model_id = None
        metadata = {}
        capabilities = {}

    monkeypatch.setattr(
        forwarder_module, "get_entry", lambda name: DummyEntry(), raising=True
    )

    async def fake_probe(self, base_url):
        return True

    monkeypatch.setattr(
        forwarder_module.ChatForwarder, "_probe", fake_probe, raising=False
    )
    forwarder.client = DummyClient(
        {
            "id": "cmpl-dummy",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi"},
                }
            ],
            "usage": {"completion_tokens": 4},
        }
    )

    payload = {
        "model": "dummy-model",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
    }

    async def run():
        result = await forwarder.handle_chat(payload)
        assert isinstance(result, tuple)

    asyncio.run(run())
    # Non-streaming path returns a tuple payload
    assert ollama_stub.calls == ["unload"]


def test_forwarder_unloads_when_switching_from_ollama(monkeypatch):
    cfg = ProxyConfig()
    cfg.autostart_enabled = False
    metrics = MetricsAggregator()
    autostart = AutostartManager(None, cfg, None)
    ollama_stub = OllamaStub()
    forwarder = ChatForwarder(cfg, metrics, autostart, DummyLogger(), None, ollama_stub)
    forwarder._last_backend = "ollama"

    class DummyEntry:
        name = "dummy-model"
        display_name = "Dummy Model"
        backend = "vllm"
        probes = SimpleNamespace(vision=None)
        chat_template = SimpleNamespace(path="templates/dummy.jinja")
        backend_config = SimpleNamespace(port=12345)
        served_model_id = None
        metadata = {}
        capabilities = {}

    monkeypatch.setattr(
        forwarder_module, "get_entry", lambda name: DummyEntry(), raising=True
    )

    async def fake_probe(self, base_url):
        return True

    monkeypatch.setattr(
        forwarder_module.ChatForwarder, "_probe", fake_probe, raising=False
    )
    forwarder.client = DummyClient(
        {
            "id": "cmpl-dummy",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi"},
                }
            ],
            "usage": {"completion_tokens": 4},
        }
    )

    payload = {
        "model": "dummy-model",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
    }

    async def run():
        result = await forwarder.handle_chat(payload)
        assert isinstance(result, tuple)

    asyncio.run(run())
    assert ollama_stub.calls == ["unload"]


def test_forwarder_deactivates_vllm_before_ollama(monkeypatch):
    cfg = ProxyConfig()
    cfg.autostart_enabled = False
    metrics = MetricsAggregator()
    vllm_stub = VllmStub()
    ollama_stub = OllamaStub()
    autostart = AutostartManager(None, cfg, vllm_stub)
    forwarder = ChatForwarder(
        cfg, metrics, autostart, DummyLogger(), vllm_stub, ollama_stub
    )

    class DummyEntry:
        name = "ollama-model"
        display_name = "Ollama Model"
        backend = "ollama"
        probes = SimpleNamespace(vision=None)
        chat_template = None
        backend_config = SimpleNamespace(port=11434)
        served_model_id = None
        metadata = {}
        capabilities = {}

    monkeypatch.setattr(
        forwarder_module, "get_entry", lambda name: DummyEntry(), raising=True
    )

    async def fake_probe(self, base_url):
        return True

    monkeypatch.setattr(
        forwarder_module.ChatForwarder, "_probe", fake_probe, raising=False
    )
    forwarder.client = DummyClient(
        {
            "id": "ollama-cmpl",
            "message": {"role": "assistant", "content": "hello"},
        }
    )

    payload = {
        "model": "ollama-model",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
    }

    forwarder._last_backend = "vllm"

    async def run():
        result = await forwarder.handle_chat(payload)
        assert isinstance(result, tuple)

    asyncio.run(run())
    assert vllm_stub.calls == ["deactivate"]
    assert ollama_stub.calls == ["ensure"]
