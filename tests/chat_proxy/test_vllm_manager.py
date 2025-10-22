import time
from types import SimpleNamespace

import pytest

from imageworks.chat_proxy.config import ProxyConfig
from imageworks.chat_proxy import forwarder as forwarder_module
from imageworks.chat_proxy.autostart import AutostartManager
from imageworks.chat_proxy.logging_utils import JsonlLogger
from imageworks.chat_proxy.metrics import MetricsAggregator
from imageworks.chat_proxy.vllm_manager import (
    ActiveVllmState,
    VllmActivationError,
    VllmManager,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _dummy_entry(name: str = "qwen-mini") -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        backend="vllm",
        backend_config=SimpleNamespace(
            model_path=None, extra_args=[], host="127.0.0.1"
        ),
        chat_template=SimpleNamespace(path=None),
        download_path=None,
        served_model_id=None,
    )


async def test_activate_reuses_running_model(tmp_path, monkeypatch):
    cfg = ProxyConfig(vllm_state_path=str(tmp_path / "state.json"), vllm_port=25055)
    manager = VllmManager(cfg)
    alive: set[int] = set()
    launch_calls: list[int] = []

    async def fake_launch(self, entry, served_model_id, port):
        pid = 4242 + len(launch_calls)
        launch_calls.append(pid)
        alive.add(pid)
        return pid

    async def fake_wait(self, port, started_at):
        return True

    async def fake_terminate(self, pid, force=False):
        alive.discard(pid)

    def fake_alive(self, pid):
        return pid in alive

    monkeypatch.setattr(VllmManager, "_launch", fake_launch, raising=False)
    monkeypatch.setattr(VllmManager, "_wait_for_health", fake_wait, raising=False)
    monkeypatch.setattr(
        VllmManager, "_terminate_process", fake_terminate, raising=False
    )
    monkeypatch.setattr(VllmManager, "_process_alive", fake_alive, raising=False)

    entry = _dummy_entry()
    try:
        state1 = await manager.activate(entry)
        assert isinstance(state1, ActiveVllmState)
        assert state1.pid in alive
        assert state1.port == cfg.vllm_port
        assert (tmp_path / "state.json").exists()

        state2 = await manager.activate(entry)
        assert state2.pid == state1.pid
        assert launch_calls.count(state1.pid) == 1

        current = await manager.current_state()
        assert current and current.logical_name == entry.name

        await manager.deactivate()
        assert not (tmp_path / "state.json").exists()
        assert not alive
    finally:
        await manager.aclose()


async def test_forwarder_invokes_vllm_manager(tmp_path, monkeypatch):
    cfg = ProxyConfig(
        log_path=str(tmp_path / "chat.log"),
        vllm_state_path=str(tmp_path / "state.json"),
        vllm_port=27001,
        autostart_enabled=False,
        vllm_single_port=True,
    )

    class DummyManager:
        def __init__(self, cfg):
            self.cfg = cfg
            self.calls: list[str] = []

        async def activate(self, entry):
            self.calls.append(entry.name)
            return ActiveVllmState(
                logical_name=entry.name,
                served_model_id=entry.name,
                port=self.cfg.vllm_port,
                pid=9100,
                started_at=time.time(),
            )

        async def deactivate(self):
            return None

        async def current_state(self):
            return None

        async def aclose(self):
            return None

    dummy_manager = DummyManager(cfg)
    metrics = MetricsAggregator()
    autostart = AutostartManager(None, cfg, dummy_manager)
    logger = JsonlLogger(str(tmp_path / "proxy.log"))
    forwarder = forwarder_module.ChatForwarder(
        cfg, metrics, autostart, logger, dummy_manager
    )

    entry = SimpleNamespace(
        name="switcher",
        display_name="switcher",
        backend="vllm",
        probes=SimpleNamespace(vision=None),
        chat_template=SimpleNamespace(path="templates/chat.jinja"),
        backend_config=SimpleNamespace(port=23456, host="127.0.0.1", base_url=None),
        served_model_id=None,
        quantization=None,
    )

    def fake_get_entry(name):
        if name == entry.name:
            return entry
        raise KeyError(name)

    async def fake_probe(self, base_url):
        return True

    class FakeResp:
        status_code = 200

        def json(self):
            return {
                "id": "resp-switch",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                    }
                ],
                "usage": {"completion_tokens": 4},
            }

    async def fake_post(self, url, json=None):  # noqa: A002
        return FakeResp()

    monkeypatch.setattr(forwarder_module, "get_entry", fake_get_entry, raising=True)
    monkeypatch.setattr(forwarder_module.ChatForwarder, "_probe", fake_probe)
    monkeypatch.setattr(forwarder_module.httpx.AsyncClient, "post", fake_post)

    try:
        result = await forwarder.handle_chat(
            {"model": entry.name, "messages": [{"role": "user", "content": "hey"}]}
        )
        assert dummy_manager.calls == [entry.name]
        norm, backend, backend_id, *_ = result
        assert backend == entry.backend
        assert backend_id == entry.name
        assert norm["choices"][0]["message"]["content"] == "ok"
    finally:
        await forwarder.aclose()


async def test_activate_health_failure_cleans_state(tmp_path, monkeypatch):
    cfg = ProxyConfig(vllm_state_path=str(tmp_path / "state.json"), vllm_port=25056)
    manager = VllmManager(cfg)
    alive: set[int] = set()
    terminated: list[int] = []

    async def fake_launch(self, entry, served_model_id, port):
        pid = 5151
        alive.add(pid)
        return pid

    async def fake_wait(self, port, started_at):
        return False

    async def fake_terminate(self, pid, force=False):
        terminated.append(pid)
        alive.discard(pid)

    def fake_alive(self, pid):
        return pid in alive

    monkeypatch.setattr(VllmManager, "_launch", fake_launch, raising=False)
    monkeypatch.setattr(VllmManager, "_wait_for_health", fake_wait, raising=False)
    monkeypatch.setattr(
        VllmManager, "_terminate_process", fake_terminate, raising=False
    )
    monkeypatch.setattr(VllmManager, "_process_alive", fake_alive, raising=False)

    entry = _dummy_entry("qwen-fail")
    try:
        with pytest.raises(VllmActivationError):
            await manager.activate(entry)
        assert not (tmp_path / "state.json").exists()
        assert alive == set()
        assert terminated == [5151]
    finally:
        await manager.aclose()
