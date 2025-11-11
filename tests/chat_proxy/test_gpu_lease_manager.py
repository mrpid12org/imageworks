import asyncio
from types import SimpleNamespace

import pytest

from imageworks.chat_proxy.gpu_leasing import (
    GpuLeaseManager,
    LeaseBusyError,
    LeaseTokenError,
)
from imageworks.chat_proxy.vllm_manager import ActiveVllmState


class FakeVllmManager:
    def __init__(self, state: ActiveVllmState | None):
        self.state = state
        self.deactivated = False
        self.activated_entries: list[str] = []

    async def current_state(self):
        return self.state if not self.deactivated else None

    async def deactivate(self):
        self.deactivated = True

    async def activate(self, entry):
        self.activated_entries.append(entry.name)


def test_acquire_without_active_model():
    async def _run():
        manager = GpuLeaseManager(FakeVllmManager(None))
        lease = await manager.acquire(owner="tester")
        assert lease.saved_model is None
        status = await manager.status()
        assert status["leased"] is True

    asyncio.run(_run())


def test_acquire_fails_when_busy():
    async def _run():
        manager = GpuLeaseManager(FakeVllmManager(None))
        await manager.acquire(owner="tester")
        with pytest.raises(LeaseBusyError):
            await manager.acquire(owner="other")

    asyncio.run(_run())


def test_release_restarts_previous_model():
    async def _run():
        state = ActiveVllmState("model-alpha", "served", 8000, 1234, 0.0)
        fake_entry = SimpleNamespace(name="model-alpha", backend="vllm")
        manager = GpuLeaseManager(
            FakeVllmManager(state),
            entry_resolver=lambda name: fake_entry,
        )
        lease = await manager.acquire(owner="tester")
        assert lease.saved_model.logical_name == "model-alpha"

        await manager.release(lease.token)

        status = await manager.status()
        assert status["leased"] is False

    asyncio.run(_run())


def test_invalid_release_token():
    async def _run():
        manager = GpuLeaseManager(FakeVllmManager(None))
        lease = await manager.acquire(owner="tester")
        with pytest.raises(LeaseTokenError):
            await manager.release("wrong-token")
        await manager.release(lease.token)

    asyncio.run(_run())
