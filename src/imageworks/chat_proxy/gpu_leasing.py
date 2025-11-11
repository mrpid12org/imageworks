"""GPU lease coordination for exclusive access to the accelerator."""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from dataclasses import dataclass, asdict
from typing import Callable, Optional

from imageworks.model_loader.registry import get_entry

from .vllm_manager import ActiveVllmState, VllmManager


class LeaseBusyError(RuntimeError):
    """Raised when a lease is already held."""


class LeaseTokenError(RuntimeError):
    """Raised when a release is attempted with an invalid token."""


@dataclass
class LeaseState:
    token: str
    owner: str
    reason: str | None
    granted_at: float
    restart_model: bool
    saved_model: Optional[ActiveVllmState]

    def to_dict(self) -> dict:
        payload = asdict(self)
        if self.saved_model:
            payload["saved_model"] = self.saved_model.to_dict()
        else:
            payload["saved_model"] = None
        return payload


class GpuLeaseManager:
    """Coordinate exclusive GPU usage between Judge Vision and chat proxy."""

    def __init__(
        self,
        vllm_manager: VllmManager,
        entry_resolver: Callable[[str], object] = get_entry,
        stale_max_age: float | None = None,
    ):
        self._vllm_manager = vllm_manager
        self._entry_resolver = entry_resolver
        self._lock = asyncio.Lock()
        self._current: LeaseState | None = None
        default_max_age = float(os.getenv("GPU_LEASE_STALE_MAX_AGE", "900"))
        self._stale_max_age = (
            stale_max_age if stale_max_age is not None else default_max_age
        )

    async def acquire(
        self,
        *,
        owner: str,
        reason: str | None = None,
        restart_model: bool = True,
    ) -> LeaseState:
        await self._prune_stale(owner=owner)

        async with self._lock:
            if self._current is not None:
                raise LeaseBusyError("GPU already leased")

            saved_model = await self._vllm_manager.current_state()
            if saved_model is not None:
                await self._vllm_manager.deactivate()

            lease = LeaseState(
                token=uuid.uuid4().hex,
                owner=owner,
                reason=reason,
                granted_at=time.time(),
                restart_model=restart_model,
                saved_model=saved_model if restart_model else None,
            )
            self._current = lease
            return lease

    async def release(self, token: str, *, restart_model: bool = True) -> None:
        async with self._lock:
            lease = self._current
            if lease is None or lease.token != token:
                raise LeaseTokenError("GPU lease token invalid or expired")
            self._current = None
        await self._restore_saved_model(lease, restart_model)

    async def status(self) -> dict:
        async with self._lock:
            if self._current is None:
                return {"leased": False}
            return {
                "leased": True,
                "lease": self._current.to_dict(),
            }

    async def force_release(
        self,
        *,
        token: str | None = None,
        owner: str | None = None,
        max_age: float | None = None,
        restart_model: bool = True,
    ) -> bool:
        async with self._lock:
            lease = self._current
            if lease is None:
                return False
            if token and lease.token != token:
                raise LeaseTokenError("GPU lease token invalid or expired")
            if owner and lease.owner != owner:
                raise LeaseTokenError("GPU lease token invalid or expired")
            if max_age is not None:
                age = time.time() - lease.granted_at
                if age < max_age:
                    raise LeaseTokenError("GPU lease younger than allowed max_age")
            self._current = None

        await self._restore_saved_model(lease, restart_model)
        return True

    async def _prune_stale(self, *, owner: str | None = None) -> None:
        if not self._stale_max_age or self._stale_max_age <= 0:
            return
        async with self._lock:
            lease = self._current
            if lease is None:
                return
            if owner and lease.owner != owner:
                return
            age = time.time() - lease.granted_at
            if age <= self._stale_max_age:
                return
            self._current = None
        await self._restore_saved_model(lease, restart_model=True)

    async def _restore_saved_model(
        self, lease: LeaseState, restart_model: bool
    ) -> None:
        if restart_model and lease.restart_model and lease.saved_model:
            entry = self._entry_resolver(lease.saved_model.logical_name)
            await self._vllm_manager.activate(entry)
