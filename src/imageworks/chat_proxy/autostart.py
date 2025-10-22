from __future__ import annotations

import asyncio
import json
import subprocess
from typing import Dict, List, Optional

from .config import ProxyConfig
from .vllm_manager import VllmManager, VllmActivationError


class AutostartManager:
    def __init__(
        self,
        raw_map: Optional[str],
        cfg: Optional[ProxyConfig] = None,
        vllm_manager: VllmManager | None = None,
    ):
        self.lock_map: Dict[str, asyncio.Lock] = {}
        self.start_map: Dict[str, Dict[str, List[str]]] = {}
        self.cfg = cfg
        self.vllm_manager = vllm_manager
        if raw_map:
            try:
                self.start_map = json.loads(raw_map)
            except Exception:  # noqa: BLE001
                self.start_map = {}

    def _lock(self, model: str) -> asyncio.Lock:
        if model not in self.lock_map:
            self.lock_map[model] = asyncio.Lock()
        return self.lock_map[model]

    async def ensure_started(self, model: str, entry) -> bool:
        if (
            self.cfg
            and self.cfg.vllm_single_port
            and getattr(entry, "backend", None) == "vllm"
            and self.vllm_manager
        ):
            try:
                await self.vllm_manager.activate(entry)
                return True
            except VllmActivationError:
                raise
            except Exception:  # noqa: BLE001
                return False

        cfg = self.start_map.get(model)
        if not cfg:
            return False
        lock = self._lock(model)
        if lock.locked():  # Another task starting
            # Wait for completion
            async with lock:
                return True
        async with lock:
            try:
                cmd = cfg.get("command")
                if not cmd:
                    return False
                subprocess.Popen(
                    cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )  # noqa: S603, S607
                return True
            except Exception:  # noqa: BLE001
                return False
