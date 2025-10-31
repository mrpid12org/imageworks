from __future__ import annotations

import asyncio
import logging

import httpx

from .config import ProxyConfig

logger = logging.getLogger("imageworks.ollama_manager")


class OllamaManager:
    """Best-effort controller that can release GPU memory held by Ollama."""

    def __init__(self, cfg: ProxyConfig):
        self.base_url = cfg.ollama_base_url.rstrip("/")
        self.stop_timeout_s = cfg.ollama_stop_timeout_s
        timeout = max(cfg.backend_timeout_ms / 1000, 10)
        self._http = httpx.AsyncClient(timeout=timeout)
        # Guard concurrent stop operations so multiple requests do not race.
        self._lock = asyncio.Lock()

    async def aclose(self) -> None:
        try:
            await self._http.aclose()
        except Exception:  # noqa: BLE001
            pass

    async def ensure_available(self) -> bool:
        """Ping the Ollama server to confirm it is reachable."""
        try:
            resp = await self._http.get(f"{self.base_url}/api/version")
            resp.raise_for_status()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("[ollama-manager] Health check failed: %s", exc)
            return False

    async def unload_all(self) -> None:
        """Request Ollama to stop all loaded models to free GPU memory."""

        async with self._lock:
            try:
                resp = await self._http.get(f"{self.base_url}/api/ps")
                resp.raise_for_status()
                payload = resp.json()
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[ollama-manager] Failed to query running models: %s", exc
                )
                return

            models = payload.get("models") or []
            if not models:
                return

            tasks = []
            for model in models:
                name = model.get("name")
                if not name:
                    continue
                logger.info("[ollama-manager] Requesting unload for model '%s'", name)
                tasks.append(
                    self._http.post(f"{self.base_url}/api/stop", json={"name": name})
                )

            if not tasks:
                return

            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True), self.stop_timeout_s
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "[ollama-manager] Timeout while waiting for Ollama to unload models"
                )
