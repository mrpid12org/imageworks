from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx

from .config import ProxyConfig

logger = logging.getLogger("imageworks.ollama_manager")


class OllamaManager:
    """Controller that can release GPU memory held by Ollama."""

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

    async def list_installed_models(self) -> set[str]:
        """Return the set of model names reported by `ollama list`/`/api/tags`."""

        try:
            resp = await self._http.get(f"{self.base_url}/api/tags")
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("[ollama-manager] Failed to query installed tags: %s", exc)
            return set()

        models = payload.get("models") or []
        names: set[str] = set()
        for model in models:
            name = model.get("name") or model.get("model")
            if not name:
                continue
            names.add(str(name).strip().lower())
        return names

    async def unload_all(self) -> int:
        """Request Ollama to stop all loaded models to free GPU memory.

        Returns:
            Number of models successfully stopped.

        Raises:
            RuntimeError: if any model fails to stop or Ollama never releases VRAM.
        """

        async with self._lock:
            models = await self._list_running_models()
            if not models:
                return 0

            target_names = []
            stop_errors: list[str] = []
            for model in models:
                name = model.get("name")
                if not name:
                    continue
                target_names.append(name)
                logger.info("[ollama-manager] Requesting unload for model '%s'", name)
                success = await self._stop_model_keepalive(name)
                if not success:
                    stop_errors.append(f"{name}: keep_alive unload failed")
                    continue

            if stop_errors:
                raise RuntimeError(
                    "[ollama-manager] Failed to unload model(s): "
                    + "; ".join(stop_errors)
                )

            if not await self._wait_until_idle(target_names, self.stop_timeout_s):
                raise RuntimeError(
                    "[ollama-manager] Timeout waiting for Ollama to release GPU memory "
                    f"for model(s): {', '.join(target_names)}"
                )

            logger.info(
                "[ollama-manager] Confirmed Ollama released %d model(s): %s",
                len(target_names),
                ", ".join(target_names),
            )
            return len(target_names)

    async def _list_running_models(self) -> list[dict[str, Any]]:
        try:
            resp = await self._http.get(f"{self.base_url}/api/ps")
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to query Ollama processes: {exc}") from exc
        models = payload.get("models") or []
        return models if isinstance(models, list) else []

    async def _wait_until_idle(self, names: list[str], timeout: float) -> bool:
        deadline = time.time() + max(timeout, 1)
        while time.time() < deadline:
            running = await self._list_running_models()
            remaining = [model for model in running if model.get("name") in set(names)]
            if not remaining:
                return True
            await asyncio.sleep(0.5)
        return False

    async def _stop_model_keepalive(self, name: str) -> bool:
        """Fallback: make a no-op request with keep_alive=0 so Ollama unloads."""

        generate_payload = {
            "model": name,
            "prompt": "release keepalive",
            "stream": False,
            "raw": True,
            "keep_alive": 0,
            "options": {"num_predict": 0},
        }
        chat_payload = {
            "model": name,
            "messages": [{"role": "user", "content": "release keepalive"}],
            "stream": False,
            "keep_alive": 0,
            "options": {"num_predict": 0},
        }
        for endpoint, payload in (
            ("generate", generate_payload),
            ("chat", chat_payload),
        ):
            try:
                resp = await self._http.post(
                    f"{self.base_url}/api/{endpoint}", json=payload
                )
                if resp.status_code >= 400:
                    text = (resp.text or "").strip()
                    logger.warning(
                        "[ollama-manager] keep_alive fallback via /api/%s failed for "
                        "'%s': HTTP %s %s",
                        endpoint,
                        name,
                        resp.status_code,
                        text[:200],
                    )
                    continue
                return True
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[ollama-manager] keep_alive fallback exception for '%s' via "
                    "/api/%s: %s",
                    name,
                    endpoint,
                    exc,
                )
        return False
