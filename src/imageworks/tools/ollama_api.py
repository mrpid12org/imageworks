from __future__ import annotations

import os
from contextlib import AbstractContextManager
from typing import Any, Iterable, Optional

import httpx


class OllamaError(RuntimeError):
    """Raised when the Ollama HTTP API cannot be reached or returns an error."""


def _default_base_url() -> str:
    return (
        os.environ.get("OLLAMA_BASE_URL")
        or os.environ.get("OLLAMA_HOST")
        or "http://127.0.0.1:11434"
    )


class OllamaClient(AbstractContextManager["OllamaClient"]):
    """Small synchronous client for the Ollama HTTP API."""

    def __init__(self, base_url: Optional[str] = None, timeout: float = 3.0) -> None:
        self.base_url = (base_url or _default_base_url()).rstrip("/")
        self._client = httpx.Client(timeout=timeout)

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401
        self.close()

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:  # noqa: BLE001
            pass

    def list_models(self) -> list[dict[str, Any]]:
        """Return the list of models reported by the Ollama daemon."""

        url = f"{self.base_url}/api/tags"
        try:
            resp = self._client.get(url)
            resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            raise OllamaError(f"Failed to GET {url}: {exc}") from exc

        payload = self._decode_json(resp, "list")
        models = payload.get("models")
        if not isinstance(models, Iterable):
            raise OllamaError("Malformed /api/tags response: missing 'models' array")
        result: list[dict[str, Any]] = []
        for item in models:
            if isinstance(item, dict) and item.get("name"):
                result.append(item)
        return result

    def show_model(self, name: str) -> dict[str, Any]:
        """Return detailed metadata for a single model."""

        url = f"{self.base_url}/api/show"
        try:
            resp = self._client.post(url, json={"model": name})
            resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            raise OllamaError(
                f"Failed to POST {url} for model '{name}': {exc}"
            ) from exc
        data = self._decode_json(resp, "show")
        if not isinstance(data, dict):
            raise OllamaError(f"Malformed /api/show response for '{name}'")
        return data

    @staticmethod
    def _decode_json(response: httpx.Response, op: str) -> dict[str, Any]:
        try:
            data = response.json()
        except Exception as exc:  # noqa: BLE001
            raise OllamaError(
                f"Failed to decode JSON from Ollama {op} response: {exc}"
            ) from exc
        if not isinstance(data, dict):
            raise OllamaError(f"Ollama {op} response was not an object")
        return data


__all__ = ["OllamaClient", "OllamaError"]
