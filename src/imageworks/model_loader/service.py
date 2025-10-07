"""Selection service skeleton for deterministic model serving."""

from __future__ import annotations

import logging
from typing import List, Optional

from .registry import get_entry, load_registry
from .models import SelectedModel, normalize_capabilities

logger = logging.getLogger(__name__)


class CapabilityError(RuntimeError):
    pass


def select_model(
    name: str, *, require_capabilities: Optional[List[str]] = None
) -> SelectedModel:
    """Lookup a model and return a connection descriptor.

    Minimal Phase 1: no process launching yet; returns derived endpoint from port.
    """
    entry = get_entry(name)
    capabilities = normalize_capabilities(entry.capabilities)

    if require_capabilities:
        missing = [
            cap for cap in require_capabilities if not capabilities.get(cap.strip().lower(), False)
        ]
        if missing:
            raise CapabilityError(
                f"Model '{name}' is missing required capabilities: {', '.join(missing)}"
            )

    # Endpoint synthesis differs slightly by backend; for now vllm/lmdeploy/ollama all expose OpenAI-compatible /v1.
    # Future backends could branch here if they require a proxy path.
    if entry.backend in {"vllm", "lmdeploy", "ollama"}:
        endpoint = f"http://localhost:{entry.backend_config.port}/v1"
    else:
        endpoint = f"http://localhost:{entry.backend_config.port}/v1"

    logger.info(
        "model_select",
        extra={
            "event_type": "select",
            "model": name,
            "backend": entry.backend,
            "endpoint": endpoint,
        },
    )

    internal_id = entry.served_model_id or name
    return SelectedModel(
        logical_name=name,
        endpoint_url=endpoint,
        internal_model_id=internal_id,
        backend=entry.backend,
        capabilities=capabilities,
    )


__all__ = ["select_model", "CapabilityError", "load_registry"]
