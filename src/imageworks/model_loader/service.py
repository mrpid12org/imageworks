"""Selection service skeleton for deterministic model serving."""

from __future__ import annotations

import logging
from typing import List, Optional

from .registry import get_entry, load_registry
from .models import SelectedModel, normalize_capabilities

logger = logging.getLogger(__name__)


class CapabilityError(RuntimeError):
    pass


def _resolve_endpoint(entry) -> str:
    cfg = getattr(entry, "backend_config", None)
    base_override = None
    if cfg is not None:
        raw_base = getattr(cfg, "base_url", None)
        if isinstance(raw_base, str):
            stripped = raw_base.strip()
            if stripped:
                base_override = stripped.rstrip("/")
    host_override = None
    if cfg is not None:
        raw_host = getattr(cfg, "host", None)
        if isinstance(raw_host, str):
            stripped_host = raw_host.strip()
            if stripped_host:
                host_override = stripped_host

    backend = getattr(entry, "backend", "")
    default_port = 8000
    if backend == "ollama":
        default_port = 11434
    elif backend == "lmdeploy":
        default_port = 24001
    elif backend == "triton":
        default_port = 9000

    port = getattr(cfg, "port", None) if cfg is not None else None
    if not isinstance(port, int) or port <= 0:
        port = default_port

    if base_override:
        endpoint = base_override
    elif host_override and host_override.startswith(("http://", "https://")):
        endpoint = host_override.rstrip("/")
    else:
        host = host_override or "localhost"
        endpoint = f"http://{host}:{port}/v1"

    return endpoint.rstrip("/")


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
    endpoint = _resolve_endpoint(entry)

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
