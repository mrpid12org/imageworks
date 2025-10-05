from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def supports_vision(entry: Any) -> bool:
    """Return True when registry entry indicates vision support.

    Prefer explicit probe results when available; otherwise fall back to
    declared capabilities. This keeps chat routing strict when a probe has
    confirmed failure while still allowing curated metadata to unlock
    multimodal models before probes run.
    """

    probes = getattr(entry, "probes", None)
    vision_probe = getattr(probes, "vision", None) if probes else None
    if vision_probe is not None:
        return bool(getattr(vision_probe, "vision_ok", False))

    capabilities = getattr(entry, "capabilities", None)
    if isinstance(capabilities, Mapping):
        return bool(capabilities.get("vision"))

    return False
