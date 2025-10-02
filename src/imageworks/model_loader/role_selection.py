"""Role-based model selection helpers using unified registry.

Provides convenience APIs to select a model by functional role (e.g. caption, keywords,
description) while enforcing capability requirements (vision by default for tagger roles).
"""

from __future__ import annotations

from typing import Iterable, List, Optional

from .registry import load_registry
from .service import CapabilityError

DEFAULT_ROLE_CAPABILITIES = {
    "caption": ["vision"],
    "keywords": ["vision"],
    "description": ["vision"],
    "narration": ["vision"],
}


def list_models_for_role(role: str) -> List[str]:
    reg = load_registry()
    return sorted(
        [
            name
            for name, entry in reg.items()
            if role in (entry.roles or []) and not entry.deprecated
        ]
    )


def select_by_role(
    role: str,
    *,
    require_capabilities: Optional[List[str]] = None,
    preferred: Optional[Iterable[str]] = None,
) -> str:
    """Return the logical model name for the given role.

    Resolution order:
      1. preferred list (first that exists & matches role)
      2. first non-deprecated registry entry advertising the role
    Capability requirements merged from DEFAULT_ROLE_CAPABILITIES and user provided list.
    """
    reg = load_registry()
    req_caps = list(DEFAULT_ROLE_CAPABILITIES.get(role, []))
    if require_capabilities:
        for c in require_capabilities:
            if c not in req_caps:
                req_caps.append(c)

    def _is_valid(name: str) -> bool:
        entry = reg.get(name)
        if not entry:
            return False
        if role not in (entry.roles or []):
            return False
        if entry.deprecated:
            return False
        # capabilities check
        for cap in req_caps:
            if not entry.capabilities.get(cap):
                return False
        return True

    # preferred first
    if preferred:
        for candidate in preferred:
            if _is_valid(candidate):
                return candidate

    # fallback to any matching entry
    for name, entry in reg.items():
        if _is_valid(name):
            return name

    raise CapabilityError(
        f"No model found for role '{role}' meeting capabilities {req_caps}"
    )


__all__ = ["list_models_for_role", "select_by_role", "DEFAULT_ROLE_CAPABILITIES"]
