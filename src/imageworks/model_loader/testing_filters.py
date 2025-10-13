from __future__ import annotations

import os
import re


DEFAULT_TEST_PATTERNS: tuple[str, ...] = (
    # Generic test prefixes used in unit/integration suites
    r"^testmodel(.*)$",
    r"^mini-test(.*)$",
    r"^custom-family(.*)$",
    # Synthetic/local logical-only/testing variants
    r"^synthetic-test($|-).*",
    r"^logical-only-test($|-).*",
    # Canonical explicit test tag: any name containing 'testzzz'
    r"^.*testzzz.*$",
    # Placeholder/demo variants that sometimes acquire suffix tokens (backend/format/quant)
    r"^model-awq($|-).*",
    r"^model-fp16($|-).*",
    r"^demo-model($|-).*",
    # Historical single-letter placeholder, allow common suffix pattern
    r"^r($|-).*",
)


def _load_extra_patterns() -> list[re.Pattern[str]]:
    raw = os.environ.get("IMAGEWORKS_TEST_MODEL_PATTERNS", "").strip()
    if not raw:
        return []
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return [re.compile(p) for p in parts]


_DEFAULT_RES = [re.compile(p) for p in DEFAULT_TEST_PATTERNS]
_EXTRA_RES = _load_extra_patterns()


def is_testing_name(name: str) -> bool:
    nm = name.strip()
    for rx in _DEFAULT_RES:
        if rx.match(nm):
            return True
    for rx in _EXTRA_RES:
        if rx.match(nm):
            return True
    return False


def is_testing_entry(name: str, entry) -> bool:  # entry is a RegistryEntry-like object
    # First check by name patterns
    if is_testing_name(name):
        return True
    # Allow explicit opt-in via metadata.testing flag
    try:
        meta = getattr(entry, "metadata", {}) or {}
        if isinstance(meta, dict) and meta.get("testing") is True:
            return True
    except Exception:  # noqa: BLE001
        pass
    return False


__all__ = ["is_testing_entry", "is_testing_name", "DEFAULT_TEST_PATTERNS"]
