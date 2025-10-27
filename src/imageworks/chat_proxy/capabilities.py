from __future__ import annotations

from collections.abc import Iterable, Mapping
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
        if capabilities.get("vision"):
            return True
        # Some registries store related markers such as multimodal/image flags.
        for token in ("multimodal", "mm", "image", "visual"):
            if capabilities.get(token):
                return True

    backend = getattr(entry, "backend", "")
    metadata = getattr(entry, "metadata", None) or {}
    if isinstance(metadata, Mapping):
        caps = metadata.get("ollama_capabilities")
        if isinstance(caps, Iterable) and not isinstance(caps, (str, bytes)):
            for cap in caps:
                if isinstance(cap, bytes):
                    try:
                        cap = cap.decode("utf-8", "ignore")
                    except Exception:  # noqa: BLE001
                        continue
                if isinstance(cap, str) and cap.strip().lower() == "vision":
                    return True

    if isinstance(backend, str) and backend.lower() == "ollama":
        # As a last resort, infer from naming conventions.
        label = str(getattr(entry, "display_name", "") or getattr(entry, "name", ""))
        label_l = label.lower()
        vision_markers = (
            " vision",
            "-vision",
            " pixtral",
            " llava",
            " minicpm-v",
            " internvl",
            " moondream",
            " phi-3-vision",
            "-vl",
            " vl ",
        )
        if any(marker in label_l for marker in vision_markers):
            return True

    return False


def supports_reasoning(entry: Any) -> bool:
    """Return True when registry entry indicates reasoning/thinking support.

    Reasoning models (like o1, deepseek-r1, gpt-oss) produce extended
    chain-of-thought outputs that can consume large amounts of context.
    """
    capabilities = getattr(entry, "capabilities", None)
    if isinstance(capabilities, Mapping):
        # Check for explicit reasoning markers
        reasoning_markers = (
            "thinking",
            "reasoning",
            "reason",
            "think",
            "chain_of_thought",
            "cot",
            "reasoner",
            "o1",
            "o3",
            "r1",
            "deepseek",
        )
        for marker in reasoning_markers:
            if capabilities.get(marker):
                return True

    return False
