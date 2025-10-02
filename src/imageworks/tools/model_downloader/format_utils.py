"""Shared format and quantization detection utilities for model directories.

This module centralises logic used by scan, downloader initial detection, and
normalization / rebuild passes so behaviour is consistent and testable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, List, Dict
import json
import re

# Priority order of formats we attempt to detect. Earlier = higher priority.
_FORMAT_PRIORITY = ["gguf", "awq", "gptq", "fp16"]

_AWQ_HINT_PATTERNS = [
    re.compile(r"\.awq$"),
    re.compile(r"[-_/]awq[-_]?", re.IGNORECASE),
]
_GGUF_SUFFIX = ".gguf"
_GPTQ_HINT = "gptq"
_SAFETENSORS_SUFFIX = ".safetensors"

# Quant filename tokens (lowercased). Maps canonical token -> list of patterns.
_QUANT_FILENAME_MAP: Dict[str, List[re.Pattern]] = {
    "q4_k_m": [re.compile(r"q4[_-]?k[_-]?m")],
    "q5_k_m": [re.compile(r"q5[_-]?k[_-]?m")],
    "q6_k": [re.compile(r"q6[_-]?k")],
    "int4": [re.compile(r"int4")],
    "int8": [re.compile(r"int8")],
}


def detect_format_and_quant(path: Path) -> Tuple[Optional[str], Optional[str]]:
    """Detect (format, quant) for a model directory.

    Returns:
        (format, quantization) where each may be None if not detectable.
    """
    if not path.exists() or not path.is_dir():  # fast fail
        return None, None

    # Collect file names (lowercase for heuristic matching)
    files = [p for p in path.rglob("*") if p.is_file()]
    lower_names = [p.name.lower() for p in files]
    repo_name_lower = path.name.lower()

    fmt: Optional[str] = None
    quant: Optional[str] = None

    # 1. GGUF
    if any(n.endswith(_GGUF_SUFFIX) for n in lower_names):
        fmt = "gguf"

    # 2. AWQ indicators (only if fmt still None)
    if fmt is None:
        if (
            any(p.search(n) for p in _AWQ_HINT_PATTERNS for n in lower_names)
            or "awq" in repo_name_lower
        ):
            fmt = "awq"

    # 3. GPTQ
    if fmt is None:
        if any(_GPTQ_HINT in n for n in lower_names):
            fmt = "gptq"

    # 4. Safetensors -> fp16 (last resort for these heuristics)
    if fmt is None:
        if any(n.endswith(_SAFETENSORS_SUFFIX) for n in lower_names):
            fmt = "fp16"

    # AWQ quantization_config parsing (quantization label w<bit>g<group>)
    if quant is None:
        for cfg_name in ("quantization_config.json", "quant_config.json"):
            cfg_path = path / cfg_name
            if cfg_path.exists():
                try:
                    with cfg_path.open("r", encoding="utf-8") as fh:
                        cfg = json.load(fh)
                    quant_method = (
                        cfg.get("quant_method")
                        or cfg.get("quantization_method")
                        or cfg.get("quantization", {}).get("method")
                    )
                    if isinstance(quant_method, str) and quant_method.lower() == "awq":
                        # ensure format is awq if still unset
                        if fmt is None:
                            fmt = "awq"
                        w_bit = (
                            cfg.get("w_bit")
                            or cfg.get("bits")
                            or cfg.get("weight_bit_width")
                        )
                        g_size = (
                            cfg.get("q_group_size")
                            or cfg.get("group_size")
                            or cfg.get("groupsize")
                        )
                        if isinstance(w_bit, int) and isinstance(g_size, int):
                            quant = f"w{w_bit}g{g_size}"
                except Exception:  # noqa: BLE001
                    pass
                break  # only inspect first existing

    # Fallback quant extraction from filenames
    if quant is None:
        for name in lower_names:
            for canonical, patterns in _QUANT_FILENAME_MAP.items():
                if any(p.search(name) for p in patterns):
                    quant = canonical
                    break
            if quant:
                break

    return fmt, quant


__all__ = ["detect_format_and_quant"]
