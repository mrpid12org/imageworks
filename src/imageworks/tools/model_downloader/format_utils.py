"""Shared format and quantization detection utilities for model directories.

This module centralises logic used by scan, downloader initial detection, and
normalization / rebuild passes so behaviour is consistent and testable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, List, Dict
import json
import re

# Container formats we attempt to detect. Earlier = higher priority.
# Note: AWQ and GPTQ are quantization schemes, not container formats.
_FORMAT_PRIORITY = ["gguf", "safetensors"]

_AWQ_HINT_PATTERNS = [
    re.compile(r"\.awq$", re.IGNORECASE),
    re.compile(r"[-_\/]awq([-_]|$)", re.IGNORECASE),
]
_GPTQ_HINT_PATTERNS = [
    re.compile(r"gptq", re.IGNORECASE),
    re.compile(r"[-_\/]gptq([-_]|$)", re.IGNORECASE),
]
_GGUF_SUFFIX = ".gguf"
_GPTQ_HINT = "gptq"
_SAFETENSORS_SUFFIX = ".safetensors"

# Quant filename tokens (lowercased). Maps canonical token -> list of patterns.
_QUANT_FILENAME_MAP: Dict[str, List[re.Pattern]] = {
    # llama.cpp-style tokens
    "q4_k_m": [re.compile(r"q4[_-]?k[_-]?m", re.IGNORECASE)],
    "q5_k_m": [re.compile(r"q5[_-]?k[_-]?m", re.IGNORECASE)],
    "q6_k": [re.compile(r"q6[_-]?k", re.IGNORECASE)],
    # int precisions
    "int4": [re.compile(r"int4", re.IGNORECASE)],
    "int8": [re.compile(r"int8", re.IGNORECASE)],
    # additional schemes/precisions
    "mxfp4": [re.compile(r"mxfp4", re.IGNORECASE)],
    "iq4_xs": [re.compile(r"iq4[_-]?xs", re.IGNORECASE)],
    "bnb": [re.compile(r"\bbnb\b", re.IGNORECASE)],
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
    quant_parts: list[str] = []

    # Format detection strictly by container
    if any(n.endswith(_GGUF_SUFFIX) for n in lower_names):
        fmt = "gguf"
    elif any(n.endswith(_SAFETENSORS_SUFFIX) for n in lower_names):
        fmt = "safetensors"

    # AWQ quantization_config parsing (quantization label w<bit>g<group>)
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
                if isinstance(quant_method, str):
                    qm = quant_method.lower()
                    if qm == "awq" and "awq" not in quant_parts:
                        quant_parts.append("awq")
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
                            # Append detail for AWQ
                            quant_parts.append(f"w{w_bit}g{g_size}")
                    if qm == "gptq" and "gptq" not in quant_parts:
                        quant_parts.append("gptq")
            except Exception:  # noqa: BLE001
                pass
            break  # only inspect first existing

    # Heuristics from filenames and directory name for quantization schemes
    # 1) AWQ / GPTQ tokens in names
    if any(
        p.search(n)
        for p in _AWQ_HINT_PATTERNS
        for n in ([repo_name_lower] + lower_names)
    ):
        if "awq" not in quant_parts:
            quant_parts.append("awq")
    if any(
        p.search(n)
        for p in _GPTQ_HINT_PATTERNS
        for n in ([repo_name_lower] + lower_names)
    ):
        if "gptq" not in quant_parts:
            quant_parts.append("gptq")

    # 2) FP16/BF16 tokens
    fp16_tok = any(
        re.search(r"\bf?p?16\b", n) for n in ([repo_name_lower] + lower_names)
    )
    bf16_tok = any(re.search(r"\bbf16\b", n) for n in ([repo_name_lower] + lower_names))
    fp8_tok = any(
        re.search(r"\bfp8\b", n, re.IGNORECASE)
        for n in ([repo_name_lower] + lower_names)
    )
    squeeze_tok = any(
        re.search(r"\bsqueezellm\b", n) for n in ([repo_name_lower] + lower_names)
    )
    if bf16_tok and "bf16" not in quant_parts:
        quant_parts.append("bf16")
    elif fp16_tok and "fp16" not in quant_parts:
        quant_parts.append("fp16")
    if fp8_tok and "fp8" not in quant_parts:
        quant_parts.append("fp8")
    if squeeze_tok and "squeezellm" not in quant_parts:
        quant_parts.append("squeezellm")

    # 3) Fallback numeric quant tokens (q4_k_m, int8, etc.)
    numeric_quant: Optional[str] = None
    for name in lower_names:
        for canonical, patterns in _QUANT_FILENAME_MAP.items():
            if any(p.search(name) for p in patterns):
                numeric_quant = canonical
                break
        if numeric_quant:
            break
    if numeric_quant and numeric_quant not in quant_parts:
        quant_parts.append(numeric_quant)

    # Compose final quant label
    quant: Optional[str] = None
    if quant_parts:
        # Prefer a compact label: include scheme (awq/gptq) if present, plus details
        # e.g., ["awq", "w4g128"] -> "awq-w4g128"; ["gptq", "int4"] -> "gptq-int4"
        base = []
        if "awq" in quant_parts:
            base.append("awq")
        if "gptq" in quant_parts:
            base.append("gptq")  # unlikely both; harmless if both tokens present
        # Append other tokens except duplicates and base tokens
        for tok in quant_parts:
            if tok in {"awq", "gptq"}:
                continue
            base.append(tok)
        quant = "-".join(base)

    return fmt, quant


__all__ = ["detect_format_and_quant"]
