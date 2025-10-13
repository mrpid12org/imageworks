"""Shared format and quantization detection utilities for model directories.

This module implements a deterministic, evidence-ordered algorithm for
identifying container format and quantization of locally downloaded model
repositories. It follows a strict precedence:

Level 1: File header metadata (e.g., GGUF header)
Level 2: Explicit config files (quantize_config.json, quantization_config.json, awq_config.json)
Level 3: Tensor structure hints in safetensors (e.g., *.qweight/*.qzeros)
Level 4: File naming conventions (e.g., *-GPTQ, *-AWQ, GGUF q-tokens like q4_k_m)
Level 5: Dynamic quantization hints (config.json quantization_config load_in_4bit/8bit)
Level 6: Default to full/half precision dtype (torch_dtype or tensor dtypes)

Lower level numbers override/conflict with higher ones; on ties we keep the
first seen signal of the same level.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, List, Dict
import json
import re
from contextlib import suppress

try:  # optional but recommended for Level 3/6 detection on safetensors
    from safetensors import safe_open  # type: ignore
except Exception:  # noqa: BLE001
    safe_open = None  # type: ignore[assignment]

# Optional GGUF header reader for Level 1 authoritative detection
try:  # pragma: no cover - optional dependency
    from gguf import GGUFReader  # type: ignore
except Exception:  # noqa: BLE001
    GGUFReader = None  # type: ignore[assignment]

_GGUF_SUFFIX = ".gguf"
_SAFETENSORS_SUFFIX = ".safetensors"

# File naming patterns for quantization tokens (Level 4 heuristic)
# Includes common llama.cpp GGUF quant names and general hints.
_QUANT_FILENAME_MAP: Dict[str, List[re.Pattern]] = {
    # llama.cpp-style tokens (add more as needed)
    "q2_k": [re.compile(r"q2[_-]?k\b", re.IGNORECASE)],
    "q3_k_s": [re.compile(r"q3[_-]?k[_-]?s\b", re.IGNORECASE)],
    "q3_k_m": [re.compile(r"q3[_-]?k[_-]?m\b", re.IGNORECASE)],
    "q3_k_l": [re.compile(r"q3[_-]?k[_-]?l\b", re.IGNORECASE)],
    "q4_0": [re.compile(r"q4[_-]?0\b", re.IGNORECASE)],
    "q4_1": [re.compile(r"q4[_-]?1\b", re.IGNORECASE)],
    "q4_k_m": [re.compile(r"q4[_-]?k[_-]?m\b", re.IGNORECASE)],
    "q5_0": [re.compile(r"q5[_-]?0\b", re.IGNORECASE)],
    "q5_1": [re.compile(r"q5[_-]?1\b", re.IGNORECASE)],
    "q5_k_m": [re.compile(r"q5[_-]?k[_-]?m\b", re.IGNORECASE)],
    "q6_k": [re.compile(r"q6[_-]?k\b", re.IGNORECASE)],
    "q8_0": [re.compile(r"q8[_-]?0\b", re.IGNORECASE)],
    # generic int and precision tokens sometimes included in names
    "int4": [re.compile(r"\bint4\b", re.IGNORECASE)],
    "int8": [re.compile(r"\bint8\b", re.IGNORECASE)],
    "awq": [re.compile(r"(?<![a-z0-9])awq(?![a-z0-9])", re.IGNORECASE)],
    "gptq": [re.compile(r"(?<![a-z0-9])gptq(?![a-z0-9])", re.IGNORECASE)],
    # vendor/other tokens
    "iq4_xs": [re.compile(r"iq4[_-]?xs\b", re.IGNORECASE)],
}


def _canonicalize_gguf_dtype(dtype_name: str) -> Optional[str]:
    """Map GGUF tensor dtype names to our canonical quant tokens.

    Examples: Q4_0 -> q4_0, Q4_K_M -> q4_k_m, F16 -> fp16, BF16 -> bf16
    """
    if not dtype_name:
        return None
    dn = dtype_name.strip().upper()
    # common float types
    if dn in {"F16", "FP16"}:
        return "fp16"
    if dn in {"F32", "FP32"}:
        return "fp32"
    if dn == "BF16":
        return "bf16"
    # integer quant families
    # normalize Q*_K_* and Q*_*
    dn = dn.replace("-", "_")
    # Accepted tokens list (upper) mapped to our lowercase with underscores
    mapping = {
        "Q2_K": "q2_k",
        "Q3_K_S": "q3_k_s",
        "Q3_K_M": "q3_k_m",
        "Q3_K_L": "q3_k_l",
        "Q4_0": "q4_0",
        "Q4_1": "q4_1",
        "Q4_K_M": "q4_k_m",
        "Q5_0": "q5_0",
        "Q5_1": "q5_1",
        "Q5_K_M": "q5_k_m",
        "Q6_K": "q6_k",
        "Q8_0": "q8_0",
        "IQ4_XS": "iq4_xs",
    }
    return mapping.get(dn)


def _detect_gguf_quant_from_header(gguf_path: Path) -> Optional[str]:
    """Return canonical quant token by inspecting GGUF header only.

    Strategy: read tensor infos via GGUFReader; count dtype of weight tensors
    (preferring names that include '.weight'), pick the most frequent.
    """
    if GGUFReader is None:
        return None
    with suppress(Exception):
        reader = GGUFReader(str(gguf_path))  # type: ignore[misc]
        # Access tensor infos; try multiple attribute names for compatibility
        tensor_infos = getattr(reader, "tensor_infos", None) or getattr(
            reader, "tensors", None
        )
        if not tensor_infos:
            return None
        counts: Dict[str, int] = {}
        # Support both list of objects or dict of infos
        iterable = (
            tensor_infos.items() if isinstance(tensor_infos, dict) else tensor_infos
        )
        for item in iterable:
            # Unpack name, info for dict-style
            if isinstance(item, tuple) and len(item) >= 2:
                name, info = item[0], item[1]
            else:
                info = item
                name = getattr(info, "name", "")
            name_l = str(name).lower()
            # dtype field could be in different attrs; try common ones
            dtype = (
                getattr(info, "dtype", None)
                or getattr(info, "ggml_type", None)
                or getattr(info, "tensor_type", None)
            )
            # Convert dtype to text
            if dtype is None:
                continue
            dname = str(getattr(dtype, "name", dtype))
            canon = _canonicalize_gguf_dtype(dname)
            if not canon:
                continue
            # weight preference: prioritize count if looks like a weight
            weight_bonus = (
                3 if (".weight" in name_l or name_l.endswith("weight")) else 1
            )
            counts[canon] = counts.get(canon, 0) + weight_bonus
        if counts:
            # Choose the token with highest score
            return sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[0][0]
    return None


def detect_format_and_quant(path: Path) -> Tuple[Optional[str], Optional[str]]:
    """Detect (format, quant) for a model directory using evidence precedence.

    Returns:
        (format, quantization) where each may be None if not detectable.
    """
    if not path.exists() or not path.is_dir():
        return None, None

    files = [p for p in path.rglob("*") if p.is_file()]
    lower_names = [p.name.lower() for p in files]

    fmt: Optional[str] = None
    quant: Optional[str] = None
    quant_level: int = 999  # lower is stronger

    # Step 2: Container detection - GGUF first, then safetensors
    gguf_files = [p for p in files if p.suffix.lower() == _GGUF_SUFFIX]
    st_files = [p for p in files if p.suffix.lower() == _SAFETENSORS_SUFFIX]

    if gguf_files:
        fmt = "gguf"
        # Level 1: parse GGUF header for authoritative quant, else fallback to filename tokens
        header_quant = None
        for gf in gguf_files:
            header_quant = _detect_gguf_quant_from_header(gf)
            if header_quant:
                quant = header_quant
                quant_level = 1
                break
        if quant is None:
            # Fallback to Level 4 filename heuristic for known GGUF quant tokens
            fname_tokens = [p.name.lower() for p in gguf_files]
            for name in fname_tokens:
                for canonical, patterns in _QUANT_FILENAME_MAP.items():
                    if any(p.search(name) for p in patterns):
                        quant = canonical
                        quant_level = 4
                        break
                if quant is not None:
                    break
        # If multiple GGUF shards exist and no token found, leave quant unknown

    # If not GGUF, consider safetensors container
    if fmt is None and st_files:
        fmt = "safetensors"

    # Step 3: Explicit quantization configs (Level 2)
    # awq_config.json is definitive for AWQ; also support quantize_config.json / quantization_config.json
    def _read_json(p: Path) -> Optional[Dict]:
        with suppress(Exception):
            with p.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        return None

    # 3a) config.json embedded quantization_config (awq/gptq are authoritative here)
    cfg_main = next((p for p in files if p.name.lower() == "config.json"), None)
    if quant_level > 2 and cfg_main:
        data_main = _read_json(cfg_main)
        if isinstance(data_main, dict):
            qcfg = data_main.get("quantization_config") or {}
            if isinstance(qcfg, dict):
                qmethod = (qcfg.get("quant_method") or "").strip().lower()
                if qmethod in {"awq", "gptq"}:
                    if qmethod == "awq":
                        w_bit = qcfg.get("w_bit")
                        q_group = qcfg.get("q_group_size")
                        quant = (
                            f"awq-w{w_bit}g{q_group}"
                            if isinstance(w_bit, int) and isinstance(q_group, int)
                            else "awq"
                        )
                    else:  # gptq
                        bits = qcfg.get("bits")
                        quant = f"gptq-int{bits}" if isinstance(bits, int) else "gptq"
                    quant_level = 2
                elif (
                    # AWQ signature by keys if quant_method is absent
                    (
                        isinstance(qcfg.get("w_bit"), int)
                        and isinstance(qcfg.get("q_group_size"), int)
                    )
                    or ("zero_point" in qcfg)
                ):
                    w_bit = qcfg.get("w_bit")
                    q_group = qcfg.get("q_group_size")
                    quant = (
                        f"awq-w{w_bit}g{q_group}"
                        if isinstance(w_bit, int) and isinstance(q_group, int)
                        else "awq"
                    )
                    quant_level = 2
                elif (
                    # GPTQ signature by keys if quant_method is absent
                    isinstance(qcfg.get("bits"), int)
                    and isinstance(qcfg.get("group_size"), int)
                    and ("desc_act" in qcfg or "act_order" in qcfg)
                ):
                    bits = qcfg.get("bits")
                    quant = f"gptq-int{bits}" if isinstance(bits, int) else "gptq"
                    quant_level = 2

    awq_cfg = next((p for p in files if p.name.lower() == "awq_config.json"), None)
    if awq_cfg:
        data = _read_json(awq_cfg)
        if isinstance(data, dict):
            w_bit = data.get("w_bit")
            q_group = data.get("q_group_size")
            quant = (
                f"awq-w{w_bit}g{q_group}"
                if isinstance(w_bit, int) and isinstance(q_group, int)
                else "awq"
            )
            quant_level = 2

    if quant_level > 2:
        cfg_file = next(
            (
                p
                for p in files
                if p.name.lower()
                in {
                    "quantize_config.json",
                    "quantization_config.json",
                    "quant_config.json",
                }
            ),
            None,
        )
        if cfg_file:
            cfg = _read_json(cfg_file)
            if isinstance(cfg, dict):
                # AWQ signature (prefer AWQ if both styles present)
                if (
                    (
                        isinstance(cfg.get("w_bit"), int)
                        and isinstance(cfg.get("q_group_size"), int)
                    )
                    or (cfg.get("quant_method") == "awq")
                    or ("zero_point" in cfg)
                ):
                    w_bit = cfg.get("w_bit")
                    q_group = cfg.get("q_group_size")
                    quant = (
                        f"awq-w{w_bit}g{q_group}"
                        if isinstance(w_bit, int) and isinstance(q_group, int)
                        else "awq"
                    )
                    quant_level = 2
                # GPTQ signature requires desc_act (per spec) or explicit quant_method
                elif (
                    isinstance(cfg.get("bits"), int)
                    and isinstance(cfg.get("group_size"), int)
                    and ("desc_act" in cfg or "act_order" in cfg)
                ) or (cfg.get("quant_method") == "gptq"):
                    bits = cfg.get("bits")
                    quant = f"gptq-int{bits}" if isinstance(bits, int) else "gptq"
                    quant_level = 2
                # bitsandbytes pre-quantized signature (rare)
                elif (
                    "bnb_4bit_compute_dtype" in cfg
                    or cfg.get("quant_method") == "bitsandbytes"
                ):
                    # Try to infer bit-width if present
                    if cfg.get("load_in_4bit") is True:
                        quant = "bitsandbytes-4bit"
                    elif cfg.get("load_in_8bit") is True:
                        quant = "bitsandbytes-8bit"
                    else:
                        quant = "bitsandbytes"
                    quant_level = 2

    # Step 4: Tensor structure (Level 3) for safetensors
    if quant_level > 3 and st_files and safe_open is not None:
        # open first model*.safetensors or first shard
        st_primary = next(
            (p for p in sorted(st_files) if p.name.startswith("model")), st_files[0]
        )
        with suppress(Exception):
            with safe_open(str(st_primary), framework="pt", device="cpu") as f:  # type: ignore[misc]
                keys = list(f.keys())
            if any(".qweight" in k for k in keys) and any(".qzeros" in k for k in keys):
                # Ambiguous between GPTQ/AWQ; mark as ambiguous per spec
                quant = "gptq-or-awq"
                quant_level = 3

    # Step 4 (continued): File & directory naming conventions if still unknown
    if quant_level > 4:
        dir_name = path.name.lower()
        search_space = [dir_name] + lower_names
        for name in search_space:
            for canonical, patterns in _QUANT_FILENAME_MAP.items():
                if any(p.search(name) for p in patterns):
                    quant = canonical
                    quant_level = 4
                    break
            if quant_level == 4:
                break

    # Step 5: Dynamic quantization hints in config.json (modern: quant_method)
    if quant_level > 5:
        cfg = next((p for p in files if p.name.lower() == "config.json"), None)
        if cfg:
            data = _read_json(cfg) or {}
            qcfg = data.get("quantization_config") or {}
            if isinstance(qcfg, dict):
                qmethod = qcfg.get("quant_method")
                if isinstance(qmethod, str) and qmethod.strip():
                    qm = qmethod.strip().lower()
                    # Normalize known values directly; keep unknowns as-is (lowercase)
                    if qm in {"mxfp4", "mxfp8", "bitsandbytes"}:
                        quant = qm
                    else:
                        # Future-proof: accept vendor-provided method token verbatim
                        quant = qm
                    quant_level = 5
                elif qcfg.get("load_in_4bit") is True:
                    quant = "bitsandbytes-4bit"
                    quant_level = 5
                elif qcfg.get("load_in_8bit") is True:
                    quant = "bitsandbytes-8bit"
                    quant_level = 5

    # Step 6: Default to dtype for FP16/BF16/FP32 when nothing else detected
    if quant is None:
        # Try safetensors tensor dtype
        if st_files and safe_open is not None:
            st_primary = next(
                (p for p in sorted(st_files) if p.name.startswith("model")), st_files[0]
            )
            with suppress(Exception):
                with safe_open(str(st_primary), framework="pt", device="cpu") as f:  # type: ignore[misc]
                    # Find a representative weight tensor
                    first_key = next(
                        (
                            k
                            for k in f.keys()
                            if k.endswith(".weight") or ".weight" in k
                        ),
                        None,
                    )
                    first_key = first_key or (list(f.keys())[0] if f.keys() else None)
                    if first_key:
                        info = f.get_tensor_metadata(first_key)
                        # info.dtype like F16, BF16, F32
                        dtype = getattr(info, "dtype", None)
                        dtype_str = str(dtype).lower() if dtype is not None else ""
                        if "bf16" in dtype_str:
                            quant = "bf16"
                        elif "f16" in dtype_str:
                            quant = "fp16"
                        elif "f32" in dtype_str:
                            quant = "fp32"

        # Fallback to config.json torch_dtype if still unknown
        if quant is None:
            cfg = next((p for p in files if p.name.lower() == "config.json"), None)
            if cfg:
                data = _read_json(cfg) or {}
                dtype = (
                    data.get("torch_dtype")
                    or (data.get("model_kwargs") or {}).get("torch_dtype")
                    or data.get("dtype")
                )
                if isinstance(dtype, str):
                    d = dtype.strip().lower()
                    if d in {"bfloat16", "bf16"}:
                        quant = "bf16"
                    elif d in {"float16", "half", "fp16"}:
                        quant = "fp16"
                    elif d in {"float32", "fp32"}:
                        quant = "fp32"

    return fmt, quant


__all__ = ["detect_format_and_quant"]
