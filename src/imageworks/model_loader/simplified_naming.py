"""Simplified naming helpers used for default display labels.

Generates short, human-friendly names of the form:
  "<family> <param_size>" optionally followed by " (QUANT)"

- Family is normalized from entry.family or Ollama metadata.
- Parameter size is taken from Ollama metadata when present, otherwise
  inferred from family tokens (e.g., 7B) or optionally by scanning
  safetensors to count parameters.
- Quantization comes from entry.quantization (ignored if 'unknown').

Used by download/registry code and UIs (CLI list, proxy) to show consistent
labels without backend/format noise.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Any
import re


def _extract_param_size(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    t = str(text).strip().lower()
    m = re.search(r"\b(\d{1,3})\s*b\b", t)
    if m:
        return f"{m.group(1)}B"
    return None


def _hf_param_size_label(path: Optional[str]) -> Optional[str]:
    """Best-effort parameter size from local safetensors. May return None.

    This is intentionally lightweight: if no safetensors are under the path,
    or safetensors is unavailable, we return None.
    """
    if not path:
        return None
    try:
        p = Path(path).expanduser()
    except Exception:
        return None
    if not p.exists():
        return None
    st_files = [f for f in p.rglob("*.safetensors") if f.is_file()]
    if not st_files:
        return None
    try:
        from safetensors import safe_open as _safe_open  # type: ignore
    except Exception:
        return None
    total: int = 0
    for sf in sorted(st_files):
        try:
            with _safe_open(str(sf), framework="pt", device="cpu") as f:  # type: ignore
                for tk in f.keys():
                    try:
                        meta = f.get_tensor_metadata(tk)
                        shape = getattr(meta, "shape", None)
                        if not shape:
                            t = f.get_tensor(tk)
                            shape = getattr(t, "shape", None)
                        if shape:
                            n = 1
                            for d in shape:
                                n *= int(d)
                            total += int(n)
                    except Exception:
                        continue
        except Exception:
            continue
    if total <= 0:
        return None
    # Convert to billions and round to one decimal; drop trailing .0
    val = total / 1_000_000_000.0
    rounded = round(val + 1e-12, 1)
    if abs(rounded - int(rounded)) < 1e-9:
        return f"{int(rounded)}B"
    return f"{rounded:.1f}B"


def _clean_family(base: str) -> str:
    raw = (base or "model").strip().lower()
    rm_patterns = [
        r"\b(gguf|safetensors)\b",
        r"\bq\d(?:_k(?:_[sml])?|_[01])\b",  # q4_k_m, q4_0
        r"\bq\d(?:-k(?:-[sml])?)\b",  # q4-k-m
        r"\b(awq|gptq|int4|int8|fp16|bf16|fp8|nf4|mxfp4|mxfp8)\b",
    ]
    fam = raw
    for pat in rm_patterns:
        fam = re.sub(pat, " ", fam)
    fam = re.sub(r"\s+", " ", fam).strip()
    fam = fam.replace("@", "-").replace("_", "-").replace("/", "-")
    fam = re.sub(r"-+", "-", fam).strip("-")
    return fam or "model"


def simplified_display_for_fields(
    *,
    family: Optional[str],
    backend: str,
    format_type: Optional[str],  # unused but kept for future tweaks
    quantization: Optional[str],
    metadata: Optional[dict[str, Any]] = None,
    download_path: Optional[str] = None,
    served_model_id: Optional[str] = None,
) -> str:
    """Return simplified display label based on fields and minimal context.

    Rules:
      - Prefer Ollama metadata for family/param size when backend=ollama.
      - For HF/local, use entry.family and try to extract size token; optionally
        scan safetensors to estimate parameters when available.
      - Append quant in parentheses unless it's missing/unknown.
    """
    meta = metadata or {}
    backend_l = (backend or "").strip().lower()
    q_raw = (quantization or "").strip()
    q_ok = bool(q_raw) and (q_raw.lower() != "unknown")
    q_label = q_raw.replace("_", " ").replace("-", " ").upper() if q_ok else ""

    fam = None
    param = None
    if backend_l == "ollama":
        fam = (
            meta.get("ollama_family")
            or meta.get("ollama_architecture")
            or family
            or "model"
        )
        param = meta.get("ollama_parameter_size") or _extract_param_size(
            meta.get("ollama_parameters")
        )
    else:
        fam = family or "model"
        # Token first, then optional safetensors scan
        param = _extract_param_size(fam) or _hf_param_size_label(download_path)

    fam_norm = _clean_family(str(fam))
    # If hyphenated trailing size like -7b remains, pull it into param
    if not param:
        m = re.search(r"-(\d{1,3})b$", fam_norm)
        if m:
            param = f"{m.group(1)}B"
            fam_norm = fam_norm[: -len(m.group(0))]
    if param and param.lower() not in fam_norm:
        base = f"{fam_norm} {param.lower()}"
    else:
        base = fam_norm
    base = base.strip().strip("-")
    return f"{base} ({q_label})" if q_label else base


def simplified_display_for_entry(entry) -> str:
    return simplified_display_for_fields(
        family=getattr(entry, "family", None),
        backend=getattr(entry, "backend", "unknown"),
        format_type=getattr(entry, "download_format", None),
        quantization=getattr(entry, "quantization", None),
        metadata=getattr(entry, "metadata", None) or {},
        download_path=getattr(entry, "download_path", None),
        served_model_id=getattr(entry, "served_model_id", None),
    )


__all__ = [
    "simplified_display_for_fields",
    "simplified_display_for_entry",
]


def simplified_slug_for_fields(
    *,
    family: Optional[str],
    backend: str,
    format_type: Optional[str],
    quantization: Optional[str],
    metadata: Optional[dict[str, Any]] = None,
    download_path: Optional[str] = None,
    served_model_id: Optional[str] = None,
) -> str:
    """Return a stable simplified slug from the simplified display.

    Slug is derived from the simplified display by replacing spaces with underscores
    and preserving parentheses, e.g.: "llava 7b (Q4 0)" -> "llava_7b_(Q4_0)".
    """
    disp = simplified_display_for_fields(
        family=family,
        backend=backend,
        format_type=format_type,
        quantization=quantization,
        metadata=metadata,
        download_path=download_path,
        served_model_id=served_model_id,
    )
    # Normalize multiple spaces to one before underscore conversion
    disp_norm = re.sub(r"\s+", " ", disp).strip()
    return disp_norm.replace(" ", "_")


def simplified_slug_for_entry(entry) -> str:
    return simplified_slug_for_fields(
        family=getattr(entry, "family", None),
        backend=getattr(entry, "backend", "unknown"),
        format_type=getattr(entry, "download_format", None),
        quantization=getattr(entry, "quantization", None),
        metadata=getattr(entry, "metadata", None) or {},
        download_path=getattr(entry, "download_path", None),
        served_model_id=getattr(entry, "served_model_id", None),
    )


__all__.extend(["simplified_slug_for_fields", "simplified_slug_for_entry"])
