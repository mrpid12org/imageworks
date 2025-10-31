"""Utilities for generating stable registry ids and human-friendly labels.

A single normalization pipeline ensures that every backend/import path derives
consistent identifiers before touching the registry.  Slugs remain compatible
with the existing `family-backend-format-quant` shape while the display string
is kept short and readable for tables and UIs.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import Optional

_TOKEN_RE = re.compile(r"[^a-z0-9]+")
_QUANT_TOKEN_RE = re.compile(r"[^a-z0-9_]+")


def _sanitize_family(value: Optional[str]) -> str:
    if not value:
        return "model"
    raw = str(value).strip().lower()
    raw = raw.replace("@", "-").replace("_", "-").replace(" ", "-")
    raw = re.sub(r"-+", "-", raw)
    raw = re.sub(r"[^a-z0-9\.-]+", "-", raw)
    raw = raw.strip("-")
    return raw or "model"


def _normalize_backend(value: str) -> str:
    return (value or "backend").strip().lower()


def _normalize_format(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    return _TOKEN_RE.sub("-", value.strip().lower()).strip("-") or None


def _normalize_quant(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    normalized = _QUANT_TOKEN_RE.sub("-", value.strip().lower())
    normalized = re.sub(r"-+", "-", normalized)
    return normalized.strip("-") or None


def _token_set(value: str) -> set[str]:
    tokens = set()
    for part in re.split(r"[-_]+", value):
        if part:
            tokens.add(part)
    return tokens


def _humanize_token(token: str) -> str:
    if not token:
        return token
    if token.isupper():
        return token
    lower = token.lower()
    if lower in {"vl", "llm", "hf", "gguf", "awq"}:
        return lower.upper()
    if re.fullmatch(r"[0-9]+b", lower):
        return lower.upper()
    if re.fullmatch(r"q[0-9].*", lower):
        return lower.upper().replace("-", " ").replace("_", " ")
    return token[:1].upper() + token[1:]


def _humanize_family(family: str) -> str:
    pieces = re.split(r"[-_/]+", family)
    human = [_humanize_token(p) for p in pieces if p]
    return " ".join(human) if human else family


def _label_format(fmt: Optional[str]) -> Optional[str]:
    if not fmt:
        return None
    if fmt.lower() == "safetensors":
        return "Safetensors"
    if fmt.lower() == "gguf":
        return "GGUF"
    if fmt.lower() == "awq":
        return "AWQ"
    return fmt.upper()


def _label_quant(quant: Optional[str]) -> Optional[str]:
    if not quant:
        return None
    return quant.replace("-", " ").replace("_", " ").upper()


_BACKEND_LABELS = {
    "vllm": "vLLM",
    "ollama": "Ollama",
    "lmdeploy": "LMDeploy",
    "gguf": "GGUF",
    "triton": "Triton",
}


def _label_backend(backend: str) -> str:
    return _BACKEND_LABELS.get(backend.lower(), backend.capitalize())


@dataclass(frozen=True)
class ModelIdentity:
    family_raw: str
    backend: str
    format: Optional[str] = None
    quant: Optional[str] = None
    display_override: Optional[str] = None

    family_key: str = ""
    backend_key: str = ""
    format_key: Optional[str] = None
    quant_key: Optional[str] = None
    slug: str = ""
    display_name: str = ""

    def __post_init__(self) -> None:  # type: ignore[override]
        family_key = _sanitize_family(self.family_raw)
        backend_token = _normalize_backend(self.backend)
        format_token = _normalize_format(self.format)
        quant_token = _normalize_quant(self.quant)

        family_tokens = _token_set(family_key)
        slug_parts: list[str] = [family_key, backend_token]
        if format_token and format_token not in family_tokens:
            slug_parts.append(format_token)
            family_tokens |= _token_set(format_token)
        if quant_token and quant_token not in family_tokens:
            slug_parts.append(quant_token)
        slug = "-".join(filter(None, slug_parts))

        if self.display_override and self.display_override.strip():
            display = self.display_override.strip()
        else:
            family_label = _humanize_family(family_key)
            fmt_label = _label_format(format_token)
            quant_label = _label_quant(quant_token)
            backend_label = _label_backend(backend_token)
            decorations = []
            if fmt_label and quant_label:
                decorations.append(f"{fmt_label} {quant_label}")
            else:
                if fmt_label:
                    decorations.append(fmt_label)
                if quant_label:
                    decorations.append(quant_label)
            if backend_label:
                decorations.append(backend_label)
            if decorations:
                display = f"{family_label} ({', '.join(decorations)})"
            else:
                display = family_label

        object.__setattr__(self, "family_key", family_key)
        object.__setattr__(self, "backend_key", backend_token)
        object.__setattr__(self, "format_key", format_token)
        object.__setattr__(self, "quant_key", quant_token)
        object.__setattr__(self, "slug", slug)
        object.__setattr__(self, "display_name", display)

    def with_backend(self, backend: str) -> "ModelIdentity":
        return replace(self, backend=backend)


def build_identity(
    *,
    family: str,
    backend: str,
    format_type: Optional[str],
    quantization: Optional[str],
    display_override: Optional[str] = None,
) -> ModelIdentity:
    """Return a normalized identity for a registry entry."""

    return ModelIdentity(
        family_raw=family,
        backend=backend,
        format=format_type,
        quant=quantization,
        display_override=display_override,
    )


__all__ = ["ModelIdentity", "build_identity"]
