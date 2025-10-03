"""Unified download adapter: replaces legacy downloader registry.

Responsibilities:
 - Create or update RegistryEntry records when a model is downloaded.
 - Provide listing & filtering of downloaded (or declarative) variants.
 - Support removal semantics (retain entry metadata, clear download fields unless force delete).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
import hashlib

from .registry import load_registry, update_entries, save_registry
from .models import (
    RegistryEntry,
    BackendConfig,
    Artifacts,
    ChatTemplate,
    VersionLock,
    PerformanceSummary,
    Probes,
)
from .hashing import compute_artifact_hashes


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def compute_directory_checksum(directory: Path) -> str:
    if not directory.exists():
        return ""
    hasher = hashlib.sha256()
    for p in sorted(directory.rglob("*")):
        if p.is_file():
            rel = p.relative_to(directory)
            try:
                size = p.stat().st_size
            except OSError:
                size = 0
            hasher.update(str(rel).encode())
            hasher.update(str(size).encode())
    return hasher.hexdigest()[:16]


def _infer_family(hf_id: Optional[str]) -> Optional[str]:
    if not hf_id:
        return None
    tail = hf_id.split("/")[-1]
    return tail.lower().replace("@", "-").replace("_", "-").replace(" ", "-")


def _infer_capabilities(name: str) -> Dict[str, bool]:
    n = name.lower()
    text = True
    vision = any(k in n for k in ["vl", "llava", "qwen2.5-vl", "idefics"])
    embedding = "siglip" in n or "embed" in n
    audio = False
    return {"text": text, "vision": vision, "embedding": embedding, "audio": audio}


def _build_variant_name(
    family: str, backend: str, fmt: Optional[str], quant: Optional[str]
) -> str:
    parts = [family, backend]
    if fmt:
        parts.append(fmt.lower())
    if quant and quant.lower() not in parts:
        parts.append(quant.lower())
    return "-".join(filter(None, parts))


def record_download(
    *,
    hf_id: Optional[str],
    backend: str,
    format_type: Optional[str],
    quantization: Optional[str],
    path: str,
    location: Optional[str],
    files: Optional[List[str]] = None,
    size_bytes: Optional[int] = None,
    source_provider: Optional[str] = None,
    roles: Optional[List[str]] = None,
    role_priority: Optional[Dict[str, int]] = None,
    family_override: Optional[str] = None,
    served_model_id: Optional[str] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
    display_name: Optional[str] = None,
) -> RegistryEntry:
    """Create or update a RegistryEntry to reflect a downloaded variant.

    If an existing entry matches (same name) it's enriched; otherwise a new skeleton is created.
    Name is derived from (family, backend, format, quantization). A `display_name` override
    allows callers to persist a canonical label (used by importers) without issuing a
    follow-up update.
    """
    # Allow tolerant duplicate handling for dynamic download writes
    import os as _os

    _os.environ.setdefault("IMAGEWORKS_ALLOW_REGISTRY_DUPES", "1")
    # Use default merged snapshot path; layered loader will choose appropriate files.
    # Explicit path not passed to allow layering when expected. If environment forces single-file
    # mode, caller can set IMAGEWORKS_REGISTRY_NO_LAYERING.
    reg = load_registry(force=True)
    family = family_override or _infer_family(hf_id) or "model"
    # Normalize quant to lowercase early for consistent persistence
    if quantization:
        quantization = quantization.lower()
    variant_name = _build_variant_name(family, backend, format_type, quantization)
    existing = reg.get(variant_name)
    p = Path(path).expanduser()
    if size_bytes is None and p.exists():
        size_bytes = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    if files is None and p.exists():
        files = [str(f.relative_to(p)) for f in p.rglob("*") if f.is_file()]
    directory_checksum = compute_directory_checksum(p)
    now = _now_iso()

    if existing:
        e = existing
        # update download provenance
        e.download_format = format_type or e.download_format
        e.download_location = location or e.download_location
        e.download_path = str(p)
        e.download_size_bytes = size_bytes or e.download_size_bytes
        if files:
            e.download_files = files
        e.download_directory_checksum = directory_checksum
        e.downloaded_at = e.downloaded_at or now
        e.last_accessed = now
        if quantization:
            e.quantization = quantization  # already lowercased
        e.source_provider = source_provider or e.source_provider
        if served_model_id:
            e.served_model_id = served_model_id
        if roles:
            merged_roles = set(e.roles) | set(roles)
            e.roles = sorted(merged_roles)
        if role_priority:
            e.role_priority.update(role_priority)
        if extra_metadata:
            if e.metadata is None:
                e.metadata = {}
            for k, v in extra_metadata.items():
                if v is not None and k not in e.metadata:
                    e.metadata[k] = v
        if display_name:
            e.display_name = display_name
        # compute artifacts hashes only if empty
        if not e.artifacts.files:
            e = compute_artifact_hashes(e)
        update_entries([e], save=True)
        return e

    capabilities = _infer_capabilities(variant_name)
    entry = RegistryEntry(
        name=variant_name,
        display_name=(display_name or (hf_id.split("/")[-1] if hf_id else variant_name)),
        backend=backend,
        backend_config=BackendConfig(port=0, model_path=str(p), extra_args=[]),
        capabilities=capabilities,
        artifacts=Artifacts(aggregate_sha256="", files=[]),
        chat_template=ChatTemplate(source="embedded", path=None, sha256=None),
        version_lock=VersionLock(
            locked=False, expected_aggregate_sha256=None, last_verified=None
        ),
        performance=PerformanceSummary(
            rolling_samples=0,
            ttft_ms_avg=None,
            throughput_toks_per_s_avg=None,
            last_sample=None,
        ),
        probes=Probes(vision=None),
        profiles_placeholder=None,
        metadata={"created_from_download": True, **(extra_metadata or {})},
        served_model_id=served_model_id,
        model_aliases=[hf_id] if hf_id else [],
        roles=roles or [],
        license=None,
        source={"huggingface_id": hf_id} if hf_id else None,
        deprecated=False,
        family=family,
        source_provider=source_provider or ("hf" if hf_id else None),
        quantization=quantization,
        backend_alternatives=[],
        role_priority=role_priority or {},
        download_format=format_type,
        download_location=location,
        download_path=str(p),
        download_size_bytes=size_bytes,
        download_files=files or [],
        download_directory_checksum=directory_checksum,
        downloaded_at=now,
        last_accessed=now,
    )
    entry = compute_artifact_hashes(entry)
    update_entries([entry], save=True)
    return entry


def list_downloads(*, only_installed: bool = False) -> List[RegistryEntry]:
    reg = load_registry(force=True)
    out: List[RegistryEntry] = []
    for e in reg.values():
        if e.download_path:
            if only_installed:
                if Path(e.download_path).expanduser().exists():
                    out.append(e)
            else:
                out.append(e)
    return sorted(out, key=lambda x: x.name)


def remove_download(name: str, *, keep_entry: bool = True) -> bool:
    reg = load_registry(force=True)
    e = reg.get(name)
    if not e:
        return False
    if keep_entry:
        e.download_path = None
        e.download_size_bytes = None
        e.download_files = []
        e.download_directory_checksum = None
        e.metadata.setdefault("removed_download", True)
        update_entries([e], save=True)
    else:
        from .registry import remove_entry

        remove_entry(name, save=True)
    save_registry()
    return True


__all__ = [
    "record_download",
    "list_downloads",
    "remove_download",
    "compute_directory_checksum",
]
