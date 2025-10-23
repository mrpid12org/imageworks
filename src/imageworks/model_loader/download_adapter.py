"""Unified download adapter: replaces legacy downloader registry.

Responsibilities:
 - Create or update RegistryEntry records when a model is downloaded.
 - Provide listing & filtering of downloaded (or declarative) variants.
 - Support removal semantics (retain entry metadata, clear download fields unless force delete).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict, Any, Iterable
from datetime import datetime, timezone
import hashlib
import re
import time as _time
import requests as _requests

from .registry import (
    find_by_download_identity,
    load_registry,
    remove_entry,
    save_registry,
    update_entries,
)
from .models import (
    RegistryEntry,
    BackendConfig,
    Artifacts,
    ChatTemplate,
    GenerationDefaults,
    VersionLock,
    PerformanceSummary,
    Probes,
)
from .hashing import compute_artifact_hashes
from .testing_filters import is_testing_name
from .naming import build_identity
from .simplified_naming import simplified_slug_for_fields
from .service import default_backend_port
import os as _os


class ImportSkipped(Exception):
    """Raised when an attempted import is identified as testing/demo and is skipped."""

    pass


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


def _normalize_filesystem_path(value: str | None) -> str | None:
    if not value:
        return None
    candidate = value.strip()
    if not candidate:
        return None
    if candidate.startswith("ollama:"):
        return candidate.lower()
    try:
        return str(Path(candidate).expanduser().resolve(strict=False))
    except Exception:  # noqa: BLE001
        try:
            return str(Path(candidate).expanduser())
        except Exception:  # noqa: BLE001
            return candidate


def _tiny_png_base64() -> str:
    """Return a tiny 1x1 transparent PNG as base64 string (no data: prefix)."""
    # Precomputed 1x1 transparent PNG
    return (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFGgJ+f3m0JwAA"
        "AABJRU5ErkJggg=="
    )


def _maybe_probe_and_mark_vision(entry: "RegistryEntry") -> None:
    """Best-effort vision capability probe for Ollama models.

    Sends a minimal image chat to the local Ollama API and, on success, records
    probes.vision.vision_ok with a timestamp and basic latency. Failures are ignored.
    """
    try:
        if entry.backend != "ollama":
            return
        if _os.environ.get("IMAGEWORKS_SKIP_POST_PROBE") in {"1", "true", "yes"}:
            return
        port = entry.backend_config.port or 11434
        ident = entry.served_model_id or entry.display_name or entry.name
        url = f"http://127.0.0.1:{port}/api/chat"
        img_b64 = _tiny_png_base64()
        payload = {
            "model": ident,
            "messages": [
                {
                    "role": "user",
                    "content": "vision probe",
                    "images": [img_b64],
                }
            ],
            "stream": False,
        }
        t0 = _time.time()
        resp = _requests.post(url, json=payload, timeout=6)
        latency_ms = int((_time.time() - t0) * 1000)
        ok = resp.status_code < 400
        if ok:
            data = {}
            try:
                data = resp.json()
            except Exception:  # noqa: BLE001
                pass
            # Basic heuristic: presence of message.content or non-empty response
            has_msg = bool(((data.get("message") or {}).get("content") or "").strip())
            if has_msg:
                from .models import VisionProbe

                # Avoid overwriting an existing affirmative probe
                if not entry.probes.vision or not entry.probes.vision.vision_ok:
                    entry.probes.vision = VisionProbe(
                        vision_ok=True,
                        timestamp=_now_iso(),
                        probe_version="1",
                        latency_ms=latency_ms,
                        notes="auto-probe:ollama",
                    )
                    update_entries([entry], save=True)
    except Exception:
        # Silent best-effort; do not block downloads/registration
        return


def _infer_family(hf_id: Optional[str]) -> Optional[str]:
    if not hf_id:
        return None
    tail = hf_id.split("/")[-1]
    return tail.lower().replace("@", "-").replace("_", "-").replace(" ", "-")


def _token_match(haystack: str, token: str) -> bool:
    pattern = rf"(^|[\s_\-/:]){re.escape(token)}($|[\s_\-/:])"
    return re.search(pattern, haystack) is not None


def _infer_capabilities(
    name: str, *, metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, bool]:
    n = name.lower()
    text = True
    vision_tokens = [
        "vision",
        "multimodal",
        "vl",
        "mm",
    ]
    vision = any(_token_match(n, tok) for tok in vision_tokens) or any(
        marker in n
        for marker in [
            "llava",
            "idefics",
            "minicpm-v",
            "internvl",
            "moondream",
            "phi-3-vision",
            "pixtral",
            "qwen-vl",
            "qwen2-vl",
            "qwen2.5-vl",
        ]
    )
    embedding = any(
        marker in n
        for marker in ["siglip", "embed", "embedding", "nomic", "text-embedding"]
    )
    audio = any(
        marker in n
        for marker in [
            "audio",
            "whisper",
            "wav",
            "vits",
            "sensevoice",
            "voice",
            "speech",
        ]
    )
    reasoning_markers = [
        "reason",
        "reasoning",
        "reasoner",
        "think",
        "-r1",
        " o1 ",
        " o3 ",
        "o1-",
        "o3-",
        "deepseek",
        "longthink",
    ]
    thinking = any(marker in n for marker in reasoning_markers)
    tools = any(
        marker in n
        for marker in [
            "tool",
            "tool_use",
            "tool-use",
            "tool_call",
            "tool-call",
            "toolcalling",
            "function_call",
            "function-call",
            "functioncalling",
            "function_tools",
            "with-tools",
            "withtools",
            "coder",
        ]
    ) or ("qwen" in n and "instruct" in n)

    reasoning = thinking or any(
        marker in n for marker in ["reason", "reasoning", "logic"]
    )

    meta_caps: List[str] = []
    if metadata:
        raw_caps = metadata.get("ollama_capabilities")
        iterable_caps: Iterable[Any] = []
        if isinstance(raw_caps, (list, tuple, set)):
            iterable_caps = raw_caps
        elif isinstance(raw_caps, str):
            iterable_caps = [raw_caps]
        if iterable_caps:
            for cap in iterable_caps:
                if isinstance(cap, bytes):
                    try:
                        cap = cap.decode("utf-8", "ignore")
                    except Exception:  # noqa: BLE001
                        cap = ""
                if not isinstance(cap, str):
                    continue
                token = cap.strip().lower()
                if token:
                    meta_caps.append(token)
    meta_caps_set = set(meta_caps)
    if meta_caps_set:
        if any(token in meta_caps_set for token in ["vision", "images", "image"]):
            vision = True
        if any(
            token in meta_caps_set for token in ["embedding", "embeddings", "embed"]
        ):
            embedding = True
        if any(
            token in meta_caps_set
            for token in ["audio", "speech", "voice", "multimodal"]
        ):
            audio = True
        if any(
            token in meta_caps_set
            for token in ["tool", "tools", "function_call", "function-call"]
        ):
            tools = True

    return {
        "text": text,
        "vision": vision,
        "embedding": embedding,
        "audio": audio,
        "thinking": thinking or reasoning,
        "reasoning": reasoning or thinking,
        "tools": tools,
    }


def _merge_capabilities(
    base: Dict[str, bool], updates: Dict[str, bool]
) -> Dict[str, bool]:
    """Return a new capabilities dict with any truthy updates applied."""
    merged = dict(base)
    for key, value in updates.items():
        if value:
            merged[key] = True
    return merged


def derive_capabilities(
    entry: "RegistryEntry",
    *,
    metadata: Optional[Dict[str, Any]] = None,
    seed: Optional[Dict[str, bool]] = None,
) -> Dict[str, bool]:
    """Compute capabilities using naming heuristics plus chat template hints."""
    base_metadata = metadata if metadata is not None else (entry.metadata or {})
    base = (
        seed
        if seed is not None
        else _infer_capabilities(entry.name, metadata=base_metadata)
    )
    return _merge_capabilities(base, _capabilities_from_template(entry))


def _capabilities_from_template(entry: "RegistryEntry") -> Dict[str, bool]:
    """Inspect chat template text (if available) for capability hints."""
    template_paths: List[Path] = []
    tpl_path = getattr(entry.chat_template, "path", None)
    if tpl_path:
        template_paths.append(Path(tpl_path))
    primary_tpl = (entry.metadata or {}).get("primary_chat_template_file")
    if (
        primary_tpl
        and entry.download_path
        and not str(entry.download_path).startswith("ollama://")
    ):
        template_paths.append(Path(entry.download_path) / primary_tpl)
    content = ""
    for candidate in template_paths:
        try:
            if candidate.exists():
                content = candidate.read_text(encoding="utf-8")
                break
        except Exception:  # noqa: BLE001
            continue
    if not content:
        return {}
    lowered = content.lower()
    tool_tokens = [
        "tool_calls",
        "tool call",
        "tools must",
        "tool_choice",
        "tool-choice",
        "functions.",
        "builtin_tools",
        "<|channel|>commentary",
    ]
    think_tokens = [
        "<|channel|>analysis",
        "chain of thought",
        "chain-of-thought",
        "internal reasoning",
        "thinking field",
        "analysis channel",
    ]
    has_tools = any(token in lowered for token in tool_tokens)
    has_thinking = any(token in lowered for token in think_tokens)
    if not (has_tools or has_thinking):
        return {}
    return {
        "tools": has_tools,
        "thinking": has_thinking,
        "reasoning": has_thinking,
    }


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
    # Use default merged snapshot path; layered loader will choose appropriate files.
    # Explicit path not passed to allow layering when expected. If environment forces single-file
    # mode, caller can set IMAGEWORKS_REGISTRY_NO_LAYERING.
    reg = load_registry(force=True)
    family = family_override or _infer_family(hf_id) or "model"
    identity = build_identity(
        family=family,
        backend=backend,
        format_type=format_type,
        quantization=quantization,
        display_override=display_name,
    )
    # Use simplified slug for canonical naming on fresh imports
    variant_name = simplified_slug_for_fields(
        family=family,
        backend=backend,
        format_type=format_type,
        quantization=quantization,
        metadata=None,
        download_path=str(Path(path).expanduser()),
        served_model_id=served_model_id,
    )
    backend = identity.backend_key
    format_type = identity.format_key
    quantization = identity.quant_key
    family = identity.family_key

    # Clean-before-write: strictly skip testing/demo placeholders unless explicitly allowed
    include_testing = _os.environ.get(
        "IMAGEWORKS_IMPORT_INCLUDE_TESTING", "0"
    ).lower() in {"1", "true", "yes", "on"}
    # Additional guard: if no hf_id and no explicit family, derive family from path tail and
    # treat overly-generic placeholders as testing/demo to avoid polluting registry.
    testing_by_family = False
    if hf_id is None and family_override is None:
        try:
            tail = Path(path).expanduser().name.lower().strip()
            tail_norm = tail.replace("@", "-").replace("_", "-").replace(" ", "-")
            generic_families = {"model", "demo", "demo-model", "r"}
            if tail_norm in generic_families or tail_norm.startswith("demo-"):
                testing_by_family = True
        except Exception:  # noqa: BLE001
            pass
    # Respect explicit family_override: don't apply broad name-based testing filters
    # unless the caller explicitly marks metadata.testing=True.
    name_testing = (
        False if family_override is not None else is_testing_name(variant_name)
    )
    is_testing = (
        name_testing
        or testing_by_family
        or bool((extra_metadata or {}).get("testing") is True)
    )
    if (not include_testing) and is_testing:
        # Do not persist this entry; signal a soft skip to callers
        raise ImportSkipped(f"Skipping testing/demo import: {variant_name}")
    existing = reg.get(variant_name)
    if not existing:
        normalized_target_path = _normalize_filesystem_path(path)
        if normalized_target_path:
            for candidate in reg.values():
                if candidate.name == variant_name:
                    continue
                if candidate.metadata and candidate.metadata.get(
                    "created_from_download"
                ):
                    continue
                candidate_path = _normalize_filesystem_path(
                    getattr(candidate.backend_config, "model_path", None)
                )
                if candidate_path and candidate_path == normalized_target_path:
                    existing = candidate
                    variant_name = candidate.name
                    target_display = candidate.display_name or identity.display_name
                    candidate_family = candidate.family or family
                    identity = build_identity(
                        family=candidate_family,
                        backend=candidate.backend or backend,
                        format_type=format_type,
                        quantization=quantization,
                        display_override=target_display,
                    )
                    backend = identity.backend_key
                    family = identity.family_key
                    break
    p = Path(path).expanduser()
    if size_bytes is None and p.exists():
        size_bytes = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    if files is None and p.exists():
        files = [str(f.relative_to(p)) for f in p.rglob("*") if f.is_file()]
    directory_checksum = compute_directory_checksum(p)
    now = _now_iso()

    def _apply_download_updates(entry: RegistryEntry) -> RegistryEntry:
        entry.backend = backend
        entry.backend_config.model_path = str(p)
        entry.download_format = format_type or entry.download_format
        entry.download_location = location or entry.download_location
        entry.download_path = str(p)
        entry.download_size_bytes = size_bytes or entry.download_size_bytes
        if files:
            entry.download_files = files
        entry.download_directory_checksum = directory_checksum
        entry.downloaded_at = entry.downloaded_at or now
        entry.last_accessed = now
        if quantization:
            entry.quantization = quantization
        if source_provider:
            entry.source_provider = source_provider
        if served_model_id:
            entry.served_model_id = served_model_id
        if roles:
            merged_roles = set(entry.roles) | set(roles)
            entry.roles = sorted(merged_roles)
        if role_priority:
            entry.role_priority.update(role_priority)
        if extra_metadata:
            entry.metadata = entry.metadata or {}
            for k, v in extra_metadata.items():
                if v is not None and k not in entry.metadata:
                    entry.metadata[k] = v
        if not getattr(entry.backend_config, "port", 0):
            entry.backend_config.port = default_backend_port(entry.backend)
        if entry.backend == "ollama" and not getattr(
            entry.backend_config, "host", None
        ):
            entry.backend_config.host = _os.environ.get(
                "IMAGEWORKS_OLLAMA_HOST", "host.docker.internal"
            )
        # Refresh auto-managed display names so CLI/proxy stay concise.
        should_refresh_display = False
        if (entry.metadata or {}).get("created_from_download"):
            should_refresh_display = True
        elif not getattr(entry, "display_name", None):
            should_refresh_display = True
        if should_refresh_display:
            try:
                from .simplified_naming import (
                    simplified_display_for_fields as _simple_disp,
                )

                entry.display_name = _simple_disp(
                    family=family,
                    backend=backend,
                    format_type=format_type,
                    quantization=quantization,
                    metadata=entry.metadata,
                    download_path=str(p),
                    served_model_id=served_model_id,
                )
            except Exception:
                entry.display_name = identity.display_name
        entry.family = family
        if hf_id:
            aliases = entry.model_aliases or []
            if hf_id not in aliases:
                entry.model_aliases = aliases + [hf_id]
        return entry

    # Backend reconciliation on update: if an 'unassigned' variant exists for the same
    # (family, format, quant), migrate it to the requested backend instead of duplicating.
    if not existing and backend != "unassigned":
        unassigned_name = simplified_slug_for_fields(
            family=family,
            backend="unassigned",
            format_type=format_type,
            quantization=quantization,
            metadata=None,
            download_path=str(p),
            served_model_id=served_model_id,
        )
        candidate = reg.get(unassigned_name)
        if candidate is not None:
            e = candidate
            old_name = e.name
            e.name = variant_name
            remove_entry(old_name, save=False)
            e = _apply_download_updates(e)
            if not e.artifacts.files:
                e = compute_artifact_hashes(e)
            e.capabilities = derive_capabilities(
                e, metadata=extra_metadata or e.metadata
            )
            update_entries([e], save=False)
            save_registry()
            _maybe_probe_and_mark_vision(e)
            return e

    if existing:
        e = _apply_download_updates(existing)
        if not e.artifacts.files:
            e = compute_artifact_hashes(e)
        e.capabilities = derive_capabilities(e, metadata=extra_metadata or e.metadata)
        update_entries([e], save=True)
        _maybe_probe_and_mark_vision(e)
        return e

    # Detect renamed variants: if another entry already references these weights,
    # upgrade it in-place instead of appending a duplicate record.
    checksum_lookup = directory_checksum or None
    rename_source = find_by_download_identity(
        backend=backend, download_path=str(p), checksum=checksum_lookup
    )
    if rename_source is None and checksum_lookup is not None:
        # Retry matching by path only when checksum differs between curated vs. downloaded entries.
        rename_source = find_by_download_identity(
            backend=backend, download_path=str(p), checksum=None
        )
    if rename_source and rename_source.name != variant_name:
        if not (rename_source.metadata or {}).get("created_from_download"):
            # Curated entry already manages these weights; update it in-place without renaming.
            curated = _apply_download_updates(rename_source)
            if not curated.artifacts.files:
                curated = compute_artifact_hashes(curated)
            curated.capabilities = derive_capabilities(
                curated, metadata=extra_metadata or curated.metadata
            )
            update_entries([curated], save=True)
            _maybe_probe_and_mark_vision(curated)
            return curated
        old_name = rename_source.name
        remove_entry(old_name, save=False)
        rename_source.name = variant_name
        rename_source = _apply_download_updates(rename_source)
        if not rename_source.artifacts.files:
            rename_source = compute_artifact_hashes(rename_source)
        rename_source.capabilities = derive_capabilities(
            rename_source, metadata=extra_metadata or rename_source.metadata
        )
        update_entries([rename_source], save=False)
        save_registry()
        _maybe_probe_and_mark_vision(rename_source)
        return rename_source

    capabilities = _infer_capabilities(variant_name, metadata=extra_metadata)
    entry = RegistryEntry(
        name=variant_name,
        # Seed display_name with simplified naming if helper is available
        display_name=(
            __import__(
                "imageworks.model_loader.simplified_naming",
                fromlist=["simplified_display_for_fields"],
            ).simplified_display_for_fields(  # type: ignore[attr-defined]
                family=family,
                backend=backend,
                format_type=format_type,
                quantization=quantization,
                metadata={"created_from_download": True},
                download_path=str(p),
                served_model_id=served_model_id,
            )
            if True
            else identity.display_name
        ),
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
        generation_defaults=GenerationDefaults(),
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
    if entry.backend_config.port in (None, 0):
        entry.backend_config.port = default_backend_port(entry.backend)
    if entry.backend == "ollama" and not entry.backend_config.host:
        entry.backend_config.host = _os.environ.get(
            "IMAGEWORKS_OLLAMA_HOST", "host.docker.internal"
        )
    entry = compute_artifact_hashes(entry)
    entry.capabilities = derive_capabilities(
        entry, metadata=extra_metadata, seed=entry.capabilities
    )
    update_entries([entry], save=True)
    # Post-registration best-effort vision probe (Ollama)
    _maybe_probe_and_mark_vision(entry)
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


def infer_capabilities(
    name: str, *, metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, bool]:
    """Public helper to expose capability inference for import scripts."""
    return _infer_capabilities(name, metadata=metadata)


__all__ = [
    "record_download",
    "list_downloads",
    "remove_download",
    "compute_directory_checksum",
    "infer_capabilities",
    "derive_capabilities",
]
