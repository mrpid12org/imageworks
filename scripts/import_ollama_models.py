"""Import locally pulled Ollama models into the unified registry.

Usage:
  uv run python scripts/import_ollama_models.py [--backend ollama] [--location linux_wsl] [--dry-run]

This script maps each Ollama model to a variant name using the unified naming
rules (family derived from the model name reported by Ollama). Since Ollama
stores models in its internal store, we cannot trivially map to an external
path; we still record a pseudo path pointing to the Ollama models directory
if discoverable, otherwise we set download_path to None (meaning not directly
filesystem-managed by us). For reproducibility, you may later augment entries
with explicit artifact hashes once exported.

The importer talks to the Ollama daemon over HTTP (`OLLAMA_BASE_URL`,
defaulting to `http://127.0.0.1:11434`). No local CLI installation is
required as long as the API endpoint is reachable.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from imageworks.model_loader.download_adapter import (
    record_download,
    ImportSkipped,
    derive_capabilities,
)
from imageworks.model_loader.naming import build_identity, ModelIdentity
from imageworks.model_loader.simplified_naming import simplified_slug_for_fields
from imageworks.model_loader import registry as unified_registry
from imageworks.tools.model_downloader.quant_utils import is_quant_token
from imageworks.tools.ollama_api import OllamaClient, OllamaError

DEFAULT_BACKEND = "ollama"

# Minimal sample dataset used when the Ollama CLI is unavailable during dry runs.
# This allows unit tests and documentation examples to exercise the normalization
# pipeline without requiring the external binary. The payload mirrors a
# quantized model with a slash-heavy identifier to validate normalization rules.
_FALLBACK_SAMPLE_MODELS: List[Dict[str, Any]] = [
    {
        "name": (
            "hf.co/mradermacher/L3.1-Dark-Reasoning-LewdPlay-evo-"
            "Hermes-R1-Uncensored-8B-i1-GGUF:Q6_K"
        ),
        "size": 6 * 1024**3,  # 6 GiB placeholder size for illustrative output
    }
]


@dataclass
class OllamaModelData:
    name: str
    family: str
    quant: Optional[str]
    size_bytes: int
    path: Path
    extra_metadata: Dict[str, Any]
    architecture: Optional[str]
    parameters: Optional[str]
    context_length: Optional[str]
    embedding_length: Optional[str]
    capabilities: Optional[List[str]]


def list_ollama_models(
    client: OllamaClient,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Return models reported by the daemon alongside a lookup index."""

    models = client.list_models()
    tags_index: Dict[str, Dict[str, Any]] = {}
    normalized: List[Dict[str, Any]] = []
    for item in models:
        name = item.get("name") or item.get("model")
        if not name:
            continue
        size = item.get("size")
        if isinstance(size, str):
            try:
                size = int(size)
            except ValueError:
                size = None
        normalized.append({"name": name, "size": size, "raw": item})
        tags_index[name] = item
    return normalized, tags_index


def _normalize_segment(seg: str, *, preserve_underscores: bool = False) -> str:
    """Normalize a name/tag segment into a deterministic, hyphenated token.

    Rules:
      - lower-case
      - replace '@', spaces and (optionally) underscores with '-'
      - replace forward slashes with '-'
      - collapse multiple consecutive hyphens
      - strip leading/trailing hyphens
    We deliberately preserve underscores in quant tokens when requested so that
    `q4_k_m` does not become `q4-k-m`, which aids matching & readability.
    """
    s = seg.strip().lower().replace("@", "-")
    if not preserve_underscores:
        s = s.replace("_", "-")
    # Always replace path separators & spaces
    s = s.replace("/", "-").replace(" ", "-")
    # Collapse repeats
    s = re.sub(r"-+", "-", s)
    s = s.strip("-")
    return s


def _split_name(name: str) -> tuple[str, Optional[str]]:
    if ":" not in name:
        return name, None
    base, tag = name.split(":", 1)
    return base, tag


def _derive_family_and_quant(full_name: str) -> tuple[str, Optional[str]]:
    base, tag = _split_name(full_name)
    path_segments = [seg for seg in re.split(r"[\\/]", base) if seg]
    if path_segments:
        base_candidate = path_segments[-1]
    else:
        base_candidate = base

    lower = base_candidate.lower()
    quant_from_base: Optional[str] = None

    for pattern in (
        r"q[0-9](?:[_-][a-z0-9]+)*",
        r"mxfp[0-9]+",
        r"mx?fp[0-9]+",
        r"int[0-9]+",
        r"fp[0-9]+",
    ):
        match = re.search(pattern, lower)
        if match:
            quant_from_base = match.group(0).replace("-", "_")
            lower = lower[: match.start()] + " " + lower[match.end() :]
            break

    lower = re.sub(r"(gguf|ggml|safetensors)", " ", lower)
    base_norm = _normalize_segment(lower)
    family_norm = base_norm or _normalize_segment(base_candidate)

    if tag is None:
        return family_norm, quant_from_base.lower() if quant_from_base else None

    tag_raw_norm = _normalize_segment(tag, preserve_underscores=True)
    tag_norm_for_family = _normalize_segment(tag)
    if is_quant_token(tag_raw_norm):
        return family_norm, tag_raw_norm

    family = f"{family_norm}-{tag_norm_for_family}"
    quant = quant_from_base.lower() if quant_from_base else None
    return family, quant


def _collect_model_data(
    client: Optional[OllamaClient],
    model: dict,
    tags_index: Dict[str, Dict[str, Any]],
) -> OllamaModelData | None:
    name = model.get("name") or model.get("model")
    size_bytes = model.get("size") or 0
    if not name:
        return None

    detail: Dict[str, Any] = {}
    if client is not None:
        try:
            detail = client.show_model(name) or {}
        except OllamaError:
            detail = {}

    detail_size = 0
    layers = detail.get("layers") or []
    if isinstance(layers, list):
        for layer in layers:
            if isinstance(layer, dict) and isinstance(layer.get("size"), int):
                detail_size += layer["size"]
    if detail_size:
        size_bytes = detail_size

    details_obj = (
        detail.get("details") if isinstance(detail.get("details"), dict) else {}
    )
    quant_detail = None
    if isinstance(details_obj, dict):
        # Prefer authoritative quantization_level per spec
        quant_detail = (
            details_obj.get("quantization_level")
            or details_obj.get("quantization")
            or details_obj.get("quant")
        )

    family_name = details_obj.get("family") if isinstance(details_obj, dict) else None
    architecture = (
        details_obj.get("architecture") if isinstance(details_obj, dict) else None
    )
    parameters = (
        details_obj.get("parameters") if isinstance(details_obj, dict) else None
    )
    parameter_size = None
    if isinstance(details_obj, dict):
        # Prefer explicit parameter_size if provided by Ollama
        parameter_size = details_obj.get("parameter_size")
    context_length = None
    if isinstance(details_obj, dict):
        context_length = details_obj.get("context_length") or details_obj.get(
            "context_length_tokens"
        )
    embedding_length = (
        details_obj.get("embedding_length") if isinstance(details_obj, dict) else None
    )

    capabilities = None
    modelfile = detail.get("modelfile")
    if isinstance(modelfile, dict):
        params = modelfile.get("parameters")
        if isinstance(params, list):
            capabilities = params
    # Fallback: if quant still unknown and modelfile is a string, try to infer from FROM line
    inferred_from_modelfile: Optional[str] = None
    if not quant_detail and isinstance(modelfile, str) and modelfile.strip():
        try:
            for line in modelfile.splitlines():
                if line.strip().upper().startswith("FROM"):
                    # Try to extract a token like Q4_0 / Q5_K_M / Q8_0 from filename
                    parts = line.split()
                    ref = parts[-1] if parts else ""
                    tokens = [p for p in re.split(r"[/\\.]", ref) if p]
                    for tok in tokens:
                        t = tok.strip().upper()
                        if re.match(r"^Q[0-9]_[01]$", t) or re.match(
                            r"^Q[0-9]_K(_[SML])?$", t
                        ):
                            inferred_from_modelfile = t
                            break
                    if inferred_from_modelfile:
                        break
        except Exception:  # noqa: BLE001
            inferred_from_modelfile = None
    if not capabilities and isinstance(detail.get("capabilities"), list):
        capabilities = detail.get("capabilities")

    family, quant_inferred = _derive_family_and_quant(name)
    # Prefer authoritative quantization_level, but ignore placeholders like 'unknown'
    quant = None
    for candidate in (quant_detail, inferred_from_modelfile, quant_inferred):
        if isinstance(candidate, str) and candidate.strip():
            if candidate.strip().lower() == "unknown":
                continue
            quant = candidate
            break
    if isinstance(quant, str):
        quant = quant.lower()

    extra_meta: Dict[str, Any] = {
        "ollama_family": family_name,
        "ollama_architecture": architecture,
        "ollama_parameters": parameters,
        "ollama_parameter_size": parameter_size
        or parameters,  # retain legacy key compatibility
        "ollama_context_length": context_length,
        "ollama_embedding_length": embedding_length,
        "ollama_capabilities": capabilities,
    }

    tag_entry = tags_index.get(name) or {}
    if not quant and isinstance(tag_entry, dict):
        tag_details = tag_entry.get("details") or {}
        if isinstance(tag_details, dict):
            # Keep this as an additional fallback
            quant_candidate = tag_details.get("quantization_level")
            if isinstance(quant_candidate, str):
                quant = quant_candidate.lower()
            if extra_meta.get("ollama_parameters") is None:
                extra_meta["ollama_parameters"] = tag_details.get("parameter_size")

    path = Path(f"ollama://{name}")

    return OllamaModelData(
        name=name,
        family=family,
        quant=quant,
        size_bytes=size_bytes,
        path=path,
        extra_metadata=extra_meta,
        architecture=architecture,
        parameters=parameters,
        context_length=context_length,
        embedding_length=embedding_length,
        capabilities=capabilities,
    )


def _print_dry_run(data: OllamaModelData) -> None:
    quant_display = data.quant or "-"
    print(
        "DRY RUN: would import"
        f" {data.name} -> variant_family={data.family} quant={quant_display} size={data.size_bytes}"
    )
    if data.architecture or data.parameters or data.context_length:
        print(
            "         arch="
            f"{data.architecture or '-'} params={data.parameters or '-'} ctx={data.context_length or '-'}"
        )


def _purge_existing_ollama_entries(backend: str) -> int:
    """Remove all existing discovered entries for the given backend (ollama) so we can
    rebuild cleanly without relying on ad-hoc duplicate collapse logic.

    We intentionally do NOT attempt to merge metadata; the discovery pass is
    considered authoritative for current local state. Curated entries (if any)
    remain untouched because we only remove entries whose backend matches and
    which are not in the curated layer (handled implicitly by save policy).
    """
    from imageworks.model_loader import registry as _reg

    reg = _reg.load_registry(force=True)
    to_remove = [
        name
        for name, e in reg.items()
        if e.backend == backend and e.source_provider == "ollama"
    ]
    removed = 0
    for name in to_remove:
        _reg.remove_entry(name, save=False)
        removed += 1
    if removed:
        _reg.save_registry()
    return removed


def _prepare_registry_for_import(
    backend: str, *, purge: bool, dry_run: bool
) -> Dict[str, Any]:
    if purge and not dry_run:
        removed = _purge_existing_ollama_entries(backend)
        if removed:
            print(f"Purged {removed} existing ollama entries before re-import.")
    return unified_registry.load_registry(force=True)


def _persist_existing_entry(
    existing, data: OllamaModelData, identity: ModelIdentity, *, location: str
) -> None:
    # Use simplified slug for canonical name
    existing.name = simplified_slug_for_fields(
        family=identity.family_key,
        backend=identity.backend_key,
        format_type=identity.format_key or "gguf",
        quantization=identity.quant_key,
        metadata=existing.metadata,
        download_path=str(data.path),
        served_model_id=data.name,
    )
    existing.family = identity.family_key
    existing.backend = identity.backend_key
    existing.quantization = identity.quant_key or existing.quantization
    existing.download_path = str(data.path)
    existing.download_format = identity.format_key or "gguf"
    existing.download_location = location
    if data.size_bytes:
        existing.download_size_bytes = data.size_bytes
    existing.served_model_id = data.name
    existing.source_provider = "ollama"
    if data.extra_metadata:
        existing.metadata = existing.metadata or {}
        for key, value in data.extra_metadata.items():
            if value is not None and key not in existing.metadata:
                existing.metadata[key] = value
    # Refresh display name using simplified naming so CLI / proxy stay concise.
    try:
        from imageworks.model_loader.simplified_naming import (
            simplified_display_for_fields as _simple_disp,
        )

        display_meta = existing.metadata or {}
        existing.display_name = _simple_disp(
            family=existing.family,
            backend=existing.backend,
            format_type=identity.format_key or "gguf",
            quantization=existing.quantization,
            metadata=display_meta,
            download_path=str(data.path),
            served_model_id=data.name,
        )
    except Exception:
        existing.display_name = identity.display_name
    existing.capabilities = derive_capabilities(existing, metadata=existing.metadata)
    if existing.backend == "ollama" and not getattr(
        existing.backend_config, "host", None
    ):
        existing.backend_config.host = os.environ.get(
            "IMAGEWORKS_OLLAMA_HOST", "imageworks-ollama"
        )
    unified_registry.update_entries([existing], save=True)


def _persist_new_entry(
    data: OllamaModelData, identity: ModelIdentity, *, location: str
):
    entry = record_download(
        hf_id=None,
        backend=identity.backend_key,
        format_type=identity.format_key or "gguf",
        quantization=identity.quant_key,
        path=str(data.path),
        location=location,
        files=None,
        size_bytes=data.size_bytes,
        source_provider="ollama",
        roles=None,
        role_priority=None,
        family_override=identity.family_key,
        served_model_id=data.name,
        extra_metadata=data.extra_metadata,
        display_name=identity.display_name,
    )
    return entry


def _persist_model(
    data: OllamaModelData,
    *,
    backend: str,
    location: str,
    registry: Dict[str, Any],
) -> None:
    identity = build_identity(
        family=data.family,
        backend=backend,
        format_type="gguf",
        quantization=data.quant,
    )
    variant_name = simplified_slug_for_fields(
        family=identity.family_key,
        backend=backend,
        format_type="gguf",
        quantization=identity.quant_key,
        metadata=None,
        download_path=str(data.path),
        served_model_id=data.name,
    )
    existing = registry.get(variant_name)
    if not existing and identity.quant_key:
        legacy_identity = build_identity(
            family=data.family,
            backend=backend,
            format_type="gguf",
            quantization=None,
        )
        legacy_name = legacy_identity.slug
        legacy_entry = registry.get(legacy_name)
        if legacy_entry:
            legacy_entry.name = variant_name
            registry.pop(legacy_name, None)
            existing = legacy_entry
    if existing:
        _persist_existing_entry(existing, data, identity, location=location)
        registry[existing.name] = existing
        return
    try:
        entry = _persist_new_entry(data, identity, location=location)
    except ImportSkipped:
        # Skip testing/demo placeholder entries silently
        return
    registry[entry.name] = entry


def _deprecate_placeholder_entries(backend: str) -> None:
    reg = unified_registry.load_registry(force=True)
    changed = 0
    for entry in reg.values():
        if entry.backend == backend and entry.name.startswith("model-ollama-gguf"):
            if not entry.deprecated:
                entry.deprecated = True
                changed += 1
    if changed:
        unified_registry.save_registry()
        print(f"Deprecated {changed} placeholder entries.")


def import_models(
    client: Optional[OllamaClient],
    models: list[dict],
    *,
    backend: str,
    location: str,
    dry_run: bool,
    deprecate_placeholders: bool,
    purge: bool,
    tags_index: Optional[Dict[str, Dict[str, Any]]] = None,
) -> int:
    imported = 0
    tags_index = tags_index or {}
    registry = _prepare_registry_for_import(backend, purge=purge, dry_run=dry_run)
    for model in models:
        data = _collect_model_data(client, model, tags_index)
        if data is None:
            continue
        if dry_run:
            _print_dry_run(data)
            imported += 1
            continue
        _persist_model(
            data,
            backend=backend,
            location=location,
            registry=registry,
        )
        imported += 1
    if not dry_run and deprecate_placeholders:
        _deprecate_placeholder_entries(backend)
    return imported


def run_import(
    *,
    backend: str = DEFAULT_BACKEND,
    location: str = "linux_wsl",
    dry_run: bool = False,
    deprecate_placeholders: bool = False,
    purge_existing: bool = False,
    client: Optional[OllamaClient] = None,
) -> int:
    """Programmatic entry point, optionally reusing an existing OllamaClient."""

    owns_client = client is None
    local_client = client or OllamaClient()
    tags_index: Dict[str, Dict[str, Any]] = {}

    try:
        models, tags_index = list_ollama_models(local_client)
    except OllamaError:
        if not dry_run:
            if owns_client:
                local_client.close()
            raise
        print(
            "[import-ollama] Ollama API unavailable; using sample dataset for dry run.",
            file=sys.stderr,
        )
        models = [dict(item) for item in _FALLBACK_SAMPLE_MODELS]
        tags_index = {
            item["name"]: {} for item in models if isinstance(item.get("name"), str)
        }
        if owns_client:
            local_client.close()
            local_client = None

    try:
        count = import_models(
            local_client,
            models,
            backend=backend,
            location=location,
            dry_run=dry_run,
            deprecate_placeholders=deprecate_placeholders,
            purge=purge_existing,
            tags_index=tags_index,
        )
    finally:
        if owns_client and local_client is not None:
            local_client.close()

    return count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import Ollama models into unified registry (Strategy A naming)"
    )
    parser.add_argument("--backend", default=DEFAULT_BACKEND)
    parser.add_argument("--location", default="linux_wsl")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--deprecate-placeholders",
        action="store_true",
        help="Mark legacy model-ollama-* placeholder entries deprecated after import",
    )
    parser.add_argument(
        "--purge-existing",
        action="store_true",
        help="Remove all existing discovered ollama entries before import (fresh rebuild)",
    )
    args = parser.parse_args()
    try:
        count = run_import(
            backend=args.backend,
            location=args.location,
            dry_run=args.dry_run,
            deprecate_placeholders=args.deprecate_placeholders,
            purge_existing=args.purge_existing,
        )
    except OllamaError as exc:
        print(f"Error listing Ollama models: {exc}", file=sys.stderr)
        return 1
    print(f"Imported {count} Ollama model(s){' (dry run)' if args.dry_run else ''}.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
