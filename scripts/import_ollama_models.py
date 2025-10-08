"""Import locally pulled Ollama models into the unified registry.

Usage:
  uv run python scripts/import_ollama_models.py [--backend ollama] [--location linux_wsl] [--dry-run]

Requirements:
  - Ollama CLI installed and accessible in PATH.
  - `ollama list --format json` supported (Ollama >= 0.1.32).

This script maps each Ollama model to a variant name using the unified naming
rules (family derived from the model name reported by Ollama). Since Ollama
stores models in its internal store, we cannot trivially map to an external
path; we still record a pseudo path pointing to the Ollama models directory
if discoverable, otherwise we set download_path to None (meaning not directly
filesystem-managed by us). For reproducibility, you may later augment entries
with explicit artifact hashes once exported.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from imageworks.model_loader.download_adapter import record_download, ImportSkipped
from imageworks.model_loader.naming import build_identity, ModelIdentity
from imageworks.model_loader import registry as unified_registry
from imageworks.tools.model_downloader.quant_utils import is_quant_token

DEFAULT_BACKEND = "ollama"

_TAGS_CACHE: Dict[str, Dict] | None = None

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
    license_text: Optional[str]


def _fetch_tags() -> Dict[str, Dict]:
    global _TAGS_CACHE
    if _TAGS_CACHE is not None:
        return _TAGS_CACHE
    tags_index: Dict[str, Dict] = {}
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    # HTTP first
    try:
        with urllib.request.urlopen(
            f"{base_url}/api/tags", timeout=1.5
        ) as resp:  # nosec B310
            data = json.loads(resp.read().decode())
            if isinstance(data, dict) and isinstance(data.get("models"), list):
                for m in data["models"]:
                    name = m.get("name")
                    if not name:
                        continue
                    tags_index[name] = m
    except Exception:  # noqa: BLE001
        pass
    # Fallback: use list_ollama_models() parsed output for size only (we already do elsewhere)
    _TAGS_CACHE = tags_index
    return tags_index


def find_ollama_store() -> Optional[Path]:
    # Common Linux location ~/.ollama/models
    candidates = [
        Path(os.environ.get("OLLAMA_MODELS", "")),
        Path.home() / ".ollama" / "models",
    ]
    for c in candidates:
        if c and c.exists():
            return c
    return None


SIZE_UNITS = {"KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}


def _parse_size(value: str, unit: str) -> int:
    try:
        return int(float(value) * SIZE_UNITS[unit.upper()])
    except Exception:  # noqa: BLE001
        return 0


def _fallback_parse_plain_list(output: str) -> List[Dict]:
    """Parse the legacy plain-text table output of `ollama list`.

    Expected format (columns separated by variable spaces):

    NAME    ID    SIZE    MODIFIED
    qwen2.5vl:7b  5ced39dfa4ba  6.0 GB  7 hours ago
    """
    lines = [line.rstrip() for line in output.splitlines() if line.strip()]
    if not lines:
        return []
    # Remove header if first line starts with NAME
    if lines and lines[0].lower().startswith("name"):
        lines = lines[1:]
    models: List[Dict] = []
    pattern = re.compile(
        r"^(?P<name>\S+)\s+(?P<id>[0-9a-f]{6,})\s+(?P<size_val>\d+(?:\.\d+)?)\s+(?P<size_unit>[KMGTP]B)\s+(?P<modified>.+)$"
    )
    for line in lines:
        m = pattern.match(line)
        if not m:
            # Skip lines we cannot parse; continue
            continue
        size_bytes = _parse_size(m.group("size_val"), m.group("size_unit"))
        models.append(
            {
                "name": m.group("name"),
                "id": m.group("id"),
                "size": size_bytes,
                "modified": m.group("modified"),
            }
        )
    return models


def list_ollama_models() -> list[dict]:
    """Return ollama models, trying JSON first then falling back to plain text.

    Older Ollama versions do not support `--format json`; we detect this and
    parse the textual table instead.
    """
    # First attempt JSON mode
    try:
        proc = subprocess.run(
            ["ollama", "list", "--format", "json"],
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError as exc:  # noqa: BLE001
        raise RuntimeError("'ollama' binary not found in PATH") from exc
    except subprocess.CalledProcessError:
        proc = None  # fall back

    if proc and proc.returncode == 0:
        try:
            data = json.loads(proc.stdout)
            if isinstance(data, list):
                return data
        except Exception:
            # fall through to plaintext parsing
            pass

    # Plain text fallback
    try:
        proc_txt = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, check=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:  # noqa: BLE001
        raise RuntimeError(
            "Failed to run 'ollama list' in either JSON or plain modes"
        ) from exc

    models = _fallback_parse_plain_list(proc_txt.stdout)
    if not models:
        raise RuntimeError(
            "Could not parse output of 'ollama list' (no models parsed). Consider upgrading Ollama."
        )
    return models


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
    base_norm = _normalize_segment(base)
    if tag is None:
        return base_norm, None
    # For quant detection we preserve underscores first
    tag_raw_norm = _normalize_segment(tag, preserve_underscores=True)
    tag_norm_for_family = _normalize_segment(tag)
    # Strategy A: if tag is quant token → quant; else tag becomes part of family
    if is_quant_token(tag_raw_norm):
        return base_norm, tag_raw_norm
    # treat long tag names similarly (e.g., 7b, latest)
    family = f"{base_norm}-{tag_norm_for_family}"
    return family, None


def _collect_model_data(
    model: dict, tags_index: Dict[str, Dict]
) -> OllamaModelData | None:
    name = model.get("name") or model.get("model")
    size_bytes = model.get("size") or 0
    if not name:
        return None

    detail = _ollama_show(name) or {}
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
        quant_detail = details_obj.get("quantization") or details_obj.get("quant")

    architecture = (
        details_obj.get("architecture") if isinstance(details_obj, dict) else None
    )
    parameters = (
        details_obj.get("parameters") if isinstance(details_obj, dict) else None
    )
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
    if not capabilities and isinstance(detail.get("capabilities"), list):
        capabilities = detail.get("capabilities")

    license_text = None
    license_info = detail.get("license")
    if isinstance(license_info, str):
        license_text = license_info
    elif isinstance(license_info, dict):
        license_text = license_info.get("name") or license_info.get("text")

    family, quant_inferred = _derive_family_and_quant(name)
    quant = quant_detail or quant_inferred
    if isinstance(quant, str):
        quant = quant.lower()

    extra_meta: Dict[str, Any] = {
        "ollama_architecture": architecture,
        "ollama_parameters": parameters,
        "ollama_context_length": context_length,
        "ollama_embedding_length": embedding_length,
        "ollama_capabilities": capabilities,
        "ollama_license": license_text,
    }

    if not quant and name in tags_index:
        tag_details = tags_index[name].get("details") or {}
        if isinstance(tag_details, dict):
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
        license_text=license_text,
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


def _show_via_http(name: str) -> dict | None:
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        req = urllib.request.Request(
            f"{base_url}/api/show",
            data=json.dumps({"model": name}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=1.5) as resp:  # nosec B310
            body = resp.read().decode()
            data = json.loads(body)
            if isinstance(data, dict):
                return data
    except Exception:  # noqa: BLE001
        return None
    return None


def _show_via_python(name: str) -> dict | None:
    try:
        import importlib

        spec = importlib.util.find_spec("ollama")  # type: ignore[attr-defined]
        if spec is None:
            return None
        from ollama import show as _py_show  # type: ignore

        data = _py_show(name)
        if isinstance(data, dict):
            return data
    except Exception:  # noqa: BLE001
        return None
    return None


def _show_via_cli_json(name: str) -> dict | None:
    try:
        proc = subprocess.run(
            ["ollama", "show", name, "--format", "json"],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:  # noqa: BLE001
        return None
    try:
        data = json.loads(proc.stdout or "{}")
    except Exception:  # noqa: BLE001
        return None
    if isinstance(data, dict):
        return data
    return None


def _show_via_cli_text(name: str) -> dict | None:
    try:
        proc_txt = subprocess.run(
            ["ollama", "show", name], capture_output=True, text=True, check=True
        )
    except Exception:  # noqa: BLE001
        return None
    lines = proc_txt.stdout.splitlines()
    quant = None
    arch = None
    params = None
    ctx = None
    for line in lines:
        lower = line.strip().lower()
        if lower.startswith("quantization"):
            parts = line.split()
            if len(parts) >= 2:
                quant = parts[-1]
        elif lower.startswith("architecture"):
            arch = line.split()[-1]
        elif lower.startswith("parameters"):
            tokens = line.split()
            if tokens:
                params = tokens[-1]
        elif lower.startswith("context length"):
            tokens = line.split()
            if tokens:
                ctx = tokens[-1]
    return {
        "details": {
            "architecture": arch,
            "parameters": params,
            "context_length": ctx,
            "quantization": quant,
        }
    }


def _ollama_show(name: str) -> dict | None:
    """Return detailed metadata for a model using layered fallbacks."""

    for resolver in (
        _show_via_http,
        _show_via_python,
        _show_via_cli_json,
        _show_via_cli_text,
    ):
        data = resolver(name)
        if isinstance(data, dict):
            return data
    return None


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
    existing.name = identity.slug
    existing.family = identity.family_key
    existing.backend = identity.backend_key
    existing.quantization = identity.quant_key or existing.quantization
    existing.display_name = identity.display_name
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
    variant_name = identity.slug
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
    models: list[dict],
    *,
    backend: str,
    location: str,
    dry_run: bool,
    deprecate_placeholders: bool,
    purge: bool,
) -> int:
    imported = 0
    tags_index = _fetch_tags()
    registry = _prepare_registry_for_import(backend, purge=purge, dry_run=dry_run)
    for model in models:
        data = _collect_model_data(model, tags_index)
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
        models = list_ollama_models()
    except RuntimeError as exc:
        if args.dry_run:
            print(
                f"[import-ollama] {exc}. Using sample dataset for dry run output.",
                file=sys.stderr,
            )
            models = [dict(item) for item in _FALLBACK_SAMPLE_MODELS]
        else:
            print(f"Error listing Ollama models: {exc}", file=sys.stderr)
            return 1
    count = import_models(
        models,
        backend=args.backend,
        location=args.location,
        dry_run=args.dry_run,
        deprecate_placeholders=args.deprecate_placeholders,
        purge=args.purge_existing,
    )
    print(f"Imported {count} Ollama model(s){' (dry run)' if args.dry_run else ''}.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
