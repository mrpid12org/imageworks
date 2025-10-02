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

import json
import os
import subprocess
from pathlib import Path
from typing import Optional, List, Dict
import argparse
import re
import urllib.request

from imageworks.model_loader.download_adapter import record_download
from imageworks.model_loader import registry as unified_registry
from imageworks.tools.model_downloader.quant_utils import is_quant_token

DEFAULT_BACKEND = "ollama"

_TAGS_CACHE: Dict[str, Dict] | None = None


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


QUANT_PAT = re.compile(r".*")  # deprecated local; using is_quant_token instead


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


def _ollama_show(name: str) -> dict | None:
    """Return detailed metadata for a model using layered fallbacks.

    Order:
      1. Local HTTP API (GET /api/show) if server running
      2. Python library (ollama.show) if installed
      3. CLI `ollama show <name> --format json`
      4. CLI plain text parsing (extract quantization/architecture heuristically)
    """
    # 1. HTTP API
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    import urllib.request as _u

    try:
        req = _u.Request(
            f"{base_url}/api/show",
            data=json.dumps({"model": name}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with _u.urlopen(req, timeout=1.5) as resp:  # nosec B310
            body = resp.read().decode()
            data = json.loads(body)
            if isinstance(data, dict):
                return data
    except Exception:  # noqa: BLE001
        pass
    # 2. Python library
    try:
        import importlib

        if importlib.util.find_spec("ollama") is not None:  # type: ignore[attr-defined]
            from ollama import show as _py_show  # type: ignore

            try:
                data = _py_show(name)
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
    except Exception:
        pass
    # 3. CLI JSON
    try:
        proc = subprocess.run(
            ["ollama", "show", name, "--format", "json"],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(proc.stdout or "{}")
        if isinstance(data, dict):
            return data
    except Exception:  # noqa: BLE001
        pass
    # 4. Plain text parsing (very limited)
    try:
        proc_txt = subprocess.run(
            ["ollama", "show", name], capture_output=True, text=True, check=True
        )
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
                # Often like "parameters          8.3B"
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
    except Exception:  # noqa: BLE001
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
    # Load registry (merged) for updates
    from imageworks.model_loader import registry as _reg

    if purge and not dry_run:
        removed = _purge_existing_ollama_entries(backend)
        if removed:
            print(f"Purged {removed} existing ollama entries before re-import.")
    reg_map = _reg.load_registry(force=True)
    for m in models:
        name = m.get("name") or m.get("model")  # fields differ across versions
        size_bytes = m.get("size") or 0
        if not name:
            continue
        detail = _ollama_show(name) or {}
        # Attempt to refine size and quantization from detail if present
        detail_size = 0
        # Some versions include digest layers with sizes; sum them
        layers = detail.get("layers") or []
        if isinstance(layers, list):
            for layer in layers:
                if isinstance(layer, dict) and isinstance(layer.get("size"), int):
                    detail_size += layer["size"]
        if detail_size:
            size_bytes = detail_size
        # Attempt quant extraction directly from detail (e.g., model["details"]["quantization"])
        quant_detail = None
        details_obj = (
            detail.get("details") if isinstance(detail.get("details"), dict) else {}
        )
        if isinstance(details_obj, dict):
            quant_detail = details_obj.get("quantization") or details_obj.get("quant")
        # Additional metadata extraction
        architecture = (
            details_obj.get("architecture") if isinstance(details_obj, dict) else None
        )
        parameters = (
            details_obj.get("parameters") if isinstance(details_obj, dict) else None
        )
        context_length = (
            details_obj.get("context_length")
            or details_obj.get("context_length_tokens")
            if isinstance(details_obj, dict)
            else None
        )
        embedding_length = (
            details_obj.get("embedding_length")
            if isinstance(details_obj, dict)
            else None
        )
        capabilities = (
            detail.get("modelfile", {}).get("parameters", [])
            if isinstance(detail.get("modelfile"), dict)
            else None
        )
        # Some versions expose a top-level capabilities list
        if not capabilities and isinstance(detail.get("capabilities"), list):
            capabilities = detail.get("capabilities")
        license_text = None
        if isinstance(detail.get("license"), str):
            license_text = detail.get("license")
        elif isinstance(detail.get("license"), dict):
            # Some builds may structure license
            license_text = detail.get("license").get("name") or detail.get(
                "license"
            ).get("text")
        family, quant_inferred = _derive_family_and_quant(name)
        quant = quant_detail or quant_inferred
        if isinstance(quant, str):
            quant = quant.lower()
        # Prepare extra metadata container early so tag enrichment can modify it
        extra_meta = {
            "ollama_architecture": architecture,
            "ollama_parameters": parameters,
            "ollama_context_length": context_length,
            "ollama_embedding_length": embedding_length,
            "ollama_capabilities": capabilities,
            "ollama_license": license_text,
        }
        # If still missing quant, consult /api/tags index
        if not quant and name in tags_index:
            det = tags_index[name].get("details") or {}
            if isinstance(det, dict):
                quant = det.get("quantization_level") or quant
                # Parameter size (human form) if original parameters missing
                if extra_meta.get("ollama_parameters") is None:
                    extra_meta["ollama_parameters"] = det.get("parameter_size")
        path = Path(f"ollama://{name}")
        if dry_run:
            print(
                f"DRY RUN: would import {name} -> variant_family={family} quant={quant or '-'} size={size_bytes}"
            )
            # Show architecture/parameters in dry run if available
            if architecture or parameters:
                print(
                    f"         arch={architecture or '-'} params={parameters or '-'} ctx={context_length or '-'}"
                )
            imported += 1
            continue
        # Canonical naming rule (per user directive): variant name does NOT include quant.
        # We always use family-backend-format (no quant token) and store quant in quantization field.
        # display_name should be base family (without backend/format suffix) plus quant if present.
        base_variant_name = f"{family}-{backend}-gguf"
        existing = reg_map.get(base_variant_name)
        # The download adapter builds names including quant, so we bypass it and manually enrich existing entry if present,
        # otherwise we call record_download once (with quant) knowing it will create family-backend-format-quant; we then
        # immediately normalize its name by removing the quant segment. This keeps a single entry canonical.
        if existing:
            # Enrich existing entry
            existing.quantization = quant or existing.quantization
            existing.download_path = str(path)
            existing.download_format = "gguf"
            existing.download_location = location
            existing.download_size_bytes = size_bytes or existing.download_size_bytes
            existing.served_model_id = name
            # Merge metadata without overwriting existing keys
            if extra_meta:
                existing.metadata = existing.metadata or {}
                for k, v in extra_meta.items():
                    if v is not None and k not in existing.metadata:
                        existing.metadata[k] = v
            # Update display name pattern
            base_display = family
            if quant:
                base_display = f"{base_display}-{quant}"  # user wants quant shown
            # Trim packager prefix heuristics for hf.co style imports (hf.co-<user>-rest...)
            # We'll remove leading 'hf.co-' and first user token if base_display starts with that pattern.
            if base_display.startswith("hf.co-"):
                parts = base_display.split("-")
                if len(parts) > 2:  # hf.co, user, rest...
                    base_display = "-".join(parts[2:])
            existing.display_name = base_display
            from imageworks.model_loader.registry import update_entries, save_registry

            update_entries([existing], save=True)
            save_registry()
        else:
            # Create via record_download (will create with quant in name); then rename if necessary.
            entry = record_download(
                hf_id=None,
                backend=backend,
                format_type="gguf",
                quantization=quant,
                path=str(path),
                location=location,
                files=None,
                size_bytes=size_bytes,
                source_provider="ollama",
                roles=None,
                role_priority=None,
                family_override=family,
                served_model_id=name,
                extra_metadata=extra_meta,
            )
            # If record_download added quant to name, normalize it by collapsing back to base_variant_name.
            if entry.name != base_variant_name:
                from imageworks.model_loader.registry import (
                    update_entries,
                    remove_entry,
                    save_registry,
                )

                original_key = entry.name
                # Rename in-place by changing entry.name and re-registering
                entry.name = base_variant_name  # type: ignore[attr-defined]
                base_display = family
                if quant:
                    base_display = f"{base_display}-{quant}"
                if base_display.startswith("hf.co-"):
                    parts = base_display.split("-")
                    if len(parts) > 2:
                        base_display = "-".join(parts[2:])
                entry.display_name = base_display
                update_entries([entry], save=True)
                # Remove the old quant-suffixed key
                if original_key != base_variant_name:
                    remove_entry(original_key, save=False)
                save_registry()
        imported += 1
    if not dry_run and deprecate_placeholders:
        # Mark legacy placeholder entries deprecated
        reg = unified_registry.load_registry(force=True)
        changed = 0
        for e in reg.values():
            if e.backend == backend and e.name.startswith("model-ollama-gguf"):
                if not e.deprecated:
                    e.deprecated = True
                    changed += 1
        if changed:
            unified_registry.save_registry()
            print(f"Deprecated {changed} placeholder entries.")
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
    models = list_ollama_models()
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
