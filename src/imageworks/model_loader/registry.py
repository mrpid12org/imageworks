"""Registry provider with layered (curated + discovered) support.

Implements a two-layer model registry:

* Curated layer: `model_registry.curated.json` - hand maintained, stable metadata.
* Discovered layer: `model_registry.discovered.json` - auto generated / mutated by tooling.
* Merged snapshot: `model_registry.json` - materialized union used for backward compatibility
    and external tooling that still expects the single-file form (read-only conceptually).

Migration: If only the legacy single `model_registry.json` exists we will perform a one-time
split on first load and classify entries into curated or discovered heuristically.
The original unified file is renamed with a `.backup.pre_split.json` suffix so nothing is lost.

Classification heuristic (initial simple rule set):
    - Entries with metadata.created_from_download == True => discovered
    - Entries whose backend in {"ollama", "unassigned"} => discovered (runtime / logical presence)
    - All others => curated

Update semantics:
    - Mutations never edit the curated file in-place.
    - Any entry changed or newly created is written (full entry) into the discovered file.
        (Overlay pattern: discovered overrides curated by name on merge.)
    - Dynamic fields (download_*, last_accessed, performance, probes, metadata.created_from_download)
        trigger the entry being considered discovered even if it originated curated.

Environment overrides:
    - IMAGEWORKS_REGISTRY_DIR: base directory (defaults to `configs`). Allows tests to use temp dirs.
    - IMAGEWORKS_ALLOW_REGISTRY_DUPES: retains previous duplicate tolerance (applied after merge).

NOTE: This is an initial implementation; future refinement may introduce sparse overlays inside
the discovered file instead of whole-entry copies for curated overlays.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Iterable, Set

from .models import (
    ArtifactFile,
    Artifacts,
    BackendConfig,
    ChatTemplate,
    PerformanceLastSample,
    PerformanceSummary,
    Probes,
    RegistryEntry,
    VersionLock,
    VisionProbe,
)

_REGISTRY_CACHE: Dict[str, RegistryEntry] | None = None
_REGISTRY_PATH: Path | None = None  # Path to merged snapshot (back-compat single file)
_CURATED_NAMES: Set[str] = set()  # Names originating from curated layer (pre overlay)
_SINGLE_FILE_MODE: bool = False  # Explicit path (legacy/simple) loading


def _registry_dir() -> Path:
    return Path(os.environ.get("IMAGEWORKS_REGISTRY_DIR", "configs"))


def _curated_path() -> Path:
    return _registry_dir() / "model_registry.curated.json"


def _discovered_path() -> Path:
    return _registry_dir() / "model_registry.discovered.json"


def _merged_snapshot_path() -> Path:
    return _registry_dir() / "model_registry.json"


def _is_dynamic(entry: RegistryEntry) -> bool:
    """Return True if the entry contains dynamic / runtime-managed fields.

    This triggers writing the entry into the discovered overlay even if it originated curated.
    """
    if entry.metadata.get("created_from_download"):
        return True
    dynamic_attr_presence = any(
        [
            entry.download_path,
            entry.downloaded_at,
            entry.last_accessed,
            entry.download_format,
            entry.download_size_bytes,
            entry.performance.rolling_samples > 0,
            bool(entry.probes.vision),
        ]
    )
    return dynamic_attr_presence


def _classify_legacy(raw_entries: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split legacy unified entries into (curated, discovered) lists."""
    curated: list[dict] = []
    discovered: list[dict] = []
    for e in raw_entries:
        meta = e.get("metadata") or {}
        backend = e.get("backend")
        if meta.get("created_from_download") or backend in {"ollama", "unassigned"}:
            discovered.append(e)
        else:
            curated.append(e)
    return curated, discovered


class RegistryLoadError(RuntimeError):
    pass


def load_registry(
    path: Path | None = None, *, force: bool = False
) -> Dict[str, RegistryEntry]:
    """Load merged registry (curated + discovered) with on-demand migration.

    Args:
        path: Optional legacy single-file path (ignored once split). Provided for
              backward compatibility; if given and layered files exist it's ignored.
        force: Force reload.
    """
    global _REGISTRY_CACHE, _REGISTRY_PATH, _CURATED_NAMES, _SINGLE_FILE_MODE  # noqa: PLW0603

    if _REGISTRY_CACHE is not None and not force:
        return _REGISTRY_CACHE

    base_dir = _registry_dir()
    merged_snapshot = _merged_snapshot_path()
    # Determine if caller requested explicit standalone path different from default merged snapshot.
    explicit_path = path is not None and path != merged_snapshot
    no_layering = os.environ.get("IMAGEWORKS_REGISTRY_NO_LAYERING") == "1"
    if explicit_path or (no_layering and path is not None):
        # Standalone legacy/simple registry load (used heavily in tests). No layering, no migration.
        file_path = Path(path)  # type: ignore[arg-type]
        if not file_path.exists():
            raise RegistryLoadError(f"Registry file not found: {file_path}")
        try:
            data = json.loads(file_path.read_text())
        except Exception as exc:  # noqa: BLE001
            raise RegistryLoadError(f"Failed to parse registry: {exc}") from exc
        if not isinstance(data, list):
            raise RegistryLoadError("Registry root must be a list of entries")
        entries: Dict[str, RegistryEntry] = {}
        seen: dict[str, int] = {}
        for raw in data:
            entry = _parse_entry(raw)
            if entry.name in entries:
                # In explicit mode duplicates are errors unless tolerance env set
                tolerate = bool(
                    int(os.environ.get("IMAGEWORKS_ALLOW_REGISTRY_DUPES", "0"))
                )
                if not tolerate:
                    raise RegistryLoadError(f"Duplicate entry: {entry.name}")
                seen[entry.name] = seen.get(entry.name, 0) + 1
            entries[entry.name] = entry
        _REGISTRY_CACHE = entries
        _REGISTRY_PATH = file_path
        _SINGLE_FILE_MODE = True
        return entries

    # If no_layering requested but no explicit path provided, fall back to layered default
    # Layered mode (default path)
    base_dir.mkdir(parents=True, exist_ok=True)
    _SINGLE_FILE_MODE = False
    curated_file = _curated_path()
    discovered_file = _discovered_path()

    migrated = False
    if (
        not curated_file.exists() and not discovered_file.exists()
    ) and merged_snapshot.exists():
        try:
            legacy_data = json.loads(merged_snapshot.read_text())
            if not isinstance(legacy_data, list):
                raise ValueError("Legacy registry not a list")
        except Exception as exc:  # noqa: BLE001
            raise RegistryLoadError(
                f"Failed to parse legacy registry during migration: {exc}"
            ) from exc
        curated_raw, discovered_raw = _classify_legacy(legacy_data)
        backup_path = merged_snapshot.with_suffix(".backup.pre_split.json")
        if not backup_path.exists():
            merged_snapshot.rename(backup_path)
        curated_file.write_text(json.dumps(curated_raw, indent=2) + "\n")
        discovered_file.write_text(json.dumps(discovered_raw, indent=2) + "\n")
        migrated = True

    if not curated_file.exists() and not discovered_file.exists():
        raise RegistryLoadError(
            f"No registry files found in {base_dir} (expected curated or discovered layer)."
        )

    def _load_list(p: Path) -> list[dict]:
        if not p.exists():
            return []
        try:
            data = json.loads(p.read_text())
        except Exception as exc:  # noqa: BLE001
            raise RegistryLoadError(f"Failed to parse {p.name}: {exc}") from exc
        if not isinstance(data, list):
            raise RegistryLoadError(f"Registry fragment {p.name} must be a list")
        return data

    curated_raw = _load_list(curated_file)
    discovered_raw = _load_list(discovered_file)

    # Adopt any externally injected entries added directly to merged snapshot file (legacy tests/scripts)
    if merged_snapshot.exists():
        try:
            merged_content = json.loads(merged_snapshot.read_text())
            if isinstance(merged_content, list):
                known = {e.get("name") for e in curated_raw} | {
                    e.get("name") for e in discovered_raw
                }
                for raw in merged_content:
                    n = raw.get("name") if isinstance(raw, dict) else None
                    if n and n not in known:
                        # treat as discovered addition
                        discovered_raw.append(raw)
                        known.add(n)
        except Exception:  # noqa: BLE001
            pass
    name_index: dict[str, dict] = {}
    _CURATED_NAMES = set()
    for raw in curated_raw:
        name = str(raw.get("name") or "").strip()
        if not name:
            continue
        name_index[name] = raw
        _CURATED_NAMES.add(name)
    for raw in discovered_raw:
        name = str(raw.get("name") or "").strip()
        if not name:
            continue
        name_index[name] = raw
    merged_raw: list[dict] = list(name_index.values())

    entries: Dict[str, RegistryEntry] = {}
    duplicate_names: dict[str, int] = {}
    tolerate_dupes = bool(int(os.environ.get("IMAGEWORKS_ALLOW_REGISTRY_DUPES", "0")))
    for raw in merged_raw:
        try:
            entry = _parse_entry(raw)
        except Exception as exc:  # noqa: BLE001
            raise RegistryLoadError(f"Invalid entry: {exc}") from exc
        existing = entries.get(entry.name)
        if existing is None:
            entries[entry.name] = entry
            continue
        if not tolerate_dupes:
            raise RegistryLoadError(f"Duplicate entry after merge: {entry.name}")
        # Tolerant duplicate strategy (should be rare with layering)
        entries[entry.name] = entry
        duplicate_names[entry.name] = duplicate_names.get(entry.name, 0) + 1

    if duplicate_names and tolerate_dupes:
        import sys  # noqa: WPS433

        summary = ", ".join(f"{n} x{c+1}" for n, c in sorted(duplicate_names.items()))
        print(
            f"[imageworks.registry] Warning: duplicates after layering: {summary}",
            file=sys.stderr,
        )

    # Materialize merged snapshot only if migrated or content differs to avoid spurious file churn.
    try:
        current_content = None
        if merged_snapshot.exists():
            current_content = merged_snapshot.read_text()
        new_content = json.dumps(merged_raw, indent=2) + "\n"
        if migrated or current_content != new_content:
            merged_snapshot.write_text(new_content)
    except Exception:  # noqa: BLE001
        pass

    _REGISTRY_CACHE = entries
    _REGISTRY_PATH = merged_snapshot
    return entries


def get_entry(name: str) -> RegistryEntry:
    if _REGISTRY_CACHE is None:
        load_registry()
    assert _REGISTRY_CACHE is not None  # for type checker
    try:
        return _REGISTRY_CACHE[name]
    except KeyError as exc:  # noqa: BLE001
        raise KeyError(f"Model '{name}' not found in registry") from exc


def _parse_entry(raw: dict) -> RegistryEntry:
    backend_cfg = raw.get("backend_config", {})
    artifacts_cfg = raw.get("artifacts", {})
    chat_tpl_cfg = raw.get("chat_template", {})
    version_lock_cfg = raw.get("version_lock", {})
    perf_cfg = raw.get("performance", {})
    probes_cfg = raw.get("probes", {})

    entry = RegistryEntry(
        name=str(raw["name"]).strip(),
        display_name=str(raw.get("display_name") or raw.get("name") or "").strip()
        or None,
        backend=str(raw["backend"]).strip(),
        backend_config=BackendConfig(
            port=int(backend_cfg.get("port", 0)),
            model_path=str(backend_cfg.get("model_path", "")),
            extra_args=list(backend_cfg.get("extra_args", []) or []),
        ),
        capabilities=dict(raw.get("capabilities", {})),
        artifacts=Artifacts(
            aggregate_sha256=str(artifacts_cfg.get("aggregate_sha256", "")),
            files=[
                ArtifactFile(path=f.get("path", ""), sha256=f.get("sha256", ""))
                for f in artifacts_cfg.get("files", [])
                if isinstance(f, dict)
            ],
        ),
        chat_template=ChatTemplate(
            source=str(chat_tpl_cfg.get("source", "embedded")),
            path=chat_tpl_cfg.get("path"),
            sha256=chat_tpl_cfg.get("sha256"),
        ),
        version_lock=VersionLock(
            locked=bool(version_lock_cfg.get("locked", False)),
            expected_aggregate_sha256=version_lock_cfg.get("expected_aggregate_sha256"),
            last_verified=version_lock_cfg.get("last_verified"),
        ),
        performance=PerformanceSummary(
            rolling_samples=int(perf_cfg.get("rolling_samples", 0)),
            ttft_ms_avg=perf_cfg.get("ttft_ms_avg"),
            throughput_toks_per_s_avg=perf_cfg.get("throughput_toks_per_s_avg"),
            last_sample=(
                PerformanceLastSample(
                    ttft_ms=perf_cfg.get("last_sample", {}).get("ttft_ms"),
                    tokens_generated=perf_cfg.get("last_sample", {}).get(
                        "tokens_generated"
                    ),
                    duration_ms=perf_cfg.get("last_sample", {}).get("duration_ms"),
                )
                if perf_cfg.get("last_sample")
                else None
            ),
        ),
        probes=Probes(
            vision=(
                VisionProbe(
                    vision_ok=probes_cfg.get("vision", {}).get("vision_ok", False),
                    timestamp=probes_cfg.get("vision", {}).get("timestamp", ""),
                    probe_version=probes_cfg.get("vision", {}).get("probe_version", ""),
                    latency_ms=probes_cfg.get("vision", {}).get("latency_ms"),
                    notes=probes_cfg.get("vision", {}).get("notes"),
                )
                if probes_cfg.get("vision")
                else None
            )
        ),
        profiles_placeholder=raw.get("profiles_placeholder"),
        metadata=dict(raw.get("metadata", {})),
        served_model_id=str(raw.get("served_model_id", "")).strip() or None,
        model_aliases=[str(a).strip() for a in raw.get("model_aliases", []) if a],
        roles=[str(r).strip() for r in raw.get("roles", []) if r],
        license=(str(raw.get("license")).strip() if raw.get("license") else None),
        source=(raw.get("source") if isinstance(raw.get("source"), dict) else None),
        deprecated=bool(raw.get("deprecated", False)),
        family=raw.get("family"),
        source_provider=raw.get("source_provider"),
        quantization=raw.get("quantization"),
        backend_alternatives=list(raw.get("backend_alternatives", []) or []),
        role_priority=dict(raw.get("role_priority", {}) or {}),
        download_format=raw.get("download_format"),
        download_location=raw.get("download_location"),
        download_path=raw.get("download_path"),
        download_size_bytes=raw.get("download_size_bytes"),
        download_files=list(raw.get("download_files", []) or []),
        download_directory_checksum=raw.get("download_directory_checksum"),
        downloaded_at=raw.get("downloaded_at"),
        last_accessed=raw.get("last_accessed"),
    )
    if not entry.name:
        raise ValueError("Missing model name")
    return entry


def list_models() -> List[str]:
    if _REGISTRY_CACHE is None:
        load_registry()
    assert _REGISTRY_CACHE is not None
    return sorted(_REGISTRY_CACHE.keys())


def _serialize_entry(entry: RegistryEntry) -> dict:
    return {
        "name": entry.name,
        "display_name": entry.display_name,
        "backend": entry.backend,
        "backend_config": {
            "port": entry.backend_config.port,
            "model_path": entry.backend_config.model_path,
            "extra_args": entry.backend_config.extra_args,
        },
        "capabilities": entry.capabilities,
        "artifacts": {
            "aggregate_sha256": entry.artifacts.aggregate_sha256,
            "files": [
                {"path": f.path, "sha256": f.sha256} for f in entry.artifacts.files
            ],
        },
        "chat_template": {
            "source": entry.chat_template.source,
            "path": entry.chat_template.path,
            "sha256": entry.chat_template.sha256,
        },
        "version_lock": {
            "locked": entry.version_lock.locked,
            "expected_aggregate_sha256": entry.version_lock.expected_aggregate_sha256,
            "last_verified": entry.version_lock.last_verified,
        },
        "performance": {
            "rolling_samples": entry.performance.rolling_samples,
            "ttft_ms_avg": entry.performance.ttft_ms_avg,
            "throughput_toks_per_s_avg": entry.performance.throughput_toks_per_s_avg,
            "last_sample": (
                {
                    "ttft_ms": entry.performance.last_sample.ttft_ms,
                    "tokens_generated": entry.performance.last_sample.tokens_generated,
                    "duration_ms": entry.performance.last_sample.duration_ms,
                }
                if entry.performance.last_sample
                else None
            ),
        },
        "probes": {
            "vision": (
                {
                    "vision_ok": entry.probes.vision.vision_ok,
                    "timestamp": entry.probes.vision.timestamp,
                    "probe_version": entry.probes.vision.probe_version,
                    "latency_ms": entry.probes.vision.latency_ms,
                    "notes": entry.probes.vision.notes,
                }
                if entry.probes.vision
                else None
            )
        },
        "profiles_placeholder": entry.profiles_placeholder,
        "metadata": entry.metadata,
        "served_model_id": entry.served_model_id,
        "model_aliases": entry.model_aliases,
        "roles": entry.roles,
        "license": entry.license,
        "source": entry.source,
        "deprecated": entry.deprecated,
        "family": entry.family,
        "source_provider": entry.source_provider,
        "quantization": entry.quantization,
        "backend_alternatives": entry.backend_alternatives,
        "role_priority": entry.role_priority,
        "download_format": entry.download_format,
        "download_location": entry.download_location,
        "download_path": entry.download_path,
        "download_size_bytes": entry.download_size_bytes,
        "download_files": entry.download_files,
        "download_directory_checksum": entry.download_directory_checksum,
        "downloaded_at": entry.downloaded_at,
        "last_accessed": entry.last_accessed,
    }


def save_registry(path: Path | None = None) -> Path:
    """Persist registry depending on mode.

    * Single-file mode: write all entries to provided or original path.
    * Layered mode: write discovered overlay & regenerate merged snapshot.
    """
    global _REGISTRY_PATH, _CURATED_NAMES, _SINGLE_FILE_MODE  # noqa: PLW0603
    if _REGISTRY_CACHE is None:
        raise RuntimeError("Registry not loaded; cannot save")

    if _SINGLE_FILE_MODE:
        target = path or _REGISTRY_PATH
        if target is None:
            target = _merged_snapshot_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        data = [_serialize_entry(e) for e in _REGISTRY_CACHE.values()]
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2) + "\n")
        tmp.replace(target)
        _REGISTRY_PATH = target
        return target

    curated_file = _curated_path()
    discovered_file = _discovered_path()
    merged_snapshot = _merged_snapshot_path()

    # Load existing curated
    curated_raw: list[dict] = []
    if curated_file.exists():
        try:
            curated_raw = json.loads(curated_file.read_text())
        except Exception:  # noqa: BLE001
            curated_raw = []
    curated_name_set = {e.get("name") for e in curated_raw if isinstance(e, dict)}

    discovered_out: list[dict] = []
    for entry in _REGISTRY_CACHE.values():
        if entry.name not in curated_name_set or _is_dynamic(entry):
            data = _serialize_entry(entry)
            # Policy: only curated layer controls deprecation. If an entry is NOT in curated
            # and marked deprecated, clear the flag unless a curated entry with same name
            # exists and is deprecated (shadow override case).
            if data.get("deprecated") and data["name"] not in curated_name_set:
                data["deprecated"] = False
            discovered_out.append(data)

    discovered_file.parent.mkdir(parents=True, exist_ok=True)
    dtmp = discovered_file.with_suffix(".tmp")
    dtmp.write_text(json.dumps(discovered_out, indent=2) + "\n")
    dtmp.replace(discovered_file)

    name_index: dict[str, dict] = {
        e["name"]: e for e in curated_raw if isinstance(e, dict) and e.get("name")
    }
    for raw in discovered_out:
        name_index[raw["name"]] = raw
    merged_raw = list(name_index.values())
    mtmp = merged_snapshot.with_suffix(".tmp")
    mtmp.write_text(json.dumps(merged_raw, indent=2) + "\n")
    mtmp.replace(merged_snapshot)
    _REGISTRY_PATH = merged_snapshot
    return merged_snapshot


def update_entries(entries: Iterable[RegistryEntry], *, save: bool = True) -> None:
    if _REGISTRY_CACHE is None:
        raise RuntimeError("Registry not loaded")
    for entry in entries:
        _REGISTRY_CACHE[entry.name] = entry
    if save:
        save_registry()


def remove_entry(name: str, *, save: bool = True) -> bool:
    """Remove an entry from the registry by name.

    Returns True if removed, False if not present.
    """
    if _REGISTRY_CACHE is None:
        raise RuntimeError("Registry not loaded")
    removed = name in _REGISTRY_CACHE
    if removed:
        del _REGISTRY_CACHE[name]
        if save:
            save_registry()
    return removed


__all__ = [
    "RegistryLoadError",
    "load_registry",
    "get_entry",
    "list_models",
    "save_registry",
    "update_entries",
    "remove_entry",
]
