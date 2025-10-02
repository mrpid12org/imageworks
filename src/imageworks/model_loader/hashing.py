"""Hashing utilities for unified model registry (consolidated).

Provides:
1. Full recursive hashing (`compute_artifact_hashes`) producing deterministic aggregate hash
2. Targeted minimal file hashing (`update_entry_artifacts`) for legacy tests (config/tokenizer subset)
3. Version lock enforcement (`verify_model`) and lightweight verification API (`verify_entry_hash`)

Preferred path going forward is full hashing; targeted subset kept for backward compatibility with existing tests.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Iterable, List, Tuple

from .models import ArtifactFile, Artifacts, RegistryEntry
from .registry import save_registry

CHUNK_SIZE = 1024 * 1024


def hash_file(path: Path, chunk_size: int = CHUNK_SIZE) -> str:
    h = sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def iter_model_files(root: Path) -> Iterable[Path]:
    for p in sorted([p for p in root.rglob("*") if p.is_file()]):
        yield p


def compute_artifact_hashes(
    entry: RegistryEntry, *, include_patterns: List[str] | None = None
) -> RegistryEntry:
    root = Path(entry.backend_config.model_path).expanduser()
    if not root.exists():
        return entry
    if include_patterns:
        sel: List[Path] = []
        for pat in include_patterns:
            sel.extend(root.glob(pat))
        files = sorted({p for p in sel if p.is_file()})
    else:
        files = list(iter_model_files(root))
    artifact_files: List[ArtifactFile] = []
    agg_inputs: List[str] = []
    for fp in files:
        rel = fp.relative_to(root).as_posix()
        h = hash_file(fp)
        artifact_files.append(ArtifactFile(path=rel, sha256=h))
        agg_inputs.append(f"{h}  {rel}")
    agg_inputs.sort()
    aggregate = sha256("\n".join(agg_inputs).encode()).hexdigest()
    return replace(
        entry, artifacts=Artifacts(aggregate_sha256=aggregate, files=artifact_files)
    )


def _targeted_file_paths(root: Path) -> List[Path]:
    candidates = [
        "config.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "mm_projector.bin",
    ]
    found: List[Path] = []
    for name in candidates:
        p = root / name
        if p.exists():
            found.append(p)
    return found


def update_entry_artifacts(entry: RegistryEntry) -> RegistryEntry:
    root = Path(entry.backend_config.model_path).expanduser()
    if not root.exists():
        return entry
    files: List[ArtifactFile] = []
    for path in _targeted_file_paths(root):
        files.append(ArtifactFile(path=path.name, sha256=hash_file(path)))
    aggregate = (
        sha256("".join(sorted(f.sha256 for f in files)).encode()).hexdigest()
        if files
        else ""
    )
    entry.artifacts.files = files
    entry.artifacts.aggregate_sha256 = aggregate
    return entry


def update_entry_artifacts_from_download(entry: RegistryEntry) -> RegistryEntry:
    """Populate artifacts block from tracked download files if present.

    Uses full SHA256 for each listed file under entry.download_path and computes
    deterministic aggregate like compute_artifact_hashes (sorted lines "<sha>  <rel>").
    Skips if no download_path or no download_files or path missing.
    """
    if not entry.download_path or not entry.download_files:
        return entry
    root = Path(entry.download_path).expanduser()
    if not root.exists():
        return entry
    artifact_files: List[ArtifactFile] = []
    agg_inputs: List[str] = []
    for rel in sorted(entry.download_files):
        fp = root / rel
        if not fp.is_file():  # skip directories / missing
            continue
        h = hash_file(fp)
        artifact_files.append(ArtifactFile(path=rel, sha256=h))
        agg_inputs.append(f"{h}  {rel}")
    if not artifact_files:
        return entry
    aggregate = sha256("\n".join(sorted(agg_inputs)).encode()).hexdigest()
    entry.artifacts.files = artifact_files
    entry.artifacts.aggregate_sha256 = aggregate
    return entry


class VersionLockViolation(RuntimeError):
    pass


def verify_model(entry: RegistryEntry, *, enforce_lock: bool = True) -> RegistryEntry:
    # Prefer download-derived artifacts if we have download tracking and no existing artifact files
    if entry.download_path and entry.download_files and not entry.artifacts.files:
        update_entry_artifacts_from_download(entry)
    else:
        update_entry_artifacts(entry)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if (
        enforce_lock
        and entry.version_lock.locked
        and entry.version_lock.expected_aggregate_sha256
    ):
        if (
            entry.artifacts.aggregate_sha256
            != entry.version_lock.expected_aggregate_sha256
        ):
            raise VersionLockViolation(
                f"Version lock violation: expected {entry.version_lock.expected_aggregate_sha256} got {entry.artifacts.aggregate_sha256}"
            )
    entry.version_lock.last_verified = now
    if (
        entry.version_lock.locked
        and not entry.version_lock.expected_aggregate_sha256
        and entry.artifacts.aggregate_sha256
    ):
        entry.version_lock.expected_aggregate_sha256 = entry.artifacts.aggregate_sha256
    save_registry()
    return entry


def verify_entry_hash(entry: RegistryEntry) -> Tuple[bool, RegistryEntry]:
    updated = compute_artifact_hashes(entry)
    expected = entry.version_lock.expected_aggregate_sha256
    if (
        entry.version_lock.locked
        and expected
        and expected != updated.artifacts.aggregate_sha256
    ):
        return False, updated
    return True, updated


__all__ = [
    "hash_file",
    "compute_artifact_hashes",
    "update_entry_artifacts",
    "update_entry_artifacts_from_download",
    "verify_model",
    "verify_entry_hash",
    "VersionLockViolation",
]
