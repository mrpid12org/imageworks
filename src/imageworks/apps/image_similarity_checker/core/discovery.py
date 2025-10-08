"""Filesystem helpers for locating candidate and library images."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Iterator, Sequence, Set, Tuple
import json
import hashlib

logger = logging.getLogger(__name__)


def _normalise_extensions(extensions: Sequence[str]) -> Tuple[str, ...]:
    result = []
    for ext in extensions:
        if not ext:
            continue
        clean = ext.lower()
        result.append(clean if clean.startswith(".") else f".{clean}")
    return tuple(dict.fromkeys(result))


def _iter_directory(directory: Path, recursive: bool) -> Iterator[Path]:
    if recursive:
        yield from directory.rglob("*")
    else:
        yield from directory.iterdir()


def discover_images(
    paths: Sequence[Path],
    *,
    recursive: bool,
    extensions: Sequence[str],
) -> Tuple[Path, ...]:
    """Return sorted unique image files under *paths*."""

    allowed = _normalise_extensions(extensions)
    seen: Set[Path] = set()
    files: Set[Path] = set()

    for root in paths:
        if not root.exists():
            logger.debug("Skipping missing input path: %s", root)
            continue
        if root.is_file():
            if root.suffix.lower() in allowed:
                files.add(root.resolve())
            continue
        if not root.is_dir():
            logger.debug("Skipping non-file path: %s", root)
            continue
        for candidate in _iter_directory(root, recursive):
            if not candidate.is_file():
                continue
            if candidate.suffix.lower() not in allowed:
                continue
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            files.add(resolved)

    return tuple(sorted(files))


def discover_library(
    library_root: Path,
    *,
    recursive: bool,
    extensions: Sequence[str],
    exclude: Iterable[Path] = (),
    refresh_library_cache: bool = False,  # kept for compatibility; no-op
    manifest_ttl_seconds: int = 0,  # kept for compatibility; no-op
) -> Tuple[Path, ...]:
    exclude_set = {path.resolve() for path in exclude}

    # Simple manifest cache under the library root (portable and stable across filesystems)
    try:
        manifest_dir = library_root / ".imageworks_cache"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        stat = library_root.stat()
        key = json.dumps(
            {
                "root": str(library_root.resolve()),
                "recursive": bool(recursive),
                "ext": list(_normalise_extensions(extensions)),
                "mtime": stat.st_mtime,
            },
            sort_keys=True,
        )
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
        manifest_path = manifest_dir / f"library_manifest__{digest}.json"
        if manifest_path.exists():
            try:
                data = json.loads(manifest_path.read_text(encoding="utf-8"))
                files = tuple(Path(p) for p in data.get("files", []))
                return tuple(p for p in files if p.resolve() not in exclude_set)
            except Exception:
                pass
    except Exception:
        manifest_path = None  # type: ignore[assignment]

    candidates = discover_images(
        [library_root], recursive=recursive, extensions=extensions
    )

    # Save manifest
    try:
        if "manifest_path" in locals() and manifest_path is not None:
            manifest = {"files": [str(p) for p in candidates]}
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    except Exception:
        pass

    return tuple(path for path in candidates if path.resolve() not in exclude_set)
