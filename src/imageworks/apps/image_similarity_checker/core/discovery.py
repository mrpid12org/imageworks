"""Filesystem helpers for locating candidate and library images."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Iterator, Sequence, Set, Tuple

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
) -> Tuple[Path, ...]:
    exclude_set = {path.resolve() for path in exclude}
    candidates = discover_images([library_root], recursive=recursive, extensions=extensions)
    return tuple(path for path in candidates if path.resolve() not in exclude_set)
