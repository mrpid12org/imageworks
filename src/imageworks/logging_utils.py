"""Centralised logging utilities for ImageWorks applications."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

__all__ = ["configure_logging"]

_MANAGED_HANDLER_FLAG = "_imageworks_managed_handler"


def _default_log_directory() -> Path:
    """Return the default directory for ImageWorks log files."""

    env_override = os.environ.get("IMAGEWORKS_LOG_DIR")
    if env_override:
        return Path(env_override).expanduser()

    module_path = Path(__file__).resolve()
    # Prefer the project root (folder containing pyproject.toml or .git)
    for candidate in [module_path] + list(module_path.parents):
        candidate_dir = candidate if candidate.is_dir() else candidate.parent
        if (candidate_dir / "pyproject.toml").exists() or (
            candidate_dir / ".git"
        ).exists():
            return candidate_dir / "logs"

    try:
        project_root = module_path.parents[3]
    except IndexError:  # pragma: no cover - defensive fallback for unusual installs
        project_root = Path.cwd()

    return project_root / "logs"


def _remove_managed_handlers(logger: logging.Logger) -> None:
    """Detach any handlers previously installed by :func:`configure_logging`."""

    for handler in list(logger.handlers):
        if getattr(handler, _MANAGED_HANDLER_FLAG, False):
            logger.removeHandler(handler)
            handler.close()


def configure_logging(
    log_name: str,
    *,
    level: int = logging.INFO,
    log_dir: Optional[Path] = None,
    include_console: bool = True,
) -> Path:
    """Configure root logging to write to a named file inside the log directory."""

    target_directory = (
        Path(log_dir).expanduser() if log_dir else _default_log_directory()
    )
    target_directory.mkdir(parents=True, exist_ok=True)
    log_path = target_directory / f"{log_name}.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    _remove_managed_handlers(root_logger)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    setattr(file_handler, _MANAGED_HANDLER_FLAG, True)
    root_logger.addHandler(file_handler)

    if include_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        setattr(console_handler, _MANAGED_HANDLER_FLAG, True)
        root_logger.addHandler(console_handler)

    logging.captureWarnings(True)

    return log_path
