"""Progress tracking utilities for Judge Vision runs."""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from datetime import UTC, datetime

_PROGRESS_REDIRECT_NAME = ".progress.json"


def normalise_progress_path(path: Path) -> Path:
    path = Path(path)
    if path.exists() and path.is_dir():
        return path / _PROGRESS_REDIRECT_NAME
    return path


@dataclass
class ProgressState:
    total: int = 0
    processed: int = 0
    current_image: Optional[str] = None
    status: str = "idle"  # idle | running | complete | error
    message: Optional[str] = None
    phase: Optional[str] = None
    history: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "processed": self.processed,
            "current_image": self.current_image,
            "status": self.status,
            "message": self.message,
            "phase": self.phase,
            "history": self.history,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProgressState":
        return cls(
            total=int(data.get("total", 0) or 0),
            processed=int(data.get("processed", 0) or 0),
            current_image=data.get("current_image"),
            status=str(data.get("status") or "idle"),
            message=data.get("message"),
            phase=data.get("phase"),
            history=list(data.get("history") or []),
        )


class ProgressTracker:
    def __init__(self, path: Path) -> None:
        self.raw_path = Path(path)
        self.path = normalise_progress_path(self.raw_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.state = ProgressState()
        if self.path.exists():
            try:
                existing = json.loads(self.path.read_text())
                self.state = ProgressState.from_dict(existing)
            except Exception:  # noqa: BLE001
                self.state = ProgressState()

    def reset(self, total: int, *, phase: Optional[str] = None) -> None:
        history = list(self.state.history)
        self.state = ProgressState(
            total=total,
            processed=0,
            current_image=None,
            status="running",
            message=None,
            phase=phase,
            history=history,
        )
        self._write()

    def update(self, *, processed: int, current_image: Optional[str]) -> None:
        self.state.processed = processed
        self.state.current_image = current_image
        self.state.status = "running"
        self.state.message = None
        self._write()

    def complete(self) -> None:
        self.state.status = "complete"
        self.state.message = None
        self._record_history_entry(status="complete")
        self.state.phase = None
        self._write()

    def fail(
        self, *, processed: int, current_image: Optional[str], message: str
    ) -> None:
        self.state.processed = processed
        self.state.current_image = current_image
        self.state.status = "error"
        self.state.message = message
        self._record_history_entry(status="error")
        self.state.phase = None
        self._write()

    def _record_history_entry(self, *, status: str) -> None:
        if not self.state.phase:
            return
        entry = {
            "phase": self.state.phase,
            "status": status,
            "processed": self.state.processed,
            "total": self.state.total,
            "message": self.state.message,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        self.state.history.append(entry)

    def _write(self) -> None:
        # Re-resolve path in case it changed (e.g., recreated directory)
        self.path = normalise_progress_path(self.raw_path)
        tmp_path = self.path.with_suffix(".tmp")
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            try:
                self.path.unlink()
            except IsADirectoryError:
                backup = self.path.with_suffix(f".dir.{int(time.time())}")
                try:
                    self.path.rename(backup)
                except OSError:
                    backup = None
                if backup and backup.exists():
                    shutil.rmtree(backup, ignore_errors=True)
                elif self.path.exists():
                    shutil.rmtree(self.path, ignore_errors=True)
        tmp_path.write_text(json.dumps(self.state.to_dict()), encoding="utf-8")
        tmp_path.replace(self.path)


__all__ = ["ProgressTracker", "ProgressState", "normalise_progress_path"]
