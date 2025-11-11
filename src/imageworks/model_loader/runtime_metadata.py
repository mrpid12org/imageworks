"""Utilities for capturing and reconciling runtime loader metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

LOG_PATH = Path("logs/model_loader_metrics.jsonl")


@dataclass
class RuntimeEvent:
    """Structured representation of a runtime loader observation."""

    timestamp: str
    model: str
    backend: str
    served_model_id: Optional[str]
    payload: Dict[str, Any]

    @classmethod
    def from_raw(cls, obj: Dict[str, Any]) -> "RuntimeEvent | None":
        try:
            timestamp = str(obj["timestamp"])
            model = str(obj["model"])
            backend = str(obj.get("backend") or "")
            served_model_id = obj.get("served_model_id")
            payload = dict(obj.get("payload") or {})
        except Exception:  # noqa: BLE001
            return None
        if not model or not timestamp:
            return None
        return cls(
            timestamp=timestamp,
            model=model,
            backend=backend or "unknown",
            served_model_id=str(served_model_id) if served_model_id else None,
            payload=payload,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "model": self.model,
            "backend": self.backend,
            "served_model_id": self.served_model_id,
            "payload": self.payload,
        }


def _ensure_log_dir() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def log_runtime_metrics(
    *,
    entry_name: str,
    backend: str,
    served_model_id: Optional[str],
    payload: Dict[str, Any],
    log_path: Path = LOG_PATH,
) -> None:
    """Append a runtime observation to the JSONL log."""

    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": entry_name,
        "backend": backend,
        "served_model_id": served_model_id,
        "payload": payload,
    }
    _ensure_log_dir()
    try:
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001
        # Logging must never be fatal; swallow errors.
        return


def load_runtime_events(log_path: Path = LOG_PATH) -> Dict[str, RuntimeEvent]:
    """Return the latest runtime event per model from the metrics log."""

    if not log_path.exists():
        return {}

    latest: Dict[str, RuntimeEvent] = {}
    try:
        with log_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue
                event = RuntimeEvent.from_raw(raw)
                if not event:
                    continue
                latest[event.model] = event
    except Exception:  # noqa: BLE001
        return {}
    return latest


def merge_runtime_payload(
    *,
    architecture_meta: Dict[str, Any],
    runtime_payload: Dict[str, Any],
    timestamp: str,
) -> Dict[str, Any]:
    """Merge runtime payload data into architecture metadata."""

    updated = dict(architecture_meta)
    runtime_section = dict(updated.get("runtime") or {})
    metrics = runtime_payload.get("metrics") or {}

    if metrics:
        runtime_section.update(metrics)
    if "extra_args" in runtime_payload:
        runtime_section["extra_args_snapshot"] = runtime_payload["extra_args"]
    runtime_section["observed_at"] = timestamp
    runtime_section["source"] = runtime_payload.get("source", "runtime-log")
    updated["runtime"] = runtime_section

    # Track provenance
    sources: Iterable[str] = updated.get("sources") or []
    merged_sources = list(dict.fromkeys(list(sources) + ["runtime-log"]))
    updated["sources"] = merged_sources

    return updated


__all__ = [
    "RuntimeEvent",
    "log_runtime_metrics",
    "load_runtime_events",
    "merge_runtime_payload",
    "LOG_PATH",
]
