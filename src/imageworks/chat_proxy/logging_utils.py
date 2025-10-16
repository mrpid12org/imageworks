from __future__ import annotations

import json
import os
import time
from typing import Any, Dict


class JsonlLogger:
    def __init__(
        self, path: str, max_bytes: int = 25_000_000, retention_days: int = 30
    ):
        self.path = path
        self.max_bytes = max_bytes
        self.retention_days = retention_days
        # Respect relative paths while avoiding mkdir("") when only a filename is provided.
        log_dir = os.path.dirname(path)
        if log_dir:
            try:
                os.makedirs(log_dir, exist_ok=True)
            except Exception:  # noqa: BLE001
                # Fallback: best-effort when directory creation fails (e.g., read-only volume).
                pass

    def _rotate_if_needed(self):
        try:
            if (
                os.path.exists(self.path)
                and os.path.getsize(self.path) > self.max_bytes
            ):
                ts = time.strftime("%Y%m%d-%H%M%S")
                rotated = f"{self.path}.{ts}"
                os.rename(self.path, rotated)
        except Exception:  # noqa: BLE001
            pass

    def log(self, record: Dict[str, Any]):
        self._rotate_if_needed()
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:  # noqa: BLE001
            pass
