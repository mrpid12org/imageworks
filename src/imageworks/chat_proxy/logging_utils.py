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
        os.makedirs(os.path.dirname(path), exist_ok=True)

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
