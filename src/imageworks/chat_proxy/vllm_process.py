"""Process controller helpers for launching vLLM runtimes."""

from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, Iterable, Optional


class LocalProcessController:
    """Spawn and manage processes on the local host."""

    def __init__(self) -> None:
        self._procs: Dict[int, subprocess.Popen] = {}
        self._log_handles: Dict[int, object] = {}
        self._lock = threading.Lock()

    def spawn(
        self,
        command: Iterable[str],
        *,
        env: Optional[dict[str, str]] = None,
        cwd: Optional[str] = None,
        log_file: Optional[Path] = None,
    ) -> int:
        log_path = Path(log_file) if log_file else None
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_fh = open(log_path, "a", buffering=1)  # noqa: PTH123
        else:
            log_fh = subprocess.DEVNULL

        proc_env = dict(os.environ)
        if env:
            for key, value in env.items():
                if value is None:
                    continue
                proc_env[key] = str(value)

        proc = subprocess.Popen(  # noqa: S603, S607
            list(command),
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            env=proc_env,
            cwd=cwd,
        )
        with self._lock:
            self._procs[proc.pid] = proc
            if log_path:
                self._log_handles[proc.pid] = log_fh
        return proc.pid

    def is_alive(self, pid: int) -> bool:
        proc = self._procs.get(pid)
        if not proc:
            return False
        alive = proc.poll() is None
        if not alive:
            self._dispose(pid)
        return alive

    def terminate(
        self, pid: int, *, force: bool = False, timeout: float = 30.0
    ) -> None:
        proc = self._procs.get(pid)
        if not proc:
            return
        try:
            if force:
                proc.kill()
            else:
                proc.terminate()
        except ProcessLookupError:
            self._dispose(pid)
            return
        except Exception:
            if force:
                return
            try:
                proc.kill()
            except Exception:
                pass
            finally:
                self._dispose(pid)
            return

        deadline = time.time() + max(timeout, 1)
        while time.time() < deadline:
            if proc.poll() is not None:
                self._dispose(pid)
                return
            time.sleep(0.5)
        if not force:
            self.terminate(pid, force=True, timeout=timeout)

    def _dispose(self, pid: int) -> None:
        with self._lock:
            proc = self._procs.pop(pid, None)
            log_fh = self._log_handles.pop(pid, None)
        if log_fh:
            try:
                log_fh.close()
            except Exception:
                pass
        if proc:
            try:
                proc.wait(timeout=0)
            except Exception:
                pass

    def close(self) -> None:
        """Placeholder to align with RemoteProcessController API."""
        return


class RemoteProcessController:
    """Represents a remote executor reachable via HTTP."""

    def __init__(self, base_url: str, *, timeout: float = 120.0) -> None:
        import httpx

        self._base = base_url.rstrip("/")
        self._http = httpx.Client(timeout=timeout)

    def close(self) -> None:
        try:
            self._http.close()
        except Exception:
            pass

    def spawn(
        self,
        command: Iterable[str],
        *,
        env: Optional[dict[str, str]] = None,
        cwd: Optional[str] = None,
        log_file: Optional[Path] = None,
    ) -> int:
        payload = {
            "command": list(command),
            "env": {k: str(v) for k, v in (env or {}).items() if v is not None},
            "cwd": str(cwd) if cwd else None,
            "log_file": str(log_file) if log_file else None,
        }
        resp = self._http.post(f"{self._base}/spawn", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return int(data["pid"])

    def is_alive(self, pid: int) -> bool:
        resp = self._http.get(f"{self._base}/process/{pid}")
        if resp.status_code == 404:
            return False
        resp.raise_for_status()
        data = resp.json()
        return bool(data.get("alive"))

    def terminate(
        self, pid: int, *, force: bool = False, timeout: float = 30.0
    ) -> None:
        payload = {"force": force, "timeout": timeout}
        resp = self._http.post(f"{self._base}/terminate/{pid}", json=payload)
        if resp.status_code in {200, 202, 404}:
            return
        resp.raise_for_status()
