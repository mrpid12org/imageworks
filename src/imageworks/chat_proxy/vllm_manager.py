from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx

from .config import ProxyConfig

logger = logging.getLogger("imageworks.vllm_manager")


class VllmActivationError(RuntimeError):
    """Raised when the orchestrator cannot activate a requested vLLM model."""


@dataclass
class ActiveVllmState:
    logical_name: str
    served_model_id: str
    port: int
    pid: int
    started_at: float

    @classmethod
    def from_dict(cls, payload: dict) -> "ActiveVllmState | None":
        try:
            logical = str(payload["logical_name"])
            served = str(payload.get("served_model_id") or "")
            port = int(payload["port"])
            pid = int(payload["pid"])
            started_at = float(payload["started_at"])
        except Exception:  # noqa: BLE001
            return None
        if not logical or not served or port <= 0 or pid <= 0:
            return None
        return cls(
            logical_name=logical,
            served_model_id=served,
            port=port,
            pid=pid,
            started_at=started_at,
        )

    def to_dict(self) -> dict:
        return {
            "logical_name": self.logical_name,
            "served_model_id": self.served_model_id,
            "port": self.port,
            "pid": self.pid,
            "started_at": self.started_at,
        }


class _FileLock:
    """Best-effort advisory file lock for cross-process coordination."""

    def __init__(self, path: Path):
        self.path = path
        self._fh: Optional[object] = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.path, "w+")  # noqa: PTH123
        try:
            import fcntl

            fcntl.flock(fh, fcntl.LOCK_EX)
        except Exception:  # noqa: BLE001
            # Non-posix or locking failed; fall back to coarse process-level lock.
            pass
        self._fh = fh

    def release(self) -> None:
        fh = self._fh
        self._fh = None
        if fh is None:
            return
        try:
            import fcntl

            fcntl.flock(fh, fcntl.LOCK_UN)
        except Exception:  # noqa: BLE001
            pass
        try:
            fh.close()
        except Exception:  # noqa: BLE001
            pass


class VllmManager:
    """Persisted single-port vLLM process controller."""

    def __init__(self, cfg: ProxyConfig):
        self.cfg = cfg
        self._state_path = Path(cfg.vllm_state_path).expanduser()
        self._lock_path = self._state_path.with_suffix(".lock")
        self._process_lock = asyncio.Lock()
        self._http = httpx.AsyncClient(timeout=max(cfg.vllm_health_timeout_s, 10))

    async def aclose(self) -> None:
        await self._http.aclose()

    async def activate(self, entry) -> ActiveVllmState:
        """Ensure the requested registry entry is the active vLLM instance."""

        logical_name = entry.name
        served_model_id = self._served_model_id(entry)
        port = self.cfg.vllm_port

        async with self._exclusive():
            state = await self._load_state()
            if (
                state
                and state.logical_name == logical_name
                and state.port == port
                and self._process_alive(state.pid)
            ):
                logger.info(
                    "[vllm-manager] Reusing active %s (pid=%s)", logical_name, state.pid
                )
                return state

            if state and self._process_alive(state.pid):
                logger.info(
                    "[vllm-manager] Stopping existing model %s (pid=%s)",
                    state.logical_name,
                    state.pid,
                )
                await self._terminate_process(state.pid)
            await self._clear_state()

            logger.info(
                "[vllm-manager] Starting model %s (served=%s) on port %s",
                logical_name,
                served_model_id,
                port,
            )
            pid = await self._launch(entry, served_model_id, port)

            started_at = time.time()
            if not await self._wait_for_health(port, started_at):
                logger.error(
                    "[vllm-manager] Health check failed for %s (pid=%s)",
                    logical_name,
                    pid,
                )
                await self._terminate_process(pid, force=True)
                raise VllmActivationError(
                    f"vLLM model '{logical_name}' failed health check"
                )

            new_state = ActiveVllmState(
                logical_name=logical_name,
                served_model_id=served_model_id,
                port=port,
                pid=pid,
                started_at=started_at,
            )
            await self._write_state(new_state)
            logger.info(
                "[vllm-manager] Model %s active on port %s (pid=%s)",
                logical_name,
                port,
                pid,
            )
            return new_state

    async def deactivate(self) -> None:
        """Stop the currently active vLLM instance if present."""

        async with self._exclusive():
            state = await self._load_state()
            if not state:
                return
            if self._process_alive(state.pid):
                logger.info(
                    "[vllm-manager] Stopping model %s (pid=%s)",
                    state.logical_name,
                    state.pid,
                )
                await self._terminate_process(state.pid)
            await self._clear_state()

    async def current_state(self) -> Optional[ActiveVllmState]:
        """Return current state if process still alive."""

        async with self._exclusive(shared=True):
            state = await self._load_state()
            if not state:
                return None
            if not self._process_alive(state.pid):
                await self._clear_state()
                return None
            return state

    @staticmethod
    def _validate_extra_args(extra_args: list[str], model_name: str) -> None:
        """
        Validate extra_args for common issues like duplicate flags.

        Logs warnings if issues are detected but does not raise exceptions,
        since vLLM will typically handle duplicates (using last occurrence).

        Args:
            extra_args: List of extra command-line arguments
            model_name: Model name for logging context
        """
        if not extra_args:
            return

        # Track seen flags (both --flag and --flag=value forms)
        seen_flags: dict[str, list[int]] = {}

        for i, arg in enumerate(extra_args):
            if arg.startswith("--"):
                # Extract flag name (before '=' if present)
                flag = arg.split("=")[0]

                if flag not in seen_flags:
                    seen_flags[flag] = []
                seen_flags[flag].append(i)

        # Check for duplicates
        duplicates = {
            flag: positions
            for flag, positions in seen_flags.items()
            if len(positions) > 1
        }

        if duplicates:
            warnings = []
            for flag, positions in duplicates.items():
                values = [extra_args[pos] for pos in positions]
                warnings.append(
                    f"  {flag}: appears at positions {positions}\n"
                    f"    Values: {values}\n"
                    f"    vLLM typically uses the LAST occurrence"
                )

            logger.warning(
                "[vllm-manager] Duplicate flags detected in extra_args for '%s':\n%s\n"
                "This may cause unexpected behavior. Consider consolidating flags.",
                model_name,
                "\n".join(warnings),
            )

    async def _launch(self, entry, served_model_id: str, port: int) -> int:
        command = self._build_command(entry, served_model_id, port)
        stdout_path = Path("logs/vllm").expanduser()
        stdout_path.mkdir(parents=True, exist_ok=True)
        log_file = stdout_path / f"{entry.name.replace('/', '_')}.log"
        with open(log_file, "a", buffering=1) as log_fh:  # noqa: PTH123
            proc = subprocess.Popen(  # noqa: S603
                command,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                env=self._inherited_env(),
            )
        return proc.pid

    def _build_command(self, entry, served_model_id: str, port: int) -> list[str]:
        model_path = self._resolve_model_path(entry)
        host = self._resolve_host(entry)
        command: list[str] = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model_path,
            "--host",
            host,
            "--port",
            str(port),
            "--served-model-name",
            served_model_id,
        ]
        extra = list(getattr(entry.backend_config, "extra_args", []) or [])

        # Validate extra_args for common issues
        self._validate_extra_args(extra, entry.name)

        logger.info(
            "[vllm-manager][debug] backend_config.extra_args for '%s': %r",
            entry.name,
            extra,
        )
        logger.info("[vllm-manager][debug] initial command: %r", command)

        # Add GPU memory utilization if not already in extra_args
        if self.cfg.vllm_gpu_memory_utilization and not any(
            arg == "--gpu-memory-utilization"
            or arg.startswith("--gpu-memory-utilization=")
            for arg in extra
        ):
            command.extend(
                [
                    "--gpu-memory-utilization",
                    str(self.cfg.vllm_gpu_memory_utilization),
                ]
            )

        # Add max-model-len if configured and not already in extra_args
        if self.cfg.vllm_max_model_len and not any(
            arg == "--max-model-len" or arg.startswith("--max-model-len=")
            for arg in extra
        ):
            command.extend(["--max-model-len", str(self.cfg.vllm_max_model_len)])

        # Add chat template if present and not already in extra_args
        if entry.chat_template and entry.chat_template.path:
            has_chat_template_in_extra = any(
                arg == "--chat-template" or arg.startswith("--chat-template=")
                for arg in extra
            )
            if not has_chat_template_in_extra:
                original_path = Path(entry.chat_template.path).expanduser()
                template_path = self._resolve_template_path(original_path)

                if template_path:
                    command.extend(["--chat-template", str(template_path)])
                    logger.info("[vllm-manager] Using chat template: %s", template_path)
                else:
                    # Fail fast if template is explicitly configured but missing
                    if self.cfg.require_template:
                        msg = (
                            f"Chat template not found: {entry.chat_template.path}\n"
                            f"Checked: {original_path}\n"
                            f"Set require_template=False to allow missing templates."
                        )
                        logger.error("[vllm-manager] %s", msg)
                        raise FileNotFoundError(msg)
                    else:
                        logger.warning(
                            "[vllm-manager] Chat template %s not found; "
                            "continuing with vLLM's default template (require_template=False)",
                            entry.chat_template.path,
                        )

        if extra:
            logger.info(
                "[vllm-manager][debug] extending command with extra_args: %r", extra
            )
            command.extend(extra)

        logger.info(
            "[vllm-manager][debug] final vLLM launch command for '%s': %r",
            entry.name,
            command,
        )
        return command

    @staticmethod
    def _resolve_model_path(entry) -> str:
        backend_cfg = getattr(entry, "backend_config", None)
        model_path = getattr(backend_cfg, "model_path", None) if backend_cfg else None
        if model_path:
            expanded = Path(model_path).expanduser()
            if expanded.exists():
                return str(expanded)
        download_path = getattr(entry, "download_path", None)
        if download_path:
            expanded = Path(download_path).expanduser()
            if expanded.exists():
                return str(expanded)
        # Fallback to original path even if not found (vLLM will raise).
        return model_path or download_path or entry.name

    @staticmethod
    def _resolve_host(entry) -> str:
        backend_cfg = getattr(entry, "backend_config", None)
        if backend_cfg and getattr(backend_cfg, "host", None):
            host = (
                backend_cfg.host.strip() if isinstance(backend_cfg.host, str) else None
            )
            if host:
                return host
        return "0.0.0.0"

    @staticmethod
    def _served_model_id(entry) -> str:
        """
        Determine the model ID to pass to vLLM's --served-model-name.

        This controls what model name vLLM will respond with in API responses
        and what name clients can use to query this model.

        Precedence:
        1. If entry.served_model_id is a non-empty string (not "none"): Use it
        2. Otherwise: Use entry.name (the logical registry name)

        Common use cases:
        - Set to entry.name (default): Backend sees same name as registry
        - Set to custom value: Backend uses different name (e.g., aliasing)
        - Set to "None" or null: Explicitly fall back to entry.name

        Example scenarios:

        # Scenario 1: Default behavior
        {
          "name": "llama-3.1-8b-instruct_(Q4_K_M)",
          "served_model_id": null
        }
        → vLLM sees: "llama-3.1-8b-instruct_(Q4_K_M)"

        # Scenario 2: Alias for compatibility
        {
          "name": "llama-3.1-8b-instruct_(Q4_K_M)",
          "served_model_id": "llama3-8b-instruct"
        }
        → vLLM sees: "llama3-8b-instruct"
        → Clients can request either name (proxy resolves to registry entry)

        # Scenario 3: Explicit None (same as null)
        {
          "name": "custom-model",
          "served_model_id": "None"
        }
        → vLLM sees: "custom-model"

        Args:
            entry: Registry entry with potential served_model_id field

        Returns:
            The model ID to pass to --served-model-name
        """
        served = getattr(entry, "served_model_id", None)
        if isinstance(served, str):
            stripped = served.strip()
            if stripped and stripped.lower() != "none":
                return stripped
        return entry.name

    @staticmethod
    def _resolve_template_path(original: Path) -> Path | None:
        """
        Resolve template path, checking both original location and workspace-rebased path.
        Returns None if not found in either location.

        This handles cases where:
        1. Template is at absolute path on host
        2. Template needs to be rebased into workspace for container access
        """
        logger.debug("[vllm-manager] Resolving template path: %s", original)

        if original.exists():
            logger.debug("[vllm-manager] Template found at original path: %s", original)
            return original

        logger.debug(
            "[vllm-manager] Template not found at original path, trying workspace rebase..."
        )
        rebased = VllmManager._rebase_into_workspace(original)

        if rebased:
            logger.debug("[vllm-manager] Rebased path: %s", rebased)
            if rebased.exists():
                logger.debug(
                    "[vllm-manager] Template found at rebased path: %s", rebased
                )
                return rebased
            else:
                logger.debug("[vllm-manager] Rebased path does not exist: %s", rebased)
        else:
            logger.debug("[vllm-manager] Could not compute rebased path")

        logger.warning(
            "[vllm-manager] Template resolution failed for: %s\n"
            "  Original path: %s (exists: %s)\n"
            "  Rebased path: %s (exists: %s)",
            original,
            original,
            original.exists(),
            rebased,
            rebased.exists() if rebased else "N/A",
        )
        return None

    @staticmethod
    def _rebase_into_workspace(original: Path) -> Path | None:
        try:
            workspace_root = Path(__file__).resolve().parents[3]
        except IndexError:
            return None

        parts = original.parts
        if not parts:
            return None

        search_roots = {
            "_staging",
            "configs",
            "src",
            "scripts",
            "logs",
            "outputs",
        }

        for marker in search_roots:
            if marker not in parts:
                continue
            try:
                idx = parts.index(marker)
            except ValueError:
                continue
            rebased = workspace_root.joinpath(*parts[idx:])
            if rebased.exists():
                return rebased

        return None

    async def _wait_for_health(self, port: int, started_at: float) -> bool:
        deadline = started_at + self.cfg.vllm_start_timeout_s
        base_url = f"http://127.0.0.1:{port}"
        health_paths = ("/health", "/v1/health")
        backoff = 0.5
        while time.time() < deadline:
            for path in health_paths:
                url = base_url + path
                try:
                    resp = await self._http.get(url)
                except Exception:  # noqa: BLE001
                    continue
                if resp.status_code >= 500:
                    continue
                ok = False
                body_text = ""
                try:
                    body_text = resp.text or ""
                except Exception:  # noqa: BLE001
                    body_text = ""
                data = None
                if body_text:
                    try:
                        data = json.loads(body_text)
                    except Exception:  # noqa: BLE001
                        data = None
                if isinstance(data, dict):
                    status_val = str(data.get("status", "")).strip().lower()
                    if status_val == "ok":
                        ok = True
                if not ok and body_text:
                    normalized = body_text.strip().lower()
                    if normalized == "ok" and resp.status_code < 400:
                        ok = True
                if not ok and resp.status_code < 400 and not body_text.strip():
                    ok = True
                if ok:
                    return True
            await asyncio.sleep(backoff)
            backoff = min(backoff * 1.5, 5.0)
        return False

    async def _terminate_process(self, pid: int, *, force: bool = False) -> None:
        if not self._process_alive(pid):
            return
        timeout = self.cfg.vllm_stop_timeout_s
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except Exception:  # noqa: BLE001
            if force:
                try:
                    os.kill(pid, signal.SIGKILL)
                except Exception:  # noqa: BLE001
                    pass
            return
        deadline = time.time() + max(timeout, 1)
        while time.time() < deadline:
            if not self._process_alive(pid):
                return
            await asyncio.sleep(0.5)
        if not force and self._process_alive(pid):
            await self._terminate_process(pid, force=True)

    def _process_alive(self, pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except Exception:  # noqa: BLE001
            return False

    async def _load_state(self) -> Optional[ActiveVllmState]:
        if not self._state_path.exists():
            return None
        try:
            data = await asyncio.to_thread(self._state_path.read_text)
            parsed = json.loads(data)
        except Exception:  # noqa: BLE001
            logger.warning("[vllm-manager] Failed reading state; clearing.")
            await self._clear_state()
            return None
        state = ActiveVllmState.from_dict(parsed)
        if not state:
            await self._clear_state()
        return state

    async def _write_state(self, state: ActiveVllmState) -> None:
        payload = json.dumps(state.to_dict(), indent=2)
        await asyncio.to_thread(self._write_atomic, payload)

    def _write_atomic(self, payload: str) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._state_path.with_suffix(".tmp")
        tmp_path.write_text(payload)
        tmp_path.replace(self._state_path)

    async def _clear_state(self) -> None:
        await asyncio.to_thread(self._state_path.unlink, missing_ok=True)

    def _inherited_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.setdefault("VLLM_WORKER_MULTIPROC_START_METHOD", "forkserver")
        return env

    @asynccontextmanager
    async def _exclusive(self, shared: bool = False):
        lock = _FileLock(self._lock_path)
        if not shared:
            await self._process_lock.acquire()
        try:
            await asyncio.to_thread(lock.acquire)
            try:
                yield
            finally:
                await asyncio.to_thread(lock.release)
        finally:
            if not shared:
                self._process_lock.release()
