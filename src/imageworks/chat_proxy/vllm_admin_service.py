"""Admin API for managing the vLLM worker process inside the vLLM container."""

from __future__ import annotations

import asyncio
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator

app = FastAPI(title="ImageWorks vLLM Admin", version="0.1")


class ActivateRequest(BaseModel):
    command: List[str] = Field(..., min_items=1)
    env: Dict[str, str] = Field(default_factory=dict)
    cwd: Optional[str] = None
    log_file: Optional[str] = None
    logical_name: str = Field(..., min_length=1)
    served_model_id: str = Field(..., min_length=1)

    @validator("command")
    def _strip_args(cls, value: List[str]) -> List[str]:
        return [str(arg) for arg in value if str(arg).strip()]


class ActivateResponse(BaseModel):
    status: str
    pid: int
    logical_name: str
    served_model_id: str


class StateResponse(BaseModel):
    status: str
    logical_name: Optional[str] = None
    served_model_id: Optional[str] = None
    pid: Optional[int] = None
    started_at: Optional[float] = None
    log_file: Optional[str] = None


_lock = asyncio.Lock()
_process: Optional[subprocess.Popen] = None
_log_handle = None
_state: Dict[str, Optional[object]] = {
    "logical_name": None,
    "served_model_id": None,
    "pid": None,
    "started_at": None,
    "log_file": None,
}


def _build_env(extra: Dict[str, str]) -> Dict[str, str]:
    env = dict(os.environ)
    for key, value in extra.items():
        if value is None:
            continue
        env[str(key)] = str(value)
    return env


def _close_log() -> None:
    global _log_handle
    handle = _log_handle
    _log_handle = None
    if handle:
        try:
            handle.close()
        except Exception:  # noqa: BLE001
            pass


def _ensure_dead() -> None:
    global _process
    proc = _process
    if proc and proc.poll() is not None:
        _close_log()
        _process = None
        _state.update(
            {
                "pid": None,
                "logical_name": None,
                "served_model_id": None,
                "started_at": None,
            }
        )


async def _terminate_running(force: bool = False) -> None:
    global _process
    proc = _process
    if not proc:
        _close_log()
        _state.update(
            {
                "pid": None,
                "logical_name": None,
                "served_model_id": None,
                "started_at": None,
            }
        )
        return
    try:
        if force:
            proc.kill()
        else:
            proc.terminate()
    except ProcessLookupError:
        pass
    except Exception:  # noqa: BLE001
        if not force:
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass
    deadline = time.time() + 10
    while proc.poll() is None and time.time() < deadline:
        await asyncio.sleep(0.2)
    _close_log()
    _process = None
    _state.update(
        {"pid": None, "logical_name": None, "served_model_id": None, "started_at": None}
    )


def _spawn_process(req: ActivateRequest) -> int:
    global _process, _log_handle
    env = _build_env(req.env)
    cwd = req.cwd or os.getcwd()
    log_path = None
    if req.log_file:
        log_path = Path(req.log_file).expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        _log_handle = open(log_path, "a", buffering=1)  # noqa: PTH123
    else:
        _log_handle = subprocess.DEVNULL
    proc = subprocess.Popen(  # noqa: S603, S607
        req.command,
        stdout=_log_handle,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=cwd,
        start_new_session=True,
    )
    _process = proc
    _state.update(
        {
            "pid": proc.pid,
            "logical_name": req.logical_name,
            "served_model_id": req.served_model_id,
            "started_at": time.time(),
            "log_file": str(log_path) if log_path else None,
        }
    )
    return proc.pid


@app.get("/health")
def health() -> Dict[str, str]:
    _ensure_dead()
    return {"status": "ok"}


@app.get("/admin/state", response_model=StateResponse)
async def admin_state() -> StateResponse:
    async with _lock:
        _ensure_dead()
        status = "stopped" if _state["pid"] is None else "running"
        return StateResponse(
            status=status,
            logical_name=_state["logical_name"],
            served_model_id=_state["served_model_id"],
            pid=_state["pid"],
            started_at=_state["started_at"],
            log_file=_state["log_file"],
        )


@app.post("/admin/activate", response_model=ActivateResponse)
async def admin_activate(req: ActivateRequest) -> ActivateResponse:
    async with _lock:
        await _terminate_running(force=True)
        try:
            pid = _spawn_process(req)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return ActivateResponse(
            status="running",
            pid=pid,
            logical_name=req.logical_name,
            served_model_id=req.served_model_id,
        )


@app.post("/admin/deactivate")
async def admin_deactivate() -> Dict[str, str]:
    async with _lock:
        await _terminate_running(force=False)
        return {"status": "stopped"}
