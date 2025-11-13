"""HTTP service that launches vLLM worker processes on demand."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .vllm_process import LocalProcessController

app = FastAPI(title="ImageWorks vLLM Executor", version="0.1")
_controller = LocalProcessController()


class SpawnRequest(BaseModel):
    command: List[str] = Field(..., min_items=1)
    env: Dict[str, str] = Field(default_factory=dict)
    cwd: Optional[str] = None
    log_file: Optional[str] = None


class SpawnResponse(BaseModel):
    pid: int


class TerminateRequest(BaseModel):
    force: bool = False
    timeout: float = 30.0


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/spawn", response_model=SpawnResponse)
def spawn_process(req: SpawnRequest):
    try:
        pid = _controller.spawn(
            req.command,
            env=req.env,
            cwd=req.cwd,
            log_file=Path(req.log_file) if req.log_file else None,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))
    return SpawnResponse(pid=pid)


@app.get("/process/{pid}")
def process_status(pid: int):
    if pid <= 0:
        raise HTTPException(status_code=404, detail="invalid pid")
    alive = _controller.is_alive(pid)
    if not alive:
        return {"pid": pid, "alive": False}
    return {"pid": pid, "alive": True}


@app.post("/terminate/{pid}")
def terminate_process(pid: int, req: TerminateRequest):
    if pid <= 0:
        raise HTTPException(status_code=404, detail="invalid pid")
    try:
        _controller.terminate(pid, force=req.force, timeout=req.timeout)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))
    return {"pid": pid, "status": "terminated" if not req.force else "force-killed"}
