"""FastAPI service exposing deterministic model loader endpoints."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path

from .registry import load_registry, list_models, get_entry
from .service import select_model, CapabilityError
from .hashing import verify_model, VersionLockViolation
from .probe import run_vision_probe
from .metrics import RollingMetrics  # placeholder for future aggregation
from .models import normalize_capabilities

app = FastAPI(title="Imageworks Deterministic Model Loader", version="0.1")

# Simple in-memory metrics map (model_name -> RollingMetrics)
_MODEL_METRICS: dict[str, RollingMetrics] = {}


class ModelSummary(BaseModel):
    name: str
    backend: str
    capabilities: dict
    locked: bool
    vision_ok: Optional[bool]
    display_name: Optional[str] = None


class SelectRequest(BaseModel):
    name: str
    require_capabilities: Optional[List[str]] = None


class SelectResponse(BaseModel):
    endpoint: str
    backend: str
    internal_model_id: str
    capabilities: dict


class VerifyRequest(BaseModel):
    name: str


class VisionProbeRequest(BaseModel):
    name: str
    image_path: str


@app.on_event("startup")
async def _startup() -> None:  # pragma: no cover
    load_registry()


@app.get("/v1/models", response_model=List[ModelSummary])
async def api_list_models():
    result: List[ModelSummary] = []
    for name in list_models():
        entry = get_entry(name)
        # Prefer simplified naming for API consumers (optional field)
        try:
            from .simplified_naming import simplified_display_for_entry as _simple_disp

            disp = _simple_disp(entry)
        except Exception:
            disp = entry.display_name or entry.name
        result.append(
            ModelSummary(
                name=name,
                backend=entry.backend,
                capabilities=normalize_capabilities(entry.capabilities),
                locked=entry.version_lock.locked,
                vision_ok=(
                    entry.probes.vision.vision_ok if entry.probes.vision else None
                ),
                display_name=disp,
            )
        )
    return result


@app.post("/v1/select", response_model=SelectResponse)
async def api_select(req: SelectRequest):
    try:
        desc = select_model(req.name, require_capabilities=req.require_capabilities)
    except CapabilityError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return SelectResponse(
        endpoint=desc.endpoint_url,
        backend=desc.backend,
        internal_model_id=desc.internal_model_id,
        capabilities=normalize_capabilities(desc.capabilities),
    )


@app.post("/v1/verify")
async def api_verify(req: VerifyRequest):
    try:
        entry = get_entry(req.name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    try:
        verify_model(entry)
    except VersionLockViolation as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"status": "ok", "aggregate_sha256": entry.artifacts.aggregate_sha256}


@app.post("/v1/probe/vision")
async def api_probe_vision(req: VisionProbeRequest):
    try:
        result = run_vision_probe(req.name, Path(req.image_path))  # type: ignore[name-defined]
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result


@app.get("/v1/models/{name}/metrics")
async def api_metrics(name: str):
    metrics = _MODEL_METRICS.get(name)
    if metrics is None:
        return {"rolling_samples": 0}
    return metrics.summary()


# Convenience root
@app.get("/")
async def root():  # pragma: no cover
    return {"service": "imageworks-model-loader", "version": "0.1"}
