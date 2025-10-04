from __future__ import annotations

import time
import inspect

import os
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .config import ProxyConfig
from .metrics import MetricsAggregator
from .autostart import AutostartManager
from .logging_utils import JsonlLogger
from .forwarder import ChatForwarder
from .errors import ProxyError
from ..model_loader.registry import load_registry, list_models, get_entry
from ..model_loader.testing_filters import is_testing_entry


_cfg = ProxyConfig.load()
_metrics = MetricsAggregator()
_autostart = AutostartManager(_cfg.autostart_map_raw)
_logger = JsonlLogger(_cfg.log_path, _cfg.max_log_bytes)
_forwarder = ChatForwarder(_cfg, _metrics, _autostart, _logger)

app = FastAPI(title="ImageWorks Chat Proxy", version="0.1")


# Track registry file modification time to auto-reload on change
_REGISTRY_PATH = Path("configs/model_registry.json")
_REGISTRY_MTIME: float | None = None


def _refresh_registry_if_changed() -> None:
    global _REGISTRY_MTIME  # noqa: PLW0603
    try:
        mtime = _REGISTRY_PATH.stat().st_mtime
    except Exception:  # noqa: BLE001
        return
    if _REGISTRY_MTIME is None:
        _REGISTRY_MTIME = mtime
        return
    if mtime != _REGISTRY_MTIME:
        load_registry(force=True)
        _REGISTRY_MTIME = mtime


@app.on_event("startup")
async def _startup():  # pragma: no cover
    load_registry()
    if _cfg.host != "127.0.0.1":
        print(
            "[chat-proxy] WARNING: Exposed host without auth (Phase 1). Consider reverse proxy + auth."
        )
    # Initialize registry mtime after first load
    try:
        global _REGISTRY_MTIME  # noqa: PLW0603
        _REGISTRY_MTIME = _REGISTRY_PATH.stat().st_mtime
    except Exception:  # noqa: BLE001
        _REGISTRY_MTIME = None


@app.get("/v1/models")
async def list_models_api():
    # Hot-reload registry if the file changed on disk
    _refresh_registry_if_changed()
    data = []
    include_testing = os.environ.get("CHAT_PROXY_INCLUDE_TEST_MODELS", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    seen_display_ids: set[str] = set()

    for name in list_models():
        entry = get_entry(name)
        if not include_testing and is_testing_entry(name, entry):
            continue
        # Hide non-installed variants for non-Ollama backends to avoid ghost entries
        installed = False
        try:
            if entry.download_path:
                p = Path(entry.download_path).expanduser()
                installed = p.exists()
        except Exception:
            installed = False
        if entry.backend != "ollama" and not installed:
            continue
        display = entry.display_name or entry.name or ""
        display_id = display or entry.name
        if display_id in seen_display_ids:
            # fall back to logical slug if friendly label collides
            display_id = entry.name
        if display_id in seen_display_ids:
            display_id = f"{entry.name}-{entry.backend}"
        if display_id in seen_display_ids:
            continue
        seen_display_ids.add(display_id)
        modalities = ["text"]
        if entry.probes.vision and entry.probes.vision.vision_ok:
            modalities.append("vision")
        # Heuristic: for Ollama, some models are vision-capable even if not yet probed.
        # Mark common vision families as such to improve UI hints.
        if entry.backend == "ollama" and "vision" not in modalities:
            disp_l = (entry.display_name or entry.name).lower()
            vision_hints = ("-vl", " vl ", "llava", "pixtral", "phi-3-vision", "vision")
            if any(h in disp_l for h in vision_hints):
                modalities.append("vision")
        templates = []
        primary_template = None
        if entry.chat_template and entry.chat_template.path:
            templates.append(entry.chat_template.path)
            primary_template = entry.chat_template.path
        data.append(
            {
                "id": display_id,
                "object": "model",
                "created": 0,
                "owned_by": "imageworks",
                "extensions": {
                    "display_id": display_id,
                    "display_name": display or entry.name,
                    "logical_id": name,
                    "backend": entry.backend,
                    "format": getattr(entry, "download_format", None),
                    "quantization": getattr(entry, "quantization", None),
                    "size_bytes": getattr(entry, "download_size_bytes", None),
                    "capabilities": getattr(entry, "capabilities", {}) or {},
                    "modalities": modalities,
                    "has_chat_template": bool(
                        entry.chat_template and entry.chat_template.path
                    ),
                    "templates": templates,
                    "primary_template": primary_template,
                    "schema_version": _cfg.schema_version,
                },
            }
        )
    return {"object": "list", "data": data}


@app.post("/v1/chat/completions")
async def chat_completions(req: Request):
    payload = await req.json()
    try:
        result = await _forwarder.handle_chat(payload)
    except ProxyError as exc:  # structured
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    # result is either streaming generator reference or full object
    if isinstance(result, tuple) and len(result) == 6:
        first_elem = result[0]
        # Recognize streaming either by callable returning async generator OR existing async generator instance
        is_async_gen_instance = inspect.isasyncgen(first_elem)
        if callable(first_elem) or is_async_gen_instance:
            (
                event_gen_fn_or_gen,
                backend,
                backend_id,
                started_at,
                first_fn,
                tokens_fn,
            ) = result

            async def streamer():
                gen = (
                    event_gen_fn_or_gen()
                    if callable(event_gen_fn_or_gen)
                    else event_gen_fn_or_gen
                )
                async for chunk in gen:
                    yield chunk
                _forwarder.record_metrics(
                    backend, payload.get("model"), started_at, first_fn, tokens_fn, True
                )
                _logger.log(
                    {
                        "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
                        "model_logical": payload.get("model"),
                        "backend": backend,
                        "model_backend_id": backend_id,
                        "stream": True,
                    }
                )

            return StreamingResponse(streamer(), media_type="text/event-stream")
        # treat as non-stream tuple
        norm, backend, backend_id, started_at, first_fn, tokens_fn = result
    else:
        # Should not happen; defensive
        norm = result
        backend = backend_id = "unknown"
        started_at = time.time()
        first_fn = lambda: time.time()  # noqa: E731
        tokens_fn = lambda: (0, True)  # noqa: E731
    _forwarder.record_metrics(
        backend, payload.get("model"), started_at, first_fn, tokens_fn, False
    )
    _logger.log(
        {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            "model_logical": payload.get("model"),
            "backend": backend,
            "model_backend_id": backend_id,
            "stream": False,
        }
    )
    return JSONResponse(content=norm)


@app.get("/v1/metrics")
async def metrics_api():
    if not _cfg.enable_metrics:
        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "type": "disabled",
                    "code": 404,
                    "message": "Metrics disabled",
                }
            },
        )
    return _metrics.summary()


@app.get("/v1/health")
async def health():  # pragma: no cover
    return {"status": "ok", "uptime_seconds": _metrics.summary().get("uptime_seconds")}


def main():  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host=_cfg.host, port=_cfg.port)
