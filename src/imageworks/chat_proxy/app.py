from __future__ import annotations

import logging
import os
import inspect
import time
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from .config import ProxyConfig
from .metrics import MetricsAggregator
from .autostart import AutostartManager
from .logging_utils import JsonlLogger
from .forwarder import ChatForwarder
from .vllm_manager import VllmManager
from .capabilities import supports_vision
from .errors import ProxyError
from .profile_manager import ProfileManager
from .role_selector import RoleSelector
from ..model_loader.registry import (
    load_registry,
    list_models,
    get_entry,
    _curated_path,
    _discovered_path,
    _merged_snapshot_path,
)
from ..model_loader.testing_filters import is_testing_entry

logging.basicConfig(level=logging.INFO)


_cfg = ProxyConfig.load()
_metrics = MetricsAggregator()
_vllm_manager = VllmManager(_cfg)
_autostart = AutostartManager(_cfg.autostart_map_raw, _cfg, _vllm_manager)
_logger = JsonlLogger(_cfg.log_path, _cfg.max_log_bytes)
_forwarder = ChatForwarder(_cfg, _metrics, _autostart, _logger, _vllm_manager)

# Phase 2: Profile management and role-based model selection
_profile_manager: ProfileManager | None = None
_role_selector: RoleSelector | None = None


app = FastAPI(title="ImageWorks Chat Proxy", version="0.1")


# Temporary debug endpoint to dump the in-memory registry
@app.get("/v1/debug/registry")
async def debug_registry():
    from ..model_loader.registry import _REGISTRY_CACHE, load_registry

    if _REGISTRY_CACHE is None:
        load_registry()
    # Only return name and display_name for brevity
    out = {
        k: {"display_name": getattr(v, "display_name", None)}
        for k, v in _REGISTRY_CACHE.items()
    }
    return JSONResponse(content=out)


# Track all registry files (curated, discovered, merged) for auto-reload
_REGISTRY_MTIMES: dict[str, float] = {}


def _get_all_registry_paths() -> list[Path]:
    """Get paths to all registry files that may be edited."""
    return [
        _curated_path(),
        _discovered_path(),
        _merged_snapshot_path(),
    ]


def _path_exists(maybe_path: str | None) -> bool:
    if not maybe_path:
        return False
    try:
        candidate = Path(str(maybe_path)).expanduser()
        return candidate.exists()
    except Exception:  # noqa: BLE001
        return False


def _has_served_backend(entry) -> bool:
    served = getattr(entry, "served_model_id", None)
    if not served:
        return False
    return str(served).strip().lower() != "none"


def _refresh_registry_if_changed() -> None:
    """Reload registry if any of the layer files have changed."""
    global _REGISTRY_MTIMES  # noqa: PLW0603

    paths = _get_all_registry_paths()
    changed = False

    for path in paths:
        try:
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            # File doesn't exist yet (e.g., fresh init)
            continue
        except Exception:  # noqa: BLE001
            continue

        key = str(path)
        if key not in _REGISTRY_MTIMES:
            _REGISTRY_MTIMES[key] = mtime
            continue

        if mtime != _REGISTRY_MTIMES[key]:
            changed = True
            _REGISTRY_MTIMES[key] = mtime

    if changed:
        logging.info("[app] Registry file(s) changed, reloading...")
        load_registry(force=True)


@app.on_event("startup")
async def _startup():  # pragma: no cover
    global _profile_manager, _role_selector  # noqa: PLW0603

    load_registry()

    # Phase 2: Initialize profile manager and role selector
    try:
        _profile_manager = ProfileManager()
        _role_selector = RoleSelector()
        profile = _profile_manager.get_active_profile()
        if profile:
            logging.info(f"[app] Active deployment profile: {profile.name}")
        else:
            logging.warning("[app] No deployment profile active!")
    except Exception as e:
        logging.error(f"[app] Failed to initialize profile management: {e}")

    if _cfg.host != "127.0.0.1":
        print(
            "[chat-proxy] WARNING: Exposed host without auth (Phase 1). "
            "Consider reverse proxy + auth."
        )
    # Initialize registry mtimes after first load
    for path in _get_all_registry_paths():
        try:
            _REGISTRY_MTIMES[str(path)] = path.stat().st_mtime
        except Exception:  # noqa: BLE001
            pass


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

    logging.info("[DEBUG] All model names and display_names in registry:")
    for name in list_models():
        entry = get_entry(name)
        logging.info(
            f"[DEBUG] {name} | display_name={getattr(entry, 'display_name', None)}"
        )
        if getattr(entry, "deprecated", False):
            continue
        if not include_testing and is_testing_entry(name, entry):
            continue
        # Hide non-installed variants for non-Ollama backends by default; allow override
        has_assets = _path_exists(
            getattr(entry, "download_path", None)
        ) or _path_exists(getattr(entry.backend_config, "model_path", None))
        if (
            entry.backend != "ollama"
            and not _cfg.include_non_installed
            and not (has_assets or _has_served_backend(entry))
        ):
            # Hide entries that neither have local weights nor a configured backend when strict mode is on.
            continue
        # Use registry display_name for UI display
        display = entry.display_name or entry.name or ""
        if _cfg.suppress_decorations:
            display_id = display
        else:
            display_id = display or entry.name
        if display_id in seen_display_ids:
            display_id = entry.name
        if display_id in seen_display_ids:
            display_id = f"{entry.name}-{entry.backend}"
        if display_id in seen_display_ids:
            continue
        seen_display_ids.add(display_id)
        modalities = ["text"]
        if supports_vision(entry):
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
        # Optionally suppress extra decorations to mirror CLI display strictly
        fmt_value = getattr(entry, "download_format", None)
        quant_value = getattr(entry, "quantization", None)
        backend_value = entry.backend
        if _cfg.suppress_decorations:
            fmt_value = None
            quant_value = None
            backend_value = None
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
                    "backend": backend_value,
                    "format": fmt_value,
                    "quantization": quant_value,
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


@app.get("/v1/config/profile")
async def get_profile_info():
    """Get active deployment profile and detected GPU information."""
    if _profile_manager is None:
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "type": "service_unavailable",
                    "code": 503,
                    "message": "Profile manager not initialized",
                }
            },
        )

    try:
        info = _profile_manager.get_profile_info()
        return JSONResponse(content=info)
    except Exception as e:
        logging.error(f"[app] Error getting profile info: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "type": "internal_error",
                    "code": 500,
                    "message": str(e),
                }
            },
        )


@app.get("/v1/models/select_by_role")
async def select_model_by_role(role: str, top_n: int = 3):
    """Select best models for a specific task role within profile constraints."""
    if _profile_manager is None or _role_selector is None:
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "type": "service_unavailable",
                    "code": 503,
                    "message": "Profile management not initialized",
                }
            },
        )

    try:
        profile = _profile_manager.get_active_profile()
        if not profile:
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "type": "no_active_profile",
                        "code": 500,
                        "message": "No active deployment profile",
                    }
                },
            )

        # Get available roles for validation
        available_roles = _role_selector.get_available_roles()
        if role not in available_roles:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "type": "invalid_role",
                        "code": 400,
                        "message": f"Unknown role '{role}'. Available roles: {', '.join(available_roles)}",
                    }
                },
            )

        # Select models for role
        selected_models = _role_selector.select_for_role(role, profile, top_n)

        # Format response
        response = {
            "role": role,
            "profile": profile.name,
            "max_vram_mb": profile.max_vram_mb,
            "model_selection_bias": profile.model_selection_bias,
            "top_n": top_n,
            "models": [
                {
                    "id": model.get("id"),
                    "name": model.get("name"),
                    "backend": model.get("backend"),
                    "quantization": model.get("quantization"),
                    "vram_estimate_mb": model.get("vram_estimate_mb"),
                    "role_priority": model.get("role_priority", {}).get(role),
                }
                for model in selected_models
            ],
        }

        return JSONResponse(content=response)

    except Exception as e:
        logging.error(f"[app] Error selecting models by role: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "type": "internal_error",
                    "code": 500,
                    "message": str(e),
                }
            },
        )


@app.get("/v1/health")
async def health():  # pragma: no cover
    return {"status": "ok", "uptime_seconds": _metrics.summary().get("uptime_seconds")}


def main():  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host=_cfg.host, port=_cfg.port)


if __name__ == "__main__":  # pragma: no cover
    main()
