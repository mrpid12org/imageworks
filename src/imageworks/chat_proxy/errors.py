from __future__ import annotations

from fastapi import HTTPException


class ProxyError(HTTPException):
    def __init__(
        self, status_code: int, err_type: str, message: str, hint: str | None = None
    ):
        payload = {"error": {"type": err_type, "code": status_code, "message": message}}
        if hint:
            payload["error"]["hint"] = hint
        super().__init__(status_code=status_code, detail=payload)


def err_model_not_found(model: str) -> ProxyError:
    return ProxyError(
        404, "model_not_found", f"Model '{model}' not found", "Check /v1/models list"
    )


def err_capability_mismatch(reason: str) -> ProxyError:
    return ProxyError(409, "capability_mismatch", reason)


def err_backend_unavailable(model: str, hint: str | None = None) -> ProxyError:
    return ProxyError(
        424, "backend_unavailable", f"Backend unavailable for '{model}'", hint
    )


def err_model_start_timeout(model: str, hint: str | None = None) -> ProxyError:
    return ProxyError(
        424,
        "model_start_timeout",
        f"Autostart did not make '{model}' healthy in time",
        hint,
    )


def err_gpu_lease_active(owner: str) -> ProxyError:
    return ProxyError(
        423,
        "gpu_lease_active",
        f"GPU currently leased by '{owner}'. Please retry once it is released.",
    )


def err_payload_too_large(limit: int) -> ProxyError:
    return ProxyError(
        413, "payload_too_large", f"Image payload exceeds limit {limit} bytes"
    )


def err_template_required(model: str) -> ProxyError:
    return ProxyError(
        409,
        "template_required",
        f"Model '{model}' missing required chat template",
        "Re-download with template or disable requirement",
    )
