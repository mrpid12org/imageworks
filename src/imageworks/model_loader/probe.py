"""Vision probe execution logic (manual trigger Phase 1)."""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Dict, Any
import requests

from .service import select_model
from .registry import get_entry

PROMPT_TEXT = "Describe this image succinctly."


def _encode_image(path: Path) -> str:
    with path.open("rb") as handle:
        return base64.b64encode(handle.read()).decode("utf-8")


def run_vision_probe(model_name: str, image_path: Path) -> Dict[str, Any]:
    entry = get_entry(model_name)
    if not entry.capabilities.get("vision"):
        raise RuntimeError(f"Model '{model_name}' is not vision-capable")

    descriptor = select_model(model_name, require_capabilities=["vision"])
    b64 = _encode_image(image_path)

    payload = {
        "model": descriptor.internal_model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT_TEXT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                ],
            }
        ],
        "max_tokens": 64,
        "temperature": 0.0,
        "stream": False,
    }

    start = time.perf_counter()
    response = requests.post(
        f"{descriptor.endpoint_url}/chat/completions", json=payload, timeout=60
    )
    latency_ms = (time.perf_counter() - start) * 1000

    success = response.status_code == 200
    content_excerpt = ""
    if success:
        try:
            j = response.json()
            content_excerpt = (
                j.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()[:160]
            )
        except Exception:  # noqa: BLE001
            success = False

    result = {
        "model": model_name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "vision_ok": success and bool(content_excerpt),
        "latency_ms": round(latency_ms, 2),
        "excerpt": content_excerpt,
        "status_code": response.status_code,
    }
    _persist_probe_result(model_name, result)
    return result


def _persist_probe_result(model_name: str, result: Dict[str, Any]) -> None:
    out_dir = Path("outputs/probes") / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    file_path = out_dir / f"vision_{result['timestamp'].replace(':', '')}.json"
    file_path.write_text(json.dumps(result, indent=2))


__all__ = ["run_vision_probe"]
