from __future__ import annotations

import asyncio
import httpx
import time
from typing import Any, AsyncGenerator, Dict
import base64
import re

from .config import ProxyConfig
from .errors import (
    err_backend_unavailable,
    err_model_not_found,
    err_model_start_timeout,
    err_capability_mismatch,
    err_payload_too_large,
    err_template_required,
)
from .normalization import normalize_response
from .metrics import MetricsAggregator, MetricSample
from .autostart import AutostartManager
from .logging_utils import JsonlLogger
from ..model_loader.registry import get_entry
from .capabilities import supports_vision


class ChatForwarder:
    def __init__(
        self,
        cfg: ProxyConfig,
        metrics: MetricsAggregator,
        autostart: AutostartManager,
        logger: JsonlLogger,
    ):
        self.cfg = cfg
        self.metrics = metrics
        self.autostart = autostart
        self.logger = logger
        self.client = httpx.AsyncClient(timeout=cfg.backend_timeout_ms / 1000)

    async def _probe(self, base_url: str) -> bool:
        try:
            r = await self.client.get(base_url + "/models")
            return r.status_code < 500
        except Exception:  # noqa: BLE001
            return False

    async def _prepare_ollama_payload(
        self, backend_id: str, messages: list[Any], stream: bool
    ) -> dict:
        prepared: list[dict] = []
        total_bytes = 0
        for m in messages or []:
            role = m.get("role") if isinstance(m, dict) else None
            content = m.get("content") if isinstance(m, dict) else None
            msg: dict[str, Any] = {"role": role or "user", "content": ""}
            images: list[str] = []
            if isinstance(content, str):
                msg["content"] = content
            elif isinstance(content, list):
                text_parts: list[str] = []
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    ptype = part.get("type")
                    if ptype == "text":
                        t = part.get("text") or ""
                        text_parts.append(str(t))
                    elif ptype == "image_url":
                        url = (part.get("image_url") or {}).get("url") or ""
                        if not url:
                            continue
                        if url.startswith("data:image") and "," in url:
                            raw = url.split(",", 1)[-1]
                            try:
                                # Validate base64 and account size
                                b = base64.b64decode(raw, validate=False)
                                total_bytes += len(b)
                                images.append(base64.b64encode(b).decode())
                            except Exception:
                                # Fallback: pass raw portion
                                images.append(raw)
                        elif url.startswith("http://") or url.startswith("https://"):
                            try:
                                resp = await self.client.get(url)
                                if resp.status_code < 400:
                                    b = await resp.aread()
                                    total_bytes += len(b)
                                    images.append(base64.b64encode(b).decode())
                            except Exception:
                                pass
                msg["content"] = " ".join([t for t in text_parts if t]).strip()
            else:
                # Unknown content type; stringify conservatively
                msg["content"] = str(content) if content is not None else ""
            if images:
                msg["images"] = images
            prepared.append(msg)
        return {"model": backend_id, "messages": prepared, "stream": bool(stream)}

    def _detect_images(self, messages: list[Any]) -> tuple[bool, int]:
        has = False
        total = 0
        for m in messages:
            content = m.get("content") if isinstance(m, dict) else None
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        url = (part.get("image_url") or {}).get("url") or ""
                        if url:
                            has = True
                        if url.startswith("data:image"):
                            # Rough size estimate: base64 length * 3/4
                            b64 = url.split(",", 1)[-1]
                            total += int(len(b64) * 0.75)
                        else:
                            # Non-base64 URL present; count as minimal bytes to trigger image flow
                            total = max(total, 1)
        return has, total

    async def handle_chat(self, payload: dict) -> Dict[str, Any]:
        requested_model = payload.get("model")
        if not requested_model:
            raise err_model_not_found("<empty>")
        model = str(requested_model)
        # Accept either the logical id or the display id (display_name[-quant])
        try:
            entry = get_entry(model)
        except KeyError:
            # Attempt to resolve by display id -> logical id
            from ..model_loader.registry import load_registry
            from ..model_loader.simplified_naming import (
                simplified_display_for_entry as _simple_disp,
                simplified_slug_for_entry as _simple_slug,
            )

            reg = load_registry()
            resolved = None
            target_norm = _normalize_label(model)
            # Prioritize exact match on logical name first
            if model in reg:
                resolved = model
            else:
                # Fallback to iterating and checking display names
                for name, e in reg.items():
                    candidates: set[str] = set()
                    disp = getattr(e, "display_name", None) or getattr(e, "name", None)
                    if disp:
                        candidates.add(str(disp))
                    try:
                        candidates.add(_simple_disp(e))
                        candidates.add(_simple_slug(e))
                    except Exception:  # noqa: BLE001
                        pass
                    quant = getattr(e, "quantization", None)
                    if disp and quant:
                        q_label = str(quant).replace("_", " ").replace("-", " ").upper()
                        candidates.add(f"{disp} ({q_label})")
                    # legacy hyphen form display-quantization
                    if disp and quant:
                        candidates.add(f"{disp}-{quant}")
                    for cand in list(candidates):
                        if not cand:
                            continue
                        if target_norm and _normalize_label(cand) == target_norm:
                            resolved = name
                            break
                    if resolved:
                        break
            if not resolved:
                raise err_model_not_found(model)
            entry = get_entry(resolved)
            # Also rewrite the payload model to the logical id so downstream logging is consistent
            payload["model"] = resolved
            model = resolved

        # Capability checks
        has_vision = supports_vision(entry)
        has_images, image_bytes = self._detect_images(payload.get("messages") or [])
        # For Ollama, defer vision capability validation to the backend (some models support vision even if
        # our registry hasn't been probed yet). Keep strict gating for non-Ollama backends.
        if has_images and entry.backend != "ollama" and not has_vision:
            raise err_capability_mismatch("Vision content provided to non-vision model")
        if image_bytes > self.cfg.max_image_bytes:
            raise err_payload_too_large(self.cfg.max_image_bytes)
        # Require chat template unless backend has its own prompting (e.g., ollama)
        if (
            self.cfg.require_template
            and entry.backend != "ollama"
            and not (entry.chat_template and entry.chat_template.path)
        ):
            raise err_template_required(model)

        # Prefer an explicit served model id; some registries persisted the literal string "None".
        served = getattr(entry, "served_model_id", None)
        if isinstance(served, str) and served.strip().lower() == "none":
            served = None
        backend_id = served or entry.name
        # Backend-specific base URL and API path
        if entry.backend == "ollama":
            port = entry.backend_config.port or 11434
            # Prefer IPv4 loopback to avoid ::1 vs 0.0.0.0 mismatch
            base_url = f"http://127.0.0.1:{port}"
            api_path = "/api/chat"
        else:
            # Sensible defaults if port not set in registry
            default_port = 8000
            if entry.backend == "lmdeploy":
                default_port = 24001
            elif entry.backend == "triton":
                default_port = 9000
            port = entry.backend_config.port or default_port
            # Prefer IPv4 loopback to avoid ::1 vs 0.0.0.0 mismatch
            base_url = f"http://127.0.0.1:{port}/v1"
            api_path = "/chat/completions"
            # For OpenAI-compatible backends (vLLM, LMDeploy, Triton), the upstream expects the
            # concrete served model name, which may differ from our logical model id. Rewrite the
            # outgoing payload's model to the backend_id we computed above.
            try:
                # Work on a shallow copy; keep original payload for logging/metrics consistency.
                payload = dict(payload)
                payload["model"] = backend_id
            except Exception:
                # If payload isn't a dict, fall back without rewrite (defensive)
                pass

        # Ensure backend reachable or autostart
        if entry.backend != "ollama" and not await self._probe(base_url):
            started = False
            if self.cfg.autostart_enabled:
                if await self.autostart.ensure_started(model):
                    # wait grace
                    await asyncio.sleep(self.cfg.autostart_grace_period_s)
                    if await self._probe(base_url):
                        started = True
                if not started:
                    raise err_model_start_timeout(model)
            else:
                raise err_backend_unavailable(
                    model, hint="Enable autostart or start backend manually"
                )

        stream = bool(payload.get("stream"))
        # For Ollama with images, force non-stream to client to avoid chunked transfer edge-cases
        if stream and entry.backend == "ollama" and has_images:
            stream = False
        url = base_url + api_path
        started_at = time.time()
        first_token_at: float | None = None
        collected_tokens = 0
        estimated = False

        if stream:
            # Streaming path differs between backends. IMPORTANT: open the upstream stream
            # inside the generator so the connection stays alive while the client consumes it.
            if entry.backend == "ollama":
                # Ollama streaming SSE/JSONL → convert to OpenAI-style SSE chunks
                opayload = await self._prepare_ollama_payload(
                    backend_id,
                    payload.get("messages") or [],
                    # For image content, request non-streaming upstream to avoid transfer-encoding issues
                    stream=False if has_images else True,
                )

                async def event_gen() -> AsyncGenerator[bytes, None]:
                    nonlocal first_token_at, collected_tokens, estimated
                    # If upstream is non-stream (image payload), do one-shot post and re-emit as SSE
                    if opayload.get("stream") is False:
                        resp = await self.client.post(url, json=opayload)
                        if resp.status_code >= 400:
                            raise err_backend_unavailable(model, hint=resp.text[:200])
                        try:
                            import json as _json

                            obj = resp.json()
                        except Exception:
                            txt = await resp.aread()
                            raise err_backend_unavailable(
                                model, hint=txt.decode(errors="ignore")[:200]
                            )
                        delta = ((obj.get("message") or {}).get("content")) or ""
                        if delta:
                            if first_token_at is None:
                                first_token_at = time.time()
                            collected_tokens += len(delta) // 4
                            estimated = False
                            oai = {
                                "id": obj.get("id") or "cmpl-ollama",
                                "object": "chat.completion.chunk",
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": delta},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                            out = _json.dumps(oai, ensure_ascii=False)
                            yield f"data: {out}\n\n".encode()
                        # Terminate stream
                        yield b"data: [DONE]\n\n"
                        return
                    # Else stream upstream and convert
                    done = False
                    try:
                        async with self.client.stream(
                            "POST", url, json=opayload
                        ) as resp:
                            if resp.status_code >= 400:
                                data = await resp.aread()
                                raise err_backend_unavailable(
                                    model, hint=data.decode(errors="ignore")[:200]
                                )
                            async for line in resp.aiter_lines():
                                if not line:
                                    continue
                                # Accept both SSE 'data: <json>' and plain JSONL lines from Ollama
                                raw = (
                                    line[5:].strip()
                                    if line.startswith("data:")
                                    else line.strip()
                                )
                                if not raw:
                                    continue
                                if raw == "[DONE]":
                                    yield b"data: [DONE]\n\n"
                                    done = True
                                    break
                                try:
                                    import json as _json

                                    obj = _json.loads(raw)
                                    if obj.get("done") is True:
                                        yield b"data: [DONE]\n\n"
                                        done = True
                                        break
                                    # Ollama stream messages usually include a 'message': { 'content': '...' }
                                    delta = (
                                        (obj.get("message") or {}).get("content")
                                    ) or ""
                                    if delta and first_token_at is None:
                                        first_token_at = time.time()
                                    collected_tokens += len(delta) // 4
                                    estimated = True
                                    oai = {
                                        "id": obj.get("id") or "cmpl-ollama",
                                        "object": "chat.completion.chunk",
                                        "choices": [
                                            {
                                                "index": 0,
                                                "delta": {"content": delta},
                                                "finish_reason": None,
                                            }
                                        ],
                                    }
                                    out = _json.dumps(oai, ensure_ascii=False)
                                    yield f"data: {out}\n\n".encode()
                                except Exception:
                                    # Unknown data; pass through as generic SSE data
                                    yield f"data: {raw}\n\n".encode()
                    finally:
                        if not done:
                            # Ensure well-formed termination for clients expecting a final marker
                            yield b"data: [DONE]\n\n"

                # Return the callable so app can invoke it later when streaming starts
                return (
                    event_gen,
                    entry.backend,
                    backend_id,
                    started_at,
                    lambda: first_token_at,
                    lambda: (collected_tokens, estimated),
                )

            # Default OpenAI-compatible streaming
            async def event_gen() -> AsyncGenerator[bytes, None]:
                nonlocal first_token_at, collected_tokens, estimated
                done = False
                async with self.client.stream("POST", url, json=payload) as resp:
                    if resp.status_code >= 400:
                        data = await resp.aread()
                        raise err_backend_unavailable(
                            model, hint=data.decode(errors="ignore")[:200]
                        )
                    try:
                        async for line in resp.aiter_lines():
                            if not line:
                                continue
                            if line.startswith("data:"):
                                chunk = line[5:].strip()
                                if chunk == "[DONE]":
                                    yield b"data: [DONE]\n\n"
                                    done = True
                                    break
                                # Attempt normalization if JSON
                                try:
                                    import json

                                    obj = json.loads(chunk)
                                    if first_token_at is None:
                                        first_token_at = time.time()
                                    norm = normalize_response(
                                        obj,
                                        disabled=self.cfg.disable_tool_normalization,
                                    )
                                    chunk = json.dumps(norm, ensure_ascii=False)
                                    # naive token estimate
                                    collected_tokens += (
                                        len(
                                            (norm.get("choices") or [{}])[0]
                                            .get("delta", {})
                                            .get("content", "")
                                        )
                                        // 4
                                    )
                                    estimated = True
                                except Exception:  # noqa: BLE001
                                    pass
                                yield f"data: {chunk}\n\n".encode()
                    finally:
                        if not done:
                            yield b"data: [DONE]\n\n"

            return (
                event_gen,
                entry.backend,
                backend_id,
                started_at,
                lambda: first_token_at,
                lambda: (collected_tokens, estimated),
            )
        else:
            if entry.backend == "ollama":
                opayload = await self._prepare_ollama_payload(
                    backend_id,
                    payload.get("messages") or [],
                    stream=False,
                )
                resp = await self.client.post(url, json=opayload)
                if resp.status_code >= 400:
                    raise err_backend_unavailable(model, hint=resp.text[:200])
                obj = resp.json()
                # Transform Ollama response to OpenAI format
                content = ((obj.get("message") or {}).get("content")) or ""
                first_token_at = time.time()
                norm = {
                    "id": obj.get("id") or "cmpl-ollama",
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": content},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"completion_tokens": max(1, len(content) // 4)},
                }
                collected_tokens = norm["usage"]["completion_tokens"]
                return (
                    norm,
                    entry.backend,
                    backend_id,
                    started_at,
                    lambda: first_token_at,
                    lambda: (collected_tokens, False),
                )
            # Default OpenAI-compatible non-stream
            resp = await self.client.post(url, json=payload)
            if resp.status_code >= 400:
                raise err_backend_unavailable(model, hint=resp.text[:200])
            obj = resp.json()
            first_token_at = time.time()
            norm = normalize_response(obj, disabled=self.cfg.disable_tool_normalization)
            # Final guard for inline function-call content -> tool_calls
            try:
                choices = norm.get("choices") or []
                if choices:
                    msg = (choices[0] or {}).get("message") or {}
                    tc = msg.get("tool_calls") or []
                    content = msg.get("content") or ""
                    if not tc and content:
                        try:
                            from .normalization import (
                                _maybe_extract_function_call_from_content,
                                _coerce_str,
                            )

                            fc = _maybe_extract_function_call_from_content(content)
                            if fc and isinstance(fc, dict) and fc.get("name"):
                                msg["tool_calls"] = [
                                    {
                                        "id": "call_0",
                                        "type": "function",
                                        "function": {
                                            "name": fc["name"],
                                            "arguments": _coerce_str(
                                                fc.get("arguments", "{}")
                                            ),
                                        },
                                    }
                                ]
                                msg["content"] = ""
                                choices[0]["message"] = msg
                                norm["choices"] = choices
                        except Exception:
                            pass
            except Exception:
                pass
            usage = norm.get("usage") or {}
            collected_tokens = (
                usage.get("completion_tokens") or usage.get("total_tokens") or 0
            )
            return (
                norm,
                entry.backend,
                backend_id,
                started_at,
                lambda: first_token_at,
                lambda: (collected_tokens, False),
            )

    def record_metrics(
        self,
        backend: str,
        model: str,
        started_at: float,
        first_token_at_fn,
        tokens_fn,
        stream: bool,
    ):
        first = first_token_at_fn()
        if first is None:
            return
        tokens_out, estimated = tokens_fn()
        duration = time.time() - started_at
        ttft_ms = (first - started_at) * 1000
        tps = tokens_out / duration if duration > 0 and tokens_out else 0.0
        self.metrics.add(
            MetricSample(
                ts=time.time(),
                model=model,
                backend=backend,
                ttft_ms=ttft_ms,
                tokens_out=tokens_out,
                duration_ms=duration * 1000,
                tokens_per_second=tps,
                stream=stream,
                estimated_counts=estimated,
            )
        )
def _normalize_label(label: Any) -> str:
    """Return a canonical lowercase token for matching model aliases."""

    if label is None:
        return ""
    text = str(label).strip().lower()
    if not text:
        return ""
    return re.sub(r"[^a-z0-9]+", "", text)

