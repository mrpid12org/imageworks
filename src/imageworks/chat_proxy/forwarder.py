from __future__ import annotations

import asyncio
import base64
import re
import time
from typing import Any, AsyncGenerator, Dict
from urllib.parse import urlsplit, urlunsplit

import httpx

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
from .autostart import AutostartManager, VllmActivationError
from .logging_utils import JsonlLogger
from .vllm_manager import VllmManager
from ..model_loader.registry import get_entry
from .capabilities import supports_vision, supports_reasoning


class ChatForwarder:
    def __init__(
        self,
        cfg: ProxyConfig,
        metrics: MetricsAggregator,
        autostart: AutostartManager,
        logger: JsonlLogger,
        vllm_manager: VllmManager | None = None,
    ):
        self.cfg = cfg
        self.metrics = metrics
        self.autostart = autostart
        self.logger = logger
        self.client = httpx.AsyncClient(timeout=cfg.backend_timeout_ms / 1000)
        self.vllm_manager = vllm_manager

    @staticmethod
    def _clean_generation_text(text: str | None) -> str | None:
        return text

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

    def _truncate_history_for_vision(self, messages: list[Any]) -> list[Any]:
        """
        Truncate message history for vision requests to avoid context length issues.

        Strategy:
        - Keep system message (if present and config.vision_keep_system=True)
        - Keep last N conversation turns (config.vision_keep_last_n_turns)
        - Keep current message (always)

        This allows vision models with limited VRAM to handle sequential image
        analysis without accumulating massive context from previous images.

        For text-only conversations, this function should not be called, preserving
        full conversational context.
        """
        if not messages:
            return messages

        # Don't truncate if disabled
        if not self.cfg.vision_truncate_history:
            return messages

        truncated = []
        system_messages = []
        conversation = []

        # Separate system messages from conversation
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "system":
                system_messages.append(msg)
            else:
                conversation.append(msg)

        # Keep system messages if configured
        if self.cfg.vision_keep_system and system_messages:
            truncated.extend(system_messages)

        # Keep last N turns (pairs of user/assistant messages)
        # A "turn" = user message + optional assistant response
        if self.cfg.vision_keep_last_n_turns > 0 and len(conversation) > 1:
            # Keep the last N turns before the current message
            # Current message is always last, so we want to preserve some history
            keep_count = self.cfg.vision_keep_last_n_turns * 2  # user + assistant
            history_to_keep = conversation[
                -(keep_count + 1) : -1
            ]  # Exclude last (current)
            truncated.extend(history_to_keep)

        # Always keep the current message (last in conversation)
        if conversation:
            truncated.append(conversation[-1])

        return truncated

    def _truncate_history_for_reasoning(self, messages: list[Any]) -> list[Any]:
        """
        Truncate message history for reasoning/thinking model requests.

        Reasoning models (o1, deepseek-r1, gpt-oss) generate very long outputs
        with extended chain-of-thought. When these accumulate in conversation
        history, they can quickly exhaust context windows or KV cache.

        Strategy is similar to vision truncation but defaults to keeping
        1 turn of history (since reasoning may benefit from immediate context).
        """
        if not messages:
            return messages

        # Don't truncate if disabled
        if not self.cfg.reasoning_truncate_history:
            return messages

        truncated = []
        system_messages = []
        conversation = []

        # Separate system messages from conversation
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "system":
                system_messages.append(msg)
            else:
                conversation.append(msg)

        # Keep system messages if configured
        if self.cfg.reasoning_keep_system and system_messages:
            truncated.extend(system_messages)

        # Keep last N turns (pairs of user/assistant messages)
        if self.cfg.reasoning_keep_last_n_turns > 0 and len(conversation) > 1:
            keep_count = self.cfg.reasoning_keep_last_n_turns * 2
            history_to_keep = conversation[-(keep_count + 1) : -1]
            truncated.extend(history_to_keep)

        # Always keep the current message
        if conversation:
            truncated.append(conversation[-1])

        return truncated

    def _resolve_backend_base(self, entry) -> tuple[str, str]:
        cfg = getattr(entry, "backend_config", None)
        base_override_raw = getattr(cfg, "base_url", None) if cfg else None
        base_override = None
        if isinstance(base_override_raw, str):
            stripped = base_override_raw.strip()
            if stripped:
                base_override = stripped.rstrip("/")
        host_raw = getattr(cfg, "host", None) if cfg else None
        host_override: str | None = None
        if isinstance(host_raw, str):
            stripped_host = host_raw.strip()
            if stripped_host:
                host_override = stripped_host
        alias = (self.cfg.loopback_alias or "").strip()
        default_port = 8000
        if getattr(entry, "backend", "") == "ollama":
            default_port = 11434
        elif getattr(entry, "backend", "") == "lmdeploy":
            default_port = 24001
        elif getattr(entry, "backend", "") == "triton":
            default_port = 9000
        port = getattr(cfg, "port", None) if cfg else None
        if not isinstance(port, int) or port <= 0:
            port = default_port

        backend = getattr(entry, "backend", "")

        if alias:
            base_override = self._alias_loopback(base_override, alias)
            host_override = self._alias_loopback(host_override, alias)

        if base_override:
            base_url = base_override
        elif host_override and host_override.startswith(("http://", "https://")):
            base_url = host_override.rstrip("/")
        else:
            host = host_override or "127.0.0.1"
            if alias and host.strip().lower() in {"127.0.0.1", "localhost", "::1"}:
                host = alias
            if backend == "ollama":
                base_url = f"http://{host}:{port}"
            else:
                base_url = f"http://{host}:{port}/v1"

        base_url = base_url.rstrip("/")
        api_path = "/api/chat" if backend == "ollama" else "/chat/completions"
        return base_url, api_path

    @staticmethod
    def _alias_loopback(value: str | None, alias: str) -> str | None:
        if not value:
            return value
        alias_clean = alias.strip()
        if not alias_clean:
            return value
        loopbacks = {"127.0.0.1", "localhost", "::1"}
        if "://" in value:
            try:
                parsed = urlsplit(value)
            except ValueError:
                return value
            hostname = parsed.hostname
            if not hostname or hostname.lower() not in loopbacks:
                return value
            netloc = alias_clean
            if parsed.port and ":" not in alias_clean.split("]")[-1]:
                netloc = f"{alias_clean}:{parsed.port}"
            if parsed.username or parsed.password:
                creds = parsed.username or ""
                if parsed.password:
                    creds += f":{parsed.password}"
                netloc = f"{creds}@{netloc}"
            return urlunsplit(parsed._replace(netloc=netloc))
        # Treat as host[:port]
        try:
            parsed = urlsplit(f"scheme://{value}")
        except ValueError:
            return value
        host = parsed.hostname
        if not host or host.lower() not in loopbacks:
            return value
        port = parsed.port
        result = alias_clean
        if port and ":" not in alias_clean.split("]")[-1]:
            result = f"{alias_clean}:{port}"
        return result

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

            # No need to import simplified_display_for_entry for display name resolution

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
                    # Do not use simplified display/slug; only use display_name, name, and quant forms
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

        messages = list(payload.get("messages") or [])
        meta = getattr(entry, "metadata", {}) or {}
        default_system_prompt = meta.get("default_system_prompt")
        if default_system_prompt:
            has_system = any(
                isinstance(msg, dict) and msg.get("role") == "system"
                for msg in messages
            )
            if not has_system:
                messages.insert(0, {"role": "system", "content": default_system_prompt})
                payload["messages"] = messages

        # Capability checks
        has_vision = supports_vision(entry)
        has_images, image_bytes = self._detect_images(payload.get("messages") or [])
        if has_images and not has_vision:
            raise err_capability_mismatch(
                f"Model '{model}' does not support vision/image content"
            )
        if image_bytes > self.cfg.max_image_bytes:
            raise err_payload_too_large(self.cfg.max_image_bytes)

        # Truncate history for vision requests if configured
        # This helps vision models with limited VRAM handle sequential image analysis
        # without accumulating massive context from previous images
        if has_images and has_vision:
            original_count = len(payload.get("messages") or [])
            truncated_messages = self._truncate_history_for_vision(
                payload.get("messages") or []
            )
            if len(truncated_messages) < original_count:
                import logging

                logger = logging.getLogger(__name__)
                logger.info(
                    "[forwarder] Truncated vision history: %d → %d messages "
                    "(keep_system=%s, keep_last_n_turns=%d)",
                    original_count,
                    len(truncated_messages),
                    self.cfg.vision_keep_system,
                    self.cfg.vision_keep_last_n_turns,
                )
                payload["messages"] = truncated_messages

        # Truncate history for reasoning/thinking requests if configured
        # This helps reasoning models avoid context overflow from previous long outputs
        has_reasoning = supports_reasoning(entry)
        if not has_images and has_reasoning and self.cfg.reasoning_truncate_history:
            original_count = len(payload.get("messages") or [])
            truncated_messages = self._truncate_history_for_reasoning(
                payload.get("messages") or []
            )
            if len(truncated_messages) < original_count:
                import logging

                logger = logging.getLogger(__name__)
                logger.info(
                    "[forwarder] Truncated reasoning history: %d → %d messages "
                    "(keep_system=%s, keep_last_n_turns=%d)",
                    original_count,
                    len(truncated_messages),
                    self.cfg.reasoning_keep_system,
                    self.cfg.reasoning_keep_last_n_turns,
                )
                payload["messages"] = truncated_messages

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
        base_url, api_path = self._resolve_backend_base(entry)
        vllm_started = False
        if self.cfg.autostart_enabled and entry.backend == "vllm" and self.autostart:
            try:
                vllm_started = await self.autostart.ensure_started(model, entry)
            except VllmActivationError as exc:
                raise err_model_start_timeout(model) from exc
            except Exception:
                vllm_started = False

        if entry.backend != "ollama":
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

        # Apply registry generation defaults when caller did not set explicit sampling params.
        defaults = getattr(entry, "generation_defaults", None)
        if defaults:

            def _apply_or_keep(target: dict, key: str, value):
                if value is None:
                    return
                if key not in target or target[key] is None:
                    target[key] = value

            _apply_or_keep(payload, "temperature", defaults.temperature)
            _apply_or_keep(payload, "top_p", defaults.top_p)
            _apply_or_keep(payload, "top_k", defaults.top_k)
            _apply_or_keep(payload, "max_tokens", defaults.max_tokens)
            _apply_or_keep(payload, "min_tokens", defaults.min_tokens)
            _apply_or_keep(payload, "frequency_penalty", defaults.frequency_penalty)
            _apply_or_keep(payload, "presence_penalty", defaults.presence_penalty)
            if getattr(defaults, "stop_sequences", None):
                if not payload.get("stop"):
                    payload["stop"] = list(defaults.stop_sequences)

        if entry.backend == "vllm":
            extra_body = {}
            try:
                current_extra = payload.get("extra_body")  # type: ignore[attr-defined]
                if isinstance(current_extra, dict):
                    extra_body = dict(current_extra)
            except Exception:
                extra_body = {}
            extra_body.setdefault("add_generation_prompt", True)
            if extra_body:
                payload["extra_body"] = extra_body

        # Ensure backend reachable or autostart
        if entry.backend != "ollama" and not await self._probe(base_url):
            started = False
            if self.cfg.autostart_enabled:
                if vllm_started:
                    started = await self._probe(base_url)
                elif await self.autostart.ensure_started(model, entry):
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
                    import logging

                    nonlocal first_token_at, collected_tokens, estimated
                    logger = logging.getLogger("ollama-stream")
                    raw_accum = ""
                    clean_accum = ""

                    def consume(delta_raw: str) -> str:
                        nonlocal raw_accum, clean_accum
                        raw_accum += delta_raw
                        cleaned_total = self._clean_generation_text(raw_accum) or ""
                        if not cleaned_total.strip():
                            return ""
                        new_segment = cleaned_total[len(clean_accum) :]
                        clean_accum = cleaned_total
                        return new_segment

                    # If upstream is non-stream (image payload), do one-shot post and re-emit as SSE
                    if opayload.get("stream") is False:
                        logger.info("[Ollama] Non-streaming POST to %s", url)
                        resp = await self.client.post(url, json=opayload)
                        logger.info("[Ollama] Response status: %s", resp.status_code)
                        if resp.status_code >= 400:
                            logger.error(
                                "[Ollama] Backend unavailable: %s", resp.text[:200]
                            )
                            raise err_backend_unavailable(model, hint=resp.text[:200])
                        try:
                            import json as _json

                            obj = resp.json()
                        except Exception as e:
                            txt = await resp.aread()
                            logger.error(
                                "[Ollama] Exception parsing JSON: %s, raw: %s",
                                e,
                                txt[:200],
                            )
                            raise err_backend_unavailable(
                                model, hint=txt.decode(errors="ignore")[:200]
                            )
                        delta_raw = ((obj.get("message") or {}).get("content")) or ""
                        if delta_raw:
                            new_segment = consume(delta_raw)
                            if new_segment:
                                if first_token_at is None:
                                    first_token_at = time.time()
                                collected_tokens += len(new_segment) // 4
                                estimated = False
                                oai = {
                                    "id": obj.get("id") or "cmpl-ollama",
                                    "object": "chat.completion.chunk",
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {"content": new_segment},
                                            "finish_reason": None,
                                        }
                                    ],
                                }
                                out = _json.dumps(oai, ensure_ascii=False)
                                logger.info(
                                    "[Ollama] Yielding non-stream chunk: %s", out[:200]
                                )
                                yield f"data: {out}\n\n".encode()
                        # Terminate stream
                        logger.info("[Ollama] Yielding [DONE] for non-stream")
                        yield b"data: [DONE]\n\n"
                        return
                    # Else stream upstream and convert
                    done = False
                    try:
                        logger.info("[Ollama] Streaming POST to %s", url)
                        async with self.client.stream(
                            "POST", url, json=opayload
                        ) as resp:
                            logger.info(
                                "[Ollama] Streaming response status: %s",
                                resp.status_code,
                            )
                            if resp.status_code >= 400:
                                data = await resp.aread()
                                logger.error(
                                    "[Ollama] Backend unavailable: %s", data[:200]
                                )
                                raise err_backend_unavailable(
                                    model, hint=data.decode(errors="ignore")[:200]
                                )
                            async for line in resp.aiter_lines():
                                logger.info("[Ollama] Got line: %r", line)
                                if not line:
                                    continue
                                # Accept both SSE 'data: <json>' and plain JSONL lines from Ollama
                                raw = (
                                    line[5:].strip()
                                    if line.startswith("data:")
                                    else line.strip()
                                )
                                logger.info("[Ollama] Parsed raw: %r", raw)
                                if not raw:
                                    continue
                                if raw == "[DONE]":
                                    logger.info("[Ollama] Yielding [DONE] from stream")
                                    yield b"data: [DONE]\n\n"
                                    done = True
                                    break
                                try:
                                    import json as _json

                                    obj = _json.loads(raw)
                                    if obj.get("done") is True:
                                        logger.info(
                                            "[Ollama] Yielding [DONE] from JSON chunk"
                                        )
                                        yield b"data: [DONE]\n\n"
                                        done = True
                                        break
                                    # Ollama stream messages usually include a 'message': { 'content': '...' }
                                    delta_raw = (
                                        (obj.get("message") or {}).get("content")
                                    ) or ""
                                    if not delta_raw:
                                        continue
                                    new_segment = consume(delta_raw)
                                    if not new_segment:
                                        continue
                                    if first_token_at is None:
                                        first_token_at = time.time()
                                    collected_tokens += len(new_segment) // 4
                                    estimated = True
                                    oai = {
                                        "id": obj.get("id") or "cmpl-ollama",
                                        "object": "chat.completion.chunk",
                                        "choices": [
                                            {
                                                "index": 0,
                                                "delta": {"content": new_segment},
                                                "finish_reason": None,
                                            }
                                        ],
                                    }
                                    out = _json.dumps(oai, ensure_ascii=False)
                                    logger.info(
                                        "[Ollama] Yielding stream chunk: %s",
                                        out[:200],
                                    )
                                    yield f"data: {out}\n\n".encode()
                                except Exception as e:
                                    logger.error(
                                        "[Ollama] Exception parsing stream chunk: %s, raw: %r",
                                        e,
                                        raw,
                                    )
                                    # Unknown data; pass through as generic SSE data
                                    yield f"data: {raw}\n\n".encode()
                    except Exception as e:
                        logger.error(
                            "[Ollama] Exception in streaming block: %s",
                            e,
                            exc_info=True,
                        )
                        raise
                    finally:
                        if not done:
                            logger.info("[Ollama] Forcing [DONE] at stream end")
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
                raw_accum = ""
                clean_accum = ""

                def consume(delta_raw: str) -> str:
                    nonlocal raw_accum, clean_accum
                    raw_accum += delta_raw
                    cleaned_total = self._clean_generation_text(raw_accum) or ""
                    if not cleaned_total.strip():
                        return ""
                    new_segment = cleaned_total[len(clean_accum) :]
                    clean_accum = cleaned_total
                    return new_segment

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
                                    choices = norm.get("choices") or []
                                    if not choices:
                                        continue
                                    delta = (choices[0] or {}).get("delta", {})
                                    if not isinstance(delta, dict):
                                        continue
                                    content = delta.get("content")
                                    if not content:
                                        continue
                                    new_segment = consume(content)
                                    if not new_segment:
                                        continue
                                    if first_token_at is None:
                                        first_token_at = time.time()
                                    delta["content"] = new_segment
                                    chunk = json.dumps(norm, ensure_ascii=False)
                                    # naive token estimate
                                    collected_tokens += len(new_segment) // 4
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
                content_raw = ((obj.get("message") or {}).get("content")) or ""
                content = self._clean_generation_text(content_raw) or ""
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
            try:
                for choice in norm.get("choices", []):
                    message = (
                        choice.get("message") if isinstance(choice, dict) else None
                    )
                    if isinstance(message, dict):
                        content = message.get("content")
                        cleaned = self._clean_generation_text(content)
                        if cleaned is not None:
                            message["content"] = cleaned
                            if isinstance(choice, dict) and isinstance(
                                choice.get("text"), str
                            ):
                                text_clean = self._clean_generation_text(choice["text"])
                                choice["text"] = text_clean or ""
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
