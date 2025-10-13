from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


def _get_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass
class ProxyConfig:
    host: str = "127.0.0.1"
    port: int = 8100
    enable_metrics: bool = False
    log_path: str = "logs/chat_proxy.jsonl"
    max_log_bytes: int = 25_000_000
    backend_timeout_ms: int = 120_000
    stream_idle_timeout_ms: int = 60_000
    autostart_enabled: bool = False
    autostart_map_raw: Optional[str] = None
    require_template: bool = True
    max_image_bytes: int = 6_000_000
    disable_tool_normalization: bool = False
    log_prompts: bool = False
    schema_version: int = 1
    suppress_decorations: bool = True
    include_non_installed: bool = False

    @classmethod
    def load(cls) -> "ProxyConfig":
        return cls(
            host=os.environ.get("CHAT_PROXY_HOST", "127.0.0.1"),
            port=_get_int("CHAT_PROXY_PORT", 8100),
            enable_metrics=_get_bool("CHAT_PROXY_ENABLE_METRICS", False),
            log_path=os.environ.get("CHAT_PROXY_LOG_PATH", "logs/chat_proxy.jsonl"),
            max_log_bytes=_get_int("CHAT_PROXY_MAX_LOG_BYTES", 25_000_000),
            backend_timeout_ms=_get_int("CHAT_PROXY_BACKEND_TIMEOUT_MS", 120_000),
            stream_idle_timeout_ms=_get_int(
                "CHAT_PROXY_STREAM_IDLE_TIMEOUT_MS", 60_000
            ),
            autostart_enabled=_get_bool("CHAT_PROXY_AUTOSTART_ENABLED", False),
            autostart_map_raw=os.environ.get("CHAT_PROXY_AUTOSTART_MAP"),
            require_template=_get_bool("CHAT_PROXY_REQUIRE_TEMPLATE", True),
            max_image_bytes=_get_int("CHAT_PROXY_MAX_IMAGE_BYTES", 6_000_000),
            disable_tool_normalization=_get_bool(
                "CHAT_PROXY_DISABLE_TOOL_NORMALIZATION", False
            ),
            log_prompts=_get_bool("CHAT_PROXY_LOG_PROMPTS", False),
            schema_version=_get_int("CHAT_PROXY_SCHEMA_VERSION", 1),
            suppress_decorations=_get_bool("CHAT_PROXY_SUPPRESS_DECORATIONS", True),
            include_non_installed=_get_bool("CHAT_PROXY_INCLUDE_NON_INSTALLED", False),
        )
