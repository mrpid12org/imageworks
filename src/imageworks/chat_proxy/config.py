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


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _get_optional_int(name: str) -> int | None:
    val = os.environ.get(name)
    if val is None or val == "":
        return None
    try:
        return int(val)
    except ValueError:
        return None


@dataclass
class ProxyConfig:
    host: str = "127.0.0.1"
    port: int = 8100
    enable_metrics: bool = False
    log_path: str = "logs/chat_proxy.jsonl"
    max_log_bytes: int = 25_000_000
    backend_timeout_ms: int = 120_000
    stream_idle_timeout_ms: int = 60_000
    autostart_enabled: bool = True
    autostart_map_raw: Optional[str] = None
    autostart_grace_period_s: int = 120
    require_template: bool = True
    max_image_bytes: int = 6_000_000
    disable_tool_normalization: bool = False
    log_prompts: bool = False
    schema_version: int = 1
    suppress_decorations: bool = True
    include_non_installed: bool = False
    loopback_alias: Optional[str] = None
    vllm_single_port: bool = True
    vllm_port: int = 24_001
    vllm_state_path: str = "_staging/active_vllm.json"
    vllm_start_timeout_s: int = 180
    vllm_stop_timeout_s: int = 30
    vllm_health_timeout_s: int = 120
    vllm_gpu_memory_utilization: float = 0.75
    vllm_max_model_len: int | None = None
    # History truncation for vision models
    vision_truncate_history: bool = True
    vision_keep_system: bool = True
    vision_keep_last_n_turns: int = 0  # 0 = only current message
    # History truncation for reasoning/thinking models
    reasoning_truncate_history: bool = False  # Opt-in for now
    reasoning_keep_system: bool = True
    reasoning_keep_last_n_turns: int = 1  # Keep 1 turn of context
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_stop_timeout_s: int = 30

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
            autostart_enabled=_get_bool("CHAT_PROXY_AUTOSTART_ENABLED", True),
            autostart_map_raw=os.environ.get("CHAT_PROXY_AUTOSTART_MAP"),
            autostart_grace_period_s=_get_int(
                "CHAT_PROXY_AUTOSTART_GRACE_PERIOD_S", 120
            ),
            require_template=_get_bool("CHAT_PROXY_REQUIRE_TEMPLATE", True),
            max_image_bytes=_get_int("CHAT_PROXY_MAX_IMAGE_BYTES", 6_000_000),
            disable_tool_normalization=_get_bool(
                "CHAT_PROXY_DISABLE_TOOL_NORMALIZATION", False
            ),
            log_prompts=_get_bool("CHAT_PROXY_LOG_PROMPTS", False),
            schema_version=_get_int("CHAT_PROXY_SCHEMA_VERSION", 1),
            suppress_decorations=_get_bool("CHAT_PROXY_SUPPRESS_DECORATIONS", True),
            include_non_installed=_get_bool("CHAT_PROXY_INCLUDE_NON_INSTALLED", False),
            loopback_alias=os.environ.get("CHAT_PROXY_LOOPBACK_ALIAS"),
            vllm_single_port=_get_bool("CHAT_PROXY_VLLM_SINGLE_PORT", True),
            vllm_port=_get_int("CHAT_PROXY_VLLM_PORT", 24_001),
            vllm_state_path=os.environ.get(
                "CHAT_PROXY_VLLM_STATE_PATH", "_staging/active_vllm.json"
            ),
            vllm_start_timeout_s=_get_int("CHAT_PROXY_VLLM_START_TIMEOUT_S", 180),
            vllm_stop_timeout_s=_get_int("CHAT_PROXY_VLLM_STOP_TIMEOUT_S", 30),
            vllm_health_timeout_s=_get_int("CHAT_PROXY_VLLM_HEALTH_TIMEOUT_S", 120),
            vllm_gpu_memory_utilization=_get_float(
                "CHAT_PROXY_VLLM_GPU_MEMORY_UTILIZATION", 0.75
            ),
            vllm_max_model_len=_get_optional_int("CHAT_PROXY_VLLM_MAX_MODEL_LEN"),
            vision_truncate_history=_get_bool(
                "CHAT_PROXY_VISION_TRUNCATE_HISTORY", True
            ),
            vision_keep_system=_get_bool("CHAT_PROXY_VISION_KEEP_SYSTEM", True),
            vision_keep_last_n_turns=_get_int("CHAT_PROXY_VISION_KEEP_LAST_N_TURNS", 0),
            reasoning_truncate_history=_get_bool(
                "CHAT_PROXY_REASONING_TRUNCATE_HISTORY", False
            ),
            reasoning_keep_system=_get_bool("CHAT_PROXY_REASONING_KEEP_SYSTEM", True),
            reasoning_keep_last_n_turns=_get_int(
                "CHAT_PROXY_REASONING_KEEP_LAST_N_TURNS", 1
            ),
            ollama_base_url=os.environ.get(
                "CHAT_PROXY_OLLAMA_BASE_URL", "http://127.0.0.1:11434"
            ),
            ollama_stop_timeout_s=_get_int("CHAT_PROXY_OLLAMA_STOP_TIMEOUT_S", 30),
        )
