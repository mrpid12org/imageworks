from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List


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
    auto_downscale_images: bool = True
    max_image_pixels: int = 448
    image_jpeg_quality: int = 85
    vision_downscale_backends: List[str] = field(default_factory=lambda: ["vllm"])
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
    config_file_path: Optional[str] = None

    @classmethod
    def load(cls) -> "ProxyConfig":
        from .config_loader import load_proxy_config

        return load_proxy_config()
