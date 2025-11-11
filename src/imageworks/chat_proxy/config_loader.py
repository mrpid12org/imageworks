from __future__ import annotations

import os
import tempfile
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Callable

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

from .config import ProxyConfig

CONFIG_FILE_ENV = "CHAT_PROXY_CONFIG_FILE"
ENV_PREFIX = "CHAT_PROXY_"
DEFAULT_CONFIG_PATH = Path("configs/chat_proxy.toml")

_SECTION_MAP: dict[str, list[str]] = {
    "general": ["schema_version", "require_template"],
    "server": [
        "host",
        "port",
        "enable_metrics",
        "log_path",
        "max_log_bytes",
        "loopback_alias",
        "include_non_installed",
        "suppress_decorations",
        "log_prompts",
    ],
    "timeouts": [
        "backend_timeout_ms",
        "stream_idle_timeout_ms",
        "autostart_grace_period_s",
    ],
    "autostart": ["autostart_enabled", "autostart_map_raw"],
    "limits": ["max_image_bytes", "disable_tool_normalization"],
    "vision_preprocess": [
        "auto_downscale_images",
        "max_image_pixels",
        "image_jpeg_quality",
        "vision_downscale_backends",
    ],
    "vision_history": [
        "vision_truncate_history",
        "vision_keep_system",
        "vision_keep_last_n_turns",
    ],
    "reasoning_history": [
        "reasoning_truncate_history",
        "reasoning_keep_system",
        "reasoning_keep_last_n_turns",
    ],
    "vllm": [
        "vllm_single_port",
        "vllm_port",
        "vllm_state_path",
        "vllm_start_timeout_s",
        "vllm_stop_timeout_s",
        "vllm_health_timeout_s",
        "vllm_gpu_memory_utilization",
        "vllm_max_model_len",
    ],
    "ollama": ["ollama_base_url", "ollama_stop_timeout_s"],
}


def _field_types() -> dict[str, Any]:
    return {f.name: f.type for f in fields(ProxyConfig)}


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _coerce_int(value: Any) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return int(str(value))


def _coerce_float(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value))


def _coerce_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _coerce_optional(value: Any, caster: Callable[[Any], Any]) -> Any:
    if value in ("", None):
        return None
    return caster(value)


_CASTERS: dict[Any, Callable[[Any], Any]] = {
    bool: _coerce_bool,
    int: _coerce_int,
    float: _coerce_float,
    str: _coerce_str,
}


def _coerce_value(field_type: Any, value: Any) -> Any:
    origin = getattr(field_type, "__origin__", None)
    if origin is None:
        caster = _CASTERS.get(field_type)
        if caster:
            return caster(value)
        return value

    if origin is list:
        return list(value)

    if origin is dict:
        return dict(value)

    if origin is type(None):  # pragma: no cover - should not happen
        return None

    from typing import Union, get_args  # imported lazily

    if origin is Union:
        args = [
            arg for arg in get_args(field_type) if arg is not type(None)
        ]  # noqa: E721
        if len(args) == 1:
            caster = _CASTERS.get(args[0])
            if caster:
                return _coerce_optional(value, caster)
    return value


def _read_config_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as fh:
        data = tomllib.load(fh)

    out: dict[str, Any] = {}
    for section, keys in _SECTION_MAP.items():
        section_values = data.get(section, {})
        if not isinstance(section_values, dict):
            continue
        for key in keys:
            if key in section_values:
                out[key] = section_values[key]
    return out


def _apply_env_overrides(config: dict[str, Any]) -> dict[str, Any]:
    env = os.environ

    def env_bool(name: str, current: bool) -> bool:
        val = env.get(name)
        if val is None:
            return current
        return val.lower() in {"1", "true", "yes", "on"}

    def env_int(name: str, current: int) -> int:
        val = env.get(name)
        if val is None:
            return current
        try:
            return int(val)
        except ValueError:
            return current

    def env_float(name: str, current: float) -> float:
        val = env.get(name)
        if val is None:
            return current
        try:
            return float(val)
        except ValueError:
            return current

    def env_str(name: str, current: str | None) -> str | None:
        val = env.get(name)
        if val is None:
            return current
        return val

    def env_list(name: str, current: list[str]) -> list[str]:
        val = env.get(name)
        if not val:
            return current
        parts = [item.strip() for item in val.split(",")]
        return [item for item in parts if item]

    overrides = {
        "host": env_str("CHAT_PROXY_HOST", config["host"]),
        "port": env_int("CHAT_PROXY_PORT", config["port"]),
        "enable_metrics": env_bool(
            "CHAT_PROXY_ENABLE_METRICS", config["enable_metrics"]
        ),
        "log_path": env_str("CHAT_PROXY_LOG_PATH", config["log_path"]),
        "max_log_bytes": env_int("CHAT_PROXY_MAX_LOG_BYTES", config["max_log_bytes"]),
        "backend_timeout_ms": env_int(
            "CHAT_PROXY_BACKEND_TIMEOUT_MS", config["backend_timeout_ms"]
        ),
        "stream_idle_timeout_ms": env_int(
            "CHAT_PROXY_STREAM_IDLE_TIMEOUT_MS", config["stream_idle_timeout_ms"]
        ),
        "autostart_enabled": env_bool(
            "CHAT_PROXY_AUTOSTART_ENABLED", config["autostart_enabled"]
        ),
        "autostart_map_raw": env_str(
            "CHAT_PROXY_AUTOSTART_MAP", config.get("autostart_map_raw")
        ),
        "autostart_grace_period_s": env_int(
            "CHAT_PROXY_AUTOSTART_GRACE_PERIOD_S", config["autostart_grace_period_s"]
        ),
        "require_template": env_bool(
            "CHAT_PROXY_REQUIRE_TEMPLATE", config["require_template"]
        ),
        "max_image_bytes": env_int(
            "CHAT_PROXY_MAX_IMAGE_BYTES", config["max_image_bytes"]
        ),
        "auto_downscale_images": env_bool(
            "CHAT_PROXY_AUTO_DOWNSCALE_IMAGES", config["auto_downscale_images"]
        ),
        "max_image_pixels": env_int(
            "CHAT_PROXY_MAX_IMAGE_PIXELS", config["max_image_pixels"]
        ),
        "image_jpeg_quality": env_int(
            "CHAT_PROXY_IMAGE_JPEG_QUALITY", config["image_jpeg_quality"]
        ),
        "vision_downscale_backends": env_list(
            "CHAT_PROXY_VISION_DOWNSCALE_BACKENDS",
            config.get("vision_downscale_backends", []),
        ),
        "disable_tool_normalization": env_bool(
            "CHAT_PROXY_DISABLE_TOOL_NORMALIZATION",
            config["disable_tool_normalization"],
        ),
        "schema_version": env_int(
            "CHAT_PROXY_SCHEMA_VERSION", config["schema_version"]
        ),
        "suppress_decorations": env_bool(
            "CHAT_PROXY_SUPPRESS_DECORATIONS", config["suppress_decorations"]
        ),
        "include_non_installed": env_bool(
            "CHAT_PROXY_INCLUDE_NON_INSTALLED", config["include_non_installed"]
        ),
        "loopback_alias": env_str(
            "CHAT_PROXY_LOOPBACK_ALIAS", config.get("loopback_alias")
        ),
        "vllm_single_port": env_bool(
            "CHAT_PROXY_VLLM_SINGLE_PORT", config["vllm_single_port"]
        ),
        "vllm_port": env_int("CHAT_PROXY_VLLM_PORT", config["vllm_port"]),
        "vllm_state_path": env_str(
            "CHAT_PROXY_VLLM_STATE_PATH", config["vllm_state_path"]
        ),
        "vllm_start_timeout_s": env_int(
            "CHAT_PROXY_VLLM_START_TIMEOUT_S", config["vllm_start_timeout_s"]
        ),
        "vllm_stop_timeout_s": env_int(
            "CHAT_PROXY_VLLM_STOP_TIMEOUT_S", config["vllm_stop_timeout_s"]
        ),
        "vllm_health_timeout_s": env_int(
            "CHAT_PROXY_VLLM_HEALTH_TIMEOUT_S", config["vllm_health_timeout_s"]
        ),
        "vllm_gpu_memory_utilization": env_float(
            "CHAT_PROXY_VLLM_GPU_MEMORY_UTILIZATION",
            config["vllm_gpu_memory_utilization"],
        ),
        "vllm_max_model_len": (
            env_int("CHAT_PROXY_VLLM_MAX_MODEL_LEN", config["vllm_max_model_len"])
            if env.get("CHAT_PROXY_VLLM_MAX_MODEL_LEN")
            else config["vllm_max_model_len"]
        ),
        "vision_truncate_history": env_bool(
            "CHAT_PROXY_VISION_TRUNCATE_HISTORY", config["vision_truncate_history"]
        ),
        "vision_keep_system": env_bool(
            "CHAT_PROXY_VISION_KEEP_SYSTEM", config["vision_keep_system"]
        ),
        "vision_keep_last_n_turns": env_int(
            "CHAT_PROXY_VISION_KEEP_LAST_N_TURNS", config["vision_keep_last_n_turns"]
        ),
        "reasoning_truncate_history": env_bool(
            "CHAT_PROXY_REASONING_TRUNCATE_HISTORY",
            config["reasoning_truncate_history"],
        ),
        "reasoning_keep_system": env_bool(
            "CHAT_PROXY_REASONING_KEEP_SYSTEM", config["reasoning_keep_system"]
        ),
        "reasoning_keep_last_n_turns": env_int(
            "CHAT_PROXY_REASONING_KEEP_LAST_N_TURNS",
            config["reasoning_keep_last_n_turns"],
        ),
        "ollama_base_url": env_str(
            "CHAT_PROXY_OLLAMA_BASE_URL", config["ollama_base_url"]
        ),
        "ollama_stop_timeout_s": env_int(
            "CHAT_PROXY_OLLAMA_STOP_TIMEOUT_S", config["ollama_stop_timeout_s"]
        ),
    }
    config.update(overrides)
    return config


def _default_config_dict() -> dict[str, Any]:
    defaults = ProxyConfig()
    data = asdict(defaults)
    data.pop("config_file_path", None)
    return data


def _normalize(config: dict[str, Any]) -> dict[str, Any]:
    field_types = _field_types()
    normalized = {}
    for key, default_value in _default_config_dict().items():
        value = config.get(key, default_value)
        field_type = field_types.get(key)
        try:
            normalized[key] = _coerce_value(field_type, value)
        except Exception:
            normalized[key] = default_value
    # normalize optional empties
    if normalized.get("autostart_map_raw") == "":
        normalized["autostart_map_raw"] = None
    if normalized.get("loopback_alias") == "":
        normalized["loopback_alias"] = None
    return normalized


def _ensure_config_file(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    defaults = ProxyConfig()
    write_config(defaults, path)


def load_file_config() -> dict[str, Any]:
    path = Path(os.environ.get(CONFIG_FILE_ENV, DEFAULT_CONFIG_PATH)).expanduser()
    _ensure_config_file(path)
    base = _default_config_dict()
    base.update(_read_config_file(path))
    normalized = _normalize(base)
    return normalized


def load_proxy_config() -> ProxyConfig:
    candidate = Path(os.environ.get(CONFIG_FILE_ENV, DEFAULT_CONFIG_PATH))
    candidate = candidate.expanduser()
    _ensure_config_file(candidate)
    file_values = _read_config_file(candidate)
    normalized = _normalize(file_values)
    normalized = _apply_env_overrides(normalized)
    cfg = ProxyConfig(**normalized)
    cfg.config_file_path = str(candidate)
    return cfg


def _format_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        inner = ", ".join(_format_value(item) for item in value)
        return f"[{inner}]"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if value is None:
        return '""'
    escaped = str(value).replace('"', '\\"')
    return f'"{escaped}"'


def _ordered_sections(config: ProxyConfig) -> dict[str, dict[str, Any]]:
    config_dict = asdict(config)
    config_dict.pop("config_file_path", None)
    sections: dict[str, dict[str, Any]] = {}
    for section, keys in _SECTION_MAP.items():
        section_values = {}
        for key in keys:
            if key in config_dict:
                section_values[key] = config_dict[key]
        if section_values:
            sections[section] = section_values
    # include anything new not in map under misc
    remaining = {
        k: v
        for k, v in config_dict.items()
        if not any(k in keys for keys in _SECTION_MAP.values())
    }
    if remaining:
        sections["misc"] = remaining
    return sections


def write_config(config: ProxyConfig, path: Path | None = None) -> None:
    path = Path(
        path or os.environ.get(CONFIG_FILE_ENV, DEFAULT_CONFIG_PATH)
    ).expanduser()
    sections = _ordered_sections(config)
    lines: list[str] = [
        "# ImageWorks Chat Proxy configuration.",
        "# Generated automatically. Edit values as needed.",
    ]
    for section, values in sections.items():
        lines.append("")
        lines.append(f"[{section}]")
        for key, value in values.items():
            lines.append(f"{key} = {_format_value(value)}")

    tmp_fd, tmp_path = tempfile.mkstemp(prefix="chat_proxy_config_", suffix=".toml")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines))
        Path(tmp_path).replace(path)
    finally:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass


def update_config_file(updates: dict[str, Any]) -> ProxyConfig:
    path = Path(os.environ.get(CONFIG_FILE_ENV, DEFAULT_CONFIG_PATH)).expanduser()
    _ensure_config_file(path)
    base = _default_config_dict()
    current_file = _read_config_file(path)
    base.update(current_file)

    unknown = [key for key in updates if key not in base]
    if unknown:
        raise KeyError(f"Unknown configuration field(s): {', '.join(sorted(unknown))}")

    base.update(updates)
    normalized = _normalize(base)
    file_config = ProxyConfig(**normalized)
    file_config.config_file_path = str(path)
    write_config(file_config, path)
    return load_proxy_config()


def list_env_overrides() -> dict[str, str]:
    return {
        key: value for key, value in os.environ.items() if key.startswith(ENV_PREFIX)
    }
