"""Configuration helpers for the Personal Tagger application."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import tomllib


_CONFIG_ENV_PREFIX = "IMAGEWORKS_PERSONAL_TAGGER__"


def _find_pyproject(start: Optional[Path] = None) -> Optional[Path]:
    """Locate the nearest ``pyproject.toml`` relative to *start*."""

    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        pyproject = candidate / "pyproject.toml"
        if pyproject.exists():
            return pyproject
    return None


def _coerce_bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default


def _coerce_int(value: object, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except Exception:
        return default


def _coerce_float(value: object, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def _as_path(value: object) -> Optional[Path]:
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    if isinstance(value, str) and value.strip():
        return Path(value).expanduser()
    return None


def _normalise_iterable(value: object) -> Tuple[str, ...]:
    if value is None:
        return tuple()
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",")]
    elif isinstance(value, Iterable):
        parts = []
        for item in value:
            if not item:
                continue
            parts.append(str(item).strip())
    else:
        return tuple()

    return tuple(filter(None, parts))


@dataclass(frozen=True)
class PersonalTaggerSettings:
    """Default configuration values sourced from project metadata."""

    default_input_dirs: Tuple[Path, ...] = field(default_factory=tuple)
    default_output_jsonl: Path = Path("outputs/results/personal_tagger.jsonl")
    default_summary_path: Path = Path("outputs/summaries/personal_tagger_summary.md")
    default_backend: str = "lmdeploy"
    default_model: str = "Qwen2.5-VL-7B-AWQ"
    default_base_url: str = "http://localhost:24001/v1"
    default_timeout: int = 120
    default_max_new_tokens: int = 512
    default_temperature: float = 0.2
    default_top_p: float = 0.9
    default_prompt_profile: str = "default"
    default_batch_size: int = 2
    default_max_workers: int = 2
    default_recursive: bool = True
    default_dry_run: bool = False
    default_no_meta: bool = False
    default_backup_originals: bool = True
    default_overwrite_metadata: bool = False
    caption_model: str = "Qwen2.5-VL-7B-AWQ"
    keyword_model: str = "Qwen2.5-VL-7B-AWQ"
    description_model: str = "Qwen2.5-VL-7B-AWQ"
    default_api_key: str = "EMPTY"
    max_keywords: int = 15
    image_extensions: Tuple[str, ...] = (
        ".jpg",
        ".jpeg",
        ".png",
        ".tif",
        ".tiff",
        ".orf",
        ".cr2",
        ".cr3",
    )
    json_schema_version: str = "1.0"


@dataclass(frozen=True)
class PersonalTaggerConfig:
    """Fully resolved runtime configuration for a CLI invocation."""

    input_paths: Tuple[Path, ...]
    output_jsonl: Path
    summary_path: Path
    backend: str
    base_url: str
    description_model: str
    timeout: int
    max_new_tokens: int
    temperature: float
    top_p: float
    prompt_profile: str
    batch_size: int
    max_workers: int
    recursive: bool
    dry_run: bool
    no_meta: bool
    backup_originals: bool
    overwrite_metadata: bool
    image_extensions: Tuple[str, ...]
    json_schema_version: str
    caption_model: str
    keyword_model: str
    max_keywords: int
    api_key: str


def _merge_dict(
    base: Dict[str, object], override: Optional[Dict[str, object]]
) -> Dict[str, object]:
    merged = base.copy()
    if not override:
        return merged
    for key, value in override.items():
        if value is None:
            continue
        merged[key] = value
    return merged


def _load_pyproject_settings(start: Optional[Path]) -> Dict[str, object]:
    pyproject = _find_pyproject(start)
    if not pyproject:
        return {}
    try:
        with pyproject.open("rb") as handle:
            data = tomllib.load(handle)
    except Exception:
        return {}

    tool_cfg = data.get("tool", {}).get("imageworks", {})
    if not isinstance(tool_cfg, dict):
        return {}

    personal_cfg = tool_cfg.get("personal_tagger")
    return personal_cfg if isinstance(personal_cfg, dict) else {}


def _load_env_settings() -> Dict[str, object]:
    values: Dict[str, object] = {}
    for env_key, env_value in os.environ.items():
        if not env_key.startswith(_CONFIG_ENV_PREFIX):
            continue
        key = env_key[len(_CONFIG_ENV_PREFIX) :].lower()
        values[key] = env_value
    return values


def load_config(start: Optional[Path] = None) -> PersonalTaggerSettings:
    """Load project-level defaults for the personal tagger."""

    defaults = PersonalTaggerSettings()
    result: Dict[str, object] = {
        "default_input_dirs": defaults.default_input_dirs,
        "default_output_jsonl": defaults.default_output_jsonl,
        "default_summary_path": defaults.default_summary_path,
        "default_backend": defaults.default_backend,
        "default_model": defaults.default_model,
        "description_model": defaults.description_model,
        "caption_model": defaults.caption_model,
        "keyword_model": defaults.keyword_model,
        "default_api_key": defaults.default_api_key,
        "default_base_url": defaults.default_base_url,
        "default_timeout": defaults.default_timeout,
        "default_max_new_tokens": defaults.default_max_new_tokens,
        "default_temperature": defaults.default_temperature,
        "default_top_p": defaults.default_top_p,
        "default_prompt_profile": defaults.default_prompt_profile,
        "default_batch_size": defaults.default_batch_size,
        "default_max_workers": defaults.default_max_workers,
        "default_recursive": defaults.default_recursive,
        "default_dry_run": defaults.default_dry_run,
        "default_backup_originals": defaults.default_backup_originals,
        "default_overwrite_metadata": defaults.default_overwrite_metadata,
        "max_keywords": defaults.max_keywords,
        "image_extensions": defaults.image_extensions,
        "json_schema_version": defaults.json_schema_version,
    }

    result = _merge_dict(result, _load_pyproject_settings(start))
    result = _merge_dict(result, _load_env_settings())

    input_dirs = tuple(
        filter(
            None,
            (
                _as_path(path)
                for path in (
                    result.get("default_input_dirs")
                    if isinstance(result.get("default_input_dirs"), Sequence)
                    else _normalise_iterable(result.get("default_input_dirs"))
                )
            ),
        )
    )

    output_jsonl = (
        _as_path(result.get("default_output_jsonl")) or defaults.default_output_jsonl
    )
    summary_path = (
        _as_path(result.get("default_summary_path")) or defaults.default_summary_path
    )
    backend = (
        str(result.get("default_backend", defaults.default_backend)).strip()
        or defaults.default_backend
    )
    description_model = (
        str(
            result.get(
                "description_model",
                result.get("default_model", defaults.description_model),
            )
        ).strip()
        or defaults.description_model
    )
    caption_model = (
        str(result.get("caption_model", defaults.caption_model)).strip()
        or defaults.caption_model
    )
    keyword_model = (
        str(result.get("keyword_model", defaults.keyword_model)).strip()
        or defaults.keyword_model
    )
    base_url = (
        str(result.get("default_base_url", defaults.default_base_url)).strip()
        or defaults.default_base_url
    )
    api_key = (
        str(result.get("default_api_key", defaults.default_api_key)).strip()
        or defaults.default_api_key
    )
    timeout = _coerce_int(result.get("default_timeout"), defaults.default_timeout)
    max_new_tokens = _coerce_int(
        result.get("default_max_new_tokens"), defaults.default_max_new_tokens
    )
    temperature = _coerce_float(
        result.get("default_temperature"), defaults.default_temperature
    )
    top_p = _coerce_float(result.get("default_top_p"), defaults.default_top_p)
    prompt_profile = (
        str(
            result.get("default_prompt_profile", defaults.default_prompt_profile)
        ).strip()
        or defaults.default_prompt_profile
    )
    batch_size = _coerce_int(
        result.get("default_batch_size"), defaults.default_batch_size
    )
    max_workers = _coerce_int(
        result.get("default_max_workers"), defaults.default_max_workers
    )
    recursive = _coerce_bool(
        result.get("default_recursive"), defaults.default_recursive
    )
    dry_run = _coerce_bool(result.get("default_dry_run"), defaults.default_dry_run)
    backup_originals = _coerce_bool(
        result.get("default_backup_originals"), defaults.default_backup_originals
    )
    overwrite_metadata = _coerce_bool(
        result.get("default_overwrite_metadata"), defaults.default_overwrite_metadata
    )
    max_keywords = _coerce_int(result.get("max_keywords"), defaults.max_keywords)

    image_exts = _normalise_iterable(result.get("image_extensions"))
    if not image_exts:
        image_exts = defaults.image_extensions
    else:
        image_exts = tuple(
            ext.lower() if ext.startswith(".") else f".{ext.lower()}"
            for ext in image_exts
        )

    return PersonalTaggerSettings(
        default_input_dirs=input_dirs,
        default_output_jsonl=output_jsonl,
        default_summary_path=summary_path,
        default_backend=backend,
        default_model=description_model,
        default_base_url=base_url,
        default_timeout=timeout,
        default_max_new_tokens=max_new_tokens,
        default_temperature=temperature,
        default_top_p=top_p,
        default_prompt_profile=prompt_profile,
        default_batch_size=batch_size,
        default_max_workers=max_workers,
        default_recursive=recursive,
        default_dry_run=dry_run,
        default_backup_originals=backup_originals,
        default_overwrite_metadata=overwrite_metadata,
        caption_model=caption_model,
        keyword_model=keyword_model,
        description_model=description_model,
        max_keywords=max_keywords,
        image_extensions=image_exts,
        json_schema_version=str(
            result.get("json_schema_version", defaults.json_schema_version)
        ),
        default_api_key=api_key,
    )


def build_runtime_config(
    *,
    settings: PersonalTaggerSettings,
    input_dirs: Sequence[Path],
    output_jsonl: Optional[Path] = None,
    summary_path: Optional[Path] = None,
    backend: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    caption_model: Optional[str] = None,
    keyword_model: Optional[str] = None,
    description_model: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: Optional[int] = None,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    prompt_profile: Optional[str] = None,
    batch_size: Optional[int] = None,
    max_workers: Optional[int] = None,
    recursive: Optional[bool] = None,
    dry_run: Optional[bool] = None,
    no_meta: Optional[bool] = None,
    backup_originals: Optional[bool] = None,
    overwrite_metadata: Optional[bool] = None,
    image_extensions: Optional[Sequence[str]] = None,
    max_keywords: Optional[int] = None,
) -> PersonalTaggerConfig:
    """Compose a runtime configuration from defaults and CLI overrides."""

    resolved_inputs: List[Path] = []
    for path in input_dirs:
        expanded = Path(path).expanduser()
        resolved_inputs.append(expanded)

    if not resolved_inputs:
        resolved_inputs = list(settings.default_input_dirs)

    if not resolved_inputs:
        raise ValueError(
            "No input directories provided. Use --input-dir or configure defaults."
        )

    resolved_output = Path(output_jsonl or settings.default_output_jsonl).expanduser()
    resolved_summary = Path(summary_path or settings.default_summary_path).expanduser()

    resolved_backend = (backend or settings.default_backend).strip().lower()
    desc_candidate = (
        description_model
        or model
        or settings.description_model
        or settings.default_model
    )
    resolved_description_model = (
        str(desc_candidate).strip() or settings.description_model
    )
    resolved_caption_model = (
        str(caption_model or settings.caption_model).strip() or settings.caption_model
    )
    resolved_keyword_model = (
        str(keyword_model or settings.keyword_model).strip() or settings.keyword_model
    )
    resolved_base_url = (base_url or settings.default_base_url).strip()
    resolved_api_key = (api_key or settings.default_api_key).strip()

    resolved_timeout = timeout if timeout is not None else settings.default_timeout
    resolved_max_new_tokens = (
        max_new_tokens
        if max_new_tokens is not None
        else settings.default_max_new_tokens
    )
    resolved_temperature = (
        temperature if temperature is not None else settings.default_temperature
    )
    resolved_top_p = top_p if top_p is not None else settings.default_top_p
    resolved_prompt = (prompt_profile or settings.default_prompt_profile).strip()
    resolved_batch_size = (
        batch_size if batch_size is not None else settings.default_batch_size
    )
    resolved_max_workers = (
        max_workers if max_workers is not None else settings.default_max_workers
    )
    resolved_recursive = (
        recursive if recursive is not None else settings.default_recursive
    )
    resolved_dry_run = dry_run if dry_run is not None else settings.default_dry_run
    resolved_no_meta = no_meta if no_meta is not None else settings.default_no_meta
    resolved_backup = (
        backup_originals
        if backup_originals is not None
        else settings.default_backup_originals
    )
    resolved_overwrite = (
        overwrite_metadata
        if overwrite_metadata is not None
        else settings.default_overwrite_metadata
    )
    resolved_max_keywords = (
        max_keywords if max_keywords is not None else settings.max_keywords
    )

    resolved_exts: Tuple[str, ...]
    if image_extensions:
        normalised = _normalise_iterable(image_extensions)
        if normalised:
            resolved_exts = tuple(
                part if part.startswith(".") else f".{part.lower()}"
                for part in normalised
            )
        else:
            resolved_exts = settings.image_extensions
    else:
        resolved_exts = settings.image_extensions

    return PersonalTaggerConfig(
        input_paths=tuple(resolved_inputs),
        output_jsonl=resolved_output,
        summary_path=resolved_summary,
        backend=resolved_backend,
        base_url=resolved_base_url,
        description_model=resolved_description_model,
        api_key=resolved_api_key,
        timeout=resolved_timeout,
        max_new_tokens=resolved_max_new_tokens,
        temperature=resolved_temperature,
        top_p=resolved_top_p,
        prompt_profile=resolved_prompt,
        batch_size=resolved_batch_size,
        max_workers=resolved_max_workers,
        recursive=resolved_recursive,
        dry_run=resolved_dry_run,
        no_meta=resolved_no_meta,
        backup_originals=resolved_backup,
        overwrite_metadata=resolved_overwrite,
        image_extensions=resolved_exts,
        json_schema_version=settings.json_schema_version,
        caption_model=resolved_caption_model,
        keyword_model=resolved_keyword_model,
        max_keywords=resolved_max_keywords,
    )
