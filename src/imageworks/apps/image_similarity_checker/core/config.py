"""Configuration helpers for the image similarity checker module."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import tomllib


_CONFIG_ENV_PREFIX = "IMAGEWORKS_IMAGE_SIMILARITY__"


def _find_pyproject(start: Optional[Path] = None) -> Optional[Path]:
    """Locate the closest ``pyproject.toml`` relative to *start*."""

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
    except Exception:  # noqa: BLE001 - defensive coercion
        return default


def _coerce_float(value: object, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except Exception:  # noqa: BLE001 - defensive coercion
        return default


def _normalise_iterable(value: object) -> Tuple[str, ...]:
    if value is None:
        return tuple()
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
    elif isinstance(value, Iterable):
        parts = []
        for item in value:
            if not item:
                continue
            parts.append(str(item).strip())
    else:
        return tuple()

    return tuple(filter(None, parts))


def _as_path(value: object) -> Optional[Path]:
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    if isinstance(value, str) and value.strip():
        return Path(value).expanduser()
    return None


@dataclass(frozen=True)
class SimilaritySettings:
    """Default configuration values sourced from project metadata."""

    default_candidates: Tuple[Path, ...] = field(default_factory=tuple)
    # Library manifest cache defaults
    default_refresh_library_cache: bool = False
    default_manifest_ttl_seconds: int = 43200  # 12 hours
    default_library_root: Path = Path(
        "/mnt/d/Proper Photos/photos/ccc competition images"
    )
    default_output_jsonl: Path = Path("outputs/results/similarity_results.jsonl")
    default_summary_path: Path = Path("outputs/summaries/similarity_summary.md")
    default_cache_dir: Path = Path("outputs/cache/similarity")
    default_fail_threshold: float = 0.92
    default_query_threshold: float = 0.82
    default_top_matches: int = 5
    default_similarity_metric: str = "cosine"
    default_strategies: Tuple[str, ...] = ("embedding", "perceptual_hash")
    default_recursive: bool = True
    default_image_extensions: Tuple[str, ...] = (
        ".jpg",
        ".jpeg",
        ".png",
        ".tif",
        ".tiff",
    )
    default_backend: str = "vllm"
    default_model: str = "Qwen2.5-VL-7B-AWQ"
    default_base_url: str = "http://localhost:8000/v1"
    default_timeout: int = 120
    default_api_key: str = "EMPTY"
    default_prompt_profile: str = "baseline"
    default_write_metadata: bool = False
    default_backup_originals: bool = True
    default_overwrite_metadata: bool = False
    default_use_loader: bool = False
    default_registry_model: Optional[str] = None
    default_registry_capabilities: Tuple[str, ...] = ("vision",)
    default_embedding_backend: str = "simple"
    default_embedding_model: Optional[str] = None
    default_generate_explanations: bool = False
    # Performance metrics defaults
    default_enable_perf_metrics: bool = False
    default_metrics_path: Path = Path("outputs/metrics/similarity_perf.json")
    # Augmentation pooling defaults
    default_enable_augment_pooling: bool = False
    default_augment_grayscale: bool = True
    default_augment_five_crop: bool = True
    default_augment_five_crop_ratio: float = 0.875


@dataclass(frozen=True)
class SimilarityConfig:
    """Fully resolved runtime configuration for a checker invocation."""

    candidates: Tuple[Path, ...]
    library_root: Path
    output_jsonl: Path
    summary_path: Path
    cache_dir: Path
    fail_threshold: float
    query_threshold: float
    top_matches: int
    similarity_metric: str
    strategies: Tuple[str, ...]
    recursive: bool
    image_extensions: Tuple[str, ...]
    backend: str
    embedding_backend: str
    embedding_model: Optional[str]
    base_url: str
    model: str
    api_key: str
    timeout: int
    prompt_profile: str
    write_metadata: bool
    backup_originals: bool
    overwrite_metadata: bool
    dry_run: bool
    use_loader: bool
    registry_model: Optional[str]
    registry_capabilities: Tuple[str, ...]
    generate_explanations: bool
    # Performance metrics runtime settings
    enable_perf_metrics: bool
    metrics_path: Path
    # Augmentation pooling runtime settings
    enable_augment_pooling: bool
    augment_grayscale: bool
    augment_five_crop: bool
    augment_five_crop_ratio: float
    # Library discovery cache controls
    refresh_library_cache: bool
    manifest_ttl_seconds: int


def _load_pyproject_settings(start: Optional[Path]) -> Dict[str, object]:
    pyproject = _find_pyproject(start)
    if not pyproject:
        return {}

    try:
        with pyproject.open("rb") as handle:
            data = tomllib.load(handle)
    except Exception:  # noqa: BLE001 - fallback to defaults
        return {}

    tool_cfg = data.get("tool", {}).get("imageworks", {})
    if not isinstance(tool_cfg, dict):
        return {}

    similarity_cfg = tool_cfg.get("image_similarity_checker")
    return similarity_cfg if isinstance(similarity_cfg, dict) else {}


def _load_env_settings() -> Dict[str, object]:
    values: Dict[str, object] = {}
    for env_key, env_value in os.environ.items():
        if not env_key.startswith(_CONFIG_ENV_PREFIX):
            continue
        key = env_key[len(_CONFIG_ENV_PREFIX) :].lower()
        values[key] = env_value
    return values


def load_settings(start: Optional[Path] = None) -> SimilaritySettings:
    """Load project defaults, applying environment overrides when present."""

    raw = _load_pyproject_settings(start)
    raw.update(_load_env_settings())

    candidate_paths: Tuple[Path, ...] = tuple()
    default_candidate = _as_path(raw.get("default_candidate"))
    if default_candidate is not None:
        candidate_paths = (default_candidate.expanduser(),)

    if not candidate_paths:
        candidate_values = _normalise_iterable(raw.get("default_candidates"))
        candidate_paths = tuple(Path(value).expanduser() for value in candidate_values)

    library_root = (
        _as_path(raw.get("default_library_root"))
        or SimilaritySettings.default_library_root
    )
    output_jsonl = (
        _as_path(raw.get("default_output_jsonl"))
        or SimilaritySettings.default_output_jsonl
    )
    summary_path = (
        _as_path(raw.get("default_summary_path"))
        or SimilaritySettings.default_summary_path
    )
    cache_dir = (
        _as_path(raw.get("default_cache_dir")) or SimilaritySettings.default_cache_dir
    )

    fail_threshold = _coerce_float(
        raw.get("default_fail_threshold"), SimilaritySettings.default_fail_threshold
    )
    query_threshold = _coerce_float(
        raw.get("default_query_threshold"), SimilaritySettings.default_query_threshold
    )
    top_matches = _coerce_int(
        raw.get("default_top_matches"), SimilaritySettings.default_top_matches
    )
    similarity_metric = (
        str(
            raw.get("default_similarity_metric"),
        ).strip()
        or SimilaritySettings.default_similarity_metric
    )

    strategies = (
        _normalise_iterable(raw.get("default_strategies"))
        or SimilaritySettings.default_strategies
    )

    recursive = _coerce_bool(
        raw.get("default_recursive"), SimilaritySettings.default_recursive
    )
    image_extensions = (
        _normalise_iterable(raw.get("image_extensions"))
        or SimilaritySettings.default_image_extensions
    )

    backend = str(
        raw.get("default_backend") or SimilaritySettings.default_backend
    ).strip()
    embedding_backend = str(
        raw.get("default_embedding_backend")
        or SimilaritySettings.default_embedding_backend
    ).strip()
    embedding_model = raw.get("default_embedding_model")
    if isinstance(embedding_model, str):
        embedding_model = embedding_model.strip() or None
    model = str(raw.get("default_model") or SimilaritySettings.default_model).strip()
    base_url = str(
        raw.get("default_base_url") or SimilaritySettings.default_base_url
    ).strip()
    timeout = _coerce_int(
        raw.get("default_timeout"), SimilaritySettings.default_timeout
    )
    api_key = str(raw.get("default_api_key") or SimilaritySettings.default_api_key)
    prompt_profile = str(
        raw.get("default_prompt_profile") or SimilaritySettings.default_prompt_profile
    ).strip()
    write_metadata = _coerce_bool(
        raw.get("default_write_metadata"), SimilaritySettings.default_write_metadata
    )
    backup_originals = _coerce_bool(
        raw.get("default_backup_originals"), SimilaritySettings.default_backup_originals
    )
    overwrite_metadata = _coerce_bool(
        raw.get("default_overwrite_metadata"),
        SimilaritySettings.default_overwrite_metadata,
    )
    use_loader = _coerce_bool(
        raw.get("default_use_loader"), SimilaritySettings.default_use_loader
    )
    generate_explanations = _coerce_bool(
        raw.get("default_generate_explanations"),
        SimilaritySettings.default_generate_explanations,
    )
    registry_model = raw.get("default_registry_model")
    if registry_model is not None:
        registry_model = str(registry_model).strip() or None
    registry_capabilities = (
        _normalise_iterable(raw.get("default_registry_capabilities"))
        or SimilaritySettings.default_registry_capabilities
    )

    # Augment pooling defaults
    enable_augment_pooling = _coerce_bool(
        raw.get("default_enable_augment_pooling"),
        SimilaritySettings.default_enable_augment_pooling,
    )
    augment_grayscale = _coerce_bool(
        raw.get("default_augment_grayscale"),
        SimilaritySettings.default_augment_grayscale,
    )
    augment_five_crop = _coerce_bool(
        raw.get("default_augment_five_crop"),
        SimilaritySettings.default_augment_five_crop,
    )
    augment_five_crop_ratio = _coerce_float(
        raw.get("default_augment_five_crop_ratio"),
        SimilaritySettings.default_augment_five_crop_ratio,
    )
    refresh_library_cache = _coerce_bool(
        raw.get("default_refresh_library_cache"),
        SimilaritySettings.default_refresh_library_cache,
    )
    manifest_ttl_seconds = _coerce_int(
        raw.get("default_manifest_ttl_seconds"),
        SimilaritySettings.default_manifest_ttl_seconds,
    )
    # Perf metrics
    enable_perf_metrics = _coerce_bool(
        raw.get("default_enable_perf_metrics"),
        SimilaritySettings.default_enable_perf_metrics,
    )
    metrics_path = (
        _as_path(raw.get("default_metrics_path"))
        or SimilaritySettings.default_metrics_path
    )

    return SimilaritySettings(
        default_candidates=candidate_paths,
        default_library_root=library_root,
        default_output_jsonl=output_jsonl,
        default_summary_path=summary_path,
        default_cache_dir=cache_dir,
        default_fail_threshold=fail_threshold,
        default_query_threshold=query_threshold,
        default_top_matches=top_matches,
        default_similarity_metric=similarity_metric,
        default_strategies=strategies,
        default_recursive=recursive,
        default_image_extensions=image_extensions,
        default_backend=backend,
        default_embedding_backend=embedding_backend,
        default_embedding_model=embedding_model,
        default_model=model,
        default_base_url=base_url,
        default_timeout=timeout,
        default_api_key=api_key,
        default_prompt_profile=prompt_profile,
        default_write_metadata=write_metadata,
        default_backup_originals=backup_originals,
        default_overwrite_metadata=overwrite_metadata,
        default_use_loader=use_loader,
        default_registry_model=registry_model,
        default_registry_capabilities=tuple(
            cap.strip().lower() for cap in registry_capabilities if cap.strip()
        ),
        default_generate_explanations=generate_explanations,
        default_refresh_library_cache=refresh_library_cache,
        default_manifest_ttl_seconds=manifest_ttl_seconds,
        default_enable_augment_pooling=enable_augment_pooling,
        default_augment_grayscale=augment_grayscale,
        default_augment_five_crop=augment_five_crop,
        default_augment_five_crop_ratio=augment_five_crop_ratio,
        default_enable_perf_metrics=enable_perf_metrics,
        default_metrics_path=metrics_path,
    )


def build_runtime_config(
    *,
    settings: SimilaritySettings,
    candidates: Optional[Sequence[Path]] = None,
    library_root: Optional[Path] = None,
    output_jsonl: Optional[Path] = None,
    summary_path: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    fail_threshold: Optional[float] = None,
    query_threshold: Optional[float] = None,
    top_matches: Optional[int] = None,
    similarity_metric: Optional[str] = None,
    strategies: Optional[Sequence[str]] = None,
    recursive: Optional[bool] = None,
    image_extensions: Optional[Sequence[str]] = None,
    backend: Optional[str] = None,
    embedding_backend: Optional[str] = None,
    embedding_model: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: Optional[int] = None,
    prompt_profile: Optional[str] = None,
    write_metadata: Optional[bool] = None,
    backup_originals: Optional[bool] = None,
    overwrite_metadata: Optional[bool] = None,
    dry_run: bool = False,
    use_loader: Optional[bool] = None,
    registry_model: Optional[str] = None,
    registry_capabilities: Optional[Sequence[str]] = None,
    generate_explanations: Optional[bool] = None,
    enable_augment_pooling: Optional[bool] = None,
    augment_grayscale: Optional[bool] = None,
    augment_five_crop: Optional[bool] = None,
    augment_five_crop_ratio: Optional[float] = None,
    enable_perf_metrics: Optional[bool] = None,
    metrics_path: Optional[Path] = None,
    refresh_library_cache: Optional[bool] = None,
    manifest_ttl_seconds: Optional[int] = None,
) -> SimilarityConfig:
    """Merge CLI overrides with defaults to produce a runtime config."""

    resolved_candidates = tuple(
        path.expanduser() for path in (candidates or settings.default_candidates)
    )
    if not resolved_candidates:
        raise ValueError("At least one candidate image or directory must be provided")

    resolved_library = (library_root or settings.default_library_root).expanduser()
    resolved_output = (output_jsonl or settings.default_output_jsonl).expanduser()
    resolved_summary = (summary_path or settings.default_summary_path).expanduser()
    resolved_cache = (cache_dir or settings.default_cache_dir).expanduser()

    resolved_fail = (
        settings.default_fail_threshold
        if fail_threshold is None
        else float(fail_threshold)
    )
    resolved_query = (
        settings.default_query_threshold
        if query_threshold is None
        else float(query_threshold)
    )
    if not (0.0 <= resolved_query <= 1.0 and 0.0 <= resolved_fail <= 1.0):
        raise ValueError("Thresholds must be in the range [0, 1]")
    if resolved_fail < resolved_query:
        raise ValueError(
            "Fail threshold must be greater than or equal to query threshold"
        )

    resolved_top = int(top_matches or settings.default_top_matches)
    resolved_metric = (
        (similarity_metric or settings.default_similarity_metric).strip().lower()
    )
    resolved_strategies = tuple(
        strategy.strip().lower()
        for strategy in (strategies or settings.default_strategies)
        if strategy and strategy.strip()
    )
    if not resolved_strategies:
        raise ValueError("At least one similarity strategy must be configured")

    resolved_recursive = (
        settings.default_recursive if recursive is None else bool(recursive)
    )
    resolved_extensions = tuple(
        ext if ext.startswith(".") else f".{ext}"
        for ext in (image_extensions or settings.default_image_extensions)
    )

    resolved_backend = (backend or settings.default_backend).strip()
    resolved_embedding_backend = (
        (embedding_backend or settings.default_embedding_backend).strip().lower()
    )
    resolved_embedding_model: Optional[str]
    if embedding_model is not None:
        resolved_embedding_model = embedding_model.strip() or None
    else:
        # Fallback to default_embedding_model; if None, allow using explainer model for backends that expect it
        resolved_embedding_model = settings.default_embedding_model or None
    resolved_model = (model or settings.default_model).strip()
    resolved_base_url = (base_url or settings.default_base_url).strip()
    resolved_api_key = (api_key or settings.default_api_key).strip()
    resolved_timeout = int(timeout or settings.default_timeout)
    resolved_prompt = (prompt_profile or settings.default_prompt_profile).strip()
    resolved_write_meta = (
        settings.default_write_metadata
        if write_metadata is None
        else bool(write_metadata)
    )
    resolved_backup_originals = (
        settings.default_backup_originals
        if backup_originals is None
        else bool(backup_originals)
    )
    resolved_overwrite_metadata = (
        settings.default_overwrite_metadata
        if overwrite_metadata is None
        else bool(overwrite_metadata)
    )
    resolved_generate_explanations = (
        settings.default_generate_explanations
        if generate_explanations is None
        else bool(generate_explanations)
    )
    resolved_use_loader = (
        settings.default_use_loader if use_loader is None else bool(use_loader)
    )
    resolved_registry = (
        registry_model
        if registry_model is not None
        else settings.default_registry_model
    )
    if resolved_registry is not None:
        resolved_registry = resolved_registry.strip() or None
    resolved_registry_capabilities = tuple(
        cap.strip().lower()
        for cap in (registry_capabilities or settings.default_registry_capabilities)
        if cap and cap.strip()
    )
    if not resolved_registry_capabilities:
        resolved_registry_capabilities = ("vision",)

    # Augmentation pooling resolution
    resolved_enable_aug = (
        settings.default_enable_augment_pooling
        if enable_augment_pooling is None
        else bool(enable_augment_pooling)
    )
    resolved_aug_gray = (
        settings.default_augment_grayscale
        if augment_grayscale is None
        else bool(augment_grayscale)
    )
    resolved_aug_five = (
        settings.default_augment_five_crop
        if augment_five_crop is None
        else bool(augment_five_crop)
    )
    resolved_aug_ratio = (
        settings.default_augment_five_crop_ratio
        if augment_five_crop_ratio is None
        else float(augment_five_crop_ratio)
    )
    # Perf metrics resolution
    resolved_enable_perf = (
        settings.default_enable_perf_metrics
        if enable_perf_metrics is None
        else bool(enable_perf_metrics)
    )
    resolved_metrics_path = (metrics_path or settings.default_metrics_path).expanduser()
    resolved_refresh_library = (
        settings.default_refresh_library_cache
        if refresh_library_cache is None
        else bool(refresh_library_cache)
    )
    resolved_manifest_ttl = int(
        manifest_ttl_seconds or settings.default_manifest_ttl_seconds
    )

    return SimilarityConfig(
        candidates=resolved_candidates,
        library_root=resolved_library,
        output_jsonl=resolved_output,
        summary_path=resolved_summary,
        cache_dir=resolved_cache,
        fail_threshold=resolved_fail,
        query_threshold=resolved_query,
        top_matches=resolved_top,
        similarity_metric=resolved_metric,
        strategies=resolved_strategies,
        recursive=resolved_recursive,
        image_extensions=resolved_extensions,
        backend=resolved_backend,
        base_url=resolved_base_url,
        model=resolved_model,
        embedding_model=resolved_embedding_model,
        api_key=resolved_api_key,
        timeout=resolved_timeout,
        prompt_profile=resolved_prompt,
        write_metadata=resolved_write_meta,
        backup_originals=resolved_backup_originals,
        overwrite_metadata=resolved_overwrite_metadata,
        dry_run=dry_run,
        use_loader=resolved_use_loader,
        registry_model=resolved_registry,
        registry_capabilities=resolved_registry_capabilities,
        embedding_backend=resolved_embedding_backend,
        generate_explanations=resolved_generate_explanations,
        enable_augment_pooling=resolved_enable_aug,
        augment_grayscale=resolved_aug_gray,
        augment_five_crop=resolved_aug_five,
        augment_five_crop_ratio=resolved_aug_ratio,
        enable_perf_metrics=resolved_enable_perf,
        metrics_path=resolved_metrics_path,
        refresh_library_cache=resolved_refresh_library,
        manifest_ttl_seconds=resolved_manifest_ttl,
    )


def load_config(
    *,
    start: Optional[Path] = None,
    candidates: Optional[Sequence[Path]] = None,
    **overrides: object,
) -> SimilarityConfig:
    """Convenience helper used by the CLI to resolve the runtime config."""

    settings = load_settings(start)
    return build_runtime_config(settings=settings, candidates=candidates, **overrides)
