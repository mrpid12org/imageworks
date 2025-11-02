"""CLI command builders for ImageWorks tools."""

from typing import List, Dict, Any

from imageworks.apps.personal_tagger.core.config import load_config


def build_similarity_command(config: Dict[str, Any]) -> List[str]:
    """
    Build imageworks-image-similarity command from config dict.

    Args:
        config: Configuration dictionary

    Returns:
        Command as list of strings
    """
    cmd = ["uv", "run", "imageworks-image-similarity", "check"]

    # Candidates (positional)
    candidates = config.get("candidates", [])
    if isinstance(candidates, str):
        candidates = [candidates]

    for candidate in candidates:
        cmd.append(str(candidate))

    # Library root
    if config.get("library_root"):
        cmd.extend(["--library-root", str(config["library_root"])])

    # Strategies (repeatable)
    strategies = config.get("strategy", [])
    if isinstance(strategies, str):
        strategies = [strategies]

    for strategy in strategies:
        cmd.extend(["--strategy", strategy])

    # Thresholds
    if "fail_threshold" in config:
        cmd.extend(["--fail-threshold", str(config["fail_threshold"])])
    if "query_threshold" in config:
        cmd.extend(["--query-threshold", str(config["query_threshold"])])
    if "top_matches" in config:
        cmd.extend(["--top-matches", str(config["top_matches"])])

    # Embedding configuration
    if config.get("embedding_backend"):
        cmd.extend(["--embedding-backend", config["embedding_backend"]])
    if config.get("embedding_model"):
        cmd.extend(["--embedding-model", config["embedding_model"]])
    if config.get("similarity_metric"):
        cmd.extend(["--similarity-metric", config["similarity_metric"]])

    # VLM explanations
    if config.get("explain"):
        cmd.append("--explain")

        if config.get("backend"):
            cmd.extend(["--backend", config["backend"]])
        if config.get("base_url"):
            cmd.extend(["--base-url", config["base_url"]])
        if config.get("model"):
            cmd.extend(["--model", config["model"]])
        if config.get("prompt_profile"):
            cmd.extend(["--prompt-profile", config["prompt_profile"]])
    else:
        cmd.append("--no-explain")

    # Augmentation
    if config.get("augment_pooling"):
        cmd.append("--augment-pooling")
    if config.get("augment_grayscale"):
        cmd.append("--augment-grayscale")
    if config.get("augment_five_crop"):
        cmd.append("--augment-five-crop")
        if "augment_five_crop_ratio" in config:
            cmd.extend(
                ["--augment-five-crop-ratio", str(config["augment_five_crop_ratio"])]
            )

    # Metadata
    if config.get("write_metadata"):
        cmd.append("--write-metadata")
    else:
        cmd.append("--no-write-metadata")

    if config.get("backup_originals"):
        cmd.append("--backup-originals")
    else:
        cmd.append("--no-backup-originals")

    if config.get("overwrite_metadata"):
        cmd.append("--overwrite-metadata")

    # Performance
    if config.get("perf_metrics"):
        cmd.append("--perf-metrics")

    if config.get("refresh_library_cache"):
        cmd.append("--refresh-library-cache")

    # Output paths
    if config.get("output_jsonl"):
        cmd.extend(["--output-jsonl", str(config["output_jsonl"])])
    if config.get("summary"):
        cmd.extend(["--summary", str(config["summary"])])

    # Dry run
    if config.get("dry_run"):
        cmd.append("--dry-run")
    else:
        cmd.append("--no-dry-run")

    return cmd


def build_mono_command(config: Dict[str, Any]) -> List[str]:
    """
    Build imageworks-mono command from config dict.

    Args:
        config: Configuration dictionary

    Returns:
        Command as list of strings
    """
    cmd = ["uv", "run", "imageworks-mono", "check"]

    # Input paths (positional)
    input_paths = config.get("input", [])
    if isinstance(input_paths, str):
        input_paths = [input_paths]

    for path in input_paths:
        cmd.extend(["--input-dir", str(path)])

    # LAB color space thresholds
    if "lab_neutral_chroma" in config:
        cmd.extend(["--lab-neutral-chroma", str(config["lab_neutral_chroma"])])
    if "lab_chroma_mask" in config:
        cmd.extend(["--lab-chroma-mask", str(config["lab_chroma_mask"])])
    if "lab_toned_pass" in config:
        cmd.extend(["--lab-toned-pass", str(config["lab_toned_pass"])])
    if "lab_toned_query" in config:
        cmd.extend(["--lab-toned-query", str(config["lab_toned_query"])])
    if "lab_fail_c4_ratio" in config:
        cmd.extend(["--lab-fail-c4-ratio", str(config["lab_fail_c4_ratio"])])
    if "lab_fail_c4_cluster" in config:
        cmd.extend(["--lab-fail-c4-cluster", str(config["lab_fail_c4_cluster"])])

    # Neutral tolerance
    if "neutral_tol" in config:
        cmd.extend(["--neutral-tol", str(config["neutral_tol"])])

    # Options
    if config.get("recursive"):
        cmd.append("--recursive")

    if config.get("auto_heatmap"):
        cmd.append("--auto-heatmap")
    if config.get("write_xmp"):
        cmd.append("--write-xmp")
    if config.get("xmp_keywords_only"):
        cmd.append("--xmp-keywords-only")

    # Output paths
    if config.get("jsonl_out"):
        cmd.extend(["--jsonl-out", str(config["jsonl_out"])])
    if config.get("summary_out"):
        cmd.extend(["--summary-out", str(config["summary_out"])])

    # Dry run
    if config.get("dry_run"):
        cmd.append("--dry-run")

    return cmd


def build_tagger_command(config: Dict[str, Any]) -> List[str]:
    """
    Build imageworks-personal-tagger command from config dict.

    Args:
        config: Configuration dictionary

    Returns:
        Command as list of strings
    """
    settings = load_config()
    cmd = ["uv", "run", "imageworks-personal-tagger", "run"]

    # Input paths (positional)
    input_paths = config.get("input") or []
    if isinstance(input_paths, str):
        input_paths = [input_paths]
    for path in input_paths:
        cmd.extend(["--input-dir", str(path)])

    # Recursive toggle
    recursive = config.get("recursive")
    if recursive is True:
        cmd.append("--recursive")
    elif recursive is False:
        cmd.append("--no-recursive")

    # Preflight handling (explicit skip flag only)
    skip_preflight = config.get("skip_preflight")
    preflight_flag = config.get("preflight")
    if skip_preflight or (preflight_flag is not None and not preflight_flag):
        cmd.append("--skip-preflight")

    # Backend + connection parameters (always emit explicit values)
    backend = (config.get("backend") or settings.default_backend).strip()
    base_url = (config.get("base_url") or settings.default_base_url).strip()
    api_key = (config.get("api_key") or settings.default_api_key).strip()
    timeout = config.get("timeout")
    if timeout is None:
        timeout = settings.default_timeout
    max_new_tokens = config.get("max_new_tokens")
    if max_new_tokens is None:
        max_new_tokens = settings.default_max_new_tokens
    temperature = config.get("temperature")
    if temperature is None:
        temperature = settings.default_temperature
    top_p = config.get("top_p")
    if top_p is None:
        top_p = settings.default_top_p

    if backend:
        cmd.extend(["--backend", backend])
    if base_url:
        cmd.extend(["--base-url", base_url])
    if api_key and api_key != "EMPTY":
        cmd.extend(["--api-key", api_key])
    if timeout is not None:
        cmd.extend(["--timeout", str(timeout)])
    if max_new_tokens is not None:
        cmd.extend(["--max-new-tokens", str(max_new_tokens)])
    if temperature is not None:
        cmd.extend(["--temperature", str(temperature)])
    if top_p is not None:
        cmd.extend(["--top-p", str(top_p)])

    prompt_profile = config.get("prompt_profile") or settings.default_prompt_profile
    if prompt_profile:
        cmd.extend(["--prompt-profile", prompt_profile])

    if config.get("critique_title_template"):
        cmd.extend(
            [
                "--critique-title-template",
                str(config["critique_title_template"]),
            ]
        )
    if config.get("critique_category"):
        cmd.extend(["--critique-category", str(config["critique_category"])])
    if config.get("critique_notes"):
        cmd.extend(["--critique-notes", str(config["critique_notes"])])

    batch_size = config.get("batch_size")
    if batch_size is None:
        batch_size = settings.default_batch_size
    if batch_size is not None:
        cmd.extend(["--batch-size", str(batch_size)])

    max_workers = config.get("max_workers")
    if max_workers is None:
        max_workers = settings.default_max_workers
    if max_workers is not None:
        cmd.extend(["--max-workers", str(max_workers)])

    use_registry = config.get("use_registry")
    if use_registry is None:
        use_registry = settings.default_use_registry

    explicit_caption_model = config.get("caption_model")
    explicit_keyword_model = config.get("keyword_model")
    explicit_description_model = config.get("description_model")
    unified_model = config.get("model")

    if use_registry:
        cmd.append("--use-registry")
        if config.get("caption_role"):
            cmd.extend(["--caption-role", config["caption_role"]])
            if explicit_caption_model:
                cmd.extend(["--caption-model", explicit_caption_model])
            elif unified_model:
                cmd.extend(["--caption-model", unified_model])
        if config.get("keyword_role"):
            cmd.extend(["--keyword-role", config["keyword_role"]])
            if explicit_keyword_model:
                cmd.extend(["--keyword-model", explicit_keyword_model])
            elif unified_model:
                cmd.extend(["--keyword-model", unified_model])
        if config.get("description_role"):
            cmd.extend(["--description-role", config["description_role"]])
            if explicit_description_model:
                cmd.extend(["--description-model", explicit_description_model])
            elif unified_model:
                cmd.extend(["--description-model", unified_model])
    else:
        if explicit_caption_model:
            cmd.extend(["--caption-model", explicit_caption_model])
        elif unified_model:
            cmd.extend(["--caption-model", unified_model])

        if explicit_keyword_model:
            cmd.extend(["--keyword-model", explicit_keyword_model])
        elif unified_model:
            cmd.extend(["--keyword-model", unified_model])

        if explicit_description_model:
            cmd.extend(["--description-model", explicit_description_model])
        elif unified_model:
            cmd.extend(["--description-model", unified_model])

    # Metadata + behaviour toggles
    dry_run = config.get("dry_run")
    if dry_run is True:
        cmd.append("--dry-run")
    elif dry_run is False:
        cmd.append("--no-dry-run")

    if config.get("no_meta"):
        cmd.append("--no-meta")

    backup_originals = config.get("backup_originals")
    if backup_originals is True:
        cmd.append("--backup-originals")
    elif backup_originals is False:
        cmd.append("--no-backup-originals")

    overwrite_metadata = config.get("overwrite_metadata")
    if overwrite_metadata is True:
        cmd.append("--overwrite-metadata")
    elif overwrite_metadata is False:
        cmd.append("--no-overwrite-metadata")

    max_keywords = config.get("max_keywords")
    if max_keywords is not None:
        cmd.extend(["--max-keywords", str(max_keywords)])

    # Output paths
    if config.get("output_jsonl"):
        cmd.extend(["--output-jsonl", str(config["output_jsonl"])])
    if config.get("summary"):
        cmd.extend(["--summary", str(config["summary"])])

    return cmd


def build_narrator_command(config: Dict[str, Any]) -> List[str]:
    """
    Build imageworks-color-narrator command from config dict.

    Args:
        config: Configuration dictionary

    Returns:
        Command as list of strings
    """
    cmd = ["uv", "run", "imageworks-color-narrator"]

    # Image paths
    images = config.get("images", [])
    if isinstance(images, str):
        images = [images]

    for img_path in images:
        cmd.extend(["--images", str(img_path)])

    # Overlays directory
    if config.get("overlays"):
        cmd.extend(["--overlays", str(config["overlays"])])

    # Mono JSONL
    if config.get("mono_jsonl"):
        cmd.extend(["--mono-jsonl", str(config["mono_jsonl"])])

    # Backend configuration
    if config.get("backend"):
        cmd.extend(["--backend", config["backend"]])
    if config.get("vlm_base_url"):
        cmd.extend(["--vlm-base-url", config["vlm_base_url"]])
    if config.get("vlm_model"):
        cmd.extend(["--vlm-model", config["vlm_model"]])

    # Prompt configuration
    if config.get("prompt"):
        cmd.extend(["--prompt", str(config["prompt"])])

    if config.get("regions"):
        cmd.append("--regions")

    # Filtering
    if "min_contamination_level" in config:
        cmd.extend(
            ["--min-contamination-level", str(config["min_contamination_level"])]
        )

    if config.get("require_overlays"):
        cmd.append("--require-overlays")

    # Metadata options
    if config.get("backup_original_files"):
        cmd.append("--backup-original-files")

    if config.get("overwrite_existing_metadata"):
        cmd.append("--overwrite-existing-metadata")

    # Output paths
    if config.get("summary"):
        cmd.extend(["--summary", str(config["summary"])])

    # Dry run
    if config.get("dry_run"):
        cmd.append("--dry-run")

    return cmd


def build_downloader_command(config: Dict[str, Any]) -> List[str]:
    """
    Build imageworks-download command from config dict.

    Args:
        config: Configuration dictionary

    Returns:
        Command as list of strings
    """
    cmd = ["uv", "run", "imageworks-download"]

    # Model identifier (positional)
    model = config.get("model")
    if model:
        cmd.append(str(model))

    # Format preferences
    formats = config.get("format", [])
    if isinstance(formats, str):
        formats = [formats]

    if formats:
        cmd.extend(["--format", ",".join(formats)])

    # Location
    if config.get("location"):
        cmd.extend(["--location", config["location"]])

    # Output directory
    if config.get("output_dir"):
        cmd.extend(["--output-dir", str(config["output_dir"])])

    # Resume option
    if config.get("resume", True):
        cmd.append("--resume")
    else:
        cmd.append("--no-resume")

    # Registry update
    if config.get("update_registry", True):
        cmd.append("--update-registry")
    else:
        cmd.append("--no-update-registry")

    return cmd
