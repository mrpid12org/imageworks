"""Command line interface for the Personal Tagger module."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import typer

from imageworks.logging_utils import configure_logging

from ..core.config import PersonalTaggerConfig, build_runtime_config, load_config
from ..core.runner import PersonalTaggerRunner

LOG_PATH = configure_logging("personal_tagger")
logger = logging.getLogger(__name__)
logger.info("Personal tagger logging initialised → %s", LOG_PATH)

app = typer.Typer(
    help="Generate personal-library keywords, captions, and descriptions."
)


@app.command("run")
def run(  # noqa: PLR0913 - CLI surface area is intentional
    input_dir: List[Path] = typer.Option(
        [],
        "--input-dir",
        "-i",
        help="Directory containing images to process (repeatable).",
    ),
    output_jsonl: Optional[Path] = typer.Option(
        None,
        "--output-jsonl",
        help="Path to JSONL audit log (defaults to pyproject configuration).",
    ),
    summary_path: Optional[Path] = typer.Option(
        None,
        "--summary",
        help="Markdown summary output path (defaults to pyproject configuration).",
    ),
    backend: Optional[str] = typer.Option(
        None,
        "--backend",
        help="Inference backend identifier (e.g. lmdeploy, vllm).",
    ),
    base_url: Optional[str] = typer.Option(
        None,
        "--base-url",
        help="Endpoint base URL for the chosen backend.",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        help="Model identifier to use for tagging prompts.",
    ),
    description_model: Optional[str] = typer.Option(
        None,
        "--description-model",
        help="Override the long-form description generation model (alias for --model).",
    ),
    caption_model: Optional[str] = typer.Option(
        None,
        "--caption-model",
        help="Model identifier for image caption generation.",
    ),
    keyword_model: Optional[str] = typer.Option(
        None,
        "--keyword-model",
        help="Model identifier for keyword extraction/classification.",
    ),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        help="API key for the OpenAI-compatible backend (if required).",
    ),
    timeout: Optional[int] = typer.Option(
        None,
        "--timeout",
        help="Request timeout in seconds.",
    ),
    max_new_tokens: Optional[int] = typer.Option(
        None,
        "--max-new-tokens",
        help="Maximum tokens to request from the backend.",
    ),
    temperature: Optional[float] = typer.Option(
        None,
        "--temperature",
        help="Sampling temperature for generative outputs.",
    ),
    top_p: Optional[float] = typer.Option(
        None,
        "--top-p",
        help="Top-p nucleus sampling value.",
    ),
    prompt_profile: Optional[str] = typer.Option(
        None,
        "--prompt-profile",
        help="Prompt profile name to apply (defaults to configuration).",
    ),
    batch_size: Optional[int] = typer.Option(
        None,
        "--batch-size",
        help="Number of images per inference batch.",
    ),
    max_workers: Optional[int] = typer.Option(
        None,
        "--max-workers",
        help="Maximum concurrent workers for preprocessing/metadata.",
    ),
    recursive: Optional[bool] = typer.Option(  # noqa: FBT001 - clarity over style
        None,
        "--recursive/--no-recursive",
        help="Recurse into subdirectories when scanning input paths.",
    ),
    dry_run: Optional[bool] = typer.Option(  # noqa: FBT001 - clarity over style
        None,
        "--dry-run/--no-dry-run",
        help="Skip AI inference and metadata writes, use fake test data.",
    ),
    no_meta: bool = typer.Option(
        False,
        "--no-meta",
        help="Run real AI inference but skip metadata writing to image files.",
    ),
    backup_originals: Optional[
        bool
    ] = typer.Option(  # noqa: FBT001 - clarity over style
        None,
        "--backup-originals/--no-backup-originals",
        help="Create backups before writing metadata.",
    ),
    overwrite_metadata: Optional[
        bool
    ] = typer.Option(  # noqa: FBT001 - clarity over style
        None,
        "--overwrite-metadata/--no-overwrite-metadata",
        help="Allow overwriting existing keywords/captions.",
    ),
    image_exts: Optional[str] = typer.Option(
        None,
        "--image-exts",
        help="Comma-separated list of additional image extensions to consider.",
    ),
    max_keywords: Optional[int] = typer.Option(
        None,
        "--max-keywords",
        help="Maximum number of keywords to retain per image.",
    ),
    skip_preflight: bool = typer.Option(
        False,
        "--skip-preflight",
        help="Skip initial connectivity + vision capability preflight checks.",
    ),
) -> None:
    """Run the personal tagger with CLI/pyproject configuration."""

    settings = load_config(Path.cwd())
    overrides = {
        "output_jsonl": output_jsonl,
        "summary_path": summary_path,
        "backend": backend,
        "base_url": base_url,
        "model": model,
        "description_model": description_model or model,
        "caption_model": caption_model,
        "keyword_model": keyword_model,
        "api_key": api_key,
        "timeout": timeout,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "prompt_profile": prompt_profile,
        "batch_size": batch_size,
        "max_workers": max_workers,
        "recursive": recursive,
        "dry_run": dry_run,
        "no_meta": no_meta,
        "backup_originals": backup_originals,
        "overwrite_metadata": overwrite_metadata,
        "image_extensions": image_exts.split(",") if image_exts else None,
        "max_keywords": max_keywords,
        "preflight": (
            False if skip_preflight else None
        ),  # explicit override only when skipping
    }

    try:
        runtime_config: PersonalTaggerConfig = build_runtime_config(
            settings=settings,
            input_dirs=input_dir,
            **overrides,
        )
    except ValueError as exc:  # missing input directory etc.
        raise typer.BadParameter(str(exc)) from exc

    runner = PersonalTaggerRunner(runtime_config)
    runner.run()


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    app()
