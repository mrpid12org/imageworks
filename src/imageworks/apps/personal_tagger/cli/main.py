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


@app.callback(invoke_without_command=True)
def _root_callback(  # noqa: PLR0913 - mirrors run() for root compatibility
    ctx: typer.Context,
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
        None, "--backend", help="Inference backend identifier (e.g. lmdeploy, vllm)."
    ),
    base_url: Optional[str] = typer.Option(
        None, "--base-url", help="Endpoint base URL for the chosen backend."
    ),
    model: Optional[str] = typer.Option(
        None, "--model", help="Model identifier to use for tagging prompts."
    ),
    description_model: Optional[str] = typer.Option(
        None,
        "--description-model",
        help="Override the long-form description generation model (alias for --model).",
    ),
    caption_model: Optional[str] = typer.Option(
        None, "--caption-model", help="Model identifier for image caption generation."
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
        None, "--timeout", help="Request timeout in seconds."
    ),
    max_new_tokens: Optional[int] = typer.Option(
        None, "--max-new-tokens", help="Maximum tokens to request from the backend."
    ),
    temperature: Optional[float] = typer.Option(
        None, "--temperature", help="Sampling temperature for generative outputs."
    ),
    top_p: Optional[float] = typer.Option(
        None, "--top-p", help="Top-p nucleus sampling value."
    ),
    prompt_profile: Optional[str] = typer.Option(
        None,
        "--prompt-profile",
        help="Prompt profile name to apply (defaults to configuration).",
    ),
    critique_title_template: Optional[str] = typer.Option(
        None,
        "--critique-title-template",
        help="Template for critique titles (placeholders: {stem}, {name}, {caption}, {parent}).",
    ),
    critique_category: Optional[str] = typer.Option(
        None,
        "--critique-category",
        help="Default competition category to supply to the critique prompt.",
    ),
    critique_notes: Optional[str] = typer.Option(
        None,
        "--critique-notes",
        help="Additional notes or judging brief forwarded to the critique prompt.",
    ),
    batch_size: Optional[int] = typer.Option(
        None, "--batch-size", help="Number of images per inference batch."
    ),
    max_workers: Optional[int] = typer.Option(
        None,
        "--max-workers",
        help="Maximum concurrent workers for preprocessing/metadata.",
    ),
    recursive: Optional[bool] = typer.Option(
        None,
        "--recursive/--no-recursive",
        help="Recurse into subdirectories when scanning input paths.",
    ),
    dry_run: Optional[bool] = typer.Option(
        None,
        "--dry-run/--no-dry-run",
        help="Run full inference but skip metadata writes (source files untouched).",
    ),
    no_meta: bool = typer.Option(
        False,
        "--no-meta",
        help="Run real AI inference but skip metadata writing to image files.",
    ),
    backup_originals: Optional[bool] = typer.Option(
        None,
        "--backup-originals/--no-backup-originals",
        help="Create backups before writing metadata.",
    ),
    overwrite_metadata: Optional[bool] = typer.Option(
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
        None, "--max-keywords", help="Maximum number of keywords to retain per image."
    ),
    skip_preflight: bool = typer.Option(
        False,
        "--skip-preflight",
        help="Skip initial connectivity + vision capability preflight checks.",
    ),
    use_loader: bool = typer.Option(
        False,
        "--use-loader",
        help="Use deterministic model loader for selection (enables registry model flags).",
    ),
    caption_registry_model: Optional[str] = typer.Option(
        None,
        "--caption-registry-model",
        help="Logical registry model name for caption stage (overrides --caption-model).",
    ),
    keyword_registry_model: Optional[str] = typer.Option(
        None,
        "--keyword-registry-model",
        help="Logical registry model name for keyword stage (overrides --keyword-model).",
    ),
    description_registry_model: Optional[str] = typer.Option(
        None,
        "--description-registry-model",
        help="Logical registry model name for description stage (overrides --description-model/--model).",
    ),
    use_registry: bool = typer.Option(
        False,
        "--use-registry",
        help="Resolve models by role from unified registry (ignores explicit *-model unless used as preference).",
    ),
    caption_role: Optional[str] = typer.Option(
        None,
        "--caption-role",
        help="Functional role name to resolve caption model (default: caption).",
    ),
    keyword_role: Optional[str] = typer.Option(
        None,
        "--keyword-role",
        help="Functional role name to resolve keyword model (default: keywords).",
    ),
    description_role: Optional[str] = typer.Option(
        None,
        "--description-role",
        help="Functional role name to resolve description model (default: description).",
    ),
    competition_config: Optional[Path] = typer.Option(
        None,
        "--competition-config",
        help="Path to competition registry TOML for Judge Vision flows.",
    ),
    competition: Optional[str] = typer.Option(
        None,
        "--competition",
        help="Competition identifier defined in the registry (e.g. club_open_2025).",
    ),
    pairwise_rounds: Optional[int] = typer.Option(
        None,
        "--pairwise-rounds",
        help="Override the number of pairwise tournament rounds.",
    ),
) -> None:
    """Root-level option passthrough for backward compatibility.

    Allows: imageworks-personal-tagger --input-dir ... (legacy style) OR
            imageworks-personal-tagger run --input-dir ... (explicit subcommand)
    """
    if ctx.invoked_subcommand is None:
        run(
            input_dir=input_dir,
            output_jsonl=output_jsonl,
            summary_path=summary_path,
            backend=backend,
            base_url=base_url,
            model=model,
            description_model=description_model,
            caption_model=caption_model,
            keyword_model=keyword_model,
            api_key=api_key,
            timeout=timeout,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            prompt_profile=prompt_profile,
            critique_title_template=critique_title_template,
            critique_category=critique_category,
            critique_notes=critique_notes,
            batch_size=batch_size,
            max_workers=max_workers,
            recursive=recursive,
            dry_run=dry_run,
            no_meta=no_meta,
            backup_originals=backup_originals,
            overwrite_metadata=overwrite_metadata,
            image_exts=image_exts,
            max_keywords=max_keywords,
            skip_preflight=skip_preflight,
            use_loader=use_loader,
            caption_registry_model=caption_registry_model,
            keyword_registry_model=keyword_registry_model,
            description_registry_model=description_registry_model,
            competition_config=competition_config,
            competition=competition,
            pairwise_rounds=pairwise_rounds,
            # stream parameter removed (not yet supported in build_runtime_config)
        )


@app.command("list-registry")
def list_registry() -> None:
    """List deterministic registry models with backend and served id."""
    from imageworks.model_loader import load_registry

    registry = load_registry(force=True)
    for name, entry in registry.items():
        served = entry.served_model_id or name
        aliases = (
            ", ".join(entry.model_aliases)
            if getattr(entry, "model_aliases", None)
            else "-"
        )
        typer.echo(
            f"{name} -> backend={entry.backend} served_id={served} aliases=[{aliases}]"
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
    critique_title_template: Optional[str] = typer.Option(
        None,
        "--critique-title-template",
        help="Template for critique titles (placeholders: {stem}, {name}, {caption}, {parent}).",
    ),
    critique_category: Optional[str] = typer.Option(
        None,
        "--critique-category",
        help="Default competition category to supply to the critique prompt.",
    ),
    critique_notes: Optional[str] = typer.Option(
        None,
        "--critique-notes",
        help="Additional notes or judging brief forwarded to the critique prompt.",
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
        help="Run full inference but skip metadata writes (source files untouched).",
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
    use_loader: bool = typer.Option(
        False,
        "--use-loader",
        help="Use deterministic model loader for selection (enables registry model flags).",
    ),
    caption_registry_model: Optional[str] = typer.Option(
        None,
        "--caption-registry-model",
        help="Logical registry model name for caption stage (overrides --caption-model).",
    ),
    keyword_registry_model: Optional[str] = typer.Option(
        None,
        "--keyword-registry-model",
        help="Logical registry model name for keyword stage (overrides --keyword-model).",
    ),
    description_registry_model: Optional[str] = typer.Option(
        None,
        "--description-registry-model",
        help="Logical registry model name for description stage (overrides --description-model/--model).",
    ),
    use_registry: bool = typer.Option(
        False,
        "--use-registry",
        help="Resolve models by role from unified registry (ignores explicit *-model unless used as preference).",
    ),
    caption_role: Optional[str] = typer.Option(
        None,
        "--caption-role",
        help="Functional role name to resolve caption model (default: caption).",
    ),
    keyword_role: Optional[str] = typer.Option(
        None,
        "--keyword-role",
        help="Functional role name to resolve keyword model (default: keywords).",
    ),
    description_role: Optional[str] = typer.Option(
        None,
        "--description-role",
        help="Functional role name to resolve description model (default: description).",
    ),
    competition_config: Optional[Path] = typer.Option(
        None,
        "--competition-config",
        help="Path to competition registry TOML for Judge Vision flows.",
    ),
    competition: Optional[str] = typer.Option(
        None,
        "--competition",
        help="Competition identifier defined in the registry (e.g. club_open_2025).",
    ),
    pairwise_rounds: Optional[int] = typer.Option(
        None,
        "--pairwise-rounds",
        help="Override the number of pairwise tournament rounds.",
    ),
) -> None:
    """Run the personal tagger with CLI/pyproject configuration."""

    settings = load_config(Path.cwd())

    # Resolve registry selections early to feed into overrides so runtime config reflects endpoint/model names.
    selection_overrides: dict[str, object] = {}
    if use_loader or any(
        [caption_registry_model, keyword_registry_model, description_registry_model]
    ):
        try:
            from imageworks.model_loader import select_model
        except Exception as exc:  # noqa: BLE001
            logger.error("Deterministic loader unavailable: %s", exc)
        else:
            stage_map = {
                "caption_model": caption_registry_model,
                "keyword_model": keyword_registry_model,
                "description_model": description_registry_model or model,
            }
            resolved_endpoint: str | None = None
            for cfg_key, logical in stage_map.items():
                if not logical:
                    continue
                try:
                    desc = select_model(logical, require_capabilities=["vision"])
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "Loader selection failed for %s (%s): %s", cfg_key, logical, exc
                    )
                    continue
                selection_overrides[cfg_key] = desc.internal_model_id
                resolved_endpoint = resolved_endpoint or desc.endpoint_url
                selection_overrides["backend"] = desc.backend
                logger.info(
                    "loader_selection",
                    extra={
                        "event_type": "select",
                        "stage": cfg_key,
                        "logical_model": logical,
                        "endpoint": desc.endpoint_url,
                    },
                )
            if resolved_endpoint:
                selection_overrides["base_url"] = resolved_endpoint

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
        "critique_title_template": critique_title_template,
        "critique_category": critique_category,
        "critique_notes": critique_notes,
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
        "use_registry": use_registry,
        "caption_role": caption_role,
        "keyword_role": keyword_role,
        "description_role": description_role,
        "competition_config": competition_config,
        "competition": competition,
        "pairwise_rounds": pairwise_rounds,
    }

    # Merge selection overrides last so they win
    overrides.update(selection_overrides)

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
