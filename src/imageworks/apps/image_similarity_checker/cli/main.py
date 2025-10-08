"""Command line interface for the image similarity checker module."""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import List, Optional

import typer

from imageworks.logging_utils import configure_logging

from ..core.config import load_config
from ..core.engine import SimilarityEngine
from ..core.models import SimilarityVerdict
from ..core.reporting import write_jsonl, write_markdown

LOG_PATH = configure_logging("image_similarity_checker")
logger = logging.getLogger(__name__)
logger.info("Image similarity checker logging initialised → %s", LOG_PATH)

app = typer.Typer(help="Identify duplicate and near-duplicate competition images.")


@app.command()
def check(
    candidates: List[Path] = typer.Argument(
        ..., metavar="CANDIDATE...", help="Candidate image files or directories."
    ),
    library_root: Optional[Path] = typer.Option(
        None, "--library-root", "-L", help="Root directory containing historical submissions."
    ),
    output_jsonl: Optional[Path] = typer.Option(
        None, "--output-jsonl", help="Machine-readable JSONL output path."
    ),
    summary_path: Optional[Path] = typer.Option(
        None, "--summary", help="Human-readable Markdown summary path."
    ),
    fail_threshold: Optional[float] = typer.Option(
        None,
        "--fail-threshold",
        help="Similarity score ≥ this value is marked as FAIL.",
    ),
    query_threshold: Optional[float] = typer.Option(
        None,
        "--query-threshold",
        help="Similarity score ≥ this value triggers a QUERY.",
    ),
    top_matches: Optional[int] = typer.Option(
        None, "--top-matches", help="Number of matches to retain per candidate."
    ),
    similarity_metric: Optional[str] = typer.Option(
        None, "--metric", help="Similarity metric to use (cosine, euclidean, manhattan)."
    ),
    strategy: List[str] = typer.Option(
        None,
        "--strategy",
        "-s",
        help="Similarity strategy to enable (repeatable; e.g. embedding, perceptual_hash).",
    ),
    embedding_backend: Optional[str] = typer.Option(
        None, "--embedding-backend", help="Embedding backend identifier (simple, open_clip, remote)."
    ),
    backend: Optional[str] = typer.Option(
        None, "--backend", help="Inference backend name (lmdeploy, vllm, ollama, ...)."
    ),
    base_url: Optional[str] = typer.Option(
        None, "--base-url", help="Base URL for OpenAI-compatible explanation/embedding endpoints."
    ),
    model: Optional[str] = typer.Option(
        None, "--model", help="Model identifier for embeddings/explanations."
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", help="API key for remote backends."
    ),
    timeout: Optional[int] = typer.Option(
        None, "--timeout", help="Request timeout for remote backends (seconds)."
    ),
    prompt_profile: Optional[str] = typer.Option(
        None, "--prompt-profile", help="Prompt profile for explanation generation."
    ),
    write_metadata: Optional[bool] = typer.Option(
        None, "--write-metadata/--no-write-metadata", help="Write similarity verdicts to image metadata."
    ),
    backup_originals: Optional[bool] = typer.Option(
        None, "--backup-originals/--no-backup-originals", help="Create backups before metadata writes."
    ),
    overwrite_metadata: Optional[bool] = typer.Option(
        None, "--overwrite-metadata/--no-overwrite-metadata", help="Overwrite existing metadata keywords."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run/--no-dry-run", help="Skip similarity computation and emit placeholder results."
    ),
    use_loader: Optional[bool] = typer.Option(
        None, "--use-loader/--no-use-loader", help="Resolve models via the deterministic model loader."
    ),
    registry_model: Optional[str] = typer.Option(
        None,
        "--registry-model",
        help="Logical model name from the model registry (requires --use-loader).",
    ),
    registry_capability: List[str] = typer.Option(
        None,
        "--registry-capability",
        help="Additional capability requirement when resolving models via loader.",
    ),
    explain: Optional[bool] = typer.Option(
        None,
        "--explain/--no-explain",
        help="Generate natural-language rationales using the configured backend.",
    ),
) -> None:
    overrides: dict[str, object] = {
        "library_root": library_root,
        "output_jsonl": output_jsonl,
        "summary_path": summary_path,
        "fail_threshold": fail_threshold,
        "query_threshold": query_threshold,
        "top_matches": top_matches,
        "similarity_metric": similarity_metric,
        "embedding_backend": embedding_backend,
        "backend": backend,
        "base_url": base_url,
        "model": model,
        "api_key": api_key,
        "timeout": timeout,
        "prompt_profile": prompt_profile,
        "write_metadata": write_metadata,
        "backup_originals": backup_originals,
        "overwrite_metadata": overwrite_metadata,
        "dry_run": dry_run,
        "use_loader": use_loader,
        "registry_model": registry_model,
        "registry_capabilities": registry_capability,
        "generate_explanations": explain,
    }
    if strategy:
        overrides["strategies"] = strategy

    config = load_config(candidates=candidates, **{k: v for k, v in overrides.items() if v is not None})

    logger.info(
        "similarity_run_config",
        extra={
            "event_type": "config",
            "candidates": [str(path) for path in config.candidates],
            "library_root": str(config.library_root),
            "strategies": list(config.strategies),
            "metric": config.similarity_metric,
            "fail_threshold": config.fail_threshold,
            "query_threshold": config.query_threshold,
            "embedding_backend": config.embedding_backend,
            "generate_explanations": config.generate_explanations,
        },
    )

    engine = SimilarityEngine(config)
    try:
        results = engine.run()
    finally:
        engine.close()

    write_jsonl(results, config.output_jsonl)
    write_markdown(results, config.summary_path)

    _print_terminal_summary(results, config)


def _print_terminal_summary(results, config) -> None:
    verdict_counts = Counter(result.verdict for result in results)
    typer.echo("\nSummary:")
    for verdict in SimilarityVerdict:
        typer.echo(
            f"  {verdict.value.upper():>5}: {verdict_counts.get(verdict, 0)}"
        )

    typer.echo("\nTop matches:")
    for result in results:
        best = result.best_match()
        best_name = best.reference.name if best else "—"
        typer.echo(
            f"  {result.candidate.name}: {result.verdict.value.upper()} (score={result.top_score:.3f}, match={best_name})"
        )
        if result.notes:
            for note in result.notes:
                typer.echo(f"    • {note}")

    typer.echo(
        f"\nDetailed JSONL → {config.output_jsonl}\nMarkdown summary → {config.summary_path}"
    )
