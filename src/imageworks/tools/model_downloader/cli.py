"""
Command-line interface for the model downloader.

Provides a user-friendly CLI with typer for downloading and managing models
following imageworks conventions.
"""

from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

from .downloader import ModelDownloader
from .registry import get_registry
from .config import get_config
from .url_analyzer import URLAnalyzer


app = typer.Typer(
    name="imageworks-download",
    help="ImageWorks Model Downloader - Download and manage AI models",
    no_args_is_help=True,
)
console = Console()


@app.command("download")
def download_model(
    model: str = typer.Argument(
        ..., help="Model name (owner/repo) or HuggingFace URL to download"
    ),
    format_preference: Optional[str] = typer.Option(
        None,
        "--format",
        "-f",
        help="Preferred format(s) (comma separated: gguf, awq, gptq, safetensors, etc.)",
    ),
    location: Optional[str] = typer.Option(
        None,
        "--location",
        "-l",
        help="Target location (linux_wsl, windows_lmstudio, or custom path)",
    ),
    include_optional: bool = typer.Option(
        False,
        "--include-optional",
        "-o",
        help="Include optional files (documentation, examples, etc.)",
    ),
    force: bool = typer.Option(
        False, "--force", help="Force re-download even if model exists"
    ),
    non_interactive: bool = typer.Option(
        False, "--non-interactive", "-y", help="Non-interactive mode (use defaults)"
    ),
):
    """Download a model from HuggingFace or URL."""

    try:
        downloader = ModelDownloader()

        preferred_formats = None
        if format_preference:
            preferred_formats = [fmt.strip() for fmt in format_preference.split(",") if fmt.strip()]

        # Run download directly, let aria2c show its native progress
        model_entry = downloader.download(
            model_identifier=model,
            format_preference=preferred_formats,
            location_override=location,
            include_optional=include_optional,
            force_redownload=force,
            interactive=not non_interactive,
        )

        # Success message
        rprint(f"✅ [green]Successfully downloaded:[/green] {model_entry.model_name}")
        rprint(f"   📁 Location: {model_entry.path}")
        rprint(f"   🔧 Format: {model_entry.format_type}")
        rprint(f"   💾 Size: {_format_size(model_entry.size_bytes)}")

        # Show compatible backends
        config = get_config()
        backends = config.get_compatible_backends(model_entry.format_type)
        rprint(f"   ⚡ Compatible with: {', '.join(backends)}")

    except Exception as e:
        rprint(f"❌ [red]Download failed:[/red] {str(e)}")
        raise typer.Exit(code=1)


@app.command("list")
def list_models(
    format_filter: Optional[str] = typer.Option(
        None,
        "--format",
        "-f",
        help="Filter by format (gguf, awq, gptq, safetensors, etc.)",
    ),
    location_filter: Optional[str] = typer.Option(
        None,
        "--location",
        "-l",
        help="Filter by location (linux_wsl, windows_lmstudio)",
    ),
    show_details: bool = typer.Option(
        False, "--details", "-d", help="Show detailed information"
    ),
    json_output: bool = typer.Option(False, "--json", help="Output in JSON format"),
):
    """List downloaded models."""

    try:
        downloader = ModelDownloader()
        models = downloader.list_models(
            format_filter=format_filter, location_filter=location_filter
        )

        if json_output:
            import json

            model_data = [model.to_dict() for model in models]
            print(json.dumps(model_data, indent=2))
            return

        if not models:
            rprint("📭 [yellow]No models found matching criteria[/yellow]")
            return

        # Create table
        table = Table(title="Downloaded Models")
        table.add_column("Model", style="cyan", no_wrap=True)
        table.add_column("Format", style="magenta")
        table.add_column("Location", style="green")
        table.add_column("Size", justify="right", style="blue")

        if show_details:
            table.add_column("Downloaded", style="dim")
            table.add_column("Files", justify="right", style="dim")

        for model in models:
            # Check if directory is missing (defensive check result)
            is_missing = model.metadata and model.metadata.get(
                "directory_missing", False
            )

            model_name = model.model_name
            if is_missing:
                model_name = f"[red]⚠ {model.model_name} (directory missing)[/red]"

            row = [
                model_name,
                model.format_type,
                model.location,
                _format_size(model.size_bytes),
            ]

            if show_details:
                download_date = model.downloaded_at.split("T")[0]  # Just date part
                row.extend([download_date, str(len(model.files))])

            table.add_row(*row)

        console.print(table)

        # Summary
        total_size = sum(model.size_bytes for model in models)
        rprint(f"\n📊 Total: {len(models)} models, {_format_size(total_size)}")

    except Exception as e:
        rprint(f"❌ [red]Failed to list models:[/red] {str(e)}")
        raise typer.Exit(code=1)


@app.command("analyze")
def analyze_url(
    url: str = typer.Argument(..., help="URL to analyze"),
    show_files: bool = typer.Option(
        False, "--files", help="Show detailed file information"
    ),
):
    """Analyze a HuggingFace URL without downloading."""

    try:
        analyzer = URLAnalyzer()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Analyzing URL...", total=None)
            analysis = analyzer.analyze_url(url)

        repo = analysis.repository

        # Repository info
        rprint(f"📍 [bold]Repository:[/bold] {repo.owner}/{repo.repo}")
        if repo.model_type:
            rprint(f"🏷️  Model Type: {repo.model_type}")
        if repo.library_name:
            rprint(f"📚 Library: {repo.library_name}")

        # Detected formats
        rprint("\n🔧 [bold]Detected Formats:[/bold]")
        for format_info in analysis.formats:
            confidence = f"{format_info.confidence:.0%}"
            rprint(f"   • {format_info.format_type} ({confidence} confidence)")
            for evidence in format_info.evidence[:2]:  # Show first 2 pieces of evidence
                rprint(f"     - {evidence}")

        # File summary
        rprint("\n📁 [bold]Files Summary:[/bold]")
        for category, files in analysis.files.items():
            if files:
                total_size = sum(f.size for f in files)
                rprint(
                    f"   • {category}: {len(files)} files ({_format_size(total_size)})"
                )

        rprint(f"\n💾 [bold]Total Size:[/bold] {_format_size(analysis.total_size)}")

        # Show detailed files if requested
        if show_files:
            for category, files in analysis.files.items():
                if files and category in ["model_weights", "config", "tokenizer"]:
                    rprint(f"\n📄 [bold]{category.title()}:[/bold]")
                    for file in files[:5]:  # Show first 5 files
                        rprint(f"   • {file.path} ({_format_size(file.size)})")
                    if len(files) > 5:
                        rprint(f"   ... and {len(files) - 5} more files")

    except Exception as e:
        rprint(f"❌ [red]Analysis failed:[/red] {str(e)}")
        raise typer.Exit(code=1)


@app.command("remove")
def remove_model(
    model_name: str = typer.Argument(..., help="Model name to remove"),
    format_type: Optional[str] = typer.Option(
        None, "--format", "-f", help="Specific format to remove"
    ),
    location: Optional[str] = typer.Option(
        None, "--location", "-l", help="Specific location to remove from"
    ),
    delete_files: bool = typer.Option(
        False, "--delete-files", help="Also delete the model files from disk"
    ),
    force: bool = typer.Option(False, "--force", help="Don't ask for confirmation"),
):
    """Remove a model from registry and optionally delete files."""

    try:
        registry = get_registry()
        models = registry.find_model(model_name, format_type, location)

        if not models:
            rprint(f"❌ [red]Model not found:[/red] {model_name}")
            raise typer.Exit(code=1)

        # Show what will be removed
        rprint("🗑️  [yellow]Will remove:[/yellow]")
        for model in models:
            rprint(f"   • {model.model_name} ({model.format_type}, {model.location})")
            if delete_files:
                rprint(f"     📁 Files: {model.path}")

        # Confirm unless forced
        if not force:
            confirm = typer.confirm("Are you sure?")
            if not confirm:
                rprint("❌ [yellow]Cancelled[/yellow]")
                return

        # Remove models
        downloader = ModelDownloader()
        success = downloader.remove_model(
            model_name,
            format_type=format_type,
            location=location,
            delete_files=delete_files,
        )

        if success:
            rprint("✅ [green]Successfully removed[/green]")
        else:
            rprint("❌ [red]Removal failed[/red]")
            raise typer.Exit(code=1)

    except Exception as e:
        rprint(f"❌ [red]Removal failed:[/red] {str(e)}")
        raise typer.Exit(code=1)


@app.command("stats")
def show_stats():
    """Show download statistics."""

    try:
        downloader = ModelDownloader()
        stats = downloader.get_stats()

        # Create stats table
        table = Table(title="Model Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Models", str(stats["total_models"]))
        table.add_row("Total Size", _format_size(stats["total_size_bytes"]))

        console.print(table)

        # Format breakdown
        if stats["by_format"]:
            format_table = Table(title="By Format")
            format_table.add_column("Format", style="magenta")
            format_table.add_column("Count", justify="right", style="blue")

            for format_type, count in sorted(stats["by_format"].items()):
                format_table.add_row(format_type, str(count))

            console.print(format_table)

        # Location breakdown
        if stats["by_location"]:
            location_table = Table(title="By Location")
            location_table.add_column("Location", style="yellow")
            location_table.add_column("Count", justify="right", style="blue")

            for location, count in sorted(stats["by_location"].items()):
                location_table.add_row(location, str(count))

            console.print(location_table)

    except Exception as e:
        rprint(f"❌ [red]Failed to get stats:[/red] {str(e)}")
        raise typer.Exit(code=1)


@app.command("migrate")
def migrate_registry(
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be done without making changes"
    ),
    analyze_only: bool = typer.Option(
        False, "--analyze-only", help="Only analyze and report issues"
    ),
    clean_empty: bool = typer.Option(
        False, "--clean-empty", help="Also remove empty directories after migration"
    ),
):
    """Fix registry inconsistencies from legacy directory structures."""

    try:
        from .migrate import (
            analyze_registry_inconsistencies,
            fix_registry_inconsistencies,
            clean_empty_directories,
        )
        from .config import get_config

        if analyze_only:
            rprint("� [cyan]Analyzing registry inconsistencies...[/cyan]")
            inconsistencies = analyze_registry_inconsistencies()

            if not inconsistencies:
                rprint("✅ [green]No registry inconsistencies found![/green]")
                return

            # Create analysis table
            table = Table(
                title=f"Registry Inconsistencies ({len(inconsistencies)} found)"
            )
            table.add_column("Model", style="cyan", no_wrap=True)
            table.add_column("Registry Path", style="red", no_wrap=True)
            table.add_column("Expected Path", style="green", no_wrap=True)
            table.add_column("Can Migrate", style="yellow")

            for issue in inconsistencies:
                can_migrate = "✅ Yes" if issue["can_migrate"] else "⚠️  Needs attention"
                table.add_row(
                    issue["model_name"],
                    str(Path(issue["registry_path"]).relative_to(Path.home())),
                    str(Path(issue["expected_path"]).relative_to(Path.home())),
                    can_migrate,
                )

            console.print(table)
            return

        mode_text = "[yellow][DRY RUN][/yellow] " if dry_run else ""
        rprint(f"🔧 {mode_text}[cyan]Fixing registry inconsistencies...[/cyan]")

        results = fix_registry_inconsistencies(dry_run=dry_run)

        # Create results table
        results_table = Table(title="Migration Results")
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Count", justify="right", style="green")

        results_table.add_row("Total Issues", str(results["total"]))
        results_table.add_row("Migrated", str(results["migrated"]))
        results_table.add_row("Registry Updates", str(results["updated_registry"]))
        results_table.add_row("Skipped", str(results["skipped"]))
        results_table.add_row("Errors", str(results["errors"]))

        if clean_empty:
            config = get_config()
            weights_dir = config.linux_wsl.root / "weights"
            removed = clean_empty_directories(weights_dir, dry_run=dry_run)
            results_table.add_row("Empty Directories Removed", str(removed))

        console.print(results_table)

        if results["errors"] == 0:
            rprint("✅ [green]Migration completed successfully![/green]")
        else:
            rprint("⚠️  [yellow]Migration completed with errors.[/yellow]")

    except Exception as e:
        rprint(f"❌ [red]Migration failed:[/red] {str(e)}")
        raise typer.Exit(code=1)


@app.command("verify")
def verify_models(
    model_name: Optional[str] = typer.Argument(
        None, help="Specific model to verify (all models if not specified)"
    ),
    fix_missing: bool = typer.Option(
        False, "--fix-missing", help="Remove registry entries for missing models"
    ),
):
    """Verify model integrity and registry consistency."""

    try:
        registry = get_registry()

        if model_name:
            models = registry.find_model(model_name)
            if not models:
                rprint(f"❌ [red]Model not found:[/red] {model_name}")
                raise typer.Exit(code=1)
        else:
            models = registry.get_all_models()

        rprint(f"🔍 [bold]Verifying {len(models)} models...[/bold]\n")

        valid_models = []
        invalid_models = []

        for model in models:
            path_exists = Path(model.path).exists()

            if path_exists:
                integrity_ok = registry.verify_model_integrity(
                    model.model_name, model.format_type, model.location
                )
                if integrity_ok:
                    valid_models.append(model)
                    rprint(f"✅ {model.model_name} ({model.format_type})")
                else:
                    invalid_models.append(model)
                    rprint(
                        f"⚠️  {model.model_name} ({model.format_type}) - integrity check failed"
                    )
            else:
                invalid_models.append(model)
                rprint(
                    f"❌ {model.model_name} ({model.format_type}) - path not found: {model.path}"
                )

        rprint("\n📊 [bold]Summary:[/bold]")
        rprint(f"   ✅ Valid: {len(valid_models)}")
        rprint(f"   ❌ Invalid: {len(invalid_models)}")

        if invalid_models and fix_missing:
            rprint(
                f"\n🔧 [yellow]Cleaning up {len(invalid_models)} invalid entries...[/yellow]"
            )
            removed_keys = registry.cleanup_missing_models()
            rprint(f"✅ Removed {len(removed_keys)} entries from registry")

    except Exception as e:
        rprint(f"❌ [red]Verification failed:[/red] {str(e)}")
        raise typer.Exit(code=1)


@app.command("config")
def show_config():
    """Show current configuration."""

    try:
        config = get_config()

        rprint("⚙️ [bold]ImageWorks Model Downloader Configuration[/bold]\n")

        # Directories
        rprint("📁 [bold]Directories:[/bold]")
        rprint(f"   Linux WSL: {config.linux_wsl.root}")
        rprint(f"   Windows LM Studio: {config.windows_lmstudio.root}")
        rprint(f"   Registry: {config.registry_path}")
        rprint(f"   Cache: {config.cache_path}\n")

        # Download settings
        rprint("⚡ [bold]Download Settings:[/bold]")
        rprint(f"   Max connections per server: {config.max_connections_per_server}")
        rprint(f"   Max concurrent downloads: {config.max_concurrent_downloads}")
        rprint(f"   Resume enabled: {config.enable_resume}")
        rprint(f"   Include optional files: {config.include_optional_files}\n")

        # Format preferences
        rprint("🔧 [bold]Format Preferences:[/bold]")
        for i, fmt in enumerate(config.preferred_formats, 1):
            rprint(f"   {i}. {fmt}")

    except Exception as e:
        rprint(f"❌ [red]Failed to show config:[/red] {str(e)}")
        raise typer.Exit(code=1)


def _format_size(size_bytes: int) -> str:
    """Format size in human-readable format."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
