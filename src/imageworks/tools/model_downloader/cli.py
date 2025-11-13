"""
Command-line interface for the model downloader.

Provides a user-friendly CLI with typer for downloading and managing models
following imageworks conventions.
"""

import os
from pathlib import Path
from typing import Optional, Any, List, Dict, Set
import typer
from rich.console import Console
from rich.table import Table
from rich import print as rprint
from rich.progress import Progress, SpinnerColumn, TextColumn
import json
import subprocess

from imageworks.logging_utils import configure_logging
from types import SimpleNamespace
from imageworks.model_loader.registry import load_registry as _load_unified_registry

from .format_utils import detect_format_and_quant
from .downloader import ModelDownloader, _derive_producer
from .url_analyzer import URLAnalyzer
from imageworks.model_loader import registry as unified_registry
from imageworks.model_loader.download_adapter import (
    record_download,
    ImportSkipped,
    list_downloads as unified_list_downloads,
    remove_download as unified_remove_download,
    compute_directory_checksum,
)
from imageworks.tools.model_downloader.architecture import (
    collect_architecture_metadata,
    merge_architecture_metadata,
)
from imageworks.model_loader.runtime_metadata import (
    load_runtime_events,
    merge_runtime_payload,
)
from imageworks.model_loader.models import RegistryEntry
from .config import get_config


app = typer.Typer(
    name="imageworks-download",
    help="ImageWorks Model Downloader - Download and manage AI models",
    no_args_is_help=True,
)
console = Console()
LOG_PATH = configure_logging("model_downloader")


def _detect_ollama_store_root() -> Optional[Path]:
    """Best-effort detection of the local Ollama store directory."""

    candidates: List[Path] = []

    env_override = os.environ.get("OLLAMA_MODELS")
    if env_override:
        candidates.append(Path(env_override).expanduser())

    candidates.append(Path.home() / ".ollama" / "models")

    try:
        linux_root = get_config().linux_wsl.root
        candidates.append(linux_root / "ollama-data" / "models")
    except Exception:
        pass

    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    return None


def _prune_empty_repo_and_owner_dirs(repo_dir: Path) -> None:
    """Remove empty repo directory and its immediate parent (owner) if empty.

    This only attempts up to two levels: the repo folder itself and its parent (owner).
    It will not recurse further up the tree. Safe if directories are non-empty or missing.
    """
    try:
        # Remove repo dir if it exists and is empty (in case files were deleted individually)
        if repo_dir.exists() and repo_dir.is_dir():
            try:
                # only remove if empty
                next(repo_dir.iterdir())
            except StopIteration:
                try:
                    repo_dir.rmdir()
                except Exception:
                    pass
            except Exception:
                pass
        # Remove owner dir if empty
        owner = repo_dir.parent
        if owner.exists() and owner.is_dir():
            try:
                next(owner.iterdir())
            except StopIteration:
                try:
                    owner.rmdir()
                except Exception:
                    pass
            except Exception:
                pass
    except Exception:
        # Never raise from cleanup
        pass


def _enable_dupe_tolerance():
    """Legacy no-op retained for backward compatibility."""
    return None


def _extract_hf_id(entry: Any) -> Optional[str]:
    if isinstance(entry, dict):
        source = entry.get("source")
    else:
        source = getattr(entry, "source", None)
    if isinstance(source, dict):
        hf_id = source.get("huggingface_id")
        if isinstance(hf_id, str) and hf_id:
            return hf_id
    aliases = []
    if isinstance(entry, dict):
        aliases = entry.get("model_aliases") or []
    else:
        aliases = getattr(entry, "model_aliases", None) or []
    for alias in aliases:
        if isinstance(alias, str) and "/" in alias:
            return alias
    return None


def _resolve_producer(entry: Any) -> Optional[str]:
    if isinstance(entry, dict):
        metadata = entry.get("metadata") or {}
    else:
        metadata = getattr(entry, "metadata", None) or {}
    existing = metadata.get("producer") if isinstance(metadata, dict) else None
    existing_str = str(existing) if existing else None
    hf_id = _extract_hf_id(entry)
    return _derive_producer(
        existing=existing_str,
        hf_id=hf_id,
        source_provider=(
            entry.get("source_provider")
            if isinstance(entry, dict)
            else getattr(entry, "source_provider", None)
        ),
        served_model_id=(
            entry.get("served_model_id")
            if isinstance(entry, dict)
            else getattr(entry, "served_model_id", None)
        ),
        download_path=(
            entry.get("download_path")
            if isinstance(entry, dict)
            else getattr(entry, "download_path", None)
        ),
    )


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
        True,
        "--include-optional/--no-include-optional",
        "-o/-O",
        help="Include optional small files (docs, small JSONs). Chat templates are always included.",
    ),
    include_large_optional: bool = typer.Option(
        False,
        "--include-large-optional/--no-include-large-optional",
        help="Include large optional files (>100MB). Off by default.",
    ),
    force: bool = typer.Option(
        False, "--force", help="Force re-download even if model exists"
    ),
    non_interactive: bool = typer.Option(
        False, "--non-interactive", "-y", help="Non-interactive mode (use defaults)"
    ),
    weight_variants: Optional[str] = typer.Option(
        None,
        "--weights",
        "-w",
        help="Comma-separated list of weight filenames to download (defaults to all variants).",
    ),
    support_repo: Optional[str] = typer.Option(
        None,
        "--support-repo",
        help="Optional repository to pull config/tokenizer assets from (e.g. original safetensors repo).",
    ),
):
    """Download a model from HuggingFace or direct repository URL."""

    try:
        _enable_dupe_tolerance()
        downloader = ModelDownloader()

        preferred_formats = None
        if format_preference:
            preferred_formats = [
                fmt.strip() for fmt in format_preference.split(",") if fmt.strip()
            ]

        selected_weights = None
        if weight_variants:
            selected_weights = [
                variant.strip()
                for variant in weight_variants.split(",")
                if variant.strip()
            ] or None

        entry = downloader.download(
            model_identifier=model,
            format_preference=preferred_formats,
            location_override=location,
            include_optional=include_optional,
            include_large_optional=include_large_optional,
            force_redownload=force,
            interactive=not non_interactive,
            weight_variants=selected_weights,
            support_repo=support_repo,
        )
        # Use registry display_name for user-facing confirmation
        display_name = entry.display_name or entry.name
        rprint(f"✅ [green]Successfully downloaded:[/green] {display_name}")
        if entry.download_path:
            rprint(f"   📁 Files stored at: {entry.download_path}")
        if entry.download_location:
            rprint(f"   🗂️  Location label: {entry.download_location}")

        fmt = entry.download_format or "unknown"
        quant = entry.quantization or "-"
        if quant and quant != "-":
            fmt_summary = f"{fmt} ({quant})" if fmt and fmt != "unknown" else quant
        else:
            fmt_summary = fmt
        rprint(f"   🔧 Format: {fmt_summary}")

        size_display = _format_size(entry.download_size_bytes or 0)
        rprint(f"   💾 Size: {size_display}")

        metadata: Dict[str, Any] = entry.metadata or {}
        files_downloaded = metadata.get("files_downloaded")
        if files_downloaded:
            rprint(f"   📄 Files downloaded: {files_downloaded}")

        info_bits: List[str] = []
        if metadata.get("model_type"):
            info_bits.append(str(metadata["model_type"]))
        if metadata.get("library"):
            info_bits.append(str(metadata["library"]))
        if info_bits:
            rprint(f"   🧩 Model info: {', '.join(info_bits)}")

        if metadata.get("has_chat_template"):
            detail = (
                "embedded"
                if metadata.get("has_embedded_chat_template")
                else "external file"
            )
            template_files = metadata.get("external_chat_template_files") or []
            file_note = ""
            if template_files:
                preview = ", ".join(template_files[:3])
                if len(template_files) > 3:
                    preview += ", …"
                file_note = f" ({preview})"
            rprint(f"   💬 Chat template detected: {detail}{file_note}")

    except Exception as exc:  # noqa: BLE001
        rprint(f"❌ [red]Download failed:[/red] {exc}")
        raise typer.Exit(code=1)


## undeprecate-ollama command removed in layered registry design.


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
    backend_filter: Optional[str] = typer.Option(
        None,
        "--backend",
        "-b",
        help="Filter by backend (vllm, ollama, lmdeploy, unassigned, etc.)",
    ),
    show_deprecated: bool = typer.Option(
        False,
        "--show-deprecated",
        help="Include deprecated entries in the listing",
    ),
    show_details: bool = typer.Option(
        False, "--details", "-d", help="Show detailed information"
    ),
    json_output: bool = typer.Option(False, "--json", help="Output in JSON format"),
    include_logical: bool = typer.Option(
        False,
        "--include-logical",
        help="Also include logical entries without download metadata",
    ),
    dedupe: bool = typer.Option(
        False,
        "--dedupe",
        help="Condense rows by family/backend/format/quant (optional)",
    ),
    show_internal_names: bool = typer.Option(
        False,
        "--show-internal-names",
        help="Include internal variant name column for debugging",
    ),
    show_backend: bool = typer.Option(
        False,
        "--show-backend",
        help="Include backend column (hidden by default)",
    ),
    show_installed: bool = typer.Option(
        False,
        "--show-installed",
        help="Include installed column (hidden by default)",
    ),
    include_testing: bool = typer.Option(
        False,
        "--include-testing",
        help="Include testing/demo placeholder models (hidden by default)",
    ),
):
    """List downloaded models."""

    try:
        _enable_dupe_tolerance()
        base_entries = unified_list_downloads(only_installed=False)
        entries = list(base_entries)
        if include_logical or include_testing:
            have = {e.name for e in entries}
            try:
                raw = json.loads(
                    Path("configs/model_registry.json").read_text(encoding="utf-8")
                )
            except Exception as _exc:  # noqa: BLE001
                rprint(
                    f"❌ [red]Failed to read registry for logical entries:[/red] {_exc}"
                )
                raise typer.Exit(code=1)
            reg_map = unified_registry.load_registry(force=True)
            for obj in raw:
                name = obj.get("name")
                if not name or name in have:
                    continue
                if obj.get("download_path") is not None:
                    continue  # only logical-only (no path)
                if (not show_deprecated) and obj.get("deprecated"):
                    continue
                e = reg_map.get(name)
                if e is None:
                    # Create lightweight logical-only stub
                    e = SimpleNamespace(
                        name=name,
                        backend=obj.get("backend", "unknown"),
                        download_format=obj.get("download_format"),
                        quantization=obj.get("quantization"),
                        backend_config=None,
                        download_location=obj.get("download_location"),
                        download_path=None,
                        download_size_bytes=obj.get("download_size_bytes"),
                        roles=obj.get("roles") or [],
                        family=obj.get("family"),
                        served_model_id=obj.get("served_model_id"),
                        deprecated=obj.get("deprecated", False),
                    )
                entries.append(e)
        # Apply filters (format/location) only to entries that have those fields
        filtered = []
        for e in entries:
            # Hide testing/demo placeholders unless explicitly requested
            if not include_testing:
                try:
                    from imageworks.model_loader.testing_filters import (
                        is_testing_name as _is_testing_name,
                    )

                    if _is_testing_name(e.name):
                        continue
                except Exception:  # noqa: BLE001
                    pass
            # Determine logical-only status (no real path or synthetic ollama path with no downloaded_at attr)
            synthetic = False
            try:
                dp = e.download_path
            except AttributeError:
                dp = None
            if isinstance(dp, str) and dp.startswith("ollama://"):
                synthetic = True
            # By default, hide logical-only entries (no physical path or synthetic placeholder) unless --include-logical
            is_logical = (dp is None) or synthetic
            if (not include_logical) and is_logical:
                continue
            # logical_only_flag removed (unused)
            if (not show_deprecated) and getattr(e, "deprecated", False):
                continue
            if backend_filter and e.backend != backend_filter:
                continue
            if (
                format_filter
                and e.download_format
                and e.download_format != format_filter
            ):
                continue
            if (
                location_filter
                and e.download_location
                and e.download_location != location_filter
            ):
                continue
            filtered.append(e)
        entries = sorted(
            filtered,
            key=lambda e: (getattr(e, "display_name", None) or e.name).lower(),
        )

        # --- Robust family parsing helpers (avoid collapsing distinct multi-token families) ---
        _KNOWN_BACKENDS = {"ollama", "vllm", "lmdeploy", "unassigned"}
        _KNOWN_FORMATS = {"gguf", "safetensors", "awq", "gptq", "fp16", "bf16"}

        def _parse_family(e):
            if getattr(e, "family", None):
                return e.family
            parts = e.name.split("-")
            if len(parts) < 3:
                return e.name  # nothing to parse
            # Walk from right stripping quant (heuristic: contains underscore or starts with q/digit or in known set)
            tail = parts[:]
            # Potential quant token criteria
            if tail and (
                tail[-1].lower().startswith("q")
                or "_" in tail[-1]
                or tail[-1].upper() in {tail[-1]}
            ):
                # rely on stored quantization if present to confirm
                if (
                    getattr(e, "quantization", None)
                    and tail[-1].lower() == getattr(e, "quantization").lower()
                ):
                    tail = tail[:-1]
                elif getattr(e, "quantization", None) is None and any(
                    c.isdigit() for c in parts[-1]
                ):
                    tail = tail[:-1]
            if tail and tail[-1].lower() in _KNOWN_FORMATS:
                tail = tail[:-1]
            if tail and tail[-1].lower() in _KNOWN_BACKENDS:
                tail = tail[:-1]
            return "-".join(tail) if tail else e.name

        # Deduplicate only if explicitly requested
        if dedupe:

            def _score(e):
                s = 0
                if getattr(e, "quantization", None):
                    s += 3
                if getattr(e, "download_path", None):
                    s += 2
                if getattr(e, "download_size_bytes", None):
                    s += 1
                meta = getattr(e, "metadata", {}) or {}
                s += len(meta)
                return s

            grouped = {}
            for e in entries:
                fam = _parse_family(e)
                key = (
                    fam,
                    e.backend,
                    getattr(e, "download_format", None),
                    getattr(e, "quantization", None),
                )
                grouped.setdefault(key, []).append(e)
            deduped = []
            for key, items in grouped.items():
                if len(items) == 1:
                    deduped.extend(items)
                    continue
                # Only dedup if at least one item has download_path (avoid hiding purely logical diversity)
                if not any(getattr(i, "download_path", None) for i in items):
                    deduped.extend(items)
                    continue
                best = max(items, key=_score)
                deduped.append(best)
            entries = sorted(deduped, key=lambda x: x.name)

        # Layer-aware logical inclusion: read discovered layer directly for pure logical entries
        if include_logical:
            existing_names = {e.name for e in entries}
            discovered_path = Path("configs/model_registry.discovered.json")
            logical_candidates = []
            if discovered_path.exists():
                try:
                    logical_candidates = json.loads(
                        discovered_path.read_text(encoding="utf-8")
                    )
                except Exception:  # noqa: BLE001
                    logical_candidates = []
            from types import SimpleNamespace as _NS

            for obj in logical_candidates:
                name = obj.get("name")
                if not name or name in existing_names:
                    continue
                dp = obj.get("download_path")
                downloaded_at = obj.get("downloaded_at")
                synthetic = (
                    isinstance(dp, str)
                    and dp.startswith("ollama://")
                    and not downloaded_at
                )
                is_logical = (dp is None) or synthetic
                if not is_logical:
                    continue
                if (not show_deprecated) and obj.get("deprecated"):
                    continue
                entries.append(
                    _NS(
                        name=name,
                        backend=obj.get("backend", "unknown"),
                        download_format=obj.get("download_format"),
                        quantization=obj.get("quantization"),
                        backend_config=None,
                        download_location=obj.get("download_location"),
                        download_path=dp,
                        download_size_bytes=obj.get("download_size_bytes"),
                        roles=obj.get("roles") or [],
                        family=obj.get("family"),
                        served_model_id=obj.get("served_model_id"),
                        deprecated=obj.get("deprecated", False),
                    )
                )
            entries = sorted(entries, key=lambda x: x.name)

        if json_output:
            import json as _json

            payload = []
            for e in entries:
                installed = bool(
                    e.download_path and Path(e.download_path).expanduser().exists()
                )
                if not installed and e.backend == "ollama":
                    installed = True
                display = getattr(e, "display_name", None) or e.name
                producer_value = _resolve_producer(e)
                payload.append(
                    {
                        "name": e.name,
                        "display_name": display,
                        "backend": e.backend,
                        "format": e.download_format,
                        "quantization": e.quantization,
                        "location": e.download_location,
                        "installed": installed,
                        "size_bytes": e.download_size_bytes,
                        "roles": e.roles,
                        "family": e.family,
                        "logical_only": e.download_path is None,
                        "served_model_id": e.served_model_id,
                        "deprecated": getattr(e, "deprecated", False),
                        "capabilities": getattr(e, "capabilities", {}) or {},
                        "producer": producer_value,
                    }
                )
            if include_logical:
                present = {p["name"] for p in payload}
                discovered_path = Path("configs/model_registry.discovered.json")
                logical_candidates = []
                if discovered_path.exists():
                    try:
                        logical_candidates = json.loads(
                            discovered_path.read_text(encoding="utf-8")
                        )
                    except Exception:  # noqa: BLE001
                        logical_candidates = []
                for obj in logical_candidates:
                    name = obj.get("name")
                    if not name or name in present:
                        continue
                    dp = obj.get("download_path")
                    downloaded_at = obj.get("downloaded_at")
                    synthetic = (
                        isinstance(dp, str)
                        and dp.startswith("ollama://")
                        and not downloaded_at
                    )
                    is_logical = (dp is None) or synthetic
                    if not is_logical:
                        continue
                    if (not show_deprecated) and obj.get("deprecated"):
                        continue
                    payload.append(
                        {
                            "name": name,
                            "backend": obj.get("backend"),
                            "format": obj.get("download_format"),
                            "quantization": obj.get("quantization"),
                            "location": obj.get("download_location"),
                            "installed": False,
                            "size_bytes": obj.get("download_size_bytes"),
                            "roles": obj.get("roles") or [],
                            "family": obj.get("family"),
                            "logical_only": True,
                            "served_model_id": obj.get("served_model_id"),
                            "deprecated": obj.get("deprecated", False),
                            "producer": _resolve_producer(obj),
                        }
                    )
                payload.sort(key=lambda x: x["name"])
            # Use plain print to avoid Rich wrapping inserting newlines mid-token
            print(_json.dumps(payload, indent=2))
            return

        if not entries:
            rprint("📭 [yellow]No models found matching criteria[/yellow]")
            return

        table = Table(title="Unified Downloaded Variants")
        # Make the Model column wider and prevent noisy bracketed suffixes from crowding it
        table.add_column("Model", style="cyan", min_width=40, overflow="fold")
        if show_internal_names:
            table.add_column("Registry ID", style="white dim")
        table.add_column("Fmt", style="magenta", width=4)
        table.add_column("Quant", style="magenta")
        table.add_column("Producer", style="green")
        if show_backend:
            table.add_column("Backend", style="green")
        table.add_column("Caps", style="green")
        if show_installed:
            table.add_column("Inst", style="yellow")
        table.add_column("Size", justify="right", style="blue")
        if show_details:
            table.add_column("ServedID", style="white")
            table.add_column("Roles", style="white")

        def _format_label(value: Optional[str], fallback: str = "-") -> str:
            if not value:
                return fallback
            v = str(value).lower()
            if v == "safetensors":
                return "ST"
            return value.upper()

        def _format_quant(value: Optional[str]) -> str:
            if not value:
                return "-"
            return str(value).replace("_", " ").replace("-", " ").upper()

        for e in entries:
            installed = bool(
                e.download_path and Path(e.download_path).expanduser().exists()
            )
            if not installed and e.backend == "ollama":
                installed = True
            # Use registry display_name by default; fall back to name
            display = getattr(e, "display_name", None) or e.name
            fmt_display = _format_label(getattr(e, "download_format", None))
            quant_display = _format_quant(getattr(e, "quantization", None))
            backend_display = e.backend
            caps_dict = getattr(e, "capabilities", {}) or {}
            caps_tokens = []
            if caps_dict.get("vision"):
                caps_tokens.append("V")
            if caps_dict.get("embedding"):
                caps_tokens.append("E")
            if caps_dict.get("audio"):
                caps_tokens.append("A")
            if caps_dict.get("thinking"):
                caps_tokens.append("R")  # Reasoning/Thinking
            if caps_dict.get("tools"):
                caps_tokens.append("T")  # Tool/function calling
            caps_display = "".join(caps_tokens) or "-"
            size_value = getattr(e, "download_size_bytes", None)
            size_display = _format_size(size_value) if size_value is not None else "-"
            producer_display = _resolve_producer(e) or "-"
            row = [display]
            if show_internal_names:
                row.append(e.name)
            row.extend(
                [
                    fmt_display,
                    quant_display,
                    producer_display,
                ]
            )
            if show_backend:
                row.append(backend_display)
            row.append(caps_display)
            if show_installed:
                row.append("✓" if installed else "✗")
            row.append(size_display)
            if show_details:
                row.append(getattr(e, "served_model_id", None) or "-")
                row.append(",".join(getattr(e, "roles", []) or []) or "-")
            table.add_row(*row)
        console.print(table)
        total_size = sum((e.download_size_bytes or 0) for e in entries)
        rprint(f"\n📊 Total: {len(entries)} variants, {_format_size(total_size)}")
    except Exception as e:  # noqa: BLE001
        rprint(f"❌ [red]Failed to list models:[/red] {e}")
        raise typer.Exit(code=1)


@app.command("analyze")
def analyze_url(
    url: str = typer.Argument(..., help="URL or owner/repo identifier to analyze"),
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
        branch_suffix = f"@{repo.branch}" if getattr(repo, "branch", None) else ""
        rprint(f"📍 [bold]Repository:[/bold] {repo.owner}/{repo.repo}{branch_suffix}")
        if getattr(repo, "model_type", None):
            rprint(f"🏷️  Model Type: {repo.model_type}")
        if getattr(repo, "library_name", None):
            rprint(f"📚 Library: {repo.library_name}")
        tags = getattr(repo, "tags", None) or []
        if tags:
            rprint(f"🔖 Tags: {', '.join(tags[:10])}{' …' if len(tags) > 10 else ''}")

        rprint("\n🔧 [bold]Detected Formats:[/bold]")
        if analysis.formats:
            for format_info in analysis.formats:
                confidence = f"{format_info.confidence:.0%}"
                quant_bits = ""
                quant_details = format_info.quantization_details or {}
                if quant_details:
                    rendered = ", ".join(
                        f"{key}={value}" for key, value in quant_details.items()
                    )
                    if rendered:
                        quant_bits = f" [{rendered}]"
                rprint(
                    f"   • {format_info.format_type}{quant_bits} ({confidence} confidence)"
                )
                for evidence in format_info.evidence[:3]:
                    rprint(f"     - {evidence}")
                if len(format_info.evidence) > 3:
                    remaining = len(format_info.evidence) - 3
                    rprint(f"     - … {remaining} more signals")
        else:
            rprint("   • No formats detected")

        rprint("\n📁 [bold]Files Summary:[/bold]")
        for category, files in analysis.files.items():
            if not files:
                continue
            total_size = sum(getattr(f, "size", 0) for f in files)
            rprint(f"   • {category}: {len(files)} files ({_format_size(total_size)})")

        rprint(f"\n💾 [bold]Total Size:[/bold] {_format_size(analysis.total_size)}")

        if show_files:
            interesting_categories = {"model_weights", "config", "tokenizer"}
            for category, files in analysis.files.items():
                if not files or category not in interesting_categories:
                    continue
                rprint(f"\n📄 [bold]{category.title()}:[/bold]")
                for file_info in files[:5]:
                    priority_suffix = (
                        " ⭐" if getattr(file_info, "priority", False) else ""
                    )
                    rprint(
                        f"   • {file_info.path} ({_format_size(file_info.size)}){priority_suffix}"
                    )
                if len(files) > 5:
                    rprint(f"   … and {len(files) - 5} more files")

    except Exception as exc:  # noqa: BLE001
        rprint(f"❌ [red]Analysis failed:[/red] {exc}")
        raise typer.Exit(code=1)


@app.command("prune-duplicates")
def prune_duplicates(
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be removed without writing"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed decisions"
    ),
    backend: Optional[str] = typer.Option(
        None, "--backend", help="Limit to backend (e.g. ollama)"
    ),
):
    """Remove stale duplicate variants keeping the richest metadata per (family, backend, format, quant).

    Heuristic: among entries sharing (family, backend, format, quant) keep the one with highest score where
      score = +3 if quantization present +2 if download_path present +1 if size_bytes present +len(metadata)
    Other entries with the same key are removed from the registry (discovered layer only effect for dynamic entries).
    """
    try:
        from imageworks.model_loader import registry as _reg

        reg = _reg.load_registry(force=True)
        _KNOWN_BACKENDS = {"ollama", "vllm", "lmdeploy", "unassigned"}
        _KNOWN_FORMATS = {"gguf", "safetensors", "awq", "gptq", "fp16", "bf16"}

        def _parse_family(e):
            if e.family:
                return e.family
            parts = e.name.split("-")
            if len(parts) < 3:
                return e.name
            tail = parts[:]
            # strip quant token if matches entry.quantization
            if e.quantization and tail and tail[-1].lower() == e.quantization.lower():
                tail = tail[:-1]
            if tail and tail[-1].lower() in _KNOWN_FORMATS:
                tail = tail[:-1]
            if tail and tail[-1].lower() in _KNOWN_BACKENDS:
                tail = tail[:-1]
            return "-".join(tail) if tail else e.name

        def _key(e):
            fam = _parse_family(e)
            return (fam, e.backend, e.download_format, e.quantization)

        def _score(e):
            s = 0
            if e.quantization:
                s += 3
            if e.download_path:
                s += 2
            if e.download_size_bytes:
                s += 1
            s += len(e.metadata or {})
            return s

        groups = {}
        for e in reg.values():
            if backend and e.backend != backend:
                continue
            groups.setdefault(_key(e), []).append(e)
        to_remove = []
        kept = 0
        for k, items in groups.items():
            if len(items) <= 1:
                kept += len(items)
                continue
            # Only prune if SAME parsed family string across all items (guard) and at least one has download_path
            fams = {_parse_family(i) for i in items}
            if len(fams) != 1:
                kept += len(items)
                continue
            if not any(i.download_path for i in items):
                kept += len(items)
                continue
            best = max(items, key=_score)
            kept += 1
            for other in items:
                if other.name != best.name:
                    to_remove.append(other.name)
            if verbose:
                fam, b, fmt, q = k
                rprint(
                    f"Group {fam}/{b}/{fmt or '-'}:{q or '-'} -> keep {best.name}; remove {[i.name for i in items if i.name != best.name]}"
                )
        if not to_remove:
            rprint("🛈 [cyan]No duplicates detected for pruning[/cyan]")
            return
        if dry_run:
            rprint(
                f"Would remove {len(to_remove)} duplicate entries: {', '.join(sorted(to_remove))}"
            )
            return
        from imageworks.model_loader.registry import remove_entry, save_registry

        removed_ct = 0
        for n in to_remove:
            if remove_entry(n, save=False):
                removed_ct += 1
        save_registry()
        rprint(
            f"✅ [green]Pruned {removed_ct} duplicate entries (remaining kept groups: {kept})[/green]"
        )
    except Exception as exc:  # noqa: BLE001
        rprint(f"❌ [red]Prune failed:[/red] {exc}")
        raise typer.Exit(code=1)


@app.command("restore-ollama")
def restore_ollama(
    backup_path: Optional[Path] = typer.Option(
        None,
        "--backup",
        help="Specific backup file to restore from (model_registry*.bak.json)",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be restored"
    ),
    include_deprecated: bool = typer.Option(
        False,
        "--include-deprecated",
        help="Also restore entries that were deprecated in backup",
    ),
):
    """Restore missing Ollama entries from the latest backup (or specified file).

    Strategy:
      1. Load current registry (names -> entries).
      2. Load backup JSON list.
      3. For each backup entry with backend=ollama not present now, stage for restore.
      4. Write into discovered layer (update_entries) unless dry-run.
    """
    try:
        from imageworks.model_loader import registry as _reg
        import glob as _glob

        if not backup_path:
            candidates = sorted(_glob.glob("configs/model_registry.*.bak.json"))
            if not candidates:
                rprint("❌ [red]No backup files found[/red]")
                raise typer.Exit(code=1)
            backup_path = Path(candidates[-1])
        if not backup_path.exists():
            rprint(f"❌ [red]Backup file not found:[/red] {backup_path}")
            raise typer.Exit(code=1)
        raw = []
        try:
            raw = json.loads(backup_path.read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                raise ValueError("backup not list")
        except Exception as exc:  # noqa: BLE001
            rprint(f"❌ [red]Failed to parse backup:[/red] {exc}")
            raise typer.Exit(code=1)
        reg = _reg.load_registry(force=True)
        existing = set(reg.keys())
        restore_objs = [
            e
            for e in raw
            if isinstance(e, dict)
            and e.get("backend") == "ollama"
            and e.get("name") not in existing
            and (include_deprecated or not e.get("deprecated"))
        ]
        if not restore_objs:
            rprint("🛈 [cyan]No missing Ollama entries to restore[/cyan]")
            return
        if dry_run:
            rprint(
                f"Would restore {len(restore_objs)} entries: {', '.join(o.get('name') for o in restore_objs[:12])}{' ...' if len(restore_objs) > 12 else ''}"
            )
            return
        # Convert minimal fields into RegistryEntry objects via parsing utility
        from imageworks.model_loader.registry import (
            _parse_entry,
            update_entries,
            save_registry,
        )  # type: ignore

        restored_entries = []
        for obj in restore_objs:
            try:
                restored_entries.append(_parse_entry(obj))
            except Exception:
                rprint(
                    f"⚠️  [yellow]Skip malformed backup entry {obj.get('name')}[/yellow]"
                )
        update_entries(restored_entries, save=True)
        save_registry()
        rprint(
            f"✅ [green]Restored {len(restored_entries)} Ollama entries from backup {backup_path.name}[/green]"
        )
    except Exception as exc:  # noqa: BLE001
        rprint(f"❌ [red]Restore failed:[/red] {exc}")
        raise typer.Exit(code=1)


@app.command("reset-discovered")
def reset_discovered(
    backend: str = typer.Option(
        ..., "--backend", help="Backend to reset (e.g. ollama, vllm, all)"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be removed"),
    backup: bool = typer.Option(
        True,
        "--backup/--no-backup",
        help="Create a timestamped backup before modifying",
    ),
):
    """Remove discovered layer entries for a backend so they can be freshly re-imported.

    This ONLY affects entries currently sourced from the discovered overlay (dynamic entries).
    Curated entries are untouched. Use with caution.
    Workflow after reset (ollama example):
        1. imageworks-download reset-discovered --backend ollama
        2. uv run python scripts/import_ollama_models.py
        3. (optional) imageworks-download discover-all  (or discover-local-hf)
    """
    try:
        from imageworks.model_loader import registry as _reg
        import json as _json
        import datetime as _dt

        discovered_path = Path("configs/model_registry.discovered.json")
        if not discovered_path.exists():
            rprint("❌ [red]No discovered layer file present[/red]")
            raise typer.Exit(code=1)
        raw = _json.loads(discovered_path.read_text(encoding="utf-8"))
        if backend.lower() == "all":
            to_remove = [e for e in raw]
        else:
            to_remove = [e for e in raw if e.get("backend") == backend]
        if not to_remove:
            rprint("🛈 [cyan]No discovered entries matched[/cyan]")
            return
        remaining = [e for e in raw if e not in to_remove]
        if dry_run:
            rprint(
                f"Would remove {len(to_remove)} discovered entries (backend={backend}): {', '.join(e.get('name') for e in to_remove[:10])}{' ...' if len(to_remove) > 10 else ''}"
            )
            return
        if backup:
            ts = _dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
            backup_path = discovered_path.with_name(
                f"model_registry.discovered.{ts}.bak.json"
            )
            backup_path.write_text(
                discovered_path.read_text(encoding="utf-8"), encoding="utf-8"
            )
            rprint(f"💾 Backup written: {backup_path.name}")
        discovered_path.write_text(
            _json.dumps(remaining, indent=2) + "\n", encoding="utf-8"
        )
        # Force reload / regenerate merged snapshot
        _reg.load_registry(force=True)
        rprint(
            f"✅ [green]Removed {len(to_remove)} discovered entries (backend={backend})[/green]"
        )
    except Exception as exc:  # noqa: BLE001
        rprint(f"❌ [red]Reset failed:[/red] {exc}")
        raise typer.Exit(code=1)


@app.command("backfill-ollama-paths")
def backfill_ollama_paths(
    location: str = typer.Option(
        "linux_wsl",
        "--location",
        help="Location label to set for backfilled Ollama entries",
    ),
    set_format: bool = typer.Option(
        True,
        "--set-format/--no-set-format",
        help="Force download_format=gguf when missing",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would change without writing"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show per-entry changes"
    ),
):
    """Populate download_* metadata for existing Ollama registry entries that lack it (Option A).

    Rationale:
      Earlier Strategy A imports may have created logical entries (family/backend/quant) without actual
      download_path. The normal importer script (`scripts/import_ollama_models.py`) can be re-run to enrich
      them, but this command works offline (no `ollama` binary required) and simply maps entries to the
      Ollama model store directory so they appear in `list` output.

    Behaviour:
      - For each entry with backend==ollama AND download_path is null/empty:
          * Sets download_path to <store>/<served_model_id or name>
            (store autodetected via $OLLAMA_MODELS, ~/.ollama/models, or <linux_wsl_root>/ollama-data/models)
            If store not found, uses synthetic scheme ollama://<served_model_id|name>
          * Sets download_format to 'gguf' if missing and --set-format
          * Leaves quantization untouched
          * Sets download_location to provided --location if empty
      - Treats path existence as optional (installed flag already special-cased for ollama)
    """
    try:
        # Work at raw JSON layer to update every matching entry (even duplicates)
        reg_path = Path("configs/model_registry.json")
        if not reg_path.exists():
            rprint("❌ [red]Registry file not found[/red]")
            raise typer.Exit(code=1)
        raw = json.loads(reg_path.read_text(encoding="utf-8"))
        store_root = _detect_ollama_store_root()
        changed = []
        for entry in raw:
            if entry.get("backend") != "ollama":
                continue
            if entry.get("download_path"):
                continue
            ident = entry.get("served_model_id") or entry.get("name")
            if store_root is not None:
                candidate = store_root / str(ident).replace(":", "/")
                new_path = str(candidate)
            else:
                new_path = f"ollama://{ident}"
            before = {
                "download_path": entry.get("download_path"),
                "download_format": entry.get("download_format"),
                "download_location": entry.get("download_location"),
            }
            if set_format and not entry.get("download_format"):
                entry["download_format"] = "gguf"
            if not entry.get("download_location"):
                entry["download_location"] = location
            entry["download_path"] = new_path
            after = {
                "download_path": entry.get("download_path"),
                "download_format": entry.get("download_format"),
                "download_location": entry.get("download_location"),
            }
            changed.append(
                {"entry": entry.get("name"), "before": before, "after": after}
            )
        if not changed:
            rprint("🛈 [cyan]No Ollama entries required backfill[/cyan]")
            return
        if dry_run:
            rprint(f"Would backfill {len(changed)} Ollama logical entries:")
            if verbose:
                for c in changed:
                    rprint(
                        f"  • {c['entry']} path: {c['before']['download_path']} -> {c['after']['download_path']}"
                    )
            else:
                sample = ", ".join(c["entry"] for c in changed[:6])
                if len(changed) > 6:
                    sample += " ..."
                rprint(f"  {sample}")
            rprint("Dry run: not saving registry")
            return
        # Write updated registry
        reg_path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
        # Post-write safety pass: ensure no ollama entry remains without download_path
        raw2 = json.loads(reg_path.read_text(encoding="utf-8"))
        second_pass = False
        for entry in raw2:
            if entry.get("backend") == "ollama" and not entry.get("download_path"):
                ident2 = entry.get("served_model_id") or entry.get("name")
                entry["download_path"] = f"ollama://{ident2}"
                if set_format and not entry.get("download_format"):
                    entry["download_format"] = "gguf"
                if not entry.get("download_location"):
                    entry["download_location"] = location
                second_pass = True
        if second_pass:
            reg_path.write_text(json.dumps(raw2, indent=2) + "\n", encoding="utf-8")
        # Invalidate and reload cache with strict duplicate enforcement
        try:
            unified_registry.load_registry(force=True)
        except Exception as _exc:  # noqa: BLE001
            rprint(
                f"⚠️  [yellow]Reload warning after backfill (non-fatal): {_exc}[/yellow]"
            )
        rprint(f"✅ [green]Backfilled {len(changed)} Ollama entries[/green]")
        if verbose:
            for c in changed:
                rprint(f"  • {c['entry']} path={c['after']['download_path']}")

    except Exception as e:  # noqa: BLE001
        rprint(f"❌ [red]Analysis failed:[/red] {str(e)}")
        raise typer.Exit(code=1)


@app.command("remove")
def remove_model(
    name: str = typer.Argument(
        ..., help="Variant name in unified registry (e.g. family-backend-format-quant)"
    ),
    delete_files: bool = typer.Option(
        False,
        "--delete-files",
        help="Also delete the model files from disk (or run 'ollama rm' for Ollama)",
    ),
    purge: bool = typer.Option(
        False,
        "--purge",
        help="Remove the entire entry instead of just clearing download metadata",
    ),
    force: bool = typer.Option(False, "--force", help="Don't ask for confirmation"),
):
    """Remove a downloaded variant (clears download metadata by default).

    By default keeps the logical entry (roles, capabilities, metadata) but clears the
    download_* fields so it can be re-downloaded deterministically. Use --purge to
    delete the entire entry.
    """
    try:
        reg_path = Path("configs/model_registry.json")
        if reg_path.exists():
            try:
                _ = reg_path.read_text(
                    encoding="utf-8"
                )  # read for potential future use
            except Exception:  # noqa: BLE001
                _ = None
        reg = unified_registry.load_registry(force=True)
        entry = reg.get(name)
        if not entry:
            rprint(f"❌ [red]Variant not found:[/red] {name}")
            raise typer.Exit(code=1)
        installed = bool(
            entry.download_path and Path(entry.download_path).expanduser().exists()
        )
        rprint("🗑️  [yellow]Will remove:[/yellow]")
        rprint(
            f"   • {entry.name} (format={entry.download_format or '-'} loc={entry.download_location or '-'})"
        )
        if delete_files and installed:
            rprint(f"     📁 Files: {entry.download_path}")
        if purge:
            rprint("     ⚠️  Entire entry will be deleted (purge)")
        if not force:
            if not typer.confirm("Proceed?"):
                rprint("❌ [yellow]Cancelled[/yellow]")
                return
        # Delete files if requested
        if delete_files and entry.backend == "ollama":
            # Prefer using the Ollama CLI to remove model blobs cleanly
            ident = entry.served_model_id or entry.display_name or entry.name
            try:
                rprint(f"🔧 Invoking: ollama rm {ident}")
                subprocess.run(["ollama", "rm", str(ident)], check=False)
            except Exception as exc:  # noqa: BLE001
                rprint(f"⚠️  [yellow]Failed to run 'ollama rm': {exc}[/yellow]")
        elif delete_files and installed and entry.download_path:
            try:
                import shutil

                shutil.rmtree(Path(entry.download_path).expanduser())
            except OSError as exc:  # noqa: BLE001
                rprint(f"⚠️  [yellow]Failed to delete files: {exc}[/yellow]")
            try:
                # Attempt to prune empty repo and owner directories
                _prune_empty_repo_and_owner_dirs(Path(entry.download_path).expanduser())
            except Exception:
                pass
        ok = unified_remove_download(entry.name, keep_entry=not purge)
        if ok:
            if purge:
                rprint("✅ [green]Entry purged[/green]")
            else:
                rprint("✅ [green]Download metadata cleared (entry retained)[/green]")
        else:
            rprint("❌ [red]Removal failed[/red]")
            raise typer.Exit(code=1)
    except Exception as e:  # noqa: BLE001
        rprint(f"❌ [red]Removal failed:[/red] {e}")
        raise typer.Exit(code=1)


@app.command("purge-deprecated")
def purge_deprecated(
    placeholders_only: bool = typer.Option(
        False,
        "--placeholders-only",
        help="Only purge legacy placeholder entries (model-ollama-gguf*)",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be removed without writing"
    ),
):
    """Permanently remove deprecated entries from the registry.

    Use --placeholders-only to restrict to legacy placeholder imports (e.g. model-ollama-gguf*).
    """
    try:
        reg = unified_registry.load_registry(force=True)
        to_delete: list[str] = []
        for name, entry in list(reg.items()):
            if not getattr(entry, "deprecated", False):
                continue
            if placeholders_only and not name.startswith("model-ollama-gguf"):
                continue
            to_delete.append(name)
        if not to_delete:
            rprint("🛈 [cyan]No matching deprecated entries to purge[/cyan]")
            return
        rprint(
            f"Will purge {len(to_delete)} deprecated entr{'y' if len(to_delete) == 1 else 'ies'}:"
        )
        for n in to_delete:
            rprint(f"  • {n}")
        if dry_run:
            rprint("Dry run: not modifying registry")
            return
        from imageworks.model_loader.registry import remove_entry

        removed = 0
        for n in to_delete:
            if remove_entry(n, save=False):
                removed += 1
        unified_registry.save_registry()
        rprint(f"✅ [green]Purged {removed} entries[/green]")
    except Exception as exc:  # noqa: BLE001
        rprint(f"❌ [red]Failed to purge deprecated entries:[/red] {exc}")
        raise typer.Exit(code=1)


@app.command("purge-hf")
def purge_hf(
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be removed without writing"
    ),
    weights_root: Path = typer.Option(
        Path("~/ai-models/weights").expanduser(),
        "--weights-root",
        help="Root path whose descendants are considered HF-sourced",
    ),
    backend_filter: Optional[str] = typer.Option(
        None, "--backend", help="Optional backend filter (e.g. vllm)"
    ),
):
    """Remove entries whose download_path is under the given weights root (default ~/ai-models/weights).

    This treats physical location as authoritative indicator of an HF-style local clone rather than relying on
    source_provider metadata, which may be absent or stale. Purges entries so they can be re-imported cleanly.
    """
    try:
        root = weights_root.expanduser().resolve()
        reg = unified_registry.load_registry(force=True)
        targets: list[str] = []
        for name, e in list(reg.items()):
            dp = getattr(e, "download_path", None)
            if not dp:
                continue
            try:
                p = Path(dp).expanduser().resolve()
            except Exception:
                continue
            if backend_filter and e.backend != backend_filter:
                continue
            try:
                if root in p.parents or p == root:
                    targets.append(name)
            except Exception:
                continue
        if not targets:
            rprint("🛈 [cyan]No HF (by path) entries to purge[/cyan]")
            return
        rprint(
            f"Will purge {len(targets)} HF-path entr{'y' if len(targets) == 1 else 'ies'} (root={root}):"
        )
        for n in targets:
            rprint(f"  • {n}")
        if dry_run:
            rprint("Dry run: not modifying registry")
            return
        from imageworks.model_loader.registry import remove_entry

        removed = 0
        for n in targets:
            if remove_entry(n, save=False):
                removed += 1
        unified_registry.save_registry()
        rprint(f"✅ [green]Purged {removed} entries under {root}")
    except Exception as exc:  # noqa: BLE001
        rprint(f"❌ [red]Failed to purge HF path entries:[/red] {exc}")
        raise typer.Exit(code=1)


@app.command("stats")
def show_stats():
    """Show download statistics."""

    try:
        entries = unified_list_downloads(only_installed=False)
        total_size = sum((e.download_size_bytes or 0) for e in entries)
        by_format: Dict[str, int] = {}
        by_location: Dict[str, int] = {}
        for e in entries:
            if e.download_format:
                by_format[e.download_format] = by_format.get(e.download_format, 0) + 1
            if e.download_location:
                by_location[e.download_location] = (
                    by_location.get(e.download_location, 0) + 1
                )
        table = Table(title="Unified Download Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Total Variants", str(len(entries)))
        table.add_row("Total Size", _format_size(total_size))
        console.print(table)
        if by_format:
            ft = Table(title="By Format")
            ft.add_column("Format")
            ft.add_column("Count", justify="right")
            for k, v in sorted(by_format.items()):
                ft.add_row(k, str(v))
            console.print(ft)
        if by_location:
            lt = Table(title="By Location")
            lt.add_column("Location")
            lt.add_column("Count", justify="right")
            for k, v in sorted(by_location.items()):
                lt.add_row(k, str(v))
            console.print(lt)
    except Exception as e:  # noqa: BLE001
        rprint(f"❌ [red]Failed to get stats:[/red] {e}")
        raise typer.Exit(code=1)


@app.command("purge-logical-only")
def purge_logical_only(
    include_curated: bool = typer.Option(
        True,
        "--include-curated/--discovered-only",
        help="Also remove curated logical-only entries",
    ),
    backup: bool = typer.Option(
        True,
        "--backup/--no-backup",
        help="Write timestamped backups of registry fragments",
    ),
    dry_run: bool = typer.Option(
        True, "--dry-run/--apply", help="Show what would be removed without writing"
    ),
):
    """Remove logical-only entries (no download_path or synthetic ollama:// without installed data).

    In layered mode:
      - Discovered layer: always eligible.
      - Curated layer: removed only if --include-curated.

    A backup of the JSON fragments is written by default.
    """
    try:
        reg_dir = Path("configs")
        curated_path = reg_dir / "model_registry.curated.json"
        discovered_path = reg_dir / "model_registry.discovered.json"
        removed: list[str] = []

        def _is_logical_only(obj: dict) -> bool:
            dp = obj.get("download_path")
            downloaded_at = obj.get("downloaded_at")
            if dp is None:
                return True
            if isinstance(dp, str) and dp.startswith("ollama://") and not downloaded_at:
                return True
            return False

        def _load(path: Path) -> list[dict]:
            if not path.exists():
                return []
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return data if isinstance(data, list) else []
            except Exception:
                return []

        def _save(path: Path, data: list[dict]):
            path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

        curated = _load(curated_path)
        discovered = _load(discovered_path)

        # Build new lists
        curated_new: list[dict] = []
        for e in curated:
            if include_curated and _is_logical_only(e):
                removed.append(e.get("name") or "<curated>")
            else:
                curated_new.append(e)
        discovered_new: list[dict] = []
        for e in discovered:
            if _is_logical_only(e):
                removed.append(e.get("name") or "<discovered>")
            else:
                discovered_new.append(e)

        if not removed:
            rprint("🛈 [cyan]No logical-only entries found[/cyan]")
            return
        if dry_run:
            rprint(
                f"Would remove {len(removed)} logical-only entr{'y' if len(removed) == 1 else 'ies'}: "
                + ", ".join(sorted(removed)[:12])
                + (" ..." if len(removed) > 12 else "")
            )
            return
        # Backups
        if backup:
            from datetime import datetime as _dt

            ts = _dt.utcnow().strftime("%Y%m%d-%H%M%S")
            if curated_path.exists():
                curated_path.with_name(
                    f"model_registry.curated.{ts}.bak.json"
                ).write_text(curated_path.read_text(encoding="utf-8"), encoding="utf-8")
            if discovered_path.exists():
                discovered_path.with_name(
                    f"model_registry.discovered.{ts}.bak.json"
                ).write_text(
                    discovered_path.read_text(encoding="utf-8"), encoding="utf-8"
                )
        # Save filtered curated/discovered
        _save(curated_path, curated_new)
        _save(discovered_path, discovered_new)
        # Also rewrite the merged snapshot directly to the filtered union to avoid re-adopting stale entries
        try:
            merged_union = {
                e.get("name"): e
                for e in curated_new
                if isinstance(e, dict) and e.get("name")
            }
            for e in discovered_new:
                if isinstance(e, dict) and e.get("name"):
                    merged_union[e["name"]] = e
            snapshot_path = reg_dir / "model_registry.json"
            snapshot_path.write_text(
                json.dumps(list(merged_union.values()), indent=2) + "\n",
                encoding="utf-8",
            )
        except Exception:
            pass
        # Regenerate registry cache from filtered files
        unified_registry.load_registry(force=True)
        unified_registry.save_registry()
        rprint(f"✅ [green]Removed {len(removed)} logical-only entries[/green]")
    except Exception as exc:  # noqa: BLE001
        rprint(f"❌ [red]Purge failed:[/red] {exc}")
        raise typer.Exit(code=1)


@app.command("verify")
def verify_models(
    name: Optional[str] = typer.Argument(
        None,
        help="Specific variant to verify (all downloaded variants if not specified)",
    ),
    fix_missing: bool = typer.Option(
        False, "--fix-missing", help="Clear download metadata for missing variants"
    ),
):
    """Verify downloaded variant integrity (directory existence & checksum change)."""

    try:
        entries = unified_list_downloads(only_installed=False)
        if name:
            entries = [e for e in entries if e.name == name]
            if not entries:
                rprint(f"❌ [red]Variant not found:[/red] {name}")
                raise typer.Exit(code=1)
        rprint(f"� [bold]Verifying {len(entries)} variants...[/bold]\n")
        valid: List[str] = []
        invalid: List[str] = []
        for e in entries:
            if not e.download_path:
                invalid.append(e.name)
                rprint(f"⚠️  {e.name} - no download path recorded")
                continue
            p = Path(e.download_path).expanduser()
            if not p.exists():
                invalid.append(e.name)
                rprint(f"❌ {e.name} - path missing: {e.download_path}")
                continue
            current_checksum = compute_directory_checksum(p)
            if (
                e.download_directory_checksum
                and current_checksum != e.download_directory_checksum
            ):
                invalid.append(e.name)
                rprint(
                    f"⚠️  {e.name} - checksum changed ({e.download_directory_checksum} -> {current_checksum})"
                )
            else:
                valid.append(e.name)
                rprint(f"✅ {e.name}")
        rprint("\n📊 [bold]Summary:[/bold]")
        rprint(f"   ✅ Valid: {len(valid)}")
        rprint(f"   ❌ Invalid: {len(invalid)}")
        if invalid and fix_missing:
            rprint(
                f"\n🔧 [yellow]Clearing download metadata for {len(invalid)} variants...[/yellow]"
            )
            for n in invalid:
                unified_remove_download(n, keep_entry=True)
            rprint("✅ Cleared")
    except Exception as e:  # noqa: BLE001
        rprint(f"❌ [red]Verification failed:[/red] {e}")
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


@app.command("list-roles")
def list_roles(
    registry_path: Path = typer.Option(
        Path("configs/model_registry.json"), help="Path to unified registry JSON"
    ),
    show_capabilities: bool = typer.Option(
        False, help="Show capability flags (text,vision,embedding,etc.)"
    ),
    json_output: bool = typer.Option(False, help="Emit JSON array instead of table"),
):
    """List role-capable models from the unified deterministic registry.

    A model with multiple roles is displayed once per role.
    """
    try:
        reg = _load_unified_registry(registry_path, force=True)
    except Exception as exc:  # noqa: BLE001
        rprint(f"❌ [red]Failed to load unified registry:[/red] {exc}")
        raise typer.Exit(code=1)

    rows: List[Dict[str, Any]] = []
    for entry in reg.values():
        if not entry.roles:
            continue
        for role in entry.roles:
            rows.append(
                {
                    "role": role,
                    "name": entry.name,
                    "backend": entry.backend,
                    "display_name": entry.display_name or entry.name,
                    "capabilities": entry.capabilities,
                }
            )

    rows.sort(key=lambda r: (r["role"], r["name"]))

    if json_output:
        rprint(json.dumps(rows, indent=2))
        return

    if not rows:
        rprint("📭 [yellow]No role-capable models found[/yellow]")
        return

    from rich.table import Table as _Table

    table = _Table(title="Role-capable Models")
    table.add_column("Role", style="magenta")
    table.add_column("Name", style="cyan")
    table.add_column("Backend", style="green")
    table.add_column("Display Name", style="white")
    if show_capabilities:
        table.add_column("Capabilities", style="blue")
    for r in rows:
        caps = ",".join(k for k, v in r["capabilities"].items() if v)
        row = [r["role"], r["name"], r["backend"], r["display_name"]]
        if show_capabilities:
            row.append(caps)
        table.add_row(*row)
    console.print(table)


def _format_size(size_bytes: int) -> str:
    """Format size in human-readable format."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


@app.command("preview-simple-slugs")
def preview_simple_slugs(
    include_ollama: bool = typer.Option(
        True, help="Include Ollama entries in the preview"
    ),
    include_hf: bool = typer.Option(
        True, help="Include HF/vLLM entries in the preview"
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON mapping"),
):
    """Preview simplified slug naming: `<family> <param_size>[-extra]` + quant.

    - HF: family from repo tail; parameter size inferred from family tokens (e.g., 7B/70B) when present.
    - Ollama: family and parameter_size from `ollama show` (stored in metadata), if available.
    - Quant is taken from registry (already detected by core logic).

    This does not apply any changes; it surfaces the proposed names and any collisions.
    """
    try:
        reg = unified_registry.load_registry(force=True)
        rows = []
        collisions: Dict[str, list[str]] = {}

        # Cache computed parameter sizes per download path to avoid repeated work
        _param_cache: Dict[str, Optional[str]] = {}

        def _format_param_label(total_params: int) -> str:
            # Convert to billions and round to one decimal; drop trailing .0
            if total_params <= 0:
                return ""
            val = total_params / 1_000_000_000.0
            rounded = round(val + 1e-12, 1)
            if abs(rounded - int(rounded)) < 1e-9:
                return f"{int(rounded)}B"
            return f"{rounded:.1f}B"

        def _hf_param_size_label(path: Optional[str]) -> Optional[str]:
            if not path:
                return None
            try:
                key = str(Path(path).expanduser())
            except Exception:
                key = str(path)
            if key in _param_cache:
                return _param_cache[key]
            try:
                p = Path(path).expanduser()
                if not p.exists():
                    _param_cache[key] = None
                    return None
                st_files = [f for f in p.rglob("*.safetensors") if f.is_file()]
                if not st_files:
                    _param_cache[key] = None
                    return None
                try:
                    from safetensors import safe_open as _safe_open  # type: ignore
                except Exception:
                    _param_cache[key] = None
                    return None
                total: int = 0
                for sf in sorted(st_files):
                    try:
                        with _safe_open(str(sf), framework="pt", device="cpu") as f:  # type: ignore
                            for tk in f.keys():
                                try:
                                    meta = f.get_tensor_metadata(tk)
                                    shape = getattr(meta, "shape", None)
                                    if not shape:
                                        # Fallback: get tensor (last resort)
                                        t = f.get_tensor(tk)
                                        shape = getattr(t, "shape", None)
                                    if shape:
                                        n = 1
                                        for d in shape:
                                            n *= int(d)
                                        total += int(n)
                                except Exception:
                                    continue
                    except Exception:
                        continue
                label = _format_param_label(total) if total > 0 else None
                _param_cache[key] = label
                return label
            except Exception:
                _param_cache[key] = None
                return None

        def _extract_param_size(text: str | None) -> str | None:
            if not text:
                return None
            t = text.strip().lower()
            # Common forms: 7b, 70b, 405b; also words like "8 billion" are out of scope here
            import re as _re

            m = _re.search(r"\b(\d{1,3})\s*b\b", t)
            if m:
                return f"{m.group(1)}B"
            return None

        def _propose_name(e) -> tuple[str, str]:
            backend = e.backend
            q_raw = getattr(e, "quantization", None)
            quant = q_raw or ""
            # Treat 'unknown' as no quant for display/base purposes
            q_is_valid = bool(quant) and (str(quant).lower() != "unknown")
            q_label = (
                quant.replace("_", " ").replace("-", " ").upper() if q_is_valid else ""
            )
            family = None
            param = None
            # Prefer Ollama metadata when backend=ollama
            if include_ollama and backend == "ollama":
                meta = getattr(e, "metadata", {}) or {}
                # Ollama show family if present
                family = (
                    meta.get("ollama_family")
                    or meta.get("ollama_architecture")
                    or e.family
                    or "model"
                )
                param = meta.get("ollama_parameter_size") or _extract_param_size(
                    meta.get("ollama_parameters")
                )
            elif include_hf and backend != "ollama":
                # HF/vLLM: family from repo tail as persisted; try to extract size token like 7B
                family = e.family or "model"
                # Prefer computed parameter size from local safetensors when available
                param = _hf_param_size_label(
                    getattr(e, "download_path", None)
                ) or _extract_param_size(family)
            else:
                family = e.family or "model"
            # Normalize family (keep dots and digits for things like llama3.1)
            import re as _re

            original_family = (family or "").strip().lower()
            # Remove container & quant tokens BEFORE replacing underscores so we can match full patterns like q4_k_m
            rm_patterns = [
                r"\b(gguf|safetensors)\b",
                r"\bq\d(?:_k(?:_[sml])?|_[01])\b",  # q4_k_m, q5_k_s, q4_0
                r"\bq\d(?:-k(?:-[sml])?)\b",  # q4-k-m style (if present in family)
                r"\b(awq|gptq|int4|int8|fp16|bf16|fp8|nf4|mxfp4|mxfp8)\b",
            ]
            fam_clean = original_family

            for pat in rm_patterns:
                fam_clean = _re.sub(pat, " ", fam_clean)
            fam_clean = _re.sub(r"\s+", " ", fam_clean).strip()
            fam_norm = fam_clean.replace("@", "-").replace("_", "-").replace("/", "-")
            fam_norm = _re.sub(r"-+", "-", fam_norm).strip("-")
            # If hyphenated param like '-7b' remains at end, extract it
            if not param:
                m = _re.search(r"-(\d{1,3})b$", fam_norm)
                if m:
                    param = f"{m.group(1)}B"
                    fam_norm = fam_norm[: -len(m.group(0))]
            # Robust fallback when family collapses to empty or generic
            if not fam_norm or fam_norm == "model":
                # Try served_model_id (for Ollama) or old name to recover a base token
                sid = (
                    getattr(e, "served_model_id", None)
                    or getattr(e, "name", None)
                    or "model"
                )
                sid_l = str(sid).lower()
                base = sid_l.split(":", 1)[0]
                base = base.replace("@", "-").replace("_", "-").replace("/", "-")
                base = _re.sub(r"-+", "-", base).strip("-")
                # Strip again common tokens
                for pat in rm_patterns:
                    base = _re.sub(pat, " ", base)
                base = _re.sub(r"\s+", " ", base).strip().replace(" ", "-")
                fam_norm = base or "model"
            if param and param.lower() not in fam_norm:
                base = f"{fam_norm} {param.lower()}"
            else:
                base = fam_norm
            base_clean = base.strip().strip("-")
            proposed_disp = f"{base_clean} ({q_label})" if q_label else base_clean
            # Proposed base: display preview but with underscores instead of spaces, keep parentheses
            proposed_slug = proposed_disp.replace(" ", "_")
            # For known testing/demo entries, prepend a unique guard prefix to avoid collisions with real models
            try:
                from imageworks.model_loader.testing_filters import (
                    is_testing_entry as _is_test,
                )

                if _is_test(getattr(e, "name", ""), e):
                    guard = "__TESTZZZ__"
                    proposed_slug = f"{guard}{proposed_slug}"
                    proposed_disp = f"{guard}{proposed_disp}"
            except Exception:
                pass
            return proposed_slug, proposed_disp

        for name, e in reg.items():
            if e.backend == "ollama" and not include_ollama:
                continue
            if e.backend != "ollama" and not include_hf:
                continue
            proposed, display = _propose_name(e)
            rows.append(
                {
                    "old_name": e.name,
                    "quant": e.quantization,
                    "family": e.family,
                    "proposed_slug_base": proposed,
                    "proposed_display": display,
                }
            )
            collisions.setdefault(proposed, []).append(e.name)

        # Detect collisions
        collision_list = [k for k, v in collisions.items() if len(v) > 1]
        if json_output:
            print(json.dumps({"rows": rows, "collisions": collision_list}, indent=2))
            return
        t = Table(title="Simplified Slug Preview")
        t.add_column("Old Name", style="cyan", overflow="fold")
        t.add_column("Proposed Base", style="yellow", min_width=54, overflow="fold")
        t.add_column("Display Preview", style="white", min_width=64, overflow="fold")
        for r in rows[:400]:  # cap for readability
            t.add_row(
                r["old_name"],
                r["proposed_slug_base"],
                r["proposed_display"],
            )
        console.print(t)
        if collision_list:
            rprint(
                f"⚠️  Potential collisions: {len(collision_list)} bases have multiples"
            )
        rprint(f"📊 Previewed {len(rows)} entries")
    except Exception as exc:  # noqa: BLE001
        rprint(f"❌ [red]Preview failed:[/red] {exc}")
        raise typer.Exit(code=1)


@app.command("apply-simple-slugs")
def apply_simple_slugs(
    include_ollama: bool = typer.Option(
        True, help="Include Ollama entries when applying"
    ),
    include_hf: bool = typer.Option(True, help="Include HF/vLLM entries when applying"),
    disambiguate: str = typer.Option(
        "backend",
        help="Strategy to resolve collisions among identical proposed bases",
        case_sensitive=False,
    ),
    rename_slugs: bool = typer.Option(
        False,
        "--rename-slugs/--display-only",
        help="Actually rename entry.name slugs; when off, only updates display_name",
    ),
    allow_skip_on_collision: bool = typer.Option(
        True,
        "--allow-skip-on-collision/--no-skip-on-collision",
        help="Skip colliding items that can't be disambiguated instead of failing",
    ),
    dry_run: bool = typer.Option(
        True, "--dry-run/--apply", help="Preview changes without writing (default)"
    ),
    backup: bool = typer.Option(
        True, "--backup/--no-backup", help="Write timestamped backup before modifying"
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Emit JSON summary of changes"
    ),
    tests_only: bool = typer.Option(
        False,
        "--tests-only",
        help="Limit changes to testing/demo entries only (identified by testing filters)",
    ),
):
    """Apply the simplified slug naming scheme.

    Behaviour:
      - Computes the same proposed base/display as preview-simple-slugs.
      - Resolves collisions by appending a stable disambiguator (backend/format) to the slug base when requested.
      - Updates display_name for all selected entries.
      - Optionally renames the entry.name (slug) if --rename-slugs is provided.
      - Writes a backup before applying unless --no-backup.

    Notes:
      - Collisions are detected across the selected subset; we avoid introducing duplicate names.
      - Parentheses are preserved in display; slugs use underscores. Disambiguators are appended as __<token>.
    """
    try:
        reg = unified_registry.load_registry(force=True)

        # Helper: keep naming logic consistent with preview
        _param_cache: Dict[str, Optional[str]] = {}

        def _format_param_label(total_params: int) -> str:
            if total_params <= 0:
                return ""
            val = total_params / 1_000_000_000.0
            rounded = round(val + 1e-12, 1)
            if abs(rounded - int(rounded)) < 1e-9:
                return f"{int(rounded)}B"
            return f"{rounded:.1f}B"

        def _hf_param_size_label(path: Optional[str]) -> Optional[str]:
            if not path:
                return None
            try:
                key = str(Path(path).expanduser())
            except Exception:
                key = str(path)
            if key in _param_cache:
                return _param_cache[key]
            try:
                p = Path(path).expanduser()
                if not p.exists():
                    _param_cache[key] = None
                    return None
                st_files = [f for f in p.rglob("*.safetensors") if f.is_file()]
                if not st_files:
                    _param_cache[key] = None
                    return None
                try:
                    from safetensors import safe_open as _safe_open  # type: ignore
                except Exception:
                    _param_cache[key] = None
                    return None
                total: int = 0
                for sf in sorted(st_files):
                    try:
                        with _safe_open(str(sf), framework="pt", device="cpu") as f:  # type: ignore
                            for tk in f.keys():
                                try:
                                    meta = f.get_tensor_metadata(tk)
                                    shape = getattr(meta, "shape", None)
                                    if not shape:
                                        t = f.get_tensor(tk)
                                        shape = getattr(t, "shape", None)
                                    if shape:
                                        n = 1
                                        for d in shape:
                                            n *= int(d)
                                        total += int(n)
                                except Exception:
                                    continue
                    except Exception:
                        continue
                label = _format_param_label(total) if total > 0 else None
                _param_cache[key] = label
                return label
            except Exception:
                _param_cache[key] = None
                return None

        def _extract_param_size(text: str | None) -> str | None:
            if not text:
                return None
            t = text.strip().lower()
            import re as _re

            m = _re.search(r"\b(\d{1,3})\s*b\b", t)
            if m:
                return f"{m.group(1)}B"
            return None

        def _propose_name(e) -> tuple[str, str]:
            backend = e.backend
            q_raw = getattr(e, "quantization", None)
            quant = q_raw or ""
            q_is_valid = bool(quant) and (str(quant).lower() != "unknown")
            q_label = (
                quant.replace("_", " ").replace("-", " ").upper() if q_is_valid else ""
            )
            family = None
            param = None
            if include_ollama and backend == "ollama":
                meta = getattr(e, "metadata", {}) or {}
                family = (
                    meta.get("ollama_family")
                    or meta.get("ollama_architecture")
                    or e.family
                    or "model"
                )
                param = meta.get("ollama_parameter_size") or _extract_param_size(
                    meta.get("ollama_parameters")
                )
            elif include_hf and backend != "ollama":
                family = e.family or "model"
                param = _hf_param_size_label(
                    getattr(e, "download_path", None)
                ) or _extract_param_size(family)
            else:
                family = e.family or "model"
            import re as _re

            original_family = (family or "").strip().lower()
            rm_patterns = [
                r"\b(gguf|safetensors)\b",
                r"\bq\d(?:_k(?:_[sml])?|_[01])\b",
                r"\bq\d(?:-k(?:-[sml])?)\b",
                r"\b(awq|gptq|int4|int8|fp16|bf16|fp8|nf4|mxfp4|mxfp8)\b",
            ]
            fam_clean = original_family
            for pat in rm_patterns:
                fam_clean = _re.sub(pat, " ", fam_clean)
            fam_clean = _re.sub(r"\s+", " ", fam_clean).strip()
            fam_norm = fam_clean.replace("@", "-").replace("_", "-").replace("/", "-")
            fam_norm = _re.sub(r"-+", "-", fam_norm).strip("-")
            if not param:
                m = _re.search(r"-(\d{1,3})b$", fam_norm)
                if m:
                    param = f"{m.group(1)}B"
                    fam_norm = fam_norm[: -len(m.group(0))]
            if not fam_norm or fam_norm == "model":
                sid = (
                    getattr(e, "served_model_id", None)
                    or getattr(e, "name", None)
                    or "model"
                )
                sid_l = str(sid).lower()
                base = sid_l.split(":", 1)[0]
                base = base.replace("@", "-").replace("_", "-").replace("/", "-")
                base = _re.sub(r"-+", "-", base).strip("-")
                for pat in rm_patterns:
                    base = _re.sub(pat, " ", base)
                base = _re.sub(r"\s+", " ", base).strip().replace(" ", "-")
                fam_norm = base or "model"
            if param and param.lower() not in fam_norm:
                base = f"{fam_norm} {param.lower()}"
            else:
                base = fam_norm
            base_clean = base.strip().strip("-")
            proposed_disp = f"{base_clean} ({q_label})" if q_label else base_clean
            proposed_slug = proposed_disp.replace(" ", "_")
            # Guard test/demo entries with unique prefix to avoid collisions with real models
            try:
                from imageworks.model_loader.testing_filters import (
                    is_testing_entry as _is_test,
                )

                if _is_test(getattr(e, "name", ""), e):
                    guard = "__TESTZZZ__"
                    proposed_slug = f"{guard}{proposed_slug}"
                    proposed_disp = f"{guard}{proposed_disp}"
            except Exception:
                pass
            return proposed_slug, proposed_disp

        # Build mapping
        candidates: List[dict] = []
        for name, e in reg.items():
            if e.backend == "ollama" and not include_ollama:
                continue
            if e.backend != "ollama" and not include_hf:
                continue
            new_base, new_disp = _propose_name(e)
            candidates.append(
                {
                    "old_name": e.name,
                    "backend": e.backend,
                    "format": e.download_format,
                    "quant": e.quantization,
                    "proposed_base": new_base,
                    "proposed_display": new_disp,
                }
            )

        # Detect collisions by proposed_base
        groups: Dict[str, List[dict]] = {}
        for c in candidates:
            groups.setdefault(c["proposed_base"], []).append(c)

        # Apply disambiguation
        disambiguate = (disambiguate or "backend").lower()
        valid_strategies = {"backend", "format", "none"}
        if disambiguate not in valid_strategies:
            rprint(f"❌ [red]Invalid disambiguation strategy: {disambiguate}[/red]")
            raise typer.Exit(code=1)

        final_map: Dict[str, dict] = {}
        skipped: List[dict] = []
        for base, items in groups.items():
            if len(items) == 1:
                final_map[items[0]["old_name"]] = {
                    **items[0],
                    "final_slug": items[0]["proposed_base"],
                    "final_display": items[0]["proposed_display"],
                    "collision": False,
                }
                continue
            # Collision
            if disambiguate == "none":
                if allow_skip_on_collision:
                    skipped.extend(items)
                    continue
                else:
                    rprint(
                        f"❌ [red]Collision for base '{base}' with --no-skip set: {[i['old_name'] for i in items]}[/red]"
                    )
                    raise typer.Exit(code=1)
            for it in items:
                token = (
                    (it.get("backend") or "").strip().lower()
                    if disambiguate == "backend"
                    else (it.get("format") or "").strip().lower() or "fmt"
                )
                # slug suffix uses double underscore separator for clarity
                final_slug = (
                    f"{it['proposed_base']}__{token}" if token else it["proposed_base"]
                )
                # display: minimalist, append [backend] or [fmt] only on collision
                suffix_human = token.upper() if token else ""
                final_display = (
                    f"{it['proposed_display']} [{suffix_human}]"
                    if suffix_human
                    else it["proposed_display"]
                )
                final_map[it["old_name"]] = {
                    **it,
                    "final_slug": final_slug,
                    "final_display": final_display,
                    "collision": True,
                }

        # Validate against existing names if renaming
        existing_names: Set[str] = set(reg.keys())
        rename_ops: List[dict] = []
        for old, spec in final_map.items():
            e = reg.get(old)
            # Limit to tests when requested
            if tests_only:
                try:
                    from imageworks.model_loader.testing_filters import (
                        is_testing_entry as _is_test,
                    )

                    if not _is_test(old, e):
                        continue
                except Exception:
                    # If filter unavailable, skip restricting to avoid accidental mass rename
                    continue
            new_name = spec["final_slug"] if rename_slugs else old
            rename_ops.append(
                {
                    "old": old,
                    "new": new_name,
                    "display": spec["final_display"],
                }
            )

        # Ensure no duplicate target names
        targets = [op["new"] for op in rename_ops]
        dup_targets: Set[str] = set()
        seen: Set[str] = set()
        for t in targets:
            if t in seen:
                dup_targets.add(t)
            seen.add(t)
        if dup_targets:
            rprint(f"❌ [red]Would create duplicate names: {sorted(dup_targets)}[/red]")
            raise typer.Exit(code=1)

        # Ensure target names don't conflict with unrelated existing entries when renaming
        if rename_slugs:
            conflicts = [
                op
                for op in rename_ops
                if op["new"] != op["old"] and op["new"] in existing_names
            ]
            if conflicts:
                names = ", ".join(c["new"] for c in conflicts[:10])
                rprint(
                    f"❌ [red]Target names already exist in registry (first 10): {names}[/red]"
                )
                raise typer.Exit(code=1)

        # Output preview table or JSON
        if json_output:
            print(
                json.dumps(
                    {
                        "operations": rename_ops,
                        "skipped": skipped,
                        "rename_slugs": rename_slugs,
                    },
                    indent=2,
                )
            )
            if dry_run:
                return
        else:
            t = Table(title="Apply Simplified Slugs")
            t.add_column("Old Name", style="cyan")
            t.add_column("New Name", style="yellow")
            t.add_column("New Display", style="white")
            for op in sorted(rename_ops, key=lambda x: x["old"]):
                t.add_row(op["old"], op["new"], op["display"])
            console.print(t)
            if skipped:
                rprint(
                    f"⚠️  Skipped {len(skipped)} colliding entries (use --disambiguate to resolve or --no-skip to fail)"
                )
            if dry_run:
                rprint(
                    "🛈 [cyan]Dry run: no changes written. Re-run with --apply to persist.[/cyan]"
                )
                return

        # Apply changes
        # Backup merged snapshot before mutation
        if backup:
            from datetime import datetime as _dt

            reg_path = Path("configs/model_registry.json")
            if reg_path.exists():
                ts = _dt.utcnow().strftime("%Y%m%d-%H%M%S")
                backup_path = reg_path.with_name(f"model_registry.{ts}.bak.json")
                try:
                    backup_path.write_text(
                        reg_path.read_text(encoding="utf-8"), encoding="utf-8"
                    )
                    rprint(f"💾 Backup written: {backup_path.name}")
                except Exception as exc:  # noqa: BLE001
                    rprint(f"⚠️  [yellow]Backup failed (continuing): {exc}[/yellow]")

        # Mutate in-memory registry cache
        # Update display names first
        for op in rename_ops:
            e = reg.get(op["old"])
            if e:
                e.display_name = op["display"]
        # Apply renames if requested
        if rename_slugs:
            # We must rebuild the dict to avoid key mutation issues during iteration
            new_reg: Dict[str, Any] = {}
            for name, e in reg.items():
                op = next((o for o in rename_ops if o["old"] == name), None)
                if op and op["new"] != name:
                    e.name = op["new"]
                    new_reg[op["new"]] = e
                else:
                    new_reg[name] = e
            # Replace cache content in-place
            reg.clear()
            reg.update(new_reg)

        # Persist
        unified_registry.save_registry()
        rprint(
            f"✅ [green]Applied {len(rename_ops)} display update(s){' with slug renames' if rename_slugs else ''}[/green]"
        )
    except Exception as exc:  # noqa: BLE001
        rprint(f"❌ [red]Apply failed:[/red] {exc}")
        raise typer.Exit(code=1)


@app.command("scan")
def scan_existing(
    base: Path = typer.Option(
        Path("~/ai-models/weights").expanduser(),
        help="Base directory to scan recursively for HF-style repos",
    ),
    backend: str = typer.Option(
        "vllm", help="Backend to assign for imported entries (can edit later)"
    ),
    location: str = typer.Option(
        "linux_wsl", help="Location label to record (linux_wsl/windows_lmstudio/custom)"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Report what would be imported without writing"
    ),
    format_hint: Optional[str] = typer.Option(
        None,
        "--format",
        "-f",
        help="Fallback default format ONLY when auto-detection fails (e.g. awq, gguf, fp16)",
    ),
    update_existing: bool = typer.Option(
        False,
        "--update-existing",
        help="If variant already exists, update its format/quant & path instead of skipping",
    ),
    include_testing: bool = typer.Option(
        False,
        "--include-testing",
        help="Include testing/demo placeholder models during import (off by default)",
    ),
):
    """Scan an existing weights directory and import discovered model folders into the unified registry.

    Heuristics:
      - Assumes layout: <base>/<owner>/<repo>(@branch)?
      - Derives huggingface id from directory names.
      - Infers format from existing files if possible (gguf, awq, fp16 via safetensors), otherwise uses --format or leaves blank.
      - Computes size and directory checksum.
    """
    try:
        # Honor include-testing by setting adapter env flag
        if include_testing:
            import os as _os

            _os.environ["IMAGEWORKS_IMPORT_INCLUDE_TESTING"] = "1"
        if not base.exists():
            rprint(f"❌ [red]Base path does not exist:[/red] {base}")
            raise typer.Exit(code=1)
        owners = [d for d in base.iterdir() if d.is_dir()]
        planned = []
        for owner_dir in owners:
            for repo_dir in owner_dir.iterdir():
                if not repo_dir.is_dir():
                    continue
                # quick skip if empty
                if not any(repo_dir.iterdir()):
                    continue
                owner = owner_dir.name
                repo = repo_dir.name
                hf_id = f"{owner}/{repo.split('@')[0]}"  # branch simplified
                # Collect files & names once
                files = list(repo_dir.rglob("*"))
                # Use shared detection
                fmt, quant = detect_format_and_quant(repo_dir)
                if fmt is None and format_hint:
                    fmt = format_hint  # fallback semantics
                size_bytes = sum(p.stat().st_size for p in files if p.is_file())
                planned.append(
                    {
                        "hf_id": hf_id,
                        "path": str(repo_dir),
                        "format": fmt,
                        "size": size_bytes,
                        "quant": quant,
                    }
                )
        if not planned:
            rprint("📭 [yellow]No candidate repositories found[/yellow]")
            return

        # Display summary table
        # Reuse same semantics as list: Fmt = container (GGUF / SAFETENSORS / '-') ; Quant = precision/quant scheme
        def _infer_container_from_planned(pr):
            raw_fmt = (pr["format"] or "").lower() if pr.get("format") else ""
            if raw_fmt == "gguf":
                return "GGUF"
            # If format detection returned fp16/awq/gptq treat as quantization/precision, not container.
            # Infer safetensors container by scanning files.
            p = Path(pr["path"]).expanduser()
            try:
                for f in p.rglob("*"):
                    if f.is_file():
                        n = f.name.lower()
                        if n.endswith(".gguf"):
                            return "GGUF"
                        if n.endswith(".safetensors"):
                            return "SAFETENSORS"
            except Exception:  # noqa: BLE001
                pass
            return "-"

        def _infer_quant_from_planned(pr):
            raw_fmt = (pr["format"] or "").lower() if pr.get("format") else ""
            quant = pr.get("quant")
            if raw_fmt in {"fp16", "fp32", "bf16"} and not quant:
                return raw_fmt
            if raw_fmt in {"awq", "gptq"} and not quant:
                return raw_fmt
            return quant or "-"

        t = Table(title="Scan Results (Planned Imports)")
        t.add_column("HF ID", style="cyan")
        t.add_column("Fmt", style="magenta")
        t.add_column("Quant", style="yellow")
        t.add_column("Size", justify="right")
        t.add_column("Path", style="green")
        for pr in planned:
            container = _infer_container_from_planned(pr)
            quant_disp = _infer_quant_from_planned(pr)
            if quant_disp and quant_disp != "-":
                quant_disp = quant_disp.lower()
            t.add_row(
                pr["hf_id"], container, quant_disp, _format_size(pr["size"]), pr["path"]
            )
        console.print(t)
        if dry_run:
            rprint("🛈 [cyan]Dry run: no changes written[/cyan]")
            return
        imported = 0
        skipped = 0
        # Load registry for update-existing logic (side effect ensures cache ready)
        unified_registry.load_registry(force=True)
        for r in planned:
            # Derive existing variant name (approx) the same way record_download will
            # by family + backend + format + quant — we can't know family normalization exactly without adapter
            # so rely on record_download to unify; if update_existing is False we just call it.
            if update_existing:
                # If an entry with same family-backend-format-quant already exists we still call record_download (it updates)
                pass
            try:
                record_download(
                    hf_id=r["hf_id"],
                    backend=backend,
                    format_type=r["format"],
                    quantization=r.get("quant"),
                    path=r["path"],
                    location=location,
                    files=None,
                    size_bytes=r["size"],
                    source_provider="hf",
                    roles=None,
                    role_priority=None,
                )
                imported += 1
            except ImportSkipped as _skip:
                skipped += 1
        msg = f"✅ [green]Imported {imported} repositories into registry[/green]"
        if skipped:
            msg += f"  |  ⚠️ Skipped {skipped} testing/demo entries"
        rprint(msg)
    except Exception as e:  # noqa: BLE001
        rprint(f"❌ [red]Scan failed:[/red] {e}")
        raise typer.Exit(code=1)


@app.command("normalize-formats")
def normalize_formats(
    base: Path = typer.Option(
        Path("~/ai-models/weights").expanduser(),
        help="Base directory to scan for confirmation / rebuild",
    ),
    dry_run: bool = typer.Option(
        True, "--dry-run", help="Show proposed changes without writing"
    ),
    apply: bool = typer.Option(
        False, "--apply", help="Apply detected changes (implies not dry-run)"
    ),
    rebuild: bool = typer.Option(
        False,
        "--rebuild",
        help="Also regenerate dynamic download_* fields (size, files, checksum)",
    ),
    prune_missing: bool = typer.Option(
        False,
        "--prune-missing",
        help="Remove entries whose download_path no longer exists (else mark deprecated)",
    ),
    backup: bool = typer.Option(
        True,
        "--backup/--no-backup",
        help="Write timestamped backup before modifying registry",
    ),
):
    """Re-detect format & quantization, optionally rebuild dynamic fields, to keep registry consistent.

    Steps:
      1. Load registry
      2. For each entry with download_path, if path exists re-detect (format, quant)
      3. If --rebuild: also recompute download_files, download_size_bytes, directory checksum
      4. Produce diff table
      5. Apply if requested
    """
    try:
        reg_path = Path("configs/model_registry.json")
        original_registry_text = None
        if reg_path.exists():
            try:
                original_registry_text = reg_path.read_text(encoding="utf-8")
            except Exception:  # noqa: BLE001
                original_registry_text = None
        reg = unified_registry.load_registry(force=True)
    except Exception as exc:  # noqa: BLE001
        rprint(f"❌ [red]Failed to load registry:[/red] {exc}")
        raise typer.Exit(code=1)

    changes = []
    # deprecated list removed (policy: only curated may deprecate)
    from datetime import datetime

    for entry in reg.values():
        path_val = entry.download_path
        if not path_val:
            continue
        p = Path(path_val).expanduser()
        if not p.exists():
            if prune_missing:
                changes.append({"name": entry.name, "action": "prune-missing"})
            # else: leave untouched (no auto deprecate)
            continue
        det_fmt, det_quant = detect_format_and_quant(p)
        fmt_old = entry.download_format
        quant_old = entry.quantization
        # gather rebuild info if requested
        rebuilt_meta = {}
        if rebuild:
            files = [f for f in p.rglob("*") if f.is_file()]
            file_names = [f.name for f in files]
            size = sum(f.stat().st_size for f in files)
            # simple checksum reuse of existing helper
            checksum = compute_directory_checksum(p)
            rebuilt_meta = {
                "download_files": file_names,
                "download_size_bytes": size,
                "download_directory_checksum": checksum,
            }
        diffs = {}
        if det_fmt and det_fmt != fmt_old:
            diffs["download_format"] = {"old": fmt_old, "new": det_fmt}
        if det_quant and det_quant != quant_old:
            diffs["quantization"] = {"old": quant_old, "new": det_quant}
        if rebuild and rebuilt_meta:
            # Compare size & checksum only for change reporting
            if entry.download_size_bytes != rebuilt_meta["download_size_bytes"]:
                diffs["download_size_bytes"] = {
                    "old": entry.download_size_bytes,
                    "new": rebuilt_meta["download_size_bytes"],
                }
            if (
                entry.download_directory_checksum
                != rebuilt_meta["download_directory_checksum"]
            ):
                diffs["download_directory_checksum"] = {
                    "old": entry.download_directory_checksum,
                    "new": rebuilt_meta["download_directory_checksum"],
                }
            # Always refresh file list length diff if changed
            if entry.download_files != rebuilt_meta["download_files"]:
                diffs["download_files_count"] = {
                    "old": len(entry.download_files or []),
                    "new": len(rebuilt_meta["download_files"]),
                }
        if diffs:
            changes.append(
                {
                    "name": entry.name,
                    "path": path_val,
                    "diffs": diffs,
                    "rebuild_meta": rebuilt_meta,
                    "det_fmt": det_fmt,
                    "det_quant": det_quant,
                }
            )
    # No automatic deprecation injection

    if not changes:
        rprint("✅ [green]No changes detected[/green]")
        return

    # Present summary
    table = Table(title="Normalization / Rebuild Preview")
    table.add_column("Name", style="cyan")
    table.add_column("Action", style="magenta")
    table.add_column("Changes", style="yellow")
    for c in changes:
        action = c.get("action", "update")
        if action != "update":
            table.add_row(c["name"], action, "-")
            continue
        diff_parts = []
        for k, d in c["diffs"].items():
            diff_parts.append(f"{k}:{d['old']}→{d['new']}")
        table.add_row(c["name"], action, ", ".join(diff_parts))
    console.print(table)

    if dry_run and not apply:
        # Dry-run: discard any in-memory mutations by reloading registry cache from disk snapshot
        try:
            from imageworks.model_loader import registry as _regmod

            # If any entries were marked deprecated in memory, revert by clearing cache
            _regmod._REGISTRY_CACHE = None  # type: ignore[attr-defined]
        except Exception:
            pass
        # Restore file content precisely (defensive) if changed
        if original_registry_text is not None:
            try:
                current_text = (
                    reg_path.read_text(encoding="utf-8") if reg_path.exists() else None
                )
                if current_text != original_registry_text:
                    reg_path.write_text(original_registry_text, encoding="utf-8")
            except Exception as exc:  # noqa: BLE001
                rprint(f"⚠️  [yellow]Dry-run restore failed (non-fatal): {exc}[/yellow]")
        rprint("🛈 [cyan]Dry run (no changes written). Use --apply to persist.[/cyan]")
        return

    # Apply changes
    if backup:
        ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        reg_path = Path("configs/model_registry.json")
        backup_path = reg_path.with_name(f"model_registry.{ts}.bak.json")
        try:
            backup_path.write_text(
                reg_path.read_text(encoding="utf-8"), encoding="utf-8"
            )
            rprint(f"💾 Backup written: {backup_path}")
        except Exception as exc:  # noqa: BLE001
            rprint(f"⚠️  [yellow]Backup failed (continuing): {exc}[/yellow]")

    # Mutate entries in memory
    # We mutate the global registry cache directly (reg is the dict returned by load_registry)
    # For pruning we need to delete keys from that dict; for updates we edit entry objects in place.
    for c in changes:
        action = c.get("action", "update")
        name = c["name"]
        if action == "prune-missing":
            if name in reg:
                del reg[name]
            continue
        e = reg.get(name)
        if not e:
            continue
        if action == "mark-deprecated":
            e.deprecated = True
            continue
        diffs = c["diffs"]
        if "download_format" in diffs:
            e.download_format = diffs["download_format"]["new"]
        if "quantization" in diffs:
            e.quantization = diffs["quantization"]["new"]
        rebuilt_meta = c.get("rebuild_meta") or {}
        for k in (
            "download_files",
            "download_size_bytes",
            "download_directory_checksum",
        ):
            if k in rebuilt_meta:
                setattr(e, k, rebuilt_meta[k])
    # Persist via registry utility (uses internal cache)
    unified_registry.save_registry()
    rprint(f"✅ [green]Applied {len(changes)} changes to registry[/green]")


@app.command("migrate-architecture")
def migrate_architecture(
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Preview metadata updates without writing registry"
    ),
    limit: Optional[int] = typer.Option(
        None, "--limit", help="Process at most N downloaded models"
    ),
) -> None:
    """Backfill architecture metadata from downloaded model assets."""

    entries = unified_list_downloads()
    if not entries:
        rprint("⚠️ [yellow]No downloaded models found.[/yellow]")
        return

    processed = 0
    updated: List[RegistryEntry] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task(
            "Collecting architecture metadata…", total=len(entries)
        )
        for entry in entries:
            if limit is not None and processed >= limit:
                break
            processed += 1
            download_path = getattr(entry, "download_path", None)
            if not download_path:
                progress.advance(task)
                continue
            result = collect_architecture_metadata(
                Path(download_path).expanduser(),
                raw_path=download_path,
                served_model_id=getattr(entry, "served_model_id", None),
            )
            if dry_run:
                summary = {
                    key: result.fields.get(key)
                    for key in (
                        "num_layers",
                        "num_attention_heads",
                        "num_kv_heads",
                        "hidden_size",
                        "context_length",
                    )
                    if result.fields.get(key) is not None
                }
                rprint(
                    f"• [cyan]{entry.name}[/cyan] -> "
                    f"{summary or 'no metadata detected'}"
                )
            else:
                entry.metadata = entry.metadata or {}
                merged = merge_architecture_metadata(
                    entry.metadata.get("architecture"), result
                )
                entry.metadata["architecture"] = merged
                updated.append(entry)
            progress.advance(task)

    if dry_run:
        rprint(
            f"ℹ️ [blue]Processed {processed} models (dry-run only, no writes).[/blue]"
        )
        return

    if not updated:
        rprint("ℹ️ [blue]No metadata changes were required.[/blue]")
        return

    unified_registry.update_entries(updated, save=True)
    rprint(
        f"✅ [green]Architecture metadata updated for {len(updated)} model(s).[/green]"
    )


@app.command("reconcile-architecture")
def reconcile_architecture(
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Preview registry updates sourced from runtime loader logs",
    )
) -> None:
    """Merge runtime loader observations into registry architecture metadata."""

    events = load_runtime_events()
    if not events:
        rprint(
            "⚠️ [yellow]No runtime metrics found at logs/model_loader_metrics.jsonl[/yellow]"
        )
        return

    registry = _load_unified_registry(force=True)
    updated: List[RegistryEntry] = []
    for model_name, entry in registry.items():
        event = events.get(model_name)
        if not event:
            continue
        architecture_meta = dict((entry.metadata or {}).get("architecture") or {})
        merged = merge_runtime_payload(
            architecture_meta=architecture_meta,
            runtime_payload=event.payload,
            timestamp=event.timestamp,
        )
        if dry_run:
            rprint(
                f"• [cyan]{model_name}[/cyan] metrics: "
                f"{event.payload.get('metrics', {})}"
            )
            continue
        entry.metadata = entry.metadata or {}
        entry.metadata["architecture"] = merged
        updated.append(entry)

    if dry_run:
        rprint(
            f"ℹ️ [blue]Identified runtime metadata for {len(updated)} model(s) "
            "(dry-run only).[/blue]"
        )
        return

    if not updated:
        rprint("ℹ️ [blue]No registry entries matched runtime metrics log.[/blue]")
        return

    unified_registry.update_entries(updated, save=True)
    rprint(
        f"✅ [green]Reconciled runtime metadata for {len(updated)} model(s).[/green]"
    )


@app.command("tidy-empty-dirs")
def tidy_empty_dirs(
    weights_root: Path = typer.Option(
        Path("~/ai-models/weights").expanduser(),
        "--weights-root",
        help="Root path containing <owner>/<repo> model folders",
    ),
    dry_run: bool = typer.Option(
        True, "--dry-run/--apply", help="Preview only by default; --apply to delete"
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Show per-folder actions"),
):
    """Remove empty repo directories and their owner directories if those are empty.

    This is a safe, one-off tidy for HF-style trees: <root>/<owner>/<repo>.
    Only deletes directories that are empty at the time of the check.
    """
    try:
        root = weights_root.expanduser()
        if not root.exists() or not root.is_dir():
            rprint(f"❌ [red]Weights root not found or not a directory:[/red] {root}")
            raise typer.Exit(code=1)
        owners = [d for d in root.iterdir() if d.is_dir()]
        empty_repos: list[Path] = []
        for owner in owners:
            for repo in [r for r in owner.iterdir() if r.is_dir()]:
                try:
                    next(repo.iterdir())
                except StopIteration:
                    empty_repos.append(repo)
                except Exception:
                    pass
        if not empty_repos:
            rprint("🛈 [cyan]No empty repo directories found[/cyan]")
            return
        # Present summary
        t = Table(title="Empty Repo Directories (owner/repo)")
        t.add_column("Owner", style="green")
        t.add_column("Repo", style="cyan")
        t.add_column("Path", style="white")
        for repo in sorted(
            empty_repos, key=lambda p: (p.parent.name.lower(), p.name.lower())
        ):
            t.add_row(repo.parent.name, repo.name, str(repo))
        console.print(t)
        rprint(f"📊 Candidates: {len(empty_repos)} under {root}")

        if dry_run:
            rprint(
                "🛈 [cyan]Dry run: no changes written. Re-run with --apply to delete.[/cyan]"
            )
            return

        # Apply deletions
        removed_repos = 0
        removed_owners = 0
        touched_owners: set[Path] = set()
        for repo in empty_repos:
            try:
                repo.rmdir()
                removed_repos += 1
                touched_owners.add(repo.parent)
                if verbose:
                    rprint(f"🗑️  removed repo: {repo}")
            except Exception as exc:  # noqa: BLE001
                rprint(f"⚠️  [yellow]Failed to remove repo {repo}: {exc}[/yellow]")
        # After repo removals, attempt owner pruning if empty
        for owner in sorted(touched_owners):
            try:
                next(owner.iterdir())
            except StopIteration:
                try:
                    owner.rmdir()
                    removed_owners += 1
                    if verbose:
                        rprint(f"🗑️  removed owner: {owner}")
                except Exception as exc:  # noqa: BLE001
                    rprint(f"⚠️  [yellow]Failed to remove owner {owner}: {exc}[/yellow]")
            except Exception:
                pass
        rprint(
            f"✅ [green]Tidy complete[/green] — repos removed: {removed_repos}, owners removed: {removed_owners}"
        )
    except Exception as exc:  # noqa: BLE001
        rprint(f"❌ [red]Tidy failed:[/red] {exc}")
        raise typer.Exit(code=1)


@app.command("purge-tests")
def purge_tests(
    dry_run: bool = typer.Option(
        True, "--dry-run/--apply", help="Preview only by default; --apply to delete"
    ),
    backup: bool = typer.Option(
        True, "--backup/--no-backup", help="Write registry backups before modifying"
    ),
    list_only: bool = typer.Option(
        False, "--list-only", help="Only list test entries detected; no changes"
    ),
):
    """Remove or list test/demo entries detected by testing filters.

    - With --list-only: prints names of test entries.
    - With --apply: deletes matching entries from the registry (layered-safe with backups).
    """
    try:
        reg = unified_registry.load_registry(force=True)
        from imageworks.model_loader.testing_filters import is_testing_entry as _is_test

        tests = [name for name, e in reg.items() if _is_test(name, e)]
        if not tests:
            rprint("🛈 [cyan]No test/demo entries found[/cyan]")
            return
        if list_only or dry_run:
            rprint("Test/demo entries detected:")
            for n in sorted(tests):
                rprint(f"  • {n}")
            if dry_run and not list_only:
                rprint("🛈 [cyan]Dry run: not modifying registry[/cyan]")
            return
        # Apply purge
        if backup:
            from datetime import datetime as _dt

            merged = Path("configs/model_registry.json")
            if merged.exists():
                ts = _dt.utcnow().strftime("%Y%m%d-%H%M%S")
                b = merged.with_name(f"model_registry.{ts}.bak.json")
                b.write_text(merged.read_text(encoding="utf-8"), encoding="utf-8")
                rprint(f"💾 Backup written: {b.name}")
        from imageworks.model_loader.registry import remove_entry

        removed = 0
        for n in tests:
            if remove_entry(n, save=False):
                removed += 1
        unified_registry.save_registry()
        rprint(f"✅ [green]Purged {removed} test/demo entries[/green]")
    except Exception as exc:  # noqa: BLE001
        rprint(f"❌ [red]Purge tests failed:[/red] {exc}")
        raise typer.Exit(code=1)


@app.command("prune-no-chat-template")
def prune_no_chat_template(
    backend: str = typer.Option(
        "vllm", "--backend", "-b", help="Backend to target (default: vllm)"
    ),
    weights_root: Path = typer.Option(
        Path("~/ai-models/weights").expanduser(),
        "--weights-root",
        help="Only delete entries whose download_path is under this directory",
    ),
    dry_run: bool = typer.Option(
        True, "--dry-run/--apply", help="Dry run by default; use --apply to delete"
    ),
    force: bool = typer.Option(
        False, "--force", help="Skip confirmation when applying"
    ),
):
    """Delete locally installed models that lack a chat template.

    Criteria:
      - entry.backend matches --backend
      - entry.download_path exists under --weights-root
      - No external .jinja/chat_template* files AND no tokenizer_config.json with 'chat_template'

    Action:
      - Delete the on-disk directory
      - Remove the entry from the unified registry (purge)
    """
    try:
        root = weights_root.expanduser().resolve()
        entries = unified_list_downloads(only_installed=False)

        def _has_chat_template(path: Path) -> tuple[bool, list[str], bool]:
            has_external = False
            external_candidates: list[str] = []
            has_embedded = False
            try:
                for child in path.iterdir():
                    if not child.is_file():
                        continue
                    n = child.name.lower()
                    if (
                        n.endswith(".jinja")
                        or "chat_template" in n
                        or n.startswith("template")
                        or n.endswith("template")
                    ):
                        try:
                            head = child.read_text(encoding="utf-8", errors="ignore")[
                                :2000
                            ]
                            if "{{" in head and "}}" in head:
                                has_external = True
                                external_candidates.append(child.name)
                        except Exception:
                            pass
                tcfg = path / "tokenizer_config.json"
                if tcfg.exists():
                    try:
                        txt = tcfg.read_text(encoding="utf-8", errors="ignore")
                        if "chat_template" in txt:
                            has_embedded = True
                    except Exception:
                        pass
            except Exception:
                pass
            return (has_external or has_embedded, external_candidates, has_embedded)

        targets: list[dict] = []
        for e in entries:
            if e.backend != backend:
                continue
            dp = getattr(e, "download_path", None)
            if not dp:
                continue
            p = Path(dp).expanduser()
            if not p.exists() or not p.is_dir():
                continue
            try:
                resolved = p.resolve()
            except Exception:
                continue
            try:
                if not (resolved == root or root in resolved.parents):
                    continue
            except Exception:
                continue
            has_any, externals, embedded = _has_chat_template(p)
            if not has_any:
                targets.append(
                    {
                        "name": e.name,
                        "path": str(p),
                        "backend": e.backend,
                    }
                )

        if not targets:
            rprint("🛈 [cyan]No matching models without chat template found[/cyan]")
            return

        # Present summary
        t = Table(title="Prune Models Without Chat Templates")
        t.add_column("Name", style="cyan")
        t.add_column("Backend", style="green")
        t.add_column("Path", style="white")
        for it in targets:
            t.add_row(it["name"], it["backend"], it["path"])
        console.print(t)
        rprint(f"📊 Candidates: {len(targets)} under {root}")

        if dry_run:
            rprint(
                "🛈 [cyan]Dry run: no changes written. Re-run with --apply to delete.[/cyan]"
            )
            return
        if not force:
            if not typer.confirm(
                f"Delete {len(targets)} model folder(s) from disk and registry?"
            ):
                rprint("❌ [yellow]Cancelled[/yellow]")
                return

        # Apply deletions
        removed = 0
        errors = 0
        for it in targets:
            try:
                # Delete from disk
                try:
                    import shutil

                    target = Path(it["path"]).expanduser()
                    shutil.rmtree(target, ignore_errors=False)
                except Exception as exc:  # noqa: BLE001
                    rprint(
                        f"⚠️  [yellow]Failed to delete files for {it['name']}: {exc}[/yellow]"
                    )
                else:
                    try:
                        _prune_empty_repo_and_owner_dirs(target)
                    except Exception:
                        pass
                # Remove from registry (purge)
                ok = unified_remove_download(it["name"], keep_entry=False)
                if ok:
                    removed += 1
                else:
                    errors += 1
            except Exception as exc:  # noqa: BLE001
                errors += 1
                rprint(f"❌ [red]Failed to remove {it['name']}: {exc}[/red]")
        # Persist registry after batch
        try:
            unified_registry.save_registry()
        except Exception:
            pass
        rprint(
            f"✅ [green]Removed {removed} models[/green]{'  |  ⚠️ errors: ' + str(errors) if errors else ''}"
        )
    except Exception as exc:  # noqa: BLE001
        rprint(f"❌ [red]Prune failed:[/red] {exc}")
        raise typer.Exit(code=1)


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
