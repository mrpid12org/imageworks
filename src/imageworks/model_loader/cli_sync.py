"""CLI utilities for syncing downloader registry into unified model registry."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

import typer

from .registry import load_registry, update_entries, save_registry
from .models import (
    RegistryEntry,
    BackendConfig,
    Artifacts,
    ChatTemplate,
    VersionLock,
    PerformanceSummary,
    Probes,
)

from .hashing import compute_artifact_hashes, hash_file

app = typer.Typer(help="Model registry maintenance commands")


def _ensure_defaults(entry: RegistryEntry) -> RegistryEntry:
    # Guarantee minimal required nested structures exist (already in dataclass but defensive if partial construction).
    return entry


@app.command("sync-downloader")
def sync_downloader(
    models_json: Path = typer.Argument(..., exists=True, readable=True),
    registry_path: Path = typer.Option(
        Path("configs/model_registry.json"), help="Path to unified registry file"
    ),
    dry_run: bool = typer.Option(
        False, help="Do not write changes; just report planned mutations"
    ),
    include_file_hashes: bool = typer.Option(
        True,
        help="If true, compute sha256 for each source file under model path (may be slower)",
    ),
):
    """Merge entries from a downloader models.json into the unified registry.

    Creates or enriches existing entries with 'source' attributes.
    Does not overwrite backend_config unless missing.
    """
    # For explicit registry_path outside default merged snapshot, enforce single-file mode
    import os as _os

    if registry_path != Path("configs/model_registry.json"):
        _os.environ.setdefault("IMAGEWORKS_REGISTRY_NO_LAYERING", "1")
    registry = load_registry(registry_path, force=True)
    downloader_data: Dict[str, Any] = json.loads(models_json.read_text())

    created = 0
    updated = 0
    new_entries = []

    for key, raw in downloader_data.items():
        hf_id = raw.get("model_name")
        model_path = raw.get("path")
        size_bytes = raw.get("size_bytes")
        fmt = raw.get("format_type")
        directory_checksum = raw.get("checksum")
        files_list = raw.get("files", [])
        source_files: list[dict[str, Any]] = []
        root_path = Path(model_path).expanduser() if model_path else None
        for f in files_list:
            entry_file: dict[str, Any] = {"path": f, "size": None, "sha256": None}
            if root_path is not None:
                candidate = root_path / f
                if candidate.exists():
                    try:
                        entry_file["size"] = candidate.stat().st_size
                        if include_file_hashes and candidate.is_file():
                            entry_file["sha256"] = hash_file(candidate)
                    except OSError:
                        pass
            source_files.append(entry_file)

        source_block = {
            "huggingface_id": hf_id,
            "format": fmt,
            "path": model_path,
            "size_bytes": size_bytes,
            "directory_checksum": directory_checksum,
            "files": source_files,
        }

        logical_name = hf_id.split("/")[-1].lower().replace("@", "-") if hf_id else key
        existing = registry.get(logical_name)
        if existing:
            # Enrich existing
            modified = existing
            if not existing.source:
                modified.source = source_block
            else:
                # Merge non-destructively
                for k2, v2 in source_block.items():
                    modified.source.setdefault(k2, v2)
            new_entries.append(modified)
            updated += 1
        else:
            # Determine backend and port
            backend = "vllm"
            port = 8000
            # If format is gguf or model_path looks like ollama, use ollama
            if (fmt and fmt.lower() == "gguf") or (
                model_path and model_path.startswith("ollama:")
            ):
                backend = "ollama"
                port = 11434
            entry = RegistryEntry(
                name=logical_name,
                display_name=hf_id.split("/")[-1] if hf_id else logical_name,
                backend=backend,
                backend_config=BackendConfig(
                    port=port, model_path=model_path or "", extra_args=[]
                ),
                capabilities={
                    "text": True,
                    "vision": False,
                    "audio": False,
                    "embedding": False,
                },
                artifacts=Artifacts(aggregate_sha256="", files=[]),
                chat_template=ChatTemplate(source="embedded", path=None, sha256=None),
                version_lock=VersionLock(
                    locked=False, expected_aggregate_sha256=None, last_verified=None
                ),
                performance=PerformanceSummary(
                    rolling_samples=0,
                    ttft_ms_avg=None,
                    throughput_toks_per_s_avg=None,
                    last_sample=None,
                ),
                probes=Probes(vision=None),
                profiles_placeholder=None,
                metadata={"notes": "Imported from downloader"},
                served_model_id=None,
                model_aliases=[hf_id] if hf_id else [],
                roles=[],
                license=None,
                source=source_block,
                deprecated=False,
            )
            entry = compute_artifact_hashes(entry)
            new_entries.append(entry)
            created += 1

    if dry_run:
        typer.echo(
            f"(dry-run) Would create {created} entries, update {updated} entries"
        )
        return

    update_entries(new_entries, save=True)
    save_registry(registry_path)
    typer.echo(
        f"Created {created} entries, updated {updated} entries → {registry_path}"
    )


@app.command("verify")
def verify(
    name: str = typer.Argument(..., help="Logical model name to verify"),
    lock: bool = typer.Option(
        False,
        help="If set, lock version to current aggregate hash when verification passes",
    ),
    registry_path: Path = typer.Option(
        Path("configs/model_registry.json"), help="Path to unified registry file"
    ),
):
    """Recompute artifact hashes and optionally lock the model version."""
    registry = load_registry(registry_path, force=True)
    entry = registry.get(name)
    if not entry:
        typer.echo(f"Model '{name}' not found")
        raise typer.Exit(code=1)
    before_hash = entry.artifacts.aggregate_sha256
    entry = compute_artifact_hashes(entry)
    after_hash = entry.artifacts.aggregate_sha256
    changed = before_hash != after_hash
    if lock:
        entry.version_lock.locked = True
        entry.version_lock.expected_aggregate_sha256 = after_hash
        entry.version_lock.last_verified = datetime.now(timezone.utc).isoformat()
    update_entries([entry], save=True)
    status = "CHANGED" if changed else "UNCHANGED"
    lock_state = "(locked)" if lock else ""
    typer.echo(f"Verify {name}: {status} aggregate={after_hash} {lock_state}")


def get_typer_app() -> typer.Typer:
    return app


@app.command("regenerate-discovered")
def regenerate_discovered(
    registry_dir: Path = typer.Option(
        Path("configs"),
        help="Base registry directory (contains curated/discovered/merged)",
    ),
    dry_run: bool = typer.Option(
        True,
        "--dry-run/--apply",
        help="Preview changes (default) or apply with --apply",
    ),
    backup: bool = typer.Option(
        True,
        "--backup/--no-backup",
        help="Backup existing discovered file before overwrite",
    ),
):
    """Rebuild the discovered layer from current merged snapshot using classification heuristic.

    Use when the discovered overlay becomes noisy or you want to reclassify legacy entries
    after editing curated content. Curated file is never modified.
    """
    from . import registry as regmod

    base = registry_dir
    curated_file_path = (
        base / "model_registry.curated.json"
    )  # renamed to avoid unused lint; used for existence check
    discovered_file = base / "model_registry.discovered.json"
    merged_snapshot = base / "model_registry.json"

    if not merged_snapshot.exists() and not curated_file_path.exists():
        typer.echo("No merged or curated registry found; nothing to rebuild.")
        raise typer.Exit(code=1)

    # Ensure snapshot exists
    if not merged_snapshot.exists() and curated_file_path.exists():
        regmod.load_registry(force=True)
    try:
        merged_raw = json.loads(merged_snapshot.read_text())
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"Failed to read merged snapshot: {exc}")
        raise typer.Exit(code=1)
    if not isinstance(merged_raw, list):
        typer.echo("Merged snapshot root must be a list")
        raise typer.Exit(code=1)

    curated_raw = []
    if curated_file_path.exists():
        try:
            curated_raw = json.loads(curated_file_path.read_text())
        except Exception:
            curated_raw = []

    curated_new, discovered_new = regmod._classify_legacy(merged_raw)  # type: ignore[attr-defined]
    curated_names = {e.get("name") for e in curated_raw}
    final_discovered = [e for e in discovered_new if e.get("name") not in curated_names]

    # Diff
    def _idx(lst):
        return {d.get("name"): d for d in lst if isinstance(d, dict) and d.get("name")}

    existing_discovered = []
    if discovered_file.exists():
        try:
            existing_discovered = json.loads(discovered_file.read_text())
        except Exception:
            existing_discovered = []
    before_idx = _idx(existing_discovered)
    after_idx = _idx(final_discovered)
    added = sorted(set(after_idx) - set(before_idx))
    removed = sorted(set(before_idx) - set(after_idx))
    retained = sorted(set(before_idx) & set(after_idx))

    typer.echo("Rebuild Plan:")
    typer.echo(f"  Added:   {len(added)}")
    typer.echo(f"  Removed: {len(removed)}")
    typer.echo(f"  Retained:{len(retained)}")
    if dry_run:
        if added:
            typer.echo(
                "  + " + ", ".join(added[:8]) + (" ..." if len(added) > 8 else "")
            )
        if removed:
            typer.echo(
                "  - " + ", ".join(removed[:8]) + (" ..." if len(removed) > 8 else "")
            )
        typer.echo("(dry-run) No files written")
        return

    if backup and discovered_file.exists():
        from datetime import datetime as _dt

        ts = _dt.utcnow().strftime("%Y%m%d-%H%M%S")
        backup_path = discovered_file.with_name(
            f"model_registry.discovered.{ts}.bak.json"
        )
        try:
            backup_path.write_text(discovered_file.read_text())
            typer.echo(f"Backup: {backup_path}")
        except Exception as exc:  # noqa: BLE001
            typer.echo(f"Backup failed (continuing): {exc}")

    discovered_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = discovered_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(final_discovered, indent=2) + "\n")
    tmp.replace(discovered_file)
    regmod.load_registry(force=True)
    typer.echo(f"Regenerated discovered layer with {len(final_discovered)} entries.")


@app.command("discover-local-hf")
def discover_local_hf(
    root: Path = typer.Option(
        Path("~/ai-models/weights").expanduser(),
        help="Root directory containing <owner>/<repo> model folders",
    ),
    registry_dir: Path = typer.Option(Path("configs"), help="Registry base directory"),
    infer_format: bool = typer.Option(
        True,
        "--infer-format/--no-infer-format",
        help="Attempt to infer format & quantization from file names",
    ),
    default_backend: str = typer.Option(
        "vllm", help="Backend to assign if creating new entries"
    ),
):
    """Scan existing HuggingFace-style local directories and register any missing models.

    Assumes directory layout: <root>/<owner>/<repo>[/@branch] containing model weight files.
    Skips directories that already have a registry entry (by logical name owner/repo[@branch]).
    """
    from . import registry as regmod

    # Ensure registry is loaded and capture current entries
    reg = regmod.load_registry(force=True)
    discovered_file = registry_dir / "model_registry.discovered.json"
    # Load raw discovered overlay for append
    try:
        if discovered_file.exists():
            discovered_raw = json.loads(discovered_file.read_text())
        else:
            discovered_raw = []
    except Exception:
        discovered_raw = []

    existing_names = {e.name for e in reg.values()}
    new_entries: list[dict[str, Any]] = []

    def _infer(owner: str, repo: str, path: Path) -> dict[str, Any]:
        fmt = None
        quant = None
        size_bytes = 0
        # Simple heuristics
        if infer_format:
            for f in path.rglob("*"):
                if not f.is_file():
                    continue
                name_l = f.name.lower()
                if name_l.endswith(".gguf"):
                    fmt = fmt or "gguf"
                elif name_l.endswith(".safetensors"):
                    fmt = fmt or "safetensors"
                elif "awq" in name_l:
                    fmt = fmt or "awq"
                if "q4" in name_l:
                    quant = quant or "q4"
                elif "q5" in name_l:
                    quant = quant or "q5"
                elif "q6" in name_l:
                    quant = quant or "q6"
                elif "q8" in name_l:
                    quant = quant or "q8"
                try:
                    size_bytes += f.stat().st_size
                except OSError:
                    pass
        logical = f"{owner}/{repo}".lower()
        entry = {
            "name": logical,
            "display_name": repo,
            "backend": default_backend,
            "backend_config": {"port": 0, "model_path": str(path), "extra_args": []},
            "download_path": str(path),
            "download_format": fmt,
            "quantization": quant,
            "download_size_bytes": size_bytes or None,
            "roles": [],
            "family": None,
            "deprecated": False,
            "served_model_id": None,
            "source": {"huggingface_id": logical, "path": str(path)},
        }
        return entry

    if not root.exists():
        typer.echo(f"Root {root} does not exist")
        raise typer.Exit(code=1)

    # Expect owner/repo two-level directories; collect candidates
    for owner_dir in root.iterdir():
        if not owner_dir.is_dir():
            continue
        owner = owner_dir.name
        for repo_dir in owner_dir.iterdir():
            if not repo_dir.is_dir():
                continue
            repo = repo_dir.name
            logical = f"{owner}/{repo}".lower()
            if logical in existing_names:
                continue
            new_entries.append(_infer(owner, repo, repo_dir))

    if not new_entries:
        typer.echo("No new HF model directories discovered.")
        return

    typer.echo(f"Discovered {len(new_entries)} new local HF model(s)")
    for e in new_entries[:10]:
        typer.echo(f"  + {e['name']} ({e.get('download_format') or '-'})")
    if len(new_entries) > 10:
        typer.echo("  ...")
    # Append to discovered overlay and save
    discovered_raw.extend(new_entries)
    discovered_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = discovered_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(discovered_raw, indent=2) + "\n")
    tmp.replace(discovered_file)
    regmod.load_registry(force=True)
    typer.echo("Saved discovered HF entries and refreshed merged snapshot.")


@app.command("discover-all")
def discover_all(
    hf_root: Path = typer.Option(
        Path("~/ai-models/weights").expanduser(), help="HF root directory"
    ),
    skip_ollama: bool = typer.Option(
        False, "--skip-ollama", help="Do not run Ollama import step"
    ),
):
    """Run both adapter-backed local HuggingFace ingestion and Ollama import."""
    from typer.testing import CliRunner

    runner = CliRunner()
    hf_res = runner.invoke(app, ["ingest-local-hf", f"--root={hf_root}"])
    typer.echo(hf_res.stdout.rstrip())
    if hf_res.exit_code != 0:
        raise typer.Exit(code=hf_res.exit_code)
    if not skip_ollama:
        import subprocess as _sp

        cmd = ["python", "scripts/import_ollama_models.py"]
        try:
            out = _sp.check_output(cmd, stderr=_sp.STDOUT, text=True)
            typer.echo(out.rstrip())
        except _sp.CalledProcessError as exc:
            typer.echo(f"Ollama import failed: {exc.output}")
            raise typer.Exit(code=exc.returncode)
    typer.echo("Combined discovery complete.")


@app.command("ingest-local-hf")
def ingest_local_hf(
    root: Path = typer.Option(
        Path("~/ai-models/weights").expanduser(),
        help="Root directory containing <owner>/<repo> model folders",
    ),
    backend: str = typer.Option("vllm", help="Backend to assign to ingested variants"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Preview without writing to registry"
    ),
):
    """Ingest HF-style local directories using the unified adapter (no owner/repo stubs)."""
    from imageworks.tools.model_downloader.format_utils import detect_format_and_quant
    from .download_adapter import record_download, ImportSkipped
    from . import registry as regmod

    if not root.exists():
        typer.echo(f"Root {root} does not exist")
        raise typer.Exit(code=1)

    candidates: list[tuple[str, str, Path]] = []
    for owner_dir in root.iterdir():
        if not owner_dir.is_dir():
            continue
        owner = owner_dir.name
        for repo_dir in owner_dir.iterdir():
            if not repo_dir.is_dir():
                continue
            repo = repo_dir.name
            candidates.append((owner, repo, repo_dir))

    if not candidates:
        typer.echo("No HF model directories found.")
        return

    created = 0
    updated = 0
    skipped = 0
    for owner, repo, path in candidates:
        hf_id = f"{owner}/{repo}"
        fmt, quant = detect_format_and_quant(path)
        if dry_run:
            typer.echo(
                f"DRY RUN: would ingest {hf_id} as backend={backend} format={fmt or '-'} quant={quant or '-'}"
            )
            continue
        try:
            entry = record_download(
                hf_id=hf_id,
                backend=backend,
                format_type=fmt,
                quantization=quant,
                path=str(path),
                location="linux_wsl",
                files=None,
                size_bytes=None,
                source_provider="hf",
                roles=None,
                role_priority=None,
                family_override=None,
                served_model_id=None,
                extra_metadata=None,
                display_name=None,
            )
            # record_download updates if existing; we differentiate by presence of downloaded_at
            if entry.downloaded_at:
                updated += 1
            else:
                created += 1
        except ImportSkipped:
            skipped += 1
        except Exception as exc:  # noqa: BLE001
            typer.echo(f"Failed to ingest {hf_id}: {exc}")
    regmod.load_registry(force=True)
    typer.echo(
        f"Ingested HF directories: created/updated={created+updated} (created={created}, updated={updated}), skipped={skipped}"
    )


@app.command("purge-imported")
def purge_imported(
    providers: str = typer.Option(
        "all", help="Which imports to purge: hf, ollama, or all"
    ),
    dry_run: bool = typer.Option(True, "--dry-run/--apply", help="Preview or apply"),
    backup: bool = typer.Option(
        True, "--backup/--no-backup", help="Backup discovered layer before write"
    ),
):
    """Remove imported (discovered) entries by provider without touching curated."""
    from . import registry as regmod
    import json as _json
    from datetime import datetime as _dt

    regmod.load_registry(force=True)
    discovered_path = Path("configs/model_registry.discovered.json")
    try:
        discovered_raw = (
            _json.loads(discovered_path.read_text()) if discovered_path.exists() else []
        )
    except Exception:  # noqa: BLE001
        discovered_raw = []

    def _is_target(obj: dict) -> bool:
        backend = obj.get("backend")
        source = obj.get("source") or {}
        provider = (obj.get("source_provider") or source.get("provider") or "").lower()
        dp = obj.get("download_path") or ""
        is_hf = (
            bool(source.get("huggingface_id"))
            or (provider == "hf")
            or (
                backend != "ollama"
                and (obj.get("metadata", {}) or {}).get("created_from_download")
            )
        )
        is_ollama = (
            backend == "ollama"
            or str(dp).startswith("ollama://")
            or provider == "ollama"
        )
        if providers == "all":
            return is_hf or is_ollama
        if providers == "hf":
            return is_hf
        if providers == "ollama":
            return is_ollama
        return False

    to_remove = [o for o in discovered_raw if isinstance(o, dict) and _is_target(o)]
    remain = [o for o in discovered_raw if o not in to_remove]

    typer.echo(
        f"Purge plan: remove={len(to_remove)} keep={len(remain)} providers={providers}"
    )
    if dry_run:
        return
    if backup and discovered_path.exists():
        backup_path = discovered_path.with_name(
            f"model_registry.discovered.{_dt.utcnow().strftime('%Y%m%d-%H%M%S')}.bak.json"
        )
        try:
            backup_path.write_text(discovered_path.read_text())
            typer.echo(f"Backup: {backup_path}")
        except Exception:  # noqa: BLE001
            pass
    discovered_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = discovered_path.with_suffix(".tmp")
    tmp.write_text(_json.dumps(remain, indent=2) + "\n")
    tmp.replace(discovered_path)
    # Remove merged snapshot so reload does not re-adopt stale entries back into discovered
    merged_snapshot = Path("configs/model_registry.json")
    try:
        if merged_snapshot.exists():
            merged_snapshot.unlink()
    except Exception:  # noqa: BLE001
        pass
    regmod.load_registry(force=True)
    typer.echo("Applied purge and refreshed merged snapshot.")


@app.command("rebuild-ollama")
def rebuild_ollama(
    location: str = typer.Option(
        "linux_wsl", "--location", help="Location label for imported entries"
    ),
    show: bool = typer.Option(
        True, "--show/--no-show", help="Show resulting ollama entries after rebuild"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Only show operations without mutating files"
    ),
):
    """Reset discovered Ollama entries then re-import from local Ollama install.

    Steps:
      1. Backup & remove all discovered backend=ollama entries.
      2. Run import_ollama_models.py to repopulate.
      3. (Optional) show resulting entries.
    """
    import json as _json
    from pathlib import Path as _P

    discovered_path = _P("configs/model_registry.discovered.json")
    if not discovered_path.exists():
        typer.echo("No discovered layer file; creating new after import.")
        removed = []
    else:
        try:
            raw = _json.loads(discovered_path.read_text())
        except Exception as exc:  # noqa: BLE001
            typer.echo(f"Failed to parse discovered file: {exc}")
            raise typer.Exit(code=1)
        removed = [e for e in raw if e.get("backend") == "ollama"]
        if dry_run:
            typer.echo(
                f"Would remove {len(removed)} existing discovered ollama entries"
            )
        else:
            if removed:
                from datetime import datetime as _dt

                backup_path = discovered_path.with_name(
                    f"model_registry.discovered.{_dt.utcnow().strftime('%Y%m%d-%H%M%S')}.bak.json"
                )
                backup_path.write_text(discovered_path.read_text())
                remain = [e for e in raw if e not in removed]
                discovered_path.write_text(_json.dumps(remain, indent=2) + "\n")
                typer.echo(
                    f"Removed {len(removed)} discovered ollama entries (backup {backup_path.name})"
                )
            else:
                typer.echo("No existing discovered ollama entries to remove")
    if dry_run:
        typer.echo("Dry run complete (skipping import)")
        return
    # Run importer
    import subprocess as _sp

    cmd = ["python", "scripts/import_ollama_models.py", f"--location={location}"]
    try:
        out = _sp.check_output(cmd, stderr=_sp.STDOUT, text=True)
        typer.echo(out.rstrip())
    except _sp.CalledProcessError as exc:
        typer.echo(f"Ollama import failed: {exc.output}")
        raise typer.Exit(code=exc.returncode)
    if show:
        from . import registry as _reg

        reg = _reg.load_registry(force=True)
        names = sorted(n for n, e in reg.items() if e.backend == "ollama")
        typer.echo(f"Ollama entries ({len(names)}):")
        for n in names:
            typer.echo(f"  - {n}")


@app.command("undeprecate-discovered")
def undeprecate_discovered(
    backend: str = typer.Option(
        "ollama", "--backend", help="Backend filter (default ollama; use 'all' for any)"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would change"),
):
    """Clear deprecated=True on discovered-layer entries (policy enforcement tool)."""
    from . import registry as _reg

    reg = _reg.load_registry(force=True)
    changed = 0
    targets = []
    for e in reg.values():
        if e.name in _reg._CURATED_NAMES:  # curated untouched
            continue
        if not e.deprecated:
            continue
        if backend != "all" and e.backend != backend:
            continue
        targets.append(e)
    if dry_run:
        typer.echo(
            f"Would undeprecate {len(targets)} entries: {', '.join(t.name for t in targets[:12])}{' ...' if len(targets)>12 else ''}"
        )
        return
    for t in targets:
        t.deprecated = False
        changed += 1
    if changed:
        _reg.save_registry()
    typer.echo(f"Undeprecated {changed} entries (backend filter={backend}).")


@app.command("list-roles")
def list_roles(
    registry_path: Path = typer.Option(
        Path("configs/model_registry.json"), help="Path to unified registry file"
    ),
    show_capabilities: bool = typer.Option(
        False, help="Include capability flags for each model"
    ),
    json_output: bool = typer.Option(
        False, help="Emit JSON instead of human-readable table"
    ),
):
    """List models that declare functional roles (caption, description, embedding, etc.).

    Output columns (text mode): role, name, backend, display_name (if present)
    If a model has multiple roles it will appear once per role.
    """
    registry = load_registry(registry_path, force=True)
    rows: list[dict[str, Any]] = []
    for entry in registry.values():
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
    # sort deterministically: role then name
    rows.sort(key=lambda r: (r["role"], r["name"]))

    if json_output:
        typer.echo(json.dumps(rows, indent=2))
        return

    if not rows:
        typer.echo("No role-capable models found in registry")
        raise typer.Exit(code=0)

    # compute column widths
    role_w = max(len(r["role"]) for r in rows)
    name_w = max(len(r["name"]) for r in rows)
    backend_w = max(len(r["backend"]) for r in rows)
    header = f"{'ROLE'.ljust(role_w)}  {'NAME'.ljust(name_w)}  {'BACKEND'.ljust(backend_w)}  DISPLAY_NAME"
    typer.echo(header)
    typer.echo("-" * len(header))
    for r in rows:
        line = f"{r['role'].ljust(role_w)}  {r['name'].ljust(name_w)}  {r['backend'].ljust(backend_w)}  {r['display_name']}"
        if show_capabilities:
            caps = ",".join(k for k, v in r["capabilities"].items() if v)
            line += f"  [{caps}]"
        typer.echo(line)


__all__ = [
    "get_typer_app",
    "regenerate_discovered",
    "sync_downloader",
    "verify",
    "list_roles",
]
