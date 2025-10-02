"""Typer CLI for deterministic model loader operations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
import typer

from .registry import load_registry, list_models, get_entry, save_registry
from .service import select_model, CapabilityError
from .hashing import verify_model, VersionLockViolation
from .probe import run_vision_probe

app = typer.Typer(help="Deterministic model registry & loader utilities")


@app.command("list")
def cmd_list(
    role: Optional[str] = typer.Option(
        None, "--role", help="Filter models advertising the specified functional role"
    ),
):  # noqa: D401 - CLI
    """List logical models with basic metadata (optionally filtered by role)."""
    load_registry()
    rows = []
    for name in list_models():
        e = get_entry(name)
        if role and role not in (e.roles or []):
            continue
        rows.append(
            {
                "name": name,
                "backend": e.backend,
                "locked": e.version_lock.locked,
                "vision": e.capabilities.get("vision"),
                "roles": e.roles,
                "hash": (
                    e.artifacts.aggregate_sha256[:12]
                    if e.artifacts.aggregate_sha256
                    else ""
                ),
            }
        )
    typer.echo(json.dumps(rows, indent=2))


@app.command("select")
def cmd_select(
    name: str,
    require_vision: bool = typer.Option(
        False, "--require-vision", help="Require vision capability"
    ),
):
    caps = ["vision"] if require_vision else None
    try:
        desc = select_model(name, require_capabilities=caps)
    except CapabilityError as exc:
        typer.echo(f"Capability error: {exc}")
        raise typer.Exit(1)
    except KeyError as exc:
        typer.echo(str(exc))
        raise typer.Exit(1)
    typer.echo(
        json.dumps(
            {
                "logical_name": desc.logical_name,
                "endpoint": desc.endpoint_url,
                "backend": desc.backend,
                "internal_model_id": desc.internal_model_id,
                "capabilities": desc.capabilities,
            },
            indent=2,
        )
    )


@app.command("verify")
def cmd_verify(name: str):
    try:
        entry = get_entry(name)
    except KeyError as exc:
        typer.echo(str(exc))
        raise typer.Exit(1)
    try:
        verify_model(entry)
    except VersionLockViolation as exc:
        typer.echo(f"Lock violation: {exc}")
        raise typer.Exit(2)
    typer.echo(
        json.dumps(
            {
                "name": name,
                "aggregate_sha256": entry.artifacts.aggregate_sha256,
                "last_verified": entry.version_lock.last_verified,
                "locked": entry.version_lock.locked,
            },
            indent=2,
        )
    )


@app.command("lock")
def cmd_lock(
    name: str,
    set_expected: bool = typer.Option(
        False,
        "--set-expected",
        help="Set expected hash from current artifacts if empty",
    ),
):
    try:
        entry = get_entry(name)
    except KeyError as exc:
        typer.echo(str(exc))
        raise typer.Exit(1)
    entry.version_lock.locked = True
    if set_expected and not entry.version_lock.expected_aggregate_sha256:
        # compute if needed
        verify_model(entry, enforce_lock=False)
        entry.version_lock.expected_aggregate_sha256 = entry.artifacts.aggregate_sha256
    save_registry()
    typer.echo(f"Locked {name}")


@app.command("unlock")
def cmd_unlock(name: str):
    try:
        entry = get_entry(name)
    except KeyError as exc:
        typer.echo(str(exc))
        raise typer.Exit(1)
    entry.version_lock.locked = False
    save_registry()
    typer.echo(f"Unlocked {name}")


@app.command("probe-vision")
def cmd_probe_vision(name: str, image: Path):
    if not image.exists():
        typer.echo(f"Image not found: {image}")
        raise typer.Exit(1)
    try:
        result = run_vision_probe(name, image)
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"Probe failed: {exc}")
        raise typer.Exit(1)
    typer.echo(json.dumps(result, indent=2))


@app.command("metrics")
def cmd_metrics(name: Optional[str] = None):
    # Placeholder: metrics are not persisted yet; show artifact hash & lock state for now.
    load_registry()
    if name:
        try:
            e = get_entry(name)
        except KeyError as exc:
            typer.echo(str(exc))
            raise typer.Exit(1)
        typer.echo(
            json.dumps(
                {
                    "name": name,
                    "aggregate_sha256": e.artifacts.aggregate_sha256,
                    "locked": e.version_lock.locked,
                },
                indent=2,
            )
        )
        return
    data = []
    for n in list_models():
        e = get_entry(n)
        data.append(
            {
                "name": n,
                "hash": (
                    e.artifacts.aggregate_sha256[:12]
                    if e.artifacts.aggregate_sha256
                    else ""
                ),
                "locked": e.version_lock.locked,
            }
        )
    typer.echo(json.dumps(data, indent=2))


def main():  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
