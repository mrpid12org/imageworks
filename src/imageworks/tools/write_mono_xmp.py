from __future__ import annotations

import json
from pathlib import Path
import shlex
from typing import List, Optional

import typer


app = typer.Typer(help="Generate an ExifTool script from mono JSONL results.")


def _kw(label: str, value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    return f"mono:{label}:{value}"


@app.command()
def generate(
    jsonl: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    out: Path = typer.Option(
        Path("write_mono_xmp.sh"), help="Output shell script path"
    ),
    as_keywords: bool = typer.Option(
        False,
        help="Also add Lightroom keywords prefixed with mono: (writes XMP-dc:Subject)",
    ),
    keywords_only: bool = typer.Option(
        False,
        help="Only write mono:* keywords; skip custom XMP-MW fields",
    ),
    sidecar: bool = typer.Option(
        False,
        help="Write to sidecar XMP files instead of in-file tags",
    ),
):
    """Emit a shell script that writes XMP diagnostics with ExifTool.

    Writes custom XMP tags under the XMP-MW namespace (ExifTool will
    create it). Optionally adds lightweight keywords (mono:verdict,...)
    that Lightroom shows by default.
    """
    lines: List[str] = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "# Requires: exiftool",
    ]
    # Auto-export EXIFTOOL_HOME if repo config exists
    cfg = Path("configs/exiftool/.ExifTool_config")
    if cfg.exists():
        lines.append(f"export EXIFTOOL_HOME={shlex.quote(str(cfg.parent))}")
    for line in jsonl.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        path = rec.get("path")
        if not path:
            continue
        args: List[str] = [
            "exiftool",
            "-overwrite_original",
        ]
        if sidecar:
            args += ["-o", "%d%f.xmp"]

        if not keywords_only:
            args += [
                f"-XMP-MW:MonoVerdict={rec.get('verdict','')}",
                f"-XMP-MW:MonoMode={rec.get('mode','')}",
            ]
            if rec.get("failure_reason"):
                args.append(f"-XMP-MW:MonoReason={rec['failure_reason']}")
            top_colors = rec.get("top_colors") or []
            for i, name in enumerate(top_colors[:3], start=1):
                args.append(f"-XMP-MW:MonoTone{i}={name}")
            if rec.get("hue_std_deg") is not None:
                args.append(f"-XMP-MW:MonoHueStd={rec['hue_std_deg']:.3f}")
            if rec.get("hue_concentration") is not None:
                args.append(f"-XMP-MW:MonoHueR={rec['hue_concentration']:.3f}")
            if rec.get("hue_bimodality") is not None:
                args.append(f"-XMP-MW:MonoHueR2={rec['hue_bimodality']:.3f}")

        if as_keywords or keywords_only:
            kws: List[str] = []
            kws.append(_kw("verdict", rec.get("verdict")))
            kws.append(_kw("mode", rec.get("mode")))
            top_colors_kw = (rec.get("top_colors") or [])[:2]
            if top_colors_kw:
                kws.append(_kw("tones", "+".join(top_colors_kw)))
            if rec.get("failure_reason"):
                kws.append(_kw("reason", rec["failure_reason"]))
            for kw in filter(None, kws):
                args.append(f"-XMP-dc:Subject+={kw}")

        args.append(shlex.quote(str(path)))
        lines.append(" ".join(args))

    out.write_text("\n".join(lines) + "\n")
    typer.echo(f"Wrote {len(lines)-3} commands to {out}")


@app.command()
def clean(
    jsonl: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    out: Path = typer.Option(
        Path("clean_mono_xmp.sh"), help="Output shell script path"
    ),
    keywords_only: bool = typer.Option(
        False, help="Only remove mono:* keywords, keep XMP-MW fields"
    ),
):
    """Generate a script to remove XMP fields and/or mono:* keywords.

    If --keywords-only is set, only removes Lightroom-visible keywords
    with prefix mono:.
    """
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "# Requires: exiftool",
    ]
    for line in jsonl.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        path = rec.get("path")
        if not path:
            continue
        args = ["exiftool", "-overwrite_original"]
        if not keywords_only:
            # Delete our custom fields
            args += [
                "-XMP-MW:MonoVerdict=",
                "-XMP-MW:MonoMode=",
                "-XMP-MW:MonoReason=",
                "-XMP-MW:MonoTone1=",
                "-XMP-MW:MonoTone2=",
                "-XMP-MW:MonoTone3=",
                "-XMP-MW:MonoHueStd=",
                "-XMP-MW:MonoHueR=",
                "-XMP-MW:MonoHueR2=",
            ]
        # Remove mono:* keywords if present
        for key in ("verdict", "mode", "tones", "reason"):
            args.append(f"-XMP-dc:Subject-={{mono:{key}:*}}")
        args.append(path)
        lines.append(" ".join(args))
    out.write_text("\n".join(lines) + "\n")
    typer.echo(f"Wrote {len(lines)-3} commands to {out}")


if __name__ == "__main__":
    app()
