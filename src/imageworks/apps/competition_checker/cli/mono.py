from pathlib import Path
import json
import os
import subprocess
import sys
from typing import Optional, Any, Dict
import typer
from rich.progress import track
from imageworks.libs.vision.mono import check_monochrome
import numpy as np
from PIL import Image
import cv2
import tomllib

app = typer.Typer(help="Competition Checker - Monochrome validation")


def _iter_files(root: Path, exts_csv: str):
    exts = [e.strip().lstrip(".").lower() for e in exts_csv.split(",") if e.strip()]
    for p in root.rglob("*"):
        if p.is_file():
            suf = p.suffix.lstrip(".").lower()
            if suf in exts:
                yield p


def _find_pyproject(start: Path) -> Optional[Path]:
    """Walk up from 'start' to locate a pyproject.toml, if any."""
    cur = start.resolve()
    for parent in [cur, *cur.parents]:
        cand = parent / "pyproject.toml"
        if cand.exists():
            return cand
    return None


def _load_defaults() -> Dict[str, Any]:
    """Load defaults from [tool.imageworks.mono] in pyproject.toml (if present)."""
    try:
        cfg_p = _find_pyproject(Path.cwd())
        if not cfg_p:
            return {}
        data = tomllib.loads(cfg_p.read_text())
        mono = data.get("tool", {}).get("imageworks", {}).get("mono", {})
        return mono if isinstance(mono, dict) else {}
    except Exception:
        return {}


@app.command()
def check(
    folder: Optional[Path] = typer.Argument(
        None, help="Folder to scan (defaults from config)"
    ),
    exts: Optional[str] = typer.Option(
        None, help="Comma-separated extensions (defaults from config)"
    ),
    neutral_tol: Optional[int] = typer.Option(
        None, help="Max channel diff (8-bit) for 'neutral'"
    ),
    toned_pass: Optional[float] = typer.Option(
        None, help="Hue σ (deg) threshold for PASS toned"
    ),
    toned_query: Optional[float] = typer.Option(
        None, help="Hue σ (deg) threshold for PASS-WITH-QUERY"
    ),
    jsonl_out: Optional[Path] = typer.Option(
        None, help="Write results as JSONL to this path"
    ),
    summary_only: bool = typer.Option(False, help="Only print summary counts"),
    write_xmp: bool = typer.Option(
        False, help="Generate and run the XMP exporter after checks"
    ),
    xmp_keywords_only: bool = typer.Option(
        False, help="With --write-xmp, only write mono:* keywords"
    ),
    xmp_sidecar: bool = typer.Option(
        False, help="With --write-xmp, write .xmp sidecars instead of in-file tags"
    ),
    xmp_no_keywords: bool = typer.Option(
        False, help="With --write-xmp, skip adding mono:* Lightroom keywords"
    ),
):
    """Run monochrome checks over a folder tree, using config defaults when omitted."""
    defaults = _load_defaults()
    folder = folder or (
        Path(defaults["default_folder"]) if defaults.get("default_folder") else None
    )
    if folder is None or not folder.exists() or not folder.is_dir():
        typer.echo(
            "Folder not provided or not found. Provide FOLDER or set "
            "tool.imageworks.mono.default_folder in pyproject.toml"
        )
        raise typer.Exit(1)
    exts = exts or defaults.get("default_exts", "jpg,jpeg")
    neutral_tol = int(
        neutral_tol if neutral_tol is not None else defaults.get("neutral_tol", 2)
    )
    toned_pass = float(
        toned_pass if toned_pass is not None else defaults.get("toned_pass_deg", 6.0)
    )
    toned_query = float(
        toned_query
        if toned_query is not None
        else defaults.get("toned_query_deg", 10.0)
    )
    jsonl_out = jsonl_out or (
        Path(defaults["default_jsonl"]) if defaults.get("default_jsonl") else None
    )

    paths = list(_iter_files(folder, exts))
    if not paths:
        typer.echo(f"No files matched in {folder} for extensions: {exts}")
        raise typer.Exit(1)

    out_f = open(jsonl_out, "w") if jsonl_out else None
    counts = {"pass": 0, "pass_with_query": 0, "fail": 0}

    for p in track(paths, description="Checking"):
        res = check_monochrome(str(p), neutral_tol, toned_pass, toned_query)
        counts[res.verdict] += 1
        if not summary_only:
            extra = []
            if res.mode == "toned" and res.dominant_color:
                extra.append(f"tone={res.dominant_color}")
            if res.top_colors:
                extra.append("tones=" + "+".join(res.top_colors[:2]))
            if res.verdict == "fail" and res.failure_reason:
                extra.append(f"reason={res.failure_reason}")
            extra_s = ("  " + "  ".join(extra)) if extra else ""
            typer.echo(
                f"[{res.verdict}] {p.name:40s}  mode={res.mode:9s}  "
                f"maxΔ={res.channel_max_diff:.1f}  hueσ={res.hue_std_deg:.2f}{extra_s}"
            )
        if out_f:
            out_obj = {
                "path": str(p),
                "verdict": res.verdict,
                "mode": res.mode,
                "channel_max_diff": res.channel_max_diff,
                "hue_std_deg": res.hue_std_deg,
                # Extended diagnostics
                "dominant_hue_deg": res.dominant_hue_deg,
                "dominant_color": res.dominant_color,
                "hue_concentration": res.hue_concentration,
                "hue_bimodality": res.hue_bimodality,
                "sat_median": res.sat_median,
                "colorfulness": res.colorfulness,
                "failure_reason": res.failure_reason,
                "top_hues_deg": res.top_hues_deg,
                "top_colors": res.top_colors,
                "top_weights": res.top_weights,
            }
            out_f.write(json.dumps(out_obj) + "\n")

    if out_f:
        out_f.close()
    summary = (
        f"\nSummary: PASS={counts['pass']}  "
        f"QUERY={counts['pass_with_query']}  "
        f"FAIL={counts['fail']}"
    )
    typer.echo(summary)

    if write_xmp:
        if not jsonl_out:
            typer.echo(
                "--write-xmp requires a JSONL output path; set default_jsonl in "
                "pyproject.toml or pass --jsonl-out"
            )
            raise typer.Exit(1)
        script_name = defaults.get("default_xmp_script", "write_xmp.sh")
        script_path = Path(script_name)
        cmd = [
            sys.executable,
            "-m",
            "imageworks.tools.write_mono_xmp",
            "generate",
            str(jsonl_out),
            "--out",
            str(script_path),
        ]
        if not xmp_no_keywords:
            cmd.append("--as-keywords")
        if xmp_keywords_only:
            cmd.append("--keywords-only")
        if xmp_sidecar:
            cmd.append("--sidecar")
        typer.echo("Generating XMP script: " + " ".join(cmd))
        subprocess.run(cmd, check=True)
        typer.echo(f"Running {script_path} ...")
        subprocess.run(["bash", str(script_path)], check=True)


@app.command()
def visualize(
    folder: Optional[Path] = typer.Argument(
        None, help="Folder to scan (defaults from config)"
    ),
    exts: Optional[str] = typer.Option(
        None, help="Comma-separated extensions (defaults from config)"
    ),
    out_suffix: Optional[str] = typer.Option(
        None, help="Suffix added before extension for output overlay"
    ),
    mode: str = typer.Option(
        "channel_diff",
        help="Leak metric: one of channel_diff, saturation, hue",
    ),
    neutral_tol: Optional[int] = typer.Option(
        None, help="Threshold for channel_diff in 8-bit (only for mode=channel_diff)"
    ),
    sat_threshold: Optional[float] = typer.Option(
        None, help="Saturation threshold in [0,1] (only for mode=saturation or hue)"
    ),
    alpha: float = typer.Option(0.6, help="Overlay opacity in [0,1]"),
    quality: int = typer.Option(92, help="JPEG quality for output"),
):
    """Generate heatmap overlays that highlight color leaks.

    - channel_diff: marks pixels where max(|R-G|,|R-B|,|G-B|) > neutral_tol
    - saturation:   marks pixels where HSV S > sat_threshold
    - hue:          same as saturation but colors the overlay by hue
    """
    mode = mode.lower().strip()
    if mode not in {"channel_diff", "saturation", "hue"}:
        raise typer.BadParameter(
            "Invalid mode. Choose one of: channel_diff, saturation, hue"
        )

    # Load defaults
    defaults = _load_defaults()
    folder = folder or (
        Path(defaults["default_folder"]) if defaults.get("default_folder") else None
    )
    if folder is None or not folder.exists() or not folder.is_dir():
        typer.echo(
            "Folder not provided or not found. Provide FOLDER or set "
            "tool.imageworks.mono.default_folder in pyproject.toml"
        )
        raise typer.Exit(1)
    exts = exts or defaults.get("default_exts", "jpg,jpeg")
    out_suffix = out_suffix or defaults.get("default_visualize_suffix", "_mono_vis")
    neutral_tol = int(
        neutral_tol if neutral_tol is not None else defaults.get("neutral_tol", 2)
    )
    sat_threshold = float(
        sat_threshold
        if sat_threshold is not None
        else defaults.get("sat_threshold", 0.06)
    )

    paths = list(_iter_files(folder, exts))
    if not paths:
        typer.echo(f"No files matched in {folder} for extensions: {exts}")
        raise typer.Exit(1)

    for p in track(paths, description="Visualizing"):
        im = Image.open(p).convert("RGB")
        rgb = np.asarray(im, dtype=np.uint8)
        h, w = rgb.shape[:2]
        # Base grayscale for context
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        if mode == "channel_diff":
            r = rgb[..., 0].astype(np.int16)
            g = rgb[..., 1].astype(np.int16)
            b = rgb[..., 2].astype(np.int16)
            diff = np.maximum(np.abs(r - g), np.maximum(np.abs(r - b), np.abs(g - b)))
            # Normalize for visualization
            norm = np.clip(
                (diff.astype(np.float32) - neutral_tol) / max(1, 255 - neutral_tol),
                0,
                1,
            )
            heat = np.zeros_like(rgb, dtype=np.float32)
            heat[..., 0] = norm  # red channel
            overlay = (gray_rgb.astype(np.float32) / 255.0) * (1 - alpha) + heat * alpha
            out = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)
        else:
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            H = hsv[..., 0].astype(np.float32) * (360.0 / 180.0)
            S = hsv[..., 1].astype(np.float32) / 255.0
            mask = S > float(sat_threshold)
            if mode == "saturation":
                # red overlay scaled by saturation
                heat = np.zeros_like(rgb, dtype=np.float32)
                heat[..., 0] = np.clip(
                    (S - sat_threshold) / max(1e-6, 1 - sat_threshold), 0, 1
                )
                overlay = (gray_rgb.astype(np.float32) / 255.0) * (
                    1 - alpha
                ) + heat * alpha
                out = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)
            else:  # hue colored overlay
                # colorize by hue for masked pixels, else grayscale
                hsv_vis = np.zeros((h, w, 3), dtype=np.uint8)
                hsv_vis[..., 0] = (H / 2.0).astype(np.uint8)
                hsv_vis[..., 1] = (np.clip(S, 0, 1) * 255).astype(np.uint8)
                hsv_vis[..., 2] = 255
                color_hue = (
                    cv2.cvtColor(hsv_vis, cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
                )
                base = gray_rgb.astype(np.float32) / 255.0
                m = mask.astype(np.float32)[..., None]
                overlay = base * (1 - m * alpha) + color_hue * (m * alpha)
                out = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)

        out_path = p.with_name(p.stem + out_suffix + ".jpg")
        Image.fromarray(out).save(out_path, quality=quality, subsampling=2)
        # Preserve timestamps for easy auto-stacking by capture time
        try:
            st = os.stat(p)
            os.utime(out_path, (st.st_atime, st.st_mtime))
        except Exception:
            pass

    typer.echo(f"Wrote overlays with suffix '{out_suffix}' next to originals.")
