from pathlib import Path
import csv
import json
import os
import subprocess
import sys
from typing import Optional, Any, Dict, List, Tuple, Iterable
import typer
from rich.progress import track
from imageworks.libs.vision.mono import (
    MonoResult,
    check_monochrome,
    _describe_chroma_footprint,
    _describe_chroma_percentiles,
    _describe_hue_drift,
    _describe_hue_spread,
    LAB_HARD_FAIL_C4_CLUSTER_DEFAULT,
    LAB_HARD_FAIL_C4_RATIO_DEFAULT,
)
import numpy as np
from PIL import Image
import cv2
import tomllib

app = typer.Typer(help="Mono Checker - Monochrome validation")


def _iter_files(root: Path, exts_csv: str):
    exts = [e.strip().lstrip(".").lower() for e in exts_csv.split(",") if e.strip()]
    for p in root.rglob("*"):
        if p.is_file():
            suf = p.suffix.lstrip(".").lower()
            if suf in exts:
                # Filter for mono competition entries: filename starts with "01_" followed by number
                stem = p.stem
                if stem.startswith("01_") and len(stem) > 3:
                    # Check if character after "01_" is a digit
                    if stem[3].isdigit():
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


LAB_TONED_PASS_DEFAULT = 10.0
LAB_TONED_QUERY_DEFAULT = 14.0
IR_NEUTRAL_CHROMA_DEFAULT = 4.0
IR_TONED_PASS_DEFAULT = 80.0
IR_TONED_QUERY_DEFAULT = 120.0

_HSL_SLIDER_TITLES = {
    "red": "Red",
    "orange": "Orange",
    "yellow": "Yellow",
    "green": "Green",
    "aqua": "Aqua",
    "blue": "Blue",
    "purple": "Purple",
    "magenta": "Magenta",
}


def _lightroom_tip(res: Dict[str, Any]) -> Optional[str]:
    verdict = res.get("verdict")
    if verdict not in {"fail", "pass_with_query"}:
        return None

    dominant = (res.get("dominant_color") or "dominant tone").lower()
    slider = _HSL_SLIDER_TITLES.get(dominant, dominant.title())
    hue_drift = float(res.get("hue_drift_deg_per_l") or 0.0)
    hue_std = float(res.get("hue_std_deg") or 0.0)
    cluster = float(res.get("chroma_cluster_max_4") or 0.0)
    ratio4 = float(res.get("chroma_ratio_4") or 0.0)
    failure_reason = res.get("failure_reason")

    overlay = (
        "lab_residual"
        if abs(hue_drift) > 45.0 or failure_reason == "split_toning_suspected"
        else "lab_chroma"
    )

    tips: List[str] = []
    if verdict == "fail":
        tips.append(
            "In Develop > Basic, toggle B&W to show how much colour sits on top of the grayscale base."
        )
        tips.append(
            f"Then drag the {slider} saturation slider in HSL up and down—the {dominant} cast swings instantly."
        )
        if abs(hue_drift) > 60 or failure_reason == "split_toning_suspected":
            tips.append(
                "If tones change across the range, pull Shadow and Highlight saturation to zero in Color Grading and restore it to expose the split tone."
            )
        if cluster > 0.05 or ratio4 > 0.05:
            tips.append(
                "Zoom into the hotspot highlighted by the overlay; that region holds most of the coloured pixels."
            )
    else:
        tips.append(
            "Press B&W to compare; if the frame barely shifts, the tint is subtle but measurable."
        )
        if (
            abs(hue_drift) > 60
            or failure_reason == "split_toning_suspected"
            or hue_std > 45
        ):
            tips.append(
                "Use Color Grading to drop Shadow and Highlight saturation to zero and back—the opposing hues reveal the split tone."
            )
        else:
            tips.append(
                f"Raise the global Saturation to +40 and return to zero while watching the {slider.lower()} areas for a gentle swing."
            )
        tips.append("Leave it if you like the tone—it's flagged only for review.")

    if overlay == "lab_residual":
        tips.append("Overlay hint: lab_residual (add lab_chroma to gauge intensity).")
    else:
        tips.append(
            "Overlay hint: lab_chroma (switch to lab_residual to see hue direction)."
        )
    return " ".join(tips)


def _result_to_json(path: str, res: MonoResult) -> Dict[str, Any]:
    obj = {
        "path": path,
        "method": res.analysis_method,
        "verdict": res.verdict,
        "mode": res.mode,
        "channel_max_diff": res.channel_max_diff,
        "hue_std_deg": res.hue_std_deg,
        "dominant_hue_deg": res.dominant_hue_deg,
        "dominant_color": res.dominant_color,
        "hue_concentration": res.hue_concentration,
        "hue_bimodality": res.hue_bimodality,
        "sat_median": res.sat_median,
        "colorfulness": res.colorfulness,
        "failure_reason": res.failure_reason,
        "split_tone_name": res.split_tone_name,
        "split_tone_description": res.split_tone_description,
        "top_hues_deg": res.top_hues_deg,
        "top_colors": res.top_colors,
        "top_weights": res.top_weights,
        "loader_status": res.loader_status,
        "source_profile": res.source_profile,
        "chroma_max": res.chroma_max,
        "chroma_median": res.chroma_median,
        "chroma_p95": res.chroma_p95,
        "chroma_p99": res.chroma_p99,
        "chroma_ratio_2": res.chroma_ratio_2,
        "chroma_ratio_4": res.chroma_ratio_4,
        "chroma_cluster_max_2": res.chroma_cluster_max_2,
        "chroma_cluster_max_4": res.chroma_cluster_max_4,
        "reason_summary": res.reason_summary,
        "title": res.title,
        "author": res.author,
        "hue_peak_delta_deg": res.hue_peak_delta_deg,
        "hue_second_mass": res.hue_second_mass,
        "hue_weighting": res.hue_weighting,
        "mean_hue_highs_deg": res.mean_hue_highs_deg,
        "mean_hue_shadows_deg": res.mean_hue_shadows_deg,
        "delta_h_highs_shadows_deg": res.delta_h_highs_shadows_deg,
        "grid_regions": res.grid_regions,
        "image_width": res.image_width,
        "image_height": res.image_height,
    }
    return obj


def _format_method_result(res: MonoResult) -> str:
    extras = []
    if res.mode == "toned" and res.dominant_color:
        extras.append(f"tone={res.dominant_color}")
    if res.top_colors:
        extras.append("tones=" + "+".join(res.top_colors[:2]))
    if res.analysis_method == "lab" and res.chroma_max is not None:
        extras.append(f"Cmax={res.chroma_max:.2f}")
    if res.verdict == "fail" and res.failure_reason:
        extras.append(f"reason={res.failure_reason}")
    if res.loader_status and res.loader_status != "no_profile_assumed_srgb":
        extras.append(f"icc={res.loader_status}")
    extra_s = ("  " + "  ".join(extras)) if extras else ""
    return (
        f"{res.analysis_method}:{res.verdict}/{res.mode} "
        f"maxΔ={res.channel_max_diff:.1f}  hueσ={res.hue_std_deg:.2f}{extra_s}"
    )


def _summarize_result(res: Dict[str, Any], label: str) -> List[str]:
    if not isinstance(res, dict) or not res:
        return []

    verdict = res.get("verdict", "n/a").upper()
    mode = res.get("mode", "-").replace("_", " ")

    header = f"{label} result: {verdict} ({mode})"

    lines = [header]

    # Add ICC profile section right after the header
    loader_diag = res.get("loader_diag", {}) or {}
    icc_status = loader_diag.get("icc_status") or ""
    if icc_status.startswith("no_profile"):
        lines.append("ICC Profile: None (sRGB assumed)")
    elif icc_status.startswith("embedded_assumed"):
        lines.append("ICC Profile: Embedded profile ignored (LittleCMS unavailable)")
    elif icc_status and not icc_status.startswith("ok"):
        lines.append(f"ICC Profile: {icc_status}")
    else:
        # Check if we have ICC profile name from loader_diag
        icc_name = loader_diag.get("icc_profile_name") or loader_diag.get(
            "profile_name"
        )
        if icc_name:
            lines.append(f"ICC Profile: {icc_name}")
        else:
            lines.append("ICC Profile: Present")

    # Add tones section
    tones: List[str] = []
    dominant = res.get("dominant_color")
    if dominant:
        tones.append(f"dominant tone ≈ {dominant}")
    top_colors = res.get("top_colors") or []
    if top_colors:
        tones.append("other tones: " + ", ".join(top_colors[:3]))
    if tones:
        lines.append("Tones: " + "; ".join(tones))

    notes: List[str] = []
    hue_std = res.get("hue_std_deg")
    if isinstance(hue_std, (int, float)):
        notes.append(_describe_hue_spread(float(hue_std)))
    cp99 = res.get("chroma_p99")
    cmax = res.get("chroma_max")
    if isinstance(cp99, (int, float)):
        notes.append(
            _describe_chroma_percentiles(float(cp99), float(cmax))
            if isinstance(cmax, (int, float))
            else _describe_chroma_percentiles(float(cp99), None)
        )
    ratio2 = res.get("chroma_ratio_2")
    ratio4 = res.get("chroma_ratio_4")
    if isinstance(ratio2, (int, float)) and isinstance(ratio4, (int, float)):
        if ratio2 > 0 or ratio4 > 0:
            notes.append(_describe_chroma_footprint(float(ratio2), float(ratio4)))
        # Skip cluster size details - covered by footprint description above
    drift = res.get("hue_drift_deg_per_l")
    if isinstance(drift, (int, float)):
        notes.append(_describe_hue_drift(float(drift)))

    # Skip technical peak analysis details

    # Check for split-toning indicators and add to notes
    failure_reason = res.get("failure_reason")
    split_tone_name = res.get("split_tone_name")
    verdict = res.get("verdict", "").lower()
    if failure_reason == "split_toning_suspected" or split_tone_name:
        if split_tone_name:
            if verdict == "fail":
                notes.append(
                    f"Split-toning detected (appears to be {split_tone_name})."
                )
            else:
                notes.append(
                    f"Possible split-toning detected (may be {split_tone_name})."
                )
        else:
            notes.append("Possible split-toning detected.")

    # Skip redundant highlights/shadows analysis - covered by drift description

    # Combine notes and reason into a single section
    combined_description = []
    if notes:
        combined_description.extend(notes)

    reason = res.get("reason_summary")
    if reason:
        combined_description.append(reason)

    if combined_description:
        lines.append("Description: " + " ".join(combined_description))

    tip = _lightroom_tip(res)
    if tip:
        lines.append("Lightroom tip: " + tip)

    return lines


def _render_grouped_tables(
    results_by_file: Dict[str, Dict[str, Dict[str, Any]]],
) -> List[str]:
    lines: List[str] = []
    if not results_by_file:
        return lines

    # Group by subdirectory first, then by verdict within each subdirectory
    subdirs: Dict[str, Dict[str, List[Tuple[str, Dict[str, Any]]]]] = {}

    for name, methods in sorted(results_by_file.items()):
        lab = methods.get("lab", {})
        verdict = lab.get("verdict")
        bucket = verdict if verdict in {"fail", "pass_with_query", "pass"} else "pass"
        bucket = {"pass_with_query": "query"}.get(bucket, bucket)

        # Extract subdirectory from file path
        file_path = methods.get("_path", "")
        if file_path:
            subdir = str(Path(file_path).parent.name)
        else:
            subdir = "Unknown"

        # Initialize subdirectory structure if needed
        if subdir not in subdirs:
            subdirs[subdir] = {"fail": [], "query": [], "pass": []}

        subdirs[subdir][bucket].append((name, lab))

    # Render by subdirectory, then by verdict within each
    for subdir in sorted(subdirs.keys()):
        verdict_sections = subdirs[subdir]

        # Count total files and individual verdict counts
        fail_count = len(verdict_sections["fail"])
        query_count = len(verdict_sections["query"])
        pass_count = len(verdict_sections["pass"])
        total_files = fail_count + query_count + pass_count

        if total_files == 0:
            continue

        lines.append("")
        lines.append(
            f"## {subdir} ({total_files} files, {fail_count} Fail, {query_count} Query, {pass_count} Pass)"
        )
        lines.append("")

        for label in ("fail", "query", "pass"):
            rows = verdict_sections[label]
            if not rows:
                continue
            lines.append(f"### {label.upper()} ({len(rows)})")
            lines.append("")
            for name, lab in rows:
                chosen = lab
                title = (chosen or {}).get("title") or "—"
                author = (chosen or {}).get("author") or "—"
                prefix = name
                marker = "_Serial"
                if marker in name:
                    prefix = name.split(marker, 1)[0]
                lines.append(f"**{prefix}**")
                lines.append(f"- Title: {title}")
                lines.append(f"- Author: {author}")
                lines.append(f"- File: {name}")
                lines_summary = _summarize_result(lab, "LAB")
                for line in lines_summary:
                    lines.append("  - " + line)
                lines.append("")

    return lines


def _extract_entry_number(filename: str) -> str:
    marker = "_Serial"
    if marker in filename:
        return filename.split(marker, 1)[0]
    return Path(filename).stem


def _generate_heatmap_image(
    path: Path,
    mode: str,
    neutral_tol: int,
    sat_threshold: float,
    chroma_clip: float,
    alpha: float,
    quality: int,
    out_suffix: str,
    description: Optional[str] = None,
) -> Optional[Path]:
    mode = mode.lower().strip()
    with Image.open(path) as im_src:
        exif = im_src.getexif()
        rgb = np.asarray(im_src.convert("RGB"), dtype=np.uint8)
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    out: Optional[np.ndarray] = None

    if mode == "channel_diff":
        r = rgb[..., 0].astype(np.int16)
        g = rgb[..., 1].astype(np.int16)
        b = rgb[..., 2].astype(np.int16)
        diff = np.maximum(np.abs(r - g), np.maximum(np.abs(r - b), np.abs(g - b)))
        norm = np.clip(
            (diff.astype(np.float32) - neutral_tol) / max(1, 255 - neutral_tol),
            0,
            1,
        )
        heat = np.zeros_like(rgb, dtype=np.float32)
        heat[..., 0] = norm
        overlay = (gray_rgb.astype(np.float32) / 255.0) * (1 - alpha) + heat * alpha
        out = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)
    elif mode in {"saturation", "hue"}:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        H = hsv[..., 0].astype(np.float32) * (360.0 / 180.0)
        S = hsv[..., 1].astype(np.float32) / 255.0
        mask = S > float(sat_threshold)
        if mode == "saturation":
            heat = np.zeros_like(rgb, dtype=np.float32)
            heat[..., 0] = np.clip(
                (S - sat_threshold) / max(1e-6, 1 - sat_threshold), 0, 1
            )
            overlay = (gray_rgb.astype(np.float32) / 255.0) * (1 - alpha) + heat * alpha
            out = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)
        else:
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
    elif mode in {"lab_chroma", "lab_residual"}:
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        a = lab[..., 1] - 128.0
        b = lab[..., 2] - 128.0
        chroma = np.hypot(a, b)
        norm = np.clip(chroma / max(1e-6, chroma_clip), 0.0, 1.0)
        base = gray_rgb.astype(np.float32) / 255.0
        if mode == "lab_chroma":
            heatmap = cv2.applyColorMap(
                (norm * 255.0).astype(np.uint8), cv2.COLORMAP_INFERNO
            )
            heatmap = (
                cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            )
            overlay = base * (1 - alpha) + heatmap * alpha
            out = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)
        else:
            hue = (np.degrees(np.arctan2(b, a)) + 360.0) % 360.0
            hsv_vis = np.zeros((h, w, 3), dtype=np.uint8)
            hsv_vis[..., 0] = (hue / 2.0).astype(np.uint8)
            hsv_vis[..., 1] = (np.clip(norm, 0, 1) * 255).astype(np.uint8)
            hsv_vis[..., 2] = (np.clip(norm * 1.2, 0, 1) * 255).astype(np.uint8)
            color = cv2.cvtColor(hsv_vis, cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
            m = (norm > 0).astype(np.float32)[..., None]
            overlay = base * (1 - m * alpha) + color * (m * alpha)
            out = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)

    if out is None:
        return None

    if out_suffix:
        suffix = out_suffix
        mode_suffix = {
            "lab_chroma": "_chroma",
            "lab_residual": "_residual",
            "channel_diff": "_diff",
            "saturation": "_sat",
            "hue": "_hue",
        }.get(mode.lower(), f"_{mode.lower()}")
        if suffix.endswith(mode_suffix):
            suffix = suffix[: -len(mode_suffix)]
        suffix = suffix + mode_suffix
    else:
        suffix = {
            "lab_chroma": "_lab_chroma",
            "lab_residual": "_lab_residual",
            "channel_diff": "_diff",
            "saturation": "_sat",
            "hue": "_hue",
        }.get(mode.lower(), f"_{mode.lower()}")

    out_path = path.with_name(path.stem + suffix + ".jpg")

    # Skip generation if overlay already exists
    if out_path.exists():
        return None

    out_img = Image.fromarray(out)
    save_kwargs = {"quality": quality, "subsampling": 2}
    if description:
        exif[0x010E] = description
    if exif:
        save_kwargs["exif"] = exif.tobytes()
    out_img.save(out_path, **save_kwargs)
    try:
        st = os.stat(path)
        os.utime(out_path, (st.st_atime, st.st_mtime))
    except Exception:
        pass

    # Copy XMP/keyword metadata from the original image so overlays inherit verdict tags.
    try:
        cmd = [
            "exiftool",
            "-overwrite_original",
            "-TagsFromFile",
            str(path),
            "-XMP:all",
            "-IPTC:Keywords",
        ]
        if description:
            cmd.append(f"-ImageDescription={description}")
        cmd.append(str(out_path))
        subprocess.run(
            cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except Exception:
        # If ExifTool is unavailable or copying fails, fall back to the EXIF written above.
        pass

    return out_path


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
    lab_neutral_chroma: Optional[float] = typer.Option(
        None, help="Neutral chroma threshold (C*) for LAB pipeline"
    ),
    lab_chroma_mask: Optional[float] = typer.Option(
        None, help="Chroma mask threshold for LAB hue statistics"
    ),
    lab_toned_pass: Optional[float] = typer.Option(
        None, help="LAB hue σ (deg) threshold for PASS toned"
    ),
    lab_toned_query: Optional[float] = typer.Option(
        None, help="LAB hue σ (deg) threshold for PASS-WITH-QUERY"
    ),
    lab_fail_c4_ratio: Optional[float] = typer.Option(
        None,
        help="Force FAIL when pct_gt_c4 exceeds this fraction (default from config)",
    ),
    lab_fail_c4_cluster: Optional[float] = typer.Option(
        None,
        help="Force FAIL when largest_cluster_pct_gt_c4 exceeds this fraction",  # noqa: E501
    ),
    jsonl_out: Optional[Path] = typer.Option(
        None, help="Write results as JSONL to this path"
    ),
    csv_out: Optional[Path] = typer.Option(
        None, help="Write tabular results to this CSV (Title, Author, verdict, etc.)"
    ),
    summary_out: Optional[Path] = typer.Option(
        None,
        help="Path for the human-readable summary (defaults to mono_summary.md)",
    ),
    summary_only: bool = typer.Option(False, help="Only print summary counts"),
    table_only: bool = typer.Option(
        True,
        help="Suppress per-image output and progress; emit only summary table/counts",
    ),
    auto_heatmap: bool = typer.Option(
        True,
        help="Generate LAB chroma heatmaps for failed images",
    ),
    write_xmp: bool = typer.Option(
        True,
        help=(
            "Generate and run the XMP exporter after checks (default: on). "
            "Use --no-write-xmp to skip writing."
        ),
    ),
    xmp_keywords_only: bool = typer.Option(
        True,
        help=(
            "With --write-xmp, only write mono:* keywords (XMP Subject + IPTC Keywords). "
            "Disable with --no-xmp-keywords-only to also write custom XMP fields."
        ),
    ),
    xmp_sidecar: bool = typer.Option(
        False, help="With --write-xmp, write .xmp sidecars instead of in-file tags"
    ),
    xmp_no_keywords: bool = typer.Option(
        False, help="With --write-xmp, skip adding mono:* Lightroom keywords"
    ),
    include_grid_regions: bool = typer.Option(
        False, help="Include 3x3 grid region analysis in JSONL output"
    ),
):
    """Run monochrome checks over a folder tree, using config defaults when omitted."""
    defaults = _load_defaults()

    # --- Determine final configuration robustly ---
    # Precedence: CLI argument > pyproject.toml default > Hardcoded fallback

    # 1. Determine the scan folder
    scan_folder = folder
    if scan_folder is None:
        default_folder_path = defaults.get("default_folder")
        if default_folder_path:
            scan_folder = Path(default_folder_path)

    if scan_folder is None or not scan_folder.exists() or not scan_folder.is_dir():
        typer.echo(
            "Folder not provided or not found. Provide FOLDER or set "
            "tool.imageworks.mono.default_folder in pyproject.toml"
        )
        raise typer.Exit(1)

    # 2. Determine other settings
    final_exts = exts or defaults.get("default_exts", "jpg,jpeg,png,tif,tiff")
    final_neutral_tol = (
        neutral_tol if neutral_tol is not None else int(defaults.get("neutral_tol", 2))
    )
    final_toned_pass = (
        toned_pass
        if toned_pass is not None
        else float(defaults.get("toned_pass_deg", 6.0))
    )
    final_toned_query = (
        toned_query
        if toned_query is not None
        else float(defaults.get("toned_query_deg", 10.0))
    )
    final_lab_neutral_chroma = (
        lab_neutral_chroma
        if lab_neutral_chroma is not None
        else float(defaults.get("lab_neutral_chroma", 2.0))
    )
    final_lab_chroma_mask = (
        lab_chroma_mask
        if lab_chroma_mask is not None
        else float(defaults.get("lab_chroma_mask", 2.0))
    )
    final_lab_toned_pass = (
        lab_toned_pass
        if lab_toned_pass is not None
        else float(defaults.get("lab_toned_pass_deg", LAB_TONED_PASS_DEFAULT))
    )
    final_lab_toned_query = (
        lab_toned_query
        if lab_toned_query is not None
        else float(defaults.get("lab_toned_query_deg", LAB_TONED_QUERY_DEFAULT))
    )
    final_lab_fail_c4_ratio = (
        lab_fail_c4_ratio
        if lab_fail_c4_ratio is not None
        else float(defaults.get("lab_fail_c4_ratio", LAB_HARD_FAIL_C4_RATIO_DEFAULT))
    )
    final_lab_fail_c4_cluster = (
        lab_fail_c4_cluster
        if lab_fail_c4_cluster is not None
        else float(
            defaults.get("lab_fail_c4_cluster", LAB_HARD_FAIL_C4_CLUSTER_DEFAULT)
        )
    )
    final_jsonl_out = jsonl_out or (
        Path(defaults["default_jsonl"]) if "default_jsonl" in defaults else None
    )
    final_summary_path = summary_out or Path(
        defaults.get("default_summary", "mono_summary.md")
    )

    # ... (rest of the configuration loading for auto_heatmap etc. remains the same)
    auto_heatmap_modes_cfg = defaults.get("auto_heatmap_modes")
    if auto_heatmap_modes_cfg:
        if isinstance(auto_heatmap_modes_cfg, str):
            auto_heatmap_modes = [
                m.strip() for m in auto_heatmap_modes_cfg.split(",") if m.strip()
            ]
        else:
            auto_heatmap_modes = [str(auto_heatmap_modes_cfg).strip()]
    else:
        auto_heatmap_modes = ["lab_chroma", "lab_residual"]
    seen_modes: set[str] = set()
    auto_heatmap_modes = [
        m for m in auto_heatmap_modes if not (m in seen_modes or seen_modes.add(m))
    ]
    auto_heatmap_alpha = float(defaults.get("auto_heatmap_alpha", 0.6))
    auto_heatmap_quality = int(defaults.get("auto_heatmap_quality", 92))
    auto_heatmap_suffix = defaults.get("auto_heatmap_suffix")
    auto_heatmap_sat_threshold = float(defaults.get("auto_heatmap_sat_threshold", 0.06))
    auto_heatmap_chroma_clip = float(defaults.get("lab_chroma_clip", 8.0))

    paths = [
        p
        for p in _iter_files(scan_folder, final_exts)
        if not any(tok in p.stem.lower() for tok in OVERLAY_TOKENS)
    ]
    if not paths:
        typer.echo(f"No files matched in {scan_folder} for extensions: {final_exts}")
        raise typer.Exit(1)

    # Ensure JSONL parent directory exists before writing
    out_f = None
    if final_jsonl_out:
        try:
            final_jsonl_out.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        out_f = open(final_jsonl_out, "w")
    csv_file = None
    csv_writer = None
    if csv_out:
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        csv_file = csv_out.open("w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            [
                "title",
                "author",
                "entry_number",
                "filename",
                "method",
                "verdict",
                "mode",
                "reason_summary",
                "failure_reason",
                "split_tone_name",
                "split_tone_description",
                "dominant_color",
                "hue_sigma_deg",
                "hue_peak_delta_deg",
                "hue_second_mass",
                "delta_h_highs_shadows_deg",
                "chroma_p99",
                "chroma_max",
                "pct_gt_c2",
                "pct_gt_c4",
                "largest_cluster_pct_gt_c2",
                "largest_cluster_pct_gt_c4",
                "hue_drift_deg_per_l",
                "loader_status",
                "source_profile",
                "scale_factor",
                "image_path",
            ]
        )
    counts = {"lab": {"pass": 0, "pass_with_query": 0, "fail": 0}}
    results_by_file: Dict[str, Dict[str, Dict[str, Any]]] = {}

    iterator: Iterable[Path]
    if table_only:
        iterator = paths
    else:
        iterator = track(paths, description="Checking")

    for p in iterator:
        res = check_monochrome(
            str(p),
            final_neutral_tol,
            final_toned_pass,
            final_toned_query,
            lab_neutral_chroma=final_lab_neutral_chroma,
            lab_chroma_mask=final_lab_chroma_mask,
            lab_toned_pass_deg=final_lab_toned_pass,
            lab_toned_query_deg=final_lab_toned_query,
            lab_hard_fail_c4_ratio=final_lab_fail_c4_ratio,
            lab_hard_fail_c4_cluster=final_lab_fail_c4_cluster,
            include_grid_regions=include_grid_regions,
        )

        entry = results_by_file.setdefault(p.name, {})
        entry.setdefault("_path", str(p))

        r = res
        counts["lab"].setdefault(r.verdict, 0)
        counts["lab"][r.verdict] += 1
        entry["lab"] = r.__dict__.copy()
        if not summary_only and not table_only:
            typer.echo(f"{p.name:40s}  {_format_method_result(r)}")
        if out_f:
            out_f.write(json.dumps(_result_to_json(str(p), r)) + "\n")

    if out_f:
        out_f.close()

    summary_count_lines: List[str] = []

    # Generate per-subdirectory summary for CLI output
    subdirs: Dict[str, Dict[str, int]] = {}
    for name, methods in sorted(results_by_file.items()):
        lab = methods.get("lab", {})
        verdict = lab.get("verdict")
        bucket = verdict if verdict in {"fail", "pass_with_query", "pass"} else "pass"
        bucket = {"pass_with_query": "query"}.get(bucket, bucket)

        # Extract subdirectory from file path
        file_path = methods.get("_path", "")
        if file_path:
            subdir = str(Path(file_path).parent.name)
        else:
            subdir = "Unknown"

        # Initialize subdirectory counts if needed
        if subdir not in subdirs:
            subdirs[subdir] = {"fail": 0, "query": 0, "pass": 0}

        subdirs[subdir][bucket] += 1

    # Print per-subdirectory summaries
    for subdir in sorted(subdirs.keys()):
        counts_sub = subdirs[subdir]
        fail_count = counts_sub["fail"]
        query_count = counts_sub["query"]
        pass_count = counts_sub["pass"]
        total_files = fail_count + query_count + pass_count

        subdir_line = f"{subdir} ({total_files} files, {fail_count} Fail, {query_count} Query, {pass_count} Pass)"
        typer.echo(subdir_line)

    c = counts["lab"]
    summary_line = (
        f"Summary (lab): PASS={c['pass']}  "
        f"QUERY={c['pass_with_query']}  "
        f"FAIL={c['fail']}"
    )
    typer.echo(summary_line)
    summary_count_lines.append(summary_line)

    grouped_lines = _render_grouped_tables(results_by_file)
    final_lines: List[str] = summary_count_lines[:]
    if grouped_lines:
        if final_lines:
            final_lines.append("")
        final_lines.extend(grouped_lines)
    summary_text = "\n".join(line for line in final_lines if line is not None).strip()
    summary_path = final_summary_path
    try:
        if summary_path.parent != summary_path:
            summary_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    summary_path.write_text(
        (summary_text + "\n") if summary_text else "", encoding="utf-8"
    )

    if csv_writer:
        csv_methods = ["lab"]
        for name in sorted(results_by_file.keys()):
            entry = results_by_file[name]
            file_path = entry.get("_path", "")
            chosen = entry.get("lab")
            title = (chosen or {}).get("title") or ""
            author = (chosen or {}).get("author") or ""
            entry_number = _extract_entry_number(name)
            for method_key in csv_methods:
                res_dict = entry.get(method_key)
                if not isinstance(res_dict, dict):
                    continue
                hue_sigma = res_dict.get("hue_std_deg")
                hue_peak_delta = res_dict.get("hue_peak_delta_deg")
                hue_second_mass = res_dict.get("hue_second_mass")
                hilo_delta = res_dict.get("delta_h_highs_shadows_deg")
                chroma_p99 = res_dict.get("chroma_p99")
                chroma_max = res_dict.get("chroma_max")
                chroma_ratio_2 = res_dict.get("chroma_ratio_2")
                chroma_ratio_4 = res_dict.get("chroma_ratio_4")
                cluster_max_2 = res_dict.get("chroma_cluster_max_2")
                cluster_max_4 = res_dict.get("chroma_cluster_max_4")
                hue_drift = res_dict.get("hue_drift_deg_per_l")
                scale_factor = res_dict.get("scale_factor")

                csv_writer.writerow(
                    [
                        title,
                        author,
                        entry_number,
                        name,
                        method_key,
                        res_dict.get("verdict", ""),
                        res_dict.get("mode", ""),
                        res_dict.get("reason_summary", ""),
                        res_dict.get("failure_reason", ""),
                        res_dict.get("split_tone_name", ""),
                        res_dict.get("split_tone_description", ""),
                        res_dict.get("dominant_color", ""),
                        (
                            f"{hue_sigma:.2f}"
                            if isinstance(hue_sigma, (float, int))
                            else ""
                        ),
                        (
                            f"{hue_peak_delta:.2f}"
                            if isinstance(hue_peak_delta, (float, int))
                            else ""
                        ),
                        (
                            f"{hue_second_mass:.4f}"
                            if isinstance(hue_second_mass, (float, int))
                            else ""
                        ),
                        (
                            f"{hilo_delta:.2f}"
                            if isinstance(hilo_delta, (float, int))
                            else ""
                        ),
                        (
                            f"{chroma_p99:.2f}"
                            if isinstance(chroma_p99, (float, int))
                            else ""
                        ),
                        (
                            f"{chroma_max:.2f}"
                            if isinstance(chroma_max, (float, int))
                            else ""
                        ),
                        (
                            f"{chroma_ratio_2:.4f}"
                            if isinstance(chroma_ratio_2, (float, int))
                            else ""
                        ),
                        (
                            f"{chroma_ratio_4:.4f}"
                            if isinstance(chroma_ratio_4, (float, int))
                            else ""
                        ),
                        (
                            f"{cluster_max_2:.4f}"
                            if isinstance(cluster_max_2, (float, int))
                            else ""
                        ),
                        (
                            f"{cluster_max_4:.4f}"
                            if isinstance(cluster_max_4, (float, int))
                            else ""
                        ),
                        (
                            f"{hue_drift:.2f}"
                            if isinstance(hue_drift, (float, int))
                            else ""
                        ),
                        res_dict.get("loader_status", ""),
                        res_dict.get("source_profile", ""),
                        (
                            f"{scale_factor:.3f}"
                            if isinstance(scale_factor, (float, int))
                            else ""
                        ),
                        file_path,
                    ]
                )

    if write_xmp:
        if not final_jsonl_out:
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
            str(final_jsonl_out),
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

    if auto_heatmap:
        fail_infos: List[Tuple[Path, Optional[str]]] = []
        for entry in results_by_file.values():
            base_path = entry.get("_path")
            if not base_path:
                continue
            result_dict = entry.get("lab")
            if isinstance(result_dict, dict):
                verdict = result_dict.get("verdict")
                if verdict in {"fail", "pass_with_query"}:
                    desc = result_dict.get("reason_summary")
                    fail_infos.append(
                        (Path(base_path), desc if verdict != "pass" else desc)
                    )
        if fail_infos:
            generated = 0
            skipped = 0
            for p, summary in fail_infos:
                for mode in auto_heatmap_modes:
                    out_path = _generate_heatmap_image(
                        p,
                        mode,
                        final_neutral_tol,
                        auto_heatmap_sat_threshold,
                        auto_heatmap_chroma_clip,
                        auto_heatmap_alpha,
                        auto_heatmap_quality,
                        auto_heatmap_suffix or "",
                        summary,
                    )
                    if out_path:
                        generated += 1
                    else:
                        skipped += 1

            if generated > 0 and skipped > 0:
                typer.echo(
                    f"Generated {generated} heatmap overlay(s) for review images ({skipped} already existed)"
                )
            elif generated > 0:
                typer.echo(
                    f"Generated {generated} heatmap overlay(s) for review images"
                )
            elif skipped > 0:
                typer.echo(f"Overlays already exist for {skipped} review images")

    if csv_file:
        csv_file.close()


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
        help="Leak metric: channel_diff, saturation, hue, lab_chroma, lab_residual",
    ),
    neutral_tol: Optional[int] = typer.Option(
        None, help="Threshold for channel_diff in 8-bit (only for mode=channel_diff)"
    ),
    sat_threshold: Optional[float] = typer.Option(
        None, help="Saturation threshold in [0,1] (only for mode=saturation or hue)"
    ),
    chroma_clip: Optional[float] = typer.Option(
        None,
        help="Chroma value mapped to max intensity for LAB-based overlays",
    ),
    alpha: float = typer.Option(0.6, help="Overlay opacity in [0,1]"),
    quality: int = typer.Option(92, help="JPEG quality for output"),
    fails_only: bool = typer.Option(
        True,
        help="Only generate overlays for images that failed the checker (default)",
    ),
):
    """Generate heatmap overlays that highlight color leaks.

    - channel_diff: marks pixels where max(|R-G|,|R-B|,|G-B|) > neutral_tol
    - saturation:   marks pixels where HSV S > sat_threshold
    - hue:          same as saturation but colors the overlay by hue
    - lab_chroma:   LAB chroma intensity map (useful for faint casts)
    - lab_residual: LAB hue residual overlay for visualising tone splits
    """
    mode = mode.lower().strip()
    if mode not in {"channel_diff", "saturation", "hue", "lab_chroma", "lab_residual"}:
        raise typer.BadParameter(
            "Invalid mode. Choose one of: channel_diff, saturation, hue, lab_chroma, lab_residual"
        )

    # Load defaults
    defaults = _load_defaults()
    allowed_modes = {"channel_diff", "saturation", "hue", "lab_chroma", "lab_residual"}
    if folder is not None and folder.name in allowed_modes and mode == "channel_diff":
        mode = folder.name
        folder = None

    folder = folder or (
        Path(defaults["default_folder"]) if defaults.get("default_folder") else None
    )
    if folder is None or not folder.exists() or not folder.is_dir():
        typer.echo(
            "Folder not provided or not found. Provide FOLDER or set "
            "tool.imageworks.mono.default_folder in pyproject.toml"
        )
        raise typer.Exit(1)
    exts = exts or defaults.get("default_exts", "jpg,jpeg,png,tif,tiff")
    out_suffix = out_suffix or defaults.get("default_visualize_suffix", "_mono_vis")
    neutral_tol = int(
        neutral_tol if neutral_tol is not None else defaults.get("neutral_tol", 2)
    )
    sat_threshold = float(
        sat_threshold
        if sat_threshold is not None
        else defaults.get("sat_threshold", 0.06)
    )
    chroma_clip_val = float(
        chroma_clip if chroma_clip is not None else defaults.get("lab_chroma_clip", 8.0)
    )

    paths = list(_iter_files(folder, exts))
    if not paths:
        typer.echo(f"No files matched in {folder} for extensions: {exts}")
        raise typer.Exit(1)

    overlay_tokens = [
        "mono_vis",
        "_lab_chroma",
        "_lab_residual",
        "_diff",
        "_sat",
        "_hue",
    ]
    paths = [
        p for p in paths if not any(tok in p.stem.lower() for tok in overlay_tokens)
    ]

    if fails_only:
        verdicts: Dict[str, Dict[str, Any]] = {}
        jsonl_path = None
        if defaults.get("default_jsonl"):
            jsonl_path = Path(defaults["default_jsonl"])
        if jsonl_path and jsonl_path.exists():
            try:
                with jsonl_path.open() as f:
                    for line in f:
                        if not line.strip():
                            continue
                        obj = json.loads(line)
                        verdicts[Path(obj.get("path", "")).name] = {
                            "verdict": obj.get("verdict", ""),
                            "reason_summary": obj.get("reason_summary"),
                        }
            except Exception:
                verdicts = {}
        if not verdicts:
            typer.echo(
                "No JSONL verdicts found; running quick check to identify failures..."
            )
            for p in track(paths, description="Scanning"):
                res = check_monochrome(str(p))
                verdicts[p.name] = {
                    "verdict": res.verdict,
                    "reason_summary": res.reason_summary,
                }
            typer.echo("Scan complete.")
        filtered = []
        summaries: Dict[str, str] = {}
        for p in paths:
            info = verdicts.get(p.name)
            if info and info.get("verdict") == "fail":
                filtered.append(p)
                summaries[p.name] = info.get("reason_summary")
        paths = filtered
        if not paths:
            typer.echo("No failed images found; skipping overlay generation.")
            return

    generated = []
    for p in track(paths, description="Visualizing"):
        summary = None
        if fails_only:
            summary = summaries.get(p.name) if "summaries" in locals() else None
        out_path = _generate_heatmap_image(
            p,
            mode,
            neutral_tol,
            sat_threshold,
            chroma_clip_val,
            alpha,
            quality,
            out_suffix or "",
            summary,
        )
        if out_path:
            generated.append(out_path)

    typer.echo(
        f"Wrote {len(generated)} overlays with suffix '{out_suffix}' next to originals."
    )


OVERLAY_TOKENS = [
    "mono_vis",
    "_lab_chroma",
    "_lab_residual",
    "_diff",
    "_sat",
    "_hue",
]
