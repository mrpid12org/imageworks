"""Report generation for the image similarity checker.

Adds optional display of image metadata titles (when present) alongside
filenames in the Markdown report.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from .models import CandidateSimilarity

# Optional metadata readers
try:  # pragma: no cover - optional dependency
    import exifread  # type: ignore
except Exception:  # pragma: no cover
    exifread = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from libxmp import XMPFiles  # type: ignore
    from libxmp.consts import XMP_NS_DC  # type: ignore
except Exception:  # pragma: no cover
    XMPFiles = None  # type: ignore[assignment]
    XMP_NS_DC = None  # type: ignore[assignment]


def _decode_windows_xp(value) -> str:
    """Decode EXIF XPTitle/XP fields (UTF-16-LE with null padding)."""
    try:
        if isinstance(value, bytes):
            raw = value
        else:
            # exifread returns a Byte values object; convert via values
            raw = bytes(value.values)  # type: ignore[attr-defined]
        text = raw.decode("utf-16le", errors="ignore")
        return text.replace("\x00", "").strip()
    except Exception:
        return ""


def _read_title_from_xmp(path: Path) -> str | None:
    if XMPFiles is None:
        return None
    try:
        xmpfile = XMPFiles(file_path=str(path))
        xmp = xmpfile.get_xmp()
        xmpfile.close_file()
        if not xmp:
            return None
        # Try localized title first
        try:
            ok, title, _ = xmp.get_localized_text(XMP_NS_DC, "title", None, "x-default")  # type: ignore[arg-type]
            if ok and title:
                return str(title).strip() or None
        except Exception:
            pass
        # Fallback: first item from dc:title array
        try:
            items = xmp.get_property(XMP_NS_DC, "title")  # type: ignore[arg-type]
            if items:
                text = str(items).strip()
                return text or None
        except Exception:
            pass
    except Exception:
        return None
    return None


def _read_title_from_exif(path: Path) -> str | None:
    if exifread is None:
        return None
    try:
        with path.open("rb") as f:
            tags = exifread.process_file(f, details=False)  # type: ignore[attr-defined]
        # Common fields
        for key in (
            "Image ImageDescription",
            "EXIF ImageDescription",
        ):
            if key in tags:
                val = str(tags[key]).strip()
                if val:
                    return val
        # Windows XPTitle (UTF-16-LE)
        for key in ("Image XPTitle", "EXIF XPTitle"):
            if key in tags:
                decoded = _decode_windows_xp(tags[key])
                if decoded:
                    return decoded
    except Exception:
        return None
    return None


def _image_label(path: Path) -> str:
    """Return a human-friendly label including filename and optional title.

    Format: "<filename> — \"<title>\"" when a title is available; otherwise just filename.
    """
    title = _read_title_from_xmp(path) or _read_title_from_exif(path)
    name = path.name
    if title:
        # Guard against excessively long titles
        safe_title = title.strip()
        if len(safe_title) > 200:
            safe_title = safe_title[:197] + "..."
        return f'{name} — "{safe_title}"'
    return name


def write_jsonl(results: Sequence[CandidateSimilarity], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result.to_json(), ensure_ascii=False))
            handle.write("\n")


def write_markdown(results: Sequence[CandidateSimilarity], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = ["# Image Similarity Checker Report", ""]
    if not results:
        lines.append("No candidate images were processed.")
    else:
        lines.extend(
            [
                "| Candidate | Verdict | Top score | Best match | Strategies |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for result in results:
            best = result.best_match()
            best_path = _image_label(best.reference) if best else "—"
            strategies = (
                ", ".join(sorted(best.extra.get("strategies", {}).keys()))
                if best
                else "—"
            )
            lines.append(
                "| {candidate} | {verdict} | {score:.3f} | {best_match} | {strategies} |".format(
                    candidate=_image_label(result.candidate),
                    verdict=result.verdict.value.upper(),
                    score=result.top_score,
                    best_match=best_path,
                    strategies=strategies or "—",
                )
            )
        lines.append("")

        for result in results:
            lines.append(f"## {_image_label(result.candidate)}")
            lines.append("")
            lines.append(f"- Verdict: **{result.verdict.value.upper()}**")
            lines.append(f"- Top score: {result.top_score:.3f} ({result.metric})")
            if result.matches:
                lines.append("- Matches:")
                for match in result.matches:
                    strategies = match.extra.get("strategies", {})
                    strategy_summary = ", ".join(
                        f"{name}={score:.3f}" for name, score in strategies.items()
                    )
                    lines.append(
                        f"  - {_image_label(match.reference)}: {match.score:.3f} [{strategy_summary or match.strategy}]"
                    )
            if result.notes:
                lines.append("- Notes:")
                for note in result.notes:
                    lines.append(f"  - {note}")
            lines.append("")

    # Optional: append performance metrics if a sidecar JSON exists next to summaries dir
    try:
        metrics_path = Path("outputs/metrics/similarity_perf.json")
        if metrics_path.exists():
            with metrics_path.open("r", encoding="utf-8") as f:
                perf = json.load(f)
            lines.append("## Performance")
            stages = perf.get("stages", {})
            if stages:
                lines.append("")
                lines.append(f"- Overall: {stages.get('overall_run_s', '—')} s")
                lines.append(
                    f"- Discover candidates: {stages.get('discover_candidates_s', '—')} s"
                )
                lines.append(
                    f"- Discover library: {stages.get('discover_library_s', '—')} s"
                )
                prime = stages.get("prime_per_strategy_s", {}) or {}
                match = stages.get("match_per_strategy_s", {}) or {}
                if prime:
                    lines.append("- Prime per strategy:")
                    for k, v in prime.items():
                        lines.append(f"  - {k}: {v} s")
                if match:
                    lines.append("- Matching per strategy:")
                    for k, v in match.items():
                        lines.append(f"  - {k}: {v} s")
                lines.append("")
    except Exception:
        pass

    path.write_text("\n".join(lines), encoding="utf-8")
