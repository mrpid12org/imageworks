from __future__ import annotations
import io
from dataclasses import dataclass
import warnings
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from xml.etree import ElementTree as ET

try:
    from PIL import ImageCms  # type: ignore

    _IMAGECMS_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on Pillow build
    ImageCms = None  # type: ignore
    _IMAGECMS_AVAILABLE = False

Verdict = Literal["pass", "pass_with_query", "fail"]
Mode = Literal["neutral", "toned", "not_mono"]

LAB_TONED_PASS_DEFAULT = 10.0
LAB_TONED_QUERY_DEFAULT = 14.0
# Allow exceptionally strong but uniform tones to pass if the hue spread remains
# narrow and the frame stays single-hued.
LAB_STRONG_TONE_HUE_STD = 14.0
LAB_STRONG_TONE_CONCENTRATION = 0.85
LAB_STRONG_TONE_PRIMARY_SHARE = 0.97
LAB_STRONG_TONE_HUE_TOLERANCE = 15.0  # degrees
# Stage-lit override: identify files where most of the frame is neutral shadow
# but the lit subject carries a single hue.
LAB_SHADOW_NEUTRAL_L = 24.0
LAB_SHADOW_NEUTRAL_CHROMA = 2.0
LAB_SHADOW_QUERY_SHARE = 0.55
LAB_SHADOW_QUERY_HUE_STD = 24.0
LAB_SHADOW_QUERY_PRIMARY_SHARE = 0.95
# If strong colour covers more than ~4% of the frame we treat the image as a
# definite failure unless the caller relaxes these limits.
LAB_HARD_FAIL_C4_RATIO_DEFAULT = 0.10
LAB_HARD_FAIL_C4_CLUSTER_DEFAULT = 0.08
# Morphological kernel size (as a fraction of width/height) used when we merge
# speckles while searching for chroma clusters.
CLUSTER_KERNEL_FRACTION = 0.08


@dataclass
class LoaderDiagnostics:
    """Metadata about how an image was standardised to sRGB."""

    icc_status: str
    icc_profile_name: Optional[str]
    cms_available: bool
    title: Optional[str]
    author: Optional[str]
    scale_factor: float


@dataclass
class MonoResult:
    """Result of the monochrome analysis.

    - verdict/mode are the primary labels.
    - channel_max_diff and hue_std_deg are simple global metrics.
    - extended fields explain "why" an image passed/failed (useful for UI,
      Lightroom keywords, or tuning).
    """

    verdict: Verdict
    mode: Mode
    channel_max_diff: float
    hue_std_deg: float
    # Optional extended diagnostics (filled when computable)
    dominant_hue_deg: Optional[float] = None
    dominant_color: Optional[str] = None
    hue_concentration: float = 0.0  # resultant length R in [0,1]
    hue_bimodality: float = 0.0  # |E[e^{i 2θ}]| in [0,1]
    sat_median: float = 0.0  # median saturation in [0,1]
    colorfulness: float = 0.0  # Hasler–Süsstrunk metric
    failure_reason: Optional[str] = (
        None  # e.g., 'split_toning_suspected', 'multi_color'
    )
    split_tone_name: Optional[str] = None  # e.g., 'Teal-Orange', 'Yellow-Blue'
    split_tone_description: Optional[str] = None
    # Top hue peaks (up to 3) for diagnostics
    top_hues_deg: Optional[List[float]] = None
    top_colors: Optional[List[str]] = None
    top_weights: Optional[List[float]] = None
    # Method / pipeline metadata
    analysis_method: str = "lab"
    loader_status: Optional[str] = None
    source_profile: Optional[str] = None
    # LAB-specific diagnostics (optional)
    chroma_max: Optional[float] = None
    chroma_median: Optional[float] = None
    chroma_p95: Optional[float] = None
    chroma_p99: Optional[float] = None
    chroma_ratio_2: float = 0.0
    chroma_ratio_4: float = 0.0
    chroma_cluster_max_2: float = 0.0
    chroma_cluster_max_4: float = 0.0
    shadow_share: float = 0.0
    subject_share: float = 0.0
    reason_summary: Optional[str] = None
    title: Optional[str] = None
    author: Optional[str] = None
    scale_factor: float = 1.0
    hue_drift_deg_per_l: Optional[float] = None
    confidence: Optional[str] = None
    # New diagnostics for split-tone gating
    hue_peak_delta_deg: Optional[float] = None  # Δh between top 2 peaks (deg)
    hue_second_mass: Optional[float] = None  # mass of 2nd peak (0–1, chroma-weighted)
    mean_hue_highs_deg: Optional[float] = None  # circular mean hue in top L* quartile
    mean_hue_shadows_deg: Optional[float] = (
        None  # circular mean hue in bottom L* quartile
    )
    delta_h_highs_shadows_deg: Optional[float] = (
        None  # circular Δh between highs & shadows
    )
    hue_weighting: Optional[str] = None  # e.g., "chroma"


@dataclass
class SplitToneRecipe:
    name: str
    h1_range: Tuple[float, float]  # Warmer tone
    h2_range: Tuple[float, float]  # Cooler tone
    delta_range: Tuple[float, float]
    description: str


SPLIT_TONE_RECIPES = [
    SplitToneRecipe(
        name="Sepia-Cyanotype",
        h1_range=(25, 45),
        h2_range=(195, 215),
        delta_range=(150, 185),
        description="Historic darkroom look with aged highlights and cool, paper-like shadows.",
    ),
    SplitToneRecipe(
        name="Gold-Blue",
        h1_range=(10, 55),
        h2_range=(220, 250),
        delta_range=(150, 200),
        description="Classic cinematic look with gold highlights and deep blue shadows.",
    ),
    SplitToneRecipe(
        name="Teal-Orange",
        h1_range=(20, 70),
        h2_range=(170, 230),
        delta_range=(130, 200),
        description="Modern cinematic look with warm skin tones and cool, deep shadows.",
    ),
    SplitToneRecipe(
        name="Yellow-Blue",
        h1_range=(50, 80),
        h2_range=(210, 260),
        delta_range=(150, 200),
        description="Classic complementary split with luminous highlights and clean, cool shadows.",
    ),
    SplitToneRecipe(
        name="Yellow-Cyan",
        h1_range=(40, 70),
        h2_range=(175, 195),
        delta_range=(120, 150),
        description="Gentle, airy split with high-key feel and soft cool tones in the darks.",
    ),
    SplitToneRecipe(
        name="Gold-Indigo",
        h1_range=(45, 65),
        h2_range=(250, 270),
        delta_range=(140, 160),
        description="Royal warmth in lights and inky, dramatic shadows, suited to night cityscapes.",
    ),
    SplitToneRecipe(
        name="Magenta-Green",
        h1_range=(290, 340),
        h2_range=(90, 150),
        delta_range=(150, 210),
        description="High-impact, graphic look with electric shadows and vivid highlights.",
    ),
    SplitToneRecipe(
        name="Red-Cyan",
        h1_range=(350, 15),  # Wraps around 0
        h2_range=(180, 200),
        delta_range=(160, 180),
        description="Maximum complementary punch for a graphic, high-contrast feel.",
    ),
]


def _parse_xmp_text(node: Optional[ET.Element], ns: Dict[str, str]) -> Optional[str]:
    if node is None:
        return None
    alt = node.find("rdf:Alt", ns)
    if alt is not None:
        entries = alt.findall("rdf:li", ns)
        chosen: Optional[str] = None
        for li in entries:
            text = (li.text or "").strip()
            if not text:
                continue
            lang = li.attrib.get("{http://www.w3.org/XML/1998/namespace}lang")
            if lang in (None, "x-default"):
                return text
            if chosen is None:
                chosen = text
        if chosen:
            return chosen
    text = (node.text or "").strip() if node.text else ""
    return text or None


def _parse_xmp_metadata_text(xmp_text: str) -> Tuple[Optional[str], Optional[str]]:
    try:
        root = ET.fromstring(xmp_text)
    except ET.ParseError:
        return None, None
    ns = {
        "dc": "http://purl.org/dc/elements/1.1/",
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    }
    title = None
    for tag in ("title", "description"):
        for node in root.findall(f".//dc:{tag}", ns):
            title = _parse_xmp_text(node, ns)
            if title:
                break
        if title:
            break
    author = None
    creator = root.find(".//dc:creator", ns)
    if creator is not None:
        seq = creator.find("rdf:Seq", ns)
        if seq is not None:
            for li in seq.findall("rdf:li", ns):
                text = (li.text or "").strip()
                if text:
                    author = text
                    break
    return title, author


def _parse_xmp_metadata_bytes(data: bytes) -> Tuple[Optional[str], Optional[str]]:
    if not data:
        return None, None
    if isinstance(data, bytes):
        text = data.decode("utf-8", "ignore")
    else:
        text = str(data)
    marker = text.find("<x:xmpmeta")
    if marker != -1:
        text = text[marker:]
    return _parse_xmp_metadata_text(text)


def _read_sidecar_metadata(path: str) -> Tuple[Optional[str], Optional[str]]:
    p = Path(path)
    candidates = [Path(str(p) + ".xmp"), Path(str(p) + ".XMP")]
    try:
        candidates.append(p.with_suffix(".xmp"))
    except ValueError:
        pass
    try:
        candidates.append(p.with_suffix(".XMP"))
    except ValueError:
        pass
    seen = set()
    for cand in candidates:
        if cand in seen or not cand.exists():
            continue
        seen.add(cand)
        try:
            text = cand.read_text(encoding="utf-8")
        except Exception:
            try:
                data = cand.read_bytes()
            except Exception:
                continue
            title, author = _parse_xmp_metadata_bytes(data)
        else:
            title, author = _parse_xmp_metadata_text(text)
        if title or author:
            return title, author
    return None, None


def _srgb_to_linear(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    return np.where(arr <= 0.04045, arr / 12.92, np.power((arr + 0.055) / 1.055, 2.4))


def _linear_to_srgb(arr: np.ndarray) -> np.ndarray:
    arr = np.clip(arr, 0.0, 1.0)
    return np.where(
        arr <= 0.0031308, arr * 12.92, 1.055 * np.power(arr, 1.0 / 2.4) - 0.055
    )


def _hue_drift_deg_per_l(
    L: np.ndarray, hue_deg: np.ndarray, mask: np.ndarray
) -> Optional[float]:
    if not np.any(mask):
        return None
    L_sel = L[mask].astype(np.float32)
    hue_sel = hue_deg[mask].astype(np.float32)
    if L_sel.size < 10:
        return None
    order = np.argsort(L_sel)
    L_sorted = L_sel[order]
    if np.allclose(L_sorted[0], L_sorted[-1]):
        return None
    hue_sorted = np.deg2rad(hue_sel[order])
    hue_unwrapped = np.unwrap(hue_sorted)
    slope_rad = float(np.polyfit(L_sorted, hue_unwrapped, 1)[0])
    return float(np.rad2deg(slope_rad))


def _load_srgb_array(
    path: str, max_side: int = 1024
) -> Tuple[np.ndarray, LoaderDiagnostics]:
    """Load image as sRGB uint8, honouring embedded ICC profiles when possible."""

    title, author = _read_sidecar_metadata(path)

    with Image.open(path) as im:
        if title is None or author is None:
            xmp_bytes = None
            getxmp = getattr(im, "getxmp", None)
            if callable(getxmp):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        xmp_bytes = getxmp()
                except Exception:
                    xmp_bytes = None
            if not xmp_bytes:
                xmp_bytes = im.info.get("XML:com.adobe.xmp")
            if xmp_bytes:
                t2, a2 = _parse_xmp_metadata_bytes(xmp_bytes)
                if title is None and t2:
                    title = t2
                if author is None and a2:
                    author = a2
        icc_status = "unknown"
        profile_name: Optional[str] = None

        icc_bytes = im.info.get("icc_profile")
        if icc_bytes and _IMAGECMS_AVAILABLE:
            try:
                src_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc_bytes))
                profile_name = ImageCms.getProfileName(src_profile)
                dst_profile = ImageCms.createProfile("sRGB")
                im = ImageCms.profileToProfile(
                    im, src_profile, dst_profile, outputMode="RGB"
                )
                icc_status = "embedded_converted_to_srgb"
            except Exception:
                # Fallback to Pillow conversion when ICC handling fails
                im = im.convert("RGB")
                icc_status = "icc_error_fallback_srgb"
        else:
            im = im.convert("RGB")
            if icc_bytes and not _IMAGECMS_AVAILABLE:
                icc_status = "embedded_assumed_srgb_no_lcms"
                profile_name = "embedded (unhandled)"
            elif icc_bytes:
                icc_status = "embedded_assumed_srgb"
                profile_name = "embedded"
            else:
                icc_status = "no_profile_assumed_srgb"

        w, h = im.size
        scale = max_side / max(w, h) if max(w, h) > max_side else 1.0
        arr = np.asarray(im, dtype=np.uint8)
        if scale < 1.0:
            arr_float = arr.astype(np.float32) / 255.0
            arr_linear = _srgb_to_linear(arr_float)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            arr_linear = cv2.resize(
                arr_linear,
                (new_w, new_h),
                interpolation=cv2.INTER_AREA,
            )
            arr_float = _linear_to_srgb(arr_linear)
            arr = np.clip(arr_float * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
        else:
            scale = 1.0

    loader_diag = LoaderDiagnostics(
        icc_status=icc_status,
        icc_profile_name=profile_name,
        cms_available=_IMAGECMS_AVAILABLE,
        title=title,
        author=author,
        scale_factor=scale,
    )
    return arr, loader_diag


def _neutral_grayscale_test(rgb: np.ndarray, tol: int = 2) -> Tuple[bool, float]:
    """Early exit for true neutral grayscale.

    If every pixel has |R-G|, |R-B|, |G-B| ≤ tol (8-bit), it is neutral.
    """
    r = rgb[..., 0].astype(np.int16)
    g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)
    max_diff = np.max(np.stack([np.abs(r - g), np.abs(r - b), np.abs(g - b)], axis=0))
    return (max_diff <= tol), float(max_diff)


def _toned_monochrome_hue_std(rgb: np.ndarray) -> float:
    """Circular std-dev of hue across sufficiently saturated pixels.

    Toned mono ⇒ small hue spread; color ⇒ larger/multi-modal spread.
    """
    # OpenCV HSV: H∈[0,180); S,V∈[0,255]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h = hsv[..., 0].astype(np.float32) * (360.0 / 180.0)
    s = hsv[..., 1].astype(np.float32) / 255.0
    # ignore nearly neutral pixels
    mask = s > 0.03
    if not np.any(mask):
        return 0.0
    h_sel = h[mask]
    radians = np.deg2rad(h_sel)
    C = np.hypot(np.mean(np.cos(radians)), np.mean(np.sin(radians)))
    C = float(np.clip(C, 1e-8, 1.0))
    circ_std_deg = float(np.rad2deg(np.sqrt(-2.0 * np.log(C))))
    return circ_std_deg


def _circular_std_deg_from_angles(angles_deg: np.ndarray) -> float:
    """Circular standard deviation for arbitrary angle arrays in degrees."""

    if angles_deg.size == 0:
        return 0.0
    radians = np.deg2rad(angles_deg)
    C = np.hypot(np.mean(np.cos(radians)), np.mean(np.sin(radians)))
    C = float(np.clip(C, 1e-8, 1.0))
    return float(np.rad2deg(np.sqrt(-2.0 * np.log(C))))


def _hue_stats(rgb: np.ndarray):
    """Compute hue stats on pixels with S > 0.03.

    Returns: mask, all hue/sat arrays, mean_hue_deg, R (single-mode tightness),
    and R2 (doubled-angle concentration; high for split-toning).
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h = hsv[..., 0].astype(np.float32) * (360.0 / 180.0)
    s = hsv[..., 1].astype(np.float32) / 255.0
    mask = s > 0.03
    if not np.any(mask):
        return mask, h, s, None, 0.0, 0.0
    h_sel = h[mask]
    radians = np.deg2rad(h_sel)
    c = np.mean(np.cos(radians))
    s_ = np.mean(np.sin(radians))
    R = float(np.hypot(c, s_))
    mean_angle = float((np.degrees(np.arctan2(s_, c)) + 360.0) % 360.0)
    # bimodality via angle-doubling resultant
    c2 = np.mean(np.cos(2.0 * radians))
    s2 = np.mean(np.sin(2.0 * radians))
    R2 = float(np.hypot(c2, s2))
    return mask, h, s, mean_angle, R, R2


def _colorfulness_hasler_susstrunk(rgb: np.ndarray) -> float:
    """Hasler–Süsstrunk colorfulness metric (rough "how colorful?").

    Helps separate vivid multicolor failures from near-neutral casts.
    Reference: Hasler & Süsstrunk (2003).
    """
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    rg = r - g
    yb = 0.5 * (r + g) - b
    sigma_rg = float(np.std(rg))
    sigma_yb = float(np.std(yb))
    mu_rg = float(np.mean(rg))
    mu_yb = float(np.mean(yb))
    return float(np.hypot(sigma_rg, sigma_yb) + 0.3 * np.hypot(mu_rg, mu_yb))


LR_HUE_BUCKETS = [
    (15, "red"),
    (45, "orange"),
    (75, "yellow"),
    (105, "green"),
    (165, "aqua"),
    (225, "blue"),
    (285, "purple"),
    (330, "magenta"),
    (360, "red"),
]


def _name_color_from_hue(hue_deg: Optional[float]) -> Optional[str]:
    """Map hue (deg) to a coarse color name for human-friendly output."""
    if hue_deg is None:
        return None
    h = hue_deg % 360.0
    for cutoff, name in LR_HUE_BUCKETS:
        if h < cutoff:
            return name
    return "unknown"


def _name_split_tone_recipe(
    h1: float, h2: float, delta: float
) -> Optional[Tuple[str, str]]:
    """Name common split-tone recipes based on the two dominant hue peaks."""
    # Sort hues into a predictable order (warm, cool) for matching
    # Red/Yellow/Orange hues are considered "warm"
    h1_is_warm = (0 <= h1 <= 75) or (h1 >= 300)
    h2_is_warm = (0 <= h2 <= 75) or (h2 >= 300)
    if not h1_is_warm and h2_is_warm:
        h1, h2 = h2, h1

    for recipe in SPLIT_TONE_RECIPES:
        # Handle hue ranges that wrap around 0° (e.g., for Red)
        h1_in_range = (
            (recipe.h1_range[0] <= h1 <= recipe.h1_range[1])
            if recipe.h1_range[0] < recipe.h1_range[1]
            else (h1 >= recipe.h1_range[0] or h1 <= recipe.h1_range[1])
        )
        h2_in_range = (
            (recipe.h2_range[0] <= h2 <= recipe.h2_range[1])
            if recipe.h2_range[0] < recipe.h2_range[1]
            else (h2 >= recipe.h2_range[0] or h2 <= recipe.h2_range[1])
        )
        delta_in_range = recipe.delta_range[0] <= delta <= recipe.delta_range[1]

        if h1_in_range and h2_in_range and delta_in_range:
            return recipe.name, recipe.description

    return None


def _circular_hue_delta(hue1: float, hue2: float) -> float:
    """Calculates the shortest circular distance between two hues in degrees."""
    delta = abs(hue1 - hue2)
    return min(delta, 360 - delta)


def _top_hue_peaks(
    h_deg: np.ndarray, s: np.ndarray, mask: np.ndarray, k: int = 3
) -> Tuple[List[float], List[float]]:
    """Find up to k dominant hue peaks using a circular histogram.

    Weighted by saturation; smoothed and de-duplicated so peaks are
    well-separated. Useful to describe split-toning/mixed tones.
    Returns: (peak_hues_deg, peak_weights)
    """
    if not np.any(mask):
        return [], []
    h_sel = h_deg[mask]
    w = np.clip(s[mask], 0.0, 1.0).astype(np.float32)
    # 36 bins (10°) circular histogram
    bins = 36
    hist, edges = np.histogram(h_sel, bins=bins, range=(0.0, 360.0), weights=w)
    # circular smoothing
    smooth = np.copy(hist)
    smooth = np.roll(hist, -1) + 2 * hist + np.roll(hist, 1)
    # find local maxima
    peaks = []
    for i in range(bins):
        if smooth[i] > smooth[(i - 1) % bins] and smooth[i] >= smooth[(i + 1) % bins]:
            peaks.append((i, smooth[i]))
    peaks.sort(key=lambda t: t[1], reverse=True)
    selected: List[int] = []
    for idx, _v in peaks:
        if all(
            min((idx - j) % bins, (j - idx) % bins) * (360.0 / bins) >= 20.0
            for j in selected
        ):
            selected.append(idx)
        if len(selected) >= k:
            break
    peak_hues: List[float] = []
    peak_weights: List[float] = []
    width = int(max(1, round((20.0 / (360.0 / bins)))))  # ±~20° window
    for idx in selected:
        # indices in window
        idxs = [(idx + d) % bins for d in range(-width, width + 1)]
        mask_win = np.zeros_like(h_sel, dtype=bool)
        # build mask by checking which h_sel fall into these bins
        bin_idx = np.floor(h_sel / (360.0 / bins)).astype(int) % bins
        for bi in idxs:
            mask_win |= bin_idx == bi
        if not np.any(mask_win):
            continue
        hs = h_sel[mask_win]
        ws = w[mask_win]
        # circular mean with weights
        rad = np.deg2rad(hs)
        c = float(np.sum(np.cos(rad) * ws))
        s_ = float(np.sum(np.sin(rad) * ws))
        ang = (np.degrees(np.arctan2(s_, c)) + 360.0) % 360.0
        peak_hues.append(float(ang))
        peak_weights.append(float(np.sum(ws)))
    return peak_hues, peak_weights


def _qualitative_chroma(value: float) -> str:
    if value < 1.5:
        return "barely above neutral"
    if value < 3.0:
        return "a faint but measurable tint"
    if value < 6.0:
        return "clearly coloured"
    return "strongly coloured"


def _describe_hue_spread(hue_std: float) -> str:
    if hue_std < 12.0:
        return f"Hue variation stays tight (≈{hue_std:.1f}°), consistent with a single tint."
    if hue_std < 45.0:
        return f"Hue variation covers about {hue_std:.1f}°; the tint wanders but stays related."
    return f"Hue variation spans about {hue_std:.1f}°, so multiple colour families are in play."


def _describe_chroma_percentiles(chroma_p99: float, chroma_max: Optional[float]) -> str:
    desc = _qualitative_chroma(chroma_p99)
    if chroma_max is not None and chroma_max > 0:
        return (
            f"Bright regions reach chroma {chroma_max:.2f} and the 99th percentile sits near "
            f"{chroma_p99:.2f}, which looks {desc}."
        )
    return f"The 99th percentile chroma is {chroma_p99:.2f}, which looks {desc}."


def _describe_chroma_footprint(chroma_ratio2: float, chroma_ratio4: float) -> str:
    mild_pct = chroma_ratio2 * 100.0
    strong_pct = chroma_ratio4 * 100.0
    if chroma_ratio2 < 0.005:
        return "Only trace pixels (under 0.5%) creep past the C*2 threshold."
    if chroma_ratio2 < 0.02:
        return (
            f"About {mild_pct:.1f}% of pixels nudge past C*2, with {strong_pct:.1f}% showing "
            "stronger colour (C*4)."
        )
    if chroma_ratio2 < 0.1:
        return (
            f"Roughly {mild_pct:.1f}% of the frame carries a mild tint (C*2) and {strong_pct:.1f}% "
            "pushes into stronger colour."
        )
    return (
        f"Around {mild_pct:.1f}% of pixels sit beyond C*2 and {strong_pct:.1f}% exceed C*4, so the "
        "cast touches a noticeable portion of the frame."
    )


def _describe_hue_drift(hue_drift: float) -> str:
    drift_abs = abs(hue_drift)
    if drift_abs < 10.0:
        return "Hue stays consistent from shadows to highlights."
    if drift_abs < 45.0:
        return f"Hue shifts gently (≈{hue_drift:.1f}°) across the tonal range."
    if drift_abs < 120.0:
        return f"Hue swings by about {hue_drift:.1f}° between darks and lights, so tones respond differently."
    return f"Hue flips by roughly {hue_drift:.1f}° through the tonal range, a strong split-tone signature."


def _describe_median_chroma(chroma_med: float) -> str:
    if chroma_med <= 0.1:
        return "Overall tint is essentially nil."
    if chroma_med <= 0.6:
        return "Overall tint stays very faint."
    if chroma_med <= 1.5:
        return "Overall tint is present but still subtle."
    return "Overall tint strength is obvious across the frame."


def _largest_cluster_fraction(
    mask: np.ndarray, kernel_frac: float = CLUSTER_KERNEL_FRACTION
) -> float:
    if mask.size == 0 or np.max(mask) == 0:
        return 0.0
    mask_uint8 = mask.astype(np.uint8)
    h, w = mask_uint8.shape
    kh = max(1, min(h, int(round(h * kernel_frac))))
    kw = max(1, min(w, int(round(w * kernel_frac))))
    kernel = np.ones((kh, kw), np.uint8)
    # Closing merges tiny speckles so a localised edit shows up as one block.
    closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    num_labels, labels = cv2.connectedComponents(closed, connectivity=8)
    if num_labels <= 1:
        overlap = int(np.sum(mask_uint8[closed.astype(bool)]))
        return float(overlap) / float(mask.size)
    max_pixels = 0
    for label in range(1, num_labels):
        component_mask = labels == label
        overlap = int(np.sum(mask_uint8[component_mask]))
        if overlap > max_pixels:
            max_pixels = overlap
    return float(max_pixels) / float(mask.size)


def _summarize_reason(
    *,
    verdict: Verdict,
    mode: Mode,
    dominant_color: Optional[str],
    dominant_hue: Optional[float],
    hue_std: float,
    hue_drift_deg_per_l: Optional[float],
    chroma_p99: float,
    chroma_med: float,
    failure_reason: Optional[str],
    loader_diag: LoaderDiagnostics,
    hue_clusters: int,
    split_tone_name: Optional[str] = None,
    delta_hue_highs_shadows_deg: Optional[float] = None,
) -> str:
    pieces: List[str] = []

    if verdict == "pass" and mode == "neutral":
        pieces.append(
            "Neutral monochrome detected; residual chroma sits inside tolerance."
        )
    elif verdict in {"pass", "pass_with_query"} and mode == "toned":
        tone = dominant_color or (
            f"{dominant_hue:.0f}°" if dominant_hue is not None else "single hue"
        )
        if verdict == "pass" and hue_std <= LAB_TONED_PASS_DEFAULT:
            pieces.append(
                f"Toned monochrome with a dominant {tone} tint; hue variation stays within the relaxed limit."
            )
        elif verdict == "pass":
            pieces.append(
                f"Toned monochrome with a dominant {tone} tint; hue variation is narrow but stronger than the standard toned limit."
            )
        else:
            pieces.append(
                f"Borderline toned image dominated by {tone}; hue variation is close to the review threshold."
            )
        pieces.append(_describe_median_chroma(chroma_med))
        if hue_clusters > 1:
            pieces.append(
                "Multiple hue clusters detected—confirm the toning is intentional."
            )
    else:
        failure_map = {
            "split_toning_suspected": "Split-toned image with distinct hue families across the frame.",
            "near_neutral_color_cast": "Subtle colour cast remains measurable even though the file is near neutral.",
            "multi_color": "Multiple strong colours appear instead of a single tint.",
            "color_present": "Colour variation exceeds the toned limits for this class.",
        }
        reason_text = failure_map.get(
            failure_reason,
            "Colour structure inconsistent with monochrome criteria.",
        )
        if split_tone_name:
            reason_text += f" (likely {split_tone_name})"
        pieces.append(reason_text)
        if dominant_color:
            pieces.append(f"Dominant tone around {dominant_color}.")

    if hue_drift_deg_per_l is not None and abs(hue_drift_deg_per_l) >= 10.0:
        drift_abs = abs(hue_drift_deg_per_l)
        if drift_abs < 45.0:
            pieces.append("Hue shifts gently as tones brighten.")
        elif drift_abs < 120.0:
            pieces.append("Hue changes noticeably between darks and lights.")
        else:
            pieces.append("Hue flips between hue families across the tonal range.")

    if delta_hue_highs_shadows_deg is not None:
        if delta_hue_highs_shadows_deg < 45.0:
            pieces.append(
                f"Highlights and shadows hues are close (Δh={delta_hue_highs_shadows_deg:.1f}°), indicating a drifted single tone."
            )
        else:
            pieces.append(
                f"Highlights and shadows hues are distinct (Δh={delta_hue_highs_shadows_deg:.1f}°), supporting a split-tone interpretation."
            )

    icc_status = loader_diag.icc_status or ""
    if icc_status.startswith("no_profile"):
        pieces.append("ICC profile missing; assumed sRGB.")
    elif icc_status.startswith("embedded_assumed"):
        pieces.append("Embedded profile ignored (LittleCMS unavailable).")

    return " ".join(pieces).strip()


def _lab_stats(
    rgb: np.ndarray,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
    float,
    float,
]:
    """Return LAB components and chroma statistics."""

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[..., 0] * (100.0 / 255.0)
    a = lab[..., 1] - 128.0
    b = lab[..., 2] - 128.0
    chroma = np.hypot(a, b)
    chroma_max = float(np.max(chroma)) if chroma.size else 0.0
    chroma_med = float(np.median(chroma)) if chroma.size else 0.0
    chroma_p95 = float(np.percentile(chroma, 95)) if chroma.size else 0.0
    chroma_p99 = float(np.percentile(chroma, 99)) if chroma.size else 0.0
    return L, a, b, chroma, chroma_max, chroma_med, chroma_p95, chroma_p99


def _get_mean_hue_in_lightness_bands(
    L: np.ndarray, hue_deg: np.ndarray, mask: np.ndarray, percentile_band: float
) -> Optional[float]:
    """Calculates the mean hue for pixels within a specific lightness percentile band."""
    if not np.any(mask):
        return None

    L_masked = L[mask]
    hue_masked = hue_deg[mask]

    if L_masked.size < 10:  # Require a minimum number of pixels for reliable stats
        return None

    # Determine lightness thresholds for the band
    lower_bound = np.percentile(L_masked, 50 - percentile_band / 2)
    upper_bound = np.percentile(L_masked, 50 + percentile_band / 2)

    # Filter pixels within the lightness band
    band_mask = (L_masked >= lower_bound) & (L_masked <= upper_bound)
    if not np.any(band_mask):
        return None

    hues_in_band = hue_masked[band_mask]
    if hues_in_band.size == 0:
        return None

    # Calculate circular mean hue for the band
    radians = np.deg2rad(hues_in_band)
    c = np.mean(np.cos(radians))
    s = np.mean(np.sin(radians))
    mean_angle = (np.degrees(np.arctan2(s, c)) + 360.0) % 360.0
    return float(mean_angle)


def _check_monochrome_lab(
    rgb: np.ndarray,
    loader_diag: LoaderDiagnostics,
    neutral_tol: int,
    neutral_chroma: float,
    chroma_mask_threshold: float,
    toned_pass_deg: float,
    toned_query_deg: float,
    hard_fail_c4_ratio: float,
    hard_fail_c4_cluster: float,
) -> MonoResult:
    """LAB-based pipeline that measures chroma directly."""

    _, max_diff = _neutral_grayscale_test(rgb, tol=neutral_tol)
    L, a, b, chroma, chroma_max, chroma_med, chroma_p95, chroma_p99 = _lab_stats(rgb)
    cf = _colorfulness_hasler_susstrunk(rgb)

    mask_c2 = chroma > 2.0
    mask_c4 = chroma > 4.0
    chroma_ratio2 = float(np.mean(mask_c2))
    chroma_ratio4 = float(np.mean(mask_c4))
    cluster_max2 = _largest_cluster_fraction(mask_c2)
    cluster_max4 = _largest_cluster_fraction(mask_c4)
    shadow_share = 0.0
    subject_share = 1.0

    hue_deg_all = (np.degrees(np.arctan2(b, a)) + 360.0) % 360.0
    mask = chroma > chroma_mask_threshold
    if not np.any(mask):
        # Fallback: use all pixels with non-zero chroma to avoid empty stats
        mask = chroma > 0.0

    chroma_norm = np.zeros_like(chroma, dtype=np.float32)
    if chroma_max > 1e-6:
        chroma_norm = np.clip(chroma / chroma_max, 0.0, 1.0)

    # --- chroma-weighted circular stats ---
    hues = hue_deg_all[mask]
    hue_drift = _hue_drift_deg_per_l(L, hue_deg_all, mask)
    weights = chroma_norm[mask].astype(np.float32)  # in [0,1]
    wsum = float(np.sum(weights)) or 1.0
    rad = np.deg2rad(hues)
    c = float(np.sum(np.cos(rad) * weights) / wsum) if hues.size else 1.0
    s_ = float(np.sum(np.sin(rad) * weights) / wsum) if hues.size else 0.0
    R = float(np.hypot(c, s_))
    hue_std = float(np.rad2deg(np.sqrt(-2.0 * np.log(max(R, 1e-8)))))
    mean_hue_deg = float((np.degrees(np.arctan2(s_, c)) + 360.0) % 360.0)
    # bimodality (doubled angle), weighted
    rad2 = 2.0 * rad
    c2 = float(np.sum(np.cos(rad2) * weights) / wsum) if hues.size else 1.0
    s2 = float(np.sum(np.sin(rad2) * weights) / wsum) if hues.size else 0.0
    R2 = float(np.hypot(c2, s2))

    dom_color = _name_color_from_hue(mean_hue_deg)
    sat_median_norm = float(np.median(chroma_norm[mask])) if np.any(mask) else 0.0

    peak_hues, peak_w = _top_hue_peaks(hue_deg_all, chroma_norm, mask, k=3)

    if peak_hues and peak_w:
        merged_hues: List[float] = []
        merged_weights: List[float] = []
        for hue, weight in sorted(
            zip(peak_hues, peak_w), key=lambda hw: hw[1], reverse=True
        ):
            inserted = False
            for idx, (mh, mw) in enumerate(zip(merged_hues, merged_weights)):
                if (
                    min(abs(hue - mh), 360.0 - abs(hue - mh))
                    <= LAB_STRONG_TONE_HUE_TOLERANCE
                ):
                    # circular mean for hue, weighted sum for weight
                    total_w = mw + weight
                    if total_w > 0:
                        cos_sum = mw * np.cos(np.deg2rad(mh)) + weight * np.cos(
                            np.deg2rad(hue)
                        )
                        sin_sum = mw * np.sin(np.deg2rad(mh)) + weight * np.sin(
                            np.deg2rad(hue)
                        )
                        merged_hues[idx] = float(
                            (np.degrees(np.arctan2(sin_sum, cos_sum)) + 360.0) % 360.0
                        )
                    merged_weights[idx] = total_w
                    inserted = True
                    break
            if not inserted:
                merged_hues.append(float(hue))
                merged_weights.append(float(weight))
        peak_hues = merged_hues
        peak_w = merged_weights

    # --- post-merge two-peak separation & second mass ---
    peak_delta_deg: Optional[float] = None
    second_mass: float = 0.0
    split_name: Optional[str] = None
    split_description: Optional[str] = None
    if peak_hues and peak_w and len(peak_hues) >= 2 and sum(peak_w) > 0:
        order = np.argsort(peak_w)[::-1]
        h1 = peak_hues[order[0]] % 360.0
        h2 = peak_hues[order[1]] % 360.0
        dh = abs((h1 - h2 + 180.0) % 360.0 - 180.0)
        peak_delta_deg = float(dh)
        total_w = float(sum(peak_w))
        if total_w > 0:
            second_mass = float(peak_w[order[1]] / total_w)
        if peak_delta_deg >= 15.0 and second_mass >= 0.10:
            recipe = _name_split_tone_recipe(h1, h2, peak_delta_deg)
            if recipe:
                split_name, split_description = recipe
            else:
                c1 = _name_color_from_hue(h1)
                c2 = _name_color_from_hue(h2)
                if c1 and c2:
                    split_name = f"{c1}-{c2} Split"

    peak_names = [_name_color_from_hue(hh) or "unknown" for hh in peak_hues]

    # --- highs vs shadows (L*) circular means & Δh ---
    mean_hue_highs_deg: Optional[float] = None
    mean_hue_shadows_deg: Optional[float] = None
    delta_h_highs_shadows_deg: Optional[float] = None
    if np.any(mask):
        L_sel = L[mask]
        q25, q75 = np.percentile(L_sel, [25, 75])
        lows = L_sel <= q25
        highs = L_sel >= q75
        if np.any(lows):
            rad_lo = np.deg2rad(hues[lows])
            w_lo = weights[lows]
            c_lo = float(np.sum(np.cos(rad_lo) * w_lo) / (float(np.sum(w_lo)) or 1.0))
            s_lo = float(np.sum(np.sin(rad_lo) * w_lo) / (float(np.sum(w_lo)) or 1.0))
            mean_hue_shadows_deg = float(
                (np.degrees(np.arctan2(s_lo, c_lo)) + 360.0) % 360.0
            )
        if np.any(highs):
            rad_hi = np.deg2rad(hues[highs])
            w_hi = weights[highs]
            c_hi = float(np.sum(np.cos(rad_hi) * w_hi) / (float(np.sum(w_hi)) or 1.0))
            s_hi = float(np.sum(np.sin(rad_hi) * w_hi) / (float(np.sum(w_hi)) or 1.0))
            mean_hue_highs_deg = float(
                (np.degrees(np.arctan2(s_hi, c_hi)) + 360.0) % 360.0
            )
        if (mean_hue_highs_deg is not None) and (mean_hue_shadows_deg is not None):
            delta_h_highs_shadows_deg = float(
                min(
                    abs(mean_hue_highs_deg - mean_hue_shadows_deg),
                    360.0 - abs(mean_hue_highs_deg - mean_hue_shadows_deg),
                )
            )

    shadow_mask = (L <= LAB_SHADOW_NEUTRAL_L) & (chroma <= LAB_SHADOW_NEUTRAL_CHROMA)
    shadow_share = float(np.mean(shadow_mask)) if shadow_mask.size else 0.0
    subject_share = max(0.0, 1.0 - shadow_share)

    primary_share = 1.0
    if peak_w:
        total_weight = float(sum(peak_w)) or 1.0
        primary_share = float(peak_w[0]) / total_weight

    uniform_strong_tone = (
        hue_std <= LAB_STRONG_TONE_HUE_STD
        and R >= LAB_STRONG_TONE_CONCENTRATION
        and primary_share >= LAB_STRONG_TONE_PRIMARY_SHARE
        and chroma_ratio4 >= 0.05
    )

    single_hue_stage_lit = (
        shadow_share >= LAB_SHADOW_QUERY_SHARE
        and subject_share >= 0.05
        and hue_std <= LAB_SHADOW_QUERY_HUE_STD
        and primary_share >= LAB_SHADOW_QUERY_PRIMARY_SHARE
        and chroma_ratio4 >= 0.05
    )

    # --- two-peak gating thresholds ---
    MERGE_DEG = 12.0  # treat as one colour if <= this
    FAIL_DEG = 15.0  # treat as genuine split if >= this and second_mass big enough
    MINOR_MASS = 0.10  # second peak mass threshold (fraction of chroma-weighted pixels)
    HILO_SPLIT_DEG = 45.0  # split signature between highs and shadows

    merge_ok = (
        (peak_delta_deg is None)
        or (peak_delta_deg <= MERGE_DEG)
        or (second_mass < MINOR_MASS)
    )
    fail_two_peak = (
        (peak_delta_deg is not None)
        and (peak_delta_deg >= FAIL_DEG)
        and (second_mass >= MINOR_MASS)
    )
    hilo_split = (delta_h_highs_shadows_deg is not None) and (
        delta_h_highs_shadows_deg >= HILO_SPLIT_DEG
    )

    force_fail = (
        chroma_p99 >= 6.0
        and (
            chroma_ratio4 >= hard_fail_c4_ratio or cluster_max4 >= hard_fail_c4_cluster
        )
        and hue_std > toned_pass_deg
        and not uniform_strong_tone
    )

    # Common payload for all return paths
    base_result_args = dict(
        channel_max_diff=max_diff,
        hue_std_deg=hue_std,
        dominant_hue_deg=mean_hue_deg,
        dominant_color=dom_color,
        hue_concentration=R,
        hue_bimodality=R2,
        sat_median=sat_median_norm,
        colorfulness=cf,
        split_tone_name=split_name,
        split_tone_description=split_description,
        top_hues_deg=peak_hues,
        top_colors=peak_names,
        top_weights=peak_w,
        analysis_method="lab",
        loader_status=loader_diag.icc_status,
        source_profile=loader_diag.icc_profile_name,
        chroma_max=chroma_max,
        chroma_median=chroma_med,
        chroma_p95=chroma_p95,
        chroma_p99=chroma_p99,
        chroma_ratio_2=chroma_ratio2,
        chroma_ratio_4=chroma_ratio4,
        chroma_cluster_max_2=cluster_max2,
        chroma_cluster_max_4=cluster_max4,
        shadow_share=shadow_share,
        subject_share=subject_share,
        title=loader_diag.title,
        author=loader_diag.author,
        scale_factor=loader_diag.scale_factor,
        hue_drift_deg_per_l=hue_drift,
        hue_peak_delta_deg=peak_delta_deg,
        hue_second_mass=second_mass,
        hue_weighting="chroma",
        mean_hue_highs_deg=mean_hue_highs_deg,
        mean_hue_shadows_deg=mean_hue_shadows_deg,
        delta_h_highs_shadows_deg=delta_h_highs_shadows_deg,
    )

    if chroma_p99 <= neutral_chroma:
        confidence = (
            "low"
            if loader_diag.icc_status.startswith("no_profile")
            or loader_diag.icc_status.startswith("embedded_assumed")
            else "normal"
        )
        return MonoResult(
            verdict="pass",
            mode="neutral",
            failure_reason=None,
            reason_summary=_summarize_reason(
                verdict="pass",
                mode="neutral",
                dominant_color=None,
                dominant_hue=None,
                hue_std=0.0,
                hue_drift_deg_per_l=None,
                chroma_p99=chroma_p99,
                chroma_med=chroma_med,
                failure_reason=None,
                loader_diag=loader_diag,
                hue_clusters=1,
                split_tone_name=split_name,
                delta_hue_highs_shadows_deg=delta_h_highs_shadows_deg,
            ),
            confidence=confidence,
            **base_result_args,
        )

    # Check for toning-collapse
    if (
        fail_two_peak
        and delta_h_highs_shadows_deg is not None
        and delta_h_highs_shadows_deg < 45.0
    ):
        verdict: Verdict = "pass_with_query" if hue_std > toned_pass_deg else "pass"
        reason_summary_base = (
            "Toning collapsed to a single hue family across highlights and shadows."
        )
        if verdict == "pass_with_query":
            reason_summary_base += (
                " Hue variation is wider than typical for a pure single tone."
            )
        return MonoResult(
            verdict=verdict,
            mode="toned",
            failure_reason=None,
            reason_summary=_summarize_reason(
                verdict=verdict,
                mode="toned",
                dominant_color=dom_color,
                dominant_hue=mean_hue_deg,
                hue_std=hue_std,
                hue_drift_deg_per_l=hue_drift,
                chroma_p99=chroma_p99,
                chroma_med=chroma_med,
                failure_reason=None,
                loader_diag=loader_diag,
                hue_clusters=len(peak_hues) if peak_hues else 1,
                split_tone_name=split_name,
                delta_hue_highs_shadows_deg=delta_h_highs_shadows_deg,
            )
            + " "
            + reason_summary_base,
            confidence="review",
            **base_result_args,
        )

    if force_fail and single_hue_stage_lit:
        reason_summary = _summarize_reason(
            verdict="pass_with_query",
            mode="toned",
            dominant_color=dom_color,
            dominant_hue=mean_hue_deg,
            hue_std=hue_std,
            hue_drift_deg_per_l=hue_drift,
            chroma_p99=chroma_p99,
            chroma_med=chroma_med,
            failure_reason=None,
            loader_diag=loader_diag,
            hue_clusters=len(peak_hues) if peak_hues else 1,
            split_tone_name=split_name,
            delta_hue_highs_shadows_deg=delta_h_highs_shadows_deg,
        )
        reason_summary += f" Large neutral-shadow region (~{shadow_share*100:.1f}%) with a single-hue subject—review lighting intent."
        return MonoResult(
            verdict="pass_with_query",
            mode="toned",
            failure_reason=None,
            reason_summary=reason_summary,
            confidence="review",
            **base_result_args,
        )

    if uniform_strong_tone and hue_std > toned_pass_deg:
        return MonoResult(
            verdict="pass",
            mode="toned",
            failure_reason=None,
            reason_summary=_summarize_reason(
                verdict="pass",
                mode="toned",
                dominant_color=dom_color,
                dominant_hue=mean_hue_deg,
                hue_std=hue_std,
                hue_drift_deg_per_l=hue_drift,
                chroma_p99=chroma_p99,
                chroma_med=chroma_med,
                failure_reason=None,
                loader_diag=loader_diag,
                hue_clusters=len(peak_hues) if peak_hues else 1,
                split_tone_name=split_name,
                delta_hue_highs_shadows_deg=delta_h_highs_shadows_deg,
            ),
            confidence="review",
            **base_result_args,
        )

    if not force_fail and hue_std <= toned_pass_deg and merge_ok:
        return MonoResult(
            verdict="pass",
            mode="toned",
            failure_reason=None,
            reason_summary=_summarize_reason(
                verdict="pass",
                mode="toned",
                dominant_color=dom_color,
                dominant_hue=mean_hue_deg,
                hue_std=hue_std,
                hue_drift_deg_per_l=hue_drift,
                chroma_p99=chroma_p99,
                chroma_med=chroma_med,
                failure_reason=None,
                loader_diag=loader_diag,
                hue_clusters=len(peak_hues) if peak_hues else 1,
                split_tone_name=split_name,
                delta_hue_highs_shadows_deg=delta_h_highs_shadows_deg,
            ),
            confidence=(
                "low"
                if loader_diag.icc_status.startswith("no_profile")
                or loader_diag.icc_status.startswith("embedded_assumed")
                else "normal"
            ),
            **base_result_args,
        )

    if not force_fail and (
        hue_std <= toned_query_deg
        or (
            peak_delta_deg is not None
            and MERGE_DEG < peak_delta_deg <= 18.0
            and second_mass < 0.15
        )
    ):
        return MonoResult(
            verdict="pass_with_query",
            mode="toned",
            failure_reason=None,
            reason_summary=_summarize_reason(
                verdict="pass_with_query",
                mode="toned",
                dominant_color=dom_color,
                dominant_hue=mean_hue_deg,
                hue_std=hue_std,
                hue_drift_deg_per_l=hue_drift,
                chroma_p99=chroma_p99,
                chroma_med=chroma_med,
                failure_reason=None,
                loader_diag=loader_diag,
                hue_clusters=len(peak_hues) if peak_hues else 1,
                split_tone_name=split_name,
                delta_hue_highs_shadows_deg=delta_h_highs_shadows_deg,
            ),
            confidence=(
                "low"
                if loader_diag.icc_status.startswith("no_profile")
                or loader_diag.icc_status.startswith("embedded_assumed")
                else "review"
            ),
            **base_result_args,
        )

    # All paths from here lead to a fail verdict
    if fail_two_peak or hilo_split or (R < 0.4 and R2 > 0.6):
        failure_reason = "split_toning_suspected"
    elif cf >= 25.0 or chroma_p95 > neutral_chroma + 8.0:
        failure_reason = "multi_color"
    elif chroma_med < neutral_chroma * 0.75 and hue_std < 30.0:
        failure_reason = "near_neutral_color_cast"
    else:
        failure_reason = "color_present"

    small_footprint = (
        chroma_p99 <= 4.0 and chroma_ratio4 < 0.01 and chroma_ratio2 < 0.08
    )
    moderate_footprint = (
        chroma_p99 <= 8.0 and chroma_ratio4 < 0.05 and chroma_ratio2 < 0.18
    )
    soft_large_footprint = chroma_p99 < 6.0
    subtle_cast = chroma_p99 < 9.0 and chroma_ratio4 < 0.12
    large_drift = hue_drift is not None and abs(hue_drift) > 120.0

    if not force_fail:
        if (small_footprint or (soft_large_footprint and chroma_ratio4 < 0.12)) and (
            large_drift or hue_std < 45.0
        ):
            return MonoResult(
                verdict="pass",
                mode="toned",
                failure_reason=None,
                reason_summary=_summarize_reason(
                    verdict="pass",
                    mode="toned",
                    dominant_color=dom_color,
                    dominant_hue=mean_hue_deg,
                    hue_std=hue_std,
                    hue_drift_deg_per_l=hue_drift,
                    chroma_p99=chroma_p99,
                    chroma_med=chroma_med,
                    failure_reason=None,
                    loader_diag=loader_diag,
                    hue_clusters=len(peak_hues) if peak_hues else 1,
                    split_tone_name=split_name,
                    delta_hue_highs_shadows_deg=delta_h_highs_shadows_deg,
                ),
                confidence="review",
                **base_result_args,
            )
        if (
            moderate_footprint
            or subtle_cast
            or soft_large_footprint
            or (large_drift and chroma_ratio4 < 0.05)
        ):
            return MonoResult(
                verdict="pass_with_query",
                mode="toned",
                failure_reason=None,
                reason_summary=_summarize_reason(
                    verdict="pass_with_query",
                    mode="toned",
                    dominant_color=dom_color,
                    dominant_hue=mean_hue_deg,
                    hue_std=hue_std,
                    hue_drift_deg_per_l=hue_drift,
                    chroma_p99=chroma_p99,
                    chroma_med=chroma_med,
                    failure_reason=None,
                    loader_diag=loader_diag,
                    hue_clusters=len(peak_hues) if peak_hues else 1,
                    split_tone_name=split_name,
                    delta_hue_highs_shadows_deg=delta_h_highs_shadows_deg,
                ),
                confidence="review",
                **base_result_args,
            )

    reason_summary = _summarize_reason(
        verdict="fail",
        mode="not_mono",
        dominant_color=dom_color,
        dominant_hue=mean_hue_deg,
        hue_std=hue_std,
        hue_drift_deg_per_l=hue_drift,
        chroma_p99=chroma_p99,
        chroma_med=chroma_med,
        failure_reason=failure_reason,
        loader_diag=loader_diag,
        hue_clusters=len(peak_hues) if peak_hues else 0,
        split_tone_name=split_name,
        delta_hue_highs_shadows_deg=delta_h_highs_shadows_deg,
    )
    if force_fail:
        reason_summary += (
            f" Strong tint patch exceeds hard limit (pct>C*4 {chroma_ratio4*100:.1f}%,"
            f" largest cluster {cluster_max4*100:.1f}%, C*99 {chroma_p99:.2f})."
        )

    return MonoResult(
        verdict="fail",
        mode="not_mono",
        failure_reason=failure_reason,
        reason_summary=reason_summary,
        confidence=(
            "low"
            if loader_diag.icc_status.startswith("no_profile")
            or loader_diag.icc_status.startswith("embedded_assumed")
            else "normal"
        ),
        **base_result_args,
    )


def check_monochrome(
    path: str,
    neutral_tol: int = 2,
    toned_pass_deg: float = LAB_TONED_PASS_DEFAULT,
    toned_query_deg: float = LAB_TONED_QUERY_DEFAULT,
    *,
    max_side: int = 1024,
    lab_neutral_chroma: float = 2.0,
    lab_chroma_mask: float = 2.0,
    lab_toned_pass_deg: Optional[float] = None,
    lab_toned_query_deg: Optional[float] = None,
    lab_hard_fail_c4_ratio: float = LAB_HARD_FAIL_C4_RATIO_DEFAULT,
    lab_hard_fail_c4_cluster: float = LAB_HARD_FAIL_C4_CLUSTER_DEFAULT,
) -> MonoResult:
    """Single LAB-based pipeline (legacy/IR variants removed)."""

    rgb, loader_diag = _load_srgb_array(path, max_side=max_side)

    lab_pass = lab_toned_pass_deg if lab_toned_pass_deg is not None else toned_pass_deg
    lab_query = (
        lab_toned_query_deg if lab_toned_query_deg is not None else toned_query_deg
    )
    return _check_monochrome_lab(
        rgb,
        loader_diag,
        neutral_tol,
        lab_neutral_chroma,
        lab_chroma_mask,
        lab_pass,
        lab_query,
        lab_hard_fail_c4_ratio,
        lab_hard_fail_c4_cluster,
    )
