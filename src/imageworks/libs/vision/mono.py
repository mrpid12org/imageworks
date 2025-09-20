from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Tuple, Optional, List
import numpy as np
from PIL import Image
import cv2

Verdict = Literal["pass", "pass_with_query", "fail"]
Mode = Literal["neutral", "toned", "not_mono"]


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
    # Top hue peaks (up to 3) for diagnostics
    top_hues_deg: Optional[List[float]] = None
    top_colors: Optional[List[str]] = None
    top_weights: Optional[List[float]] = None


def _to_srgb_array(path: str, max_side: int = 1024) -> np.ndarray:
    """Load as sRGB uint8, optionally downscaling for speed.

    Downscaling preserves statistics (hue/saturation) while making large
    inputs fast to process.
    """
    im = Image.open(path).convert("RGB")
    w, h = im.size
    scale = max_side / max(w, h) if max(w, h) > max_side else 1.0
    if scale < 1.0:
        im = im.resize(
            (int(w * scale), int(h * scale)), resample=Image.Resampling.BICUBIC
        )
    return np.asarray(im, dtype=np.uint8)


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


def _name_color_from_hue(hue_deg: Optional[float]) -> Optional[str]:
    """Map hue (deg) to a coarse color name for human-friendly output."""
    if hue_deg is None:
        return None
    h = hue_deg % 360.0
    # Coarse names; adjust bins to taste
    bins = [
        (15, "red"),
        (45, "orange"),
        (65, "yellow"),
        (90, "lime"),
        (150, "green"),
        (190, "cyan"),
        (250, "blue"),
        (280, "indigo"),
        (315, "violet"),
        (345, "magenta"),
        (360, "red"),
    ]
    for cutoff, name in bins:
        if h < cutoff:
            return name
    return "unknown"


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


def check_monochrome(
    path: str,
    neutral_tol: int = 2,
    toned_pass_deg: float = 6.0,
    toned_query_deg: float = 10.0,
) -> MonoResult:
    """Decide neutral/toned/not_mono and explain why.

    Decision flow:
    - Neutral PASS if the RGB channel differences never exceed neutral_tol.
    - Else compute hue spread on saturated pixels:
      • PASS/toned if hue_std_deg ≤ toned_pass_deg
      • PASS_WITH_QUERY if between toned_pass_deg and toned_query_deg
      • otherwise FAIL/not_mono

    Failure reasons:
    - split_toning_suspected: low single-mode concentration (R) but high
      doubled-angle concentration (R2) ⇒ two opposite hues.
    - multi_color: high overall colorfulness.
    - near_neutral_color_cast: low median saturation with small hue spread
      that still exceeds thresholds (a faint tint).
    """
    rgb = _to_srgb_array(path)
    is_neutral, max_diff = _neutral_grayscale_test(rgb, tol=neutral_tol)
    if is_neutral:
        # Neutral grayscale: extended metrics mostly irrelevant
        # Compute colorfulness for completeness
        cf = _colorfulness_hasler_susstrunk(rgb)
        return MonoResult(
            "pass",
            "neutral",
            max_diff,
            0.0,
            dominant_hue_deg=None,
            dominant_color=None,
            hue_concentration=0.0,
            hue_bimodality=0.0,
            sat_median=0.0,
            colorfulness=cf,
            failure_reason=None,
            top_hues_deg=[],
            top_colors=[],
            top_weights=[],
        )

    # Compute extended hue stats once
    mask, h_deg_all, s_all, mean_hue_deg, R, R2 = _hue_stats(rgb)
    hue_std = _toned_monochrome_hue_std(rgb)
    sat_median = float(np.median(s_all[mask])) if np.any(mask) else 0.0
    dom_color = _name_color_from_hue(mean_hue_deg)
    cf = _colorfulness_hasler_susstrunk(rgb)
    peak_hues, peak_w = _top_hue_peaks(h_deg_all, s_all, mask, k=3)
    peak_names = [_name_color_from_hue(hh) or "unknown" for hh in peak_hues]

    if hue_std <= toned_pass_deg:
        return MonoResult(
            "pass",
            "toned",
            max_diff,
            hue_std,
            dominant_hue_deg=mean_hue_deg,
            dominant_color=dom_color,
            hue_concentration=R,
            hue_bimodality=R2,
            sat_median=sat_median,
            colorfulness=cf,
            failure_reason=None,
            top_hues_deg=peak_hues,
            top_colors=peak_names,
            top_weights=peak_w,
        )
    if hue_std <= toned_query_deg:
        return MonoResult(
            "pass_with_query",
            "toned",
            max_diff,
            hue_std,
            dominant_hue_deg=mean_hue_deg,
            dominant_color=dom_color,
            hue_concentration=R,
            hue_bimodality=R2,
            sat_median=sat_median,
            colorfulness=cf,
            failure_reason=None,
            top_hues_deg=peak_hues,
            top_colors=peak_names,
            top_weights=peak_w,
        )

    # Failure: try to characterise the nature of color usage
    failure_reason: Optional[str] = None
    # Split toning heuristic: low single-mode concentration
    # but strong antipodal structure in doubled-angle space
    if R < 0.4 and R2 > 0.6:
        failure_reason = "split_toning_suspected"
    else:
        # Distinguish strong multi-colour from weak colour cast
        if cf >= 25.0:
            failure_reason = "multi_color"
        elif sat_median < 0.1 and hue_std < 30.0:
            failure_reason = "near_neutral_color_cast"
        else:
            failure_reason = "color_present"

    return MonoResult(
        "fail",
        "not_mono",
        max_diff,
        hue_std,
        dominant_hue_deg=mean_hue_deg,
        dominant_color=dom_color,
        hue_concentration=R,
        hue_bimodality=R2,
        sat_median=sat_median,
        colorfulness=cf,
        failure_reason=failure_reason,
        top_hues_deg=peak_hues,
        top_colors=peak_names,
        top_weights=peak_w,
    )
