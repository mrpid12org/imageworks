"""Tonal analysis utilities for photographic images."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import cv2
import numpy as np


def compute_tonal_metrics(image_path: Path) -> Dict[str, float]:
    """Compute tonal metrics (dynamic range, clipping, local contrast)."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(
            f"Unable to read image for tonal analysis: {image_path}"
        )

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    hist = cv2.calcHist([gray], [0], None, [256], [0, 1]).ravel()
    hist /= hist.sum() if hist.sum() else 1.0
    cdf = np.cumsum(hist)

    p1 = np.searchsorted(cdf, 0.01) / 255.0
    p99 = np.searchsorted(cdf, 0.99) / 255.0
    dynamic_range = float(max(0.0, p99 - p1))

    mean_lum = float(gray.mean())
    histogram_centre = float(np.dot(hist, np.linspace(0, 1, num=256)))
    clip_low = float((gray < 0.02).mean() * 100.0)
    clip_high = float((gray > 0.98).mean() * 100.0)

    local_contrast = _compute_local_contrast(gray)

    mid_tone_range = cdf[192] - cdf[64]  # approx 25%-75%

    return {
        "mean_luminance": mean_lum,
        "dynamic_range": dynamic_range,
        "histogram_centre": histogram_centre,
        "clip_low_percent": clip_low,
        "clip_high_percent": clip_high,
        "local_contrast": local_contrast,
        "mid_tone_range": float(mid_tone_range),
    }


def tonal_summary(metrics: Dict[str, float]) -> str:
    """Generate a short textual description of tonal metrics."""
    summary = []
    dyn = metrics.get("dynamic_range", 0.0)
    if dyn < 0.5:
        summary.append("Overall contrast appears subdued.")
    elif dyn > 0.8:
        summary.append("Strong tonal spread with punchy contrast.")

    mean_lum = metrics.get("mean_luminance", 0.5)
    if mean_lum < 0.4:
        summary.append("Exposure leans dark.")
    elif mean_lum > 0.6:
        summary.append("Exposure leans bright.")

    if metrics.get("clip_high_percent", 0.0) > 1.0:
        summary.append("Highlights show minor clipping.")
    if metrics.get("clip_low_percent", 0.0) > 1.0:
        summary.append("Deep shadows verge on blockage.")

    local = metrics.get("local_contrast", 0.0)
    if local < 0.03:
        summary.append("Local contrast is a little flat.")
    elif local > 0.08:
        summary.append("Local contrast is quite aggressive.")

    if not summary:
        summary.append("Tonal balance looks well controlled with moderate contrast.")
    return " ".join(summary)


def _compute_local_contrast(gray: np.ndarray) -> float:
    try:
        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        return float(lap.std())
    except cv2.error:
        # fallback to simple gradient magnitude using numpy
        gx = np.gradient(gray, axis=0)
        gy = np.gradient(gray, axis=1)
        grad = np.sqrt(gx**2 + gy**2)
        return float(np.std(grad))


__all__ = ["compute_tonal_metrics", "tonal_summary"]
