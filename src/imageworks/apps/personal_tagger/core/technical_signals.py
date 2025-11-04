"""Stage 1 technical signal extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image, ImageFilter

from .judge_types import TechnicalSignals


@dataclass
class TechnicalSignalExtractor:
    """Compute lightweight image quality heuristics."""

    def run(self, image_path: Path) -> TechnicalSignals:
        metrics: Dict[str, float] = {}
        notes = ""

        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                grayscale = img.convert("L")
        except Exception as exc:  # noqa: BLE001
            return TechnicalSignals(metrics={}, notes=f"signal_error: {exc}")

        arr = np.asarray(grayscale, dtype=np.float32) / 255.0
        metrics["mean_luma"] = float(np.mean(arr))
        metrics["contrast"] = float(np.std(arr))

        laplacian = grayscale.filter(ImageFilter.FIND_EDGES)
        lap_arr = np.asarray(laplacian, dtype=np.float32) / 255.0
        metrics["edge_density"] = float(np.mean(lap_arr))

        saturation = _estimate_saturation(img)
        if saturation is not None:
            metrics["saturation"] = saturation

        technical_notes = []
        if metrics["contrast"] < 0.08:
            technical_notes.append("low contrast detected")
        if metrics["edge_density"] < 0.05:
            technical_notes.append("limited sharp detail (possible blur)")
        if saturation is not None and saturation < 0.15:
            technical_notes.append("very low saturation (consider mono intent)")
        if not technical_notes:
            technical_notes.append("Signals nominal")
        notes = "; ".join(technical_notes)
        return TechnicalSignals(metrics=metrics, notes=notes)


def _estimate_saturation(image: Image.Image) -> float | None:
    try:
        hsv = image.convert("HSV")
    except Exception:  # noqa: BLE001
        return None
    sat = np.asarray(hsv.split()[1], dtype=np.float32) / 255.0
    return float(np.mean(sat))

