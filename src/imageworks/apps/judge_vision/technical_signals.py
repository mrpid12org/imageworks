"""Stage 1 technical signal extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, TypeVar, Callable

import numpy as np
from PIL import Image, ImageFilter

from .judge_types import TechnicalSignals

# Import directly from submodules to avoid pulling in mono.py dependencies (cv2)
from imageworks.libs.vision import aesthetic_models
from imageworks.libs.vision import tonal


@dataclass
class TechnicalSignalExtractor:
    """Compute lightweight image quality heuristics."""

    enable_nima: bool = True
    enable_musiq: bool = True
    use_gpu: bool = False

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

        advanced_notes = self._augment_with_deterministic_models(image_path, metrics)
        technical_notes.extend(advanced_notes)

        tonal_summary = None
        try:
            tonal_metrics = tonal.compute_tonal_metrics(image_path)
            metrics.update(tonal_metrics)
            tonal_summary = tonal.tonal_summary(tonal_metrics)
            technical_notes.append(tonal_summary)
        except Exception as exc:  # noqa: BLE001
            technical_notes.append(f"tonal_error: {exc}")

        if not technical_notes:
            technical_notes.append("Signals nominal")
        notes = "; ".join(technical_notes)
        return TechnicalSignals(
            metrics=metrics, notes=notes, tonal_summary=tonal_summary
        )

    def _augment_with_deterministic_models(
        self, image_path: Path, metrics: Dict[str, float]
    ) -> list[str]:
        summaries: list[str] = []

        if self.enable_nima:
            aesthetic = _safe_call(
                aesthetic_models.score_nima,
                image_path,
                "aesthetic",
                self.use_gpu,
            )
            technical = _safe_call(
                aesthetic_models.score_nima,
                image_path,
                "technical",
                self.use_gpu,
            )
            if aesthetic:
                metrics["nima_aesthetic_mean"] = aesthetic["mean"]
                metrics["nima_aesthetic_std"] = aesthetic["std"]
                summaries.append(f"NIMA aesthetic {aesthetic['mean']:.2f}/10")
            if technical:
                metrics["nima_technical_mean"] = technical["mean"]
                metrics["nima_technical_std"] = technical["std"]
                summaries.append(f"NIMA technical {technical['mean']:.2f}/10")

        if self.enable_musiq:
            mos = _safe_call(
                aesthetic_models.score_musiq,
                image_path,
                "spaq",
                self.use_gpu,
            )
            if mos is not None:
                metrics["musiq_mos"] = mos
                summaries.append(f"MUSIQ MOS {mos:.1f}/100")

        return summaries


def _estimate_saturation(image: Image.Image) -> float | None:
    try:
        hsv = image.convert("HSV")
    except Exception:  # noqa: BLE001
        return None
    sat = np.asarray(hsv.split()[1], dtype=np.float32) / 255.0
    return float(np.mean(sat))


T = TypeVar("T")


def _safe_call(fn: Callable[..., T], *args) -> Optional[T]:
    try:
        return fn(*args)
    except Exception:  # noqa: BLE001
        return None
