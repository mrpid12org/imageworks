from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Tuple
import numpy as np
from PIL import Image
import cv2

Verdict = Literal["pass", "pass_with_query", "fail"]
Mode = Literal["neutral", "toned", "not_mono"]


@dataclass
class MonoResult:
    verdict: Verdict
    mode: Mode
    channel_max_diff: float
    hue_std_deg: float


def _to_srgb_array(path: str, max_side: int = 1024) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    w, h = im.size
    scale = max_side / max(w, h) if max(w, h) > max_side else 1.0
    if scale < 1.0:
        im = im.resize(
            (int(w * scale), int(h * scale)), resample=Image.Resampling.BICUBIC
        )
    return np.asarray(im, dtype=np.uint8)


def _neutral_grayscale_test(rgb: np.ndarray, tol: int = 2) -> Tuple[bool, float]:
    r = rgb[..., 0].astype(np.int16)
    g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)
    max_diff = np.max(np.stack([np.abs(r - g), np.abs(r - b), np.abs(g - b)], axis=0))
    return (max_diff <= tol), float(max_diff)


def _toned_monochrome_hue_std(rgb: np.ndarray) -> float:
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


def check_monochrome(
    path: str,
    neutral_tol: int = 2,
    toned_pass_deg: float = 6.0,
    toned_query_deg: float = 10.0,
) -> MonoResult:
    """
    PASS + neutral: channel_max_diff <= neutral_tol
    PASS + toned:   hue_std_deg <= toned_pass_deg
    PASS_WITH_QUERY: toned_pass_deg < hue_std_deg <= toned_query_deg
    FAIL otherwise
    """
    rgb = _to_srgb_array(path)
    is_neutral, max_diff = _neutral_grayscale_test(rgb, tol=neutral_tol)
    if is_neutral:
        return MonoResult("pass", "neutral", max_diff, 0.0)
    hue_std = _toned_monochrome_hue_std(rgb)
    if hue_std <= toned_pass_deg:
        return MonoResult("pass", "toned", max_diff, hue_std)
    if hue_std <= toned_query_deg:
        return MonoResult("pass_with_query", "toned", max_diff, hue_std)
    return MonoResult("fail", "not_mono", max_diff, hue_std)
