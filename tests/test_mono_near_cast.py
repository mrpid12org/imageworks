import numpy as np
import cv2
from PIL import Image

from imageworks.libs.vision.mono import check_monochrome


def _save(tmp_path, rgb):
    p = tmp_path / "near_cast.png"
    Image.fromarray(rgb).save(p)
    return p


def _block(hue_deg: float, w=32, h=64, sat=0.06):
    H = np.full((h, w), hue_deg / 2.0, np.float32)  # OpenCV H in [0,180)
    S = np.full((h, w), sat * 255.0, np.float32)
    V = np.linspace(96, 192, h, dtype=np.float32)[:, None].repeat(w, axis=1)
    hsv = np.stack([H, S, V], axis=-1).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def test_near_neutral_color_cast(tmp_path):
    # Low-saturation, slightly different hues across halves -> perceived cast
    left = _block(5.0, w=32)
    right = _block(20.0, w=32)
    rgb = np.concatenate([left, right], axis=1)
    p = _save(tmp_path, rgb)
    # Tight thresholds to force a fail despite low saturation
    res = check_monochrome(str(p), toned_pass_deg=4.0, toned_query_deg=6.0)
    assert res.verdict == "fail"
    assert res.sat_median < 0.1
    assert res.hue_std_deg < 30.0
    assert res.failure_reason == "near_neutral_color_cast"
