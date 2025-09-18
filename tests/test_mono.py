import numpy as np
from PIL import Image
from libs.vision.mono import check_monochrome


def _save(tmp_path, rgb):
    p = tmp_path / "im.png"
    Image.fromarray(rgb, "RGB").save(p)
    return p


def test_neutral_gray(tmp_path):
    rgb = np.full((64, 64, 3), 128, np.uint8)
    p = _save(tmp_path, rgb)
    res = check_monochrome(str(p))
    assert res.verdict == "pass" and res.mode == "neutral"


def test_toned(tmp_path):
    # constant hue-ish with varying value
    rgb = np.zeros((64, 64, 3), np.uint8)
    rgb[..., 0], rgb[..., 1] = 60, 80
    for i in range(64):
        rgb[i, :, 2] = 100 + i
    p = _save(tmp_path, rgb)
    res = check_monochrome(str(p))
    assert res.verdict in ("pass", "pass_with_query") and res.mode == "toned"


def test_color_fail(tmp_path):
    a = np.zeros((64, 64, 3), np.uint8)
    a[::2, ::2] = (200, 20, 20)
    a[1::2, 1::2] = (20, 200, 20)
    p = _save(tmp_path, a)
    res = check_monochrome(str(p))
    assert res.verdict == "fail" and res.mode == "not_mono"
