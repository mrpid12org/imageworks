import numpy as np
import cv2
from PIL import Image

from imageworks.libs.vision.mono import check_monochrome


def _save(tmp_path, rgb):
    p = tmp_path / "im.png"
    Image.fromarray(rgb).save(p)
    return p


def _toned(hue_deg: float, w=64, h=64, sat=0.2):
    H = np.full((h, w), hue_deg / 2.0, np.float32)  # OpenCV H [0,180)
    S = np.full((h, w), sat * 255.0, np.float32)
    V = np.linspace(80, 220, h, dtype=np.float32)[:, None].repeat(w, axis=1)
    hsv = np.stack([H, S, V], axis=-1).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def test_split_toned_blue_yellow(tmp_path):
    # Left half blue (~220°), right half yellow (~60°)
    left = _toned(220.0, w=32)
    right = _toned(60.0, w=32)
    rgb = np.concatenate([left, right], axis=1)
    p = _save(tmp_path, rgb)
    res = check_monochrome(str(p))
    assert res.verdict == "fail" and res.mode == "not_mono"
    assert res.failure_reason in ("split_toning_suspected", "multi_color")
    # Expect bimodality high and top tones to include blue and yellow-family names
    assert res.hue_bimodality >= 0.6
    assert res.top_colors is not None and len(res.top_colors) >= 2
    names = set(res.top_colors[:2])
    assert any(n in names for n in ("blue", "aqua"))
    assert any(n in names for n in ("yellow", "orange", "green", "purple", "magenta"))


def test_toned_boundary_pass(tmp_path):
    # Slight hue jitter around 30° to stay under toned_pass_deg (a bit lenient)
    base = _toned(30.0)
    p = _save(tmp_path, base)
    res = check_monochrome(str(p), toned_pass_deg=8.0, toned_query_deg=10.0)
    assert res.verdict == "pass" and res.mode == "toned"
    assert res.dominant_color is not None
    # High concentration expected for a narrow single tone
    assert res.hue_concentration > 0.6


def test_boundary_query_and_fail(tmp_path):
    # Construct a mild two-tone that lands between thresholds (query)
    left = _toned(20.0, w=32, sat=0.12)
    right = _toned(40.0, w=32, sat=0.12)
    rgb = np.concatenate([left, right], axis=1)
    p = _save(tmp_path, rgb)
    res_q = check_monochrome(str(p), toned_pass_deg=6.0, toned_query_deg=14.0)
    assert res_q.verdict in ("pass_with_query", "fail")

    # Push beyond query threshold to ensure fail
    res_f = check_monochrome(str(p), toned_pass_deg=4.0, toned_query_deg=6.0)
    assert res_f.verdict == "fail"


def test_uniform_strong_tone_passes(tmp_path):
    # Very strong single hue across the frame should still count as toned mono
    rgb = _toned(60.0, w=64, h=64, sat=0.65)
    p = _save(tmp_path, rgb)
    res = check_monochrome(str(p))
    assert res.verdict in {"pass", "pass_with_query"}
    assert res.mode == "toned"
    assert res.hue_std_deg <= 15.0


def test_stage_lit_single_hue_promotes_query(tmp_path):
    size = 192
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    center = size // 2
    radius = size // 3
    mask = (xx - center) ** 2 + (yy - center) ** 2 <= radius**2

    rgb = np.zeros((size, size, 3), np.uint8)
    rgb[:] = 4  # near-black background

    grad = ((xx - center) / radius).astype(np.float32)
    subject_r = np.clip(140 + grad * 120, 0, 255)
    subject_g = np.clip(20 + grad * 30, 0, 255)
    subject_b = np.clip(210 - grad * 140, 0, 255)

    rgb[..., 0][mask] = subject_r[mask].astype(np.uint8)
    rgb[..., 1][mask] = subject_g[mask].astype(np.uint8)
    rgb[..., 2][mask] = subject_b[mask].astype(np.uint8)

    p = _save(tmp_path, rgb)
    res = check_monochrome(str(p))
    assert res.verdict == "pass_with_query"
    assert res.chroma_ratio_4 > 0.1
    assert res.hue_std_deg > 6.0


def test_stage_lit_multi_hue_still_fails(tmp_path):
    size = 192
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    center = size // 2
    radius = size // 3
    mask = (xx - center) ** 2 + (yy - center) ** 2 <= radius**2
    mask_half = mask & (xx < center)
    mask_other = mask & (xx >= center)

    hsv = np.zeros((size, size, 3), np.uint8)
    hsv[..., 2] = 6
    hsv[..., 1] = 0

    hsv[..., 0][mask_half] = 135  # purple side
    hsv[..., 1][mask_half] = 220
    hsv[..., 2][mask_half] = 230

    hsv[..., 0][mask_other] = 60  # green/yellow side
    hsv[..., 1][mask_other] = 220
    hsv[..., 2][mask_other] = 230

    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    p = _save(tmp_path, rgb)
    res = check_monochrome(str(p))
    assert res.verdict == "fail"
