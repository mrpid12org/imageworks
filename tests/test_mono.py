import numpy as np
from PIL import Image
from imageworks.libs.vision.mono import check_monochrome


def _save(tmp_path, rgb):
    p = tmp_path / "im.png"
    Image.fromarray(rgb).save(p)
    return p


def test_neutral_gray(tmp_path):
    rgb = np.full((64, 64, 3), 128, np.uint8)
    p = _save(tmp_path, rgb)
    res = check_monochrome(str(p))
    assert res.verdict == "pass" and res.mode == "neutral"


def test_toned(tmp_path):
    # Gentle split tone: slight warm bias in highlights, cool bias in shadows
    rgb = np.full((64, 64, 3), 128, np.uint8)
    ramp = np.linspace(118, 138, 64, dtype=np.uint8)
    for i, val in enumerate(ramp):
        rgb[i, :, 0] = np.clip(val + 4, 0, 255)
        rgb[i, :, 1] = val
        rgb[i, :, 2] = np.clip(val - 3, 0, 255)
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


def test_lab_pipeline_neutral(tmp_path):
    rgb = np.full((128, 128, 3), 200, np.uint8)
    p = _save(tmp_path, rgb)
    res = check_monochrome(str(p))
    assert res.analysis_method == "lab"
    assert res.verdict == "pass"
    assert res.mode == "neutral"
    assert res.chroma_max is not None and res.chroma_max < 0.5
    assert res.reason_summary


def test_lab_pipeline_color_fail(tmp_path):
    rgb = np.zeros((64, 64, 3), np.uint8)
    rgb[::2, ::2] = (180, 40, 20)
    rgb[1::2, 1::2] = (40, 180, 220)
    p = _save(tmp_path, rgb)
    res = check_monochrome(str(p))
    assert res.verdict == "fail"
    assert res.failure_reason in {
        "multi_color",
        "split_toning_suspected",
        "color_present",
    }
    assert res.chroma_max is not None and res.chroma_max > 10.0
    assert res.reason_summary
    assert res.chroma_cluster_max_4 > 0.0


def test_lab_relaxes_to_query_for_small_chroma(tmp_path):
    # Tiny patch of hue leak should produce a query rather than a fail
    rgb = np.full((64, 64, 3), 160, np.uint8)
    rgb[0:8, 0:8, 1] = np.clip(rgb[0:8, 0:8, 1] + 4, 0, 255)  # subtle green tint
    rgb[56:64, 56:64, 0] = np.clip(
        rgb[56:64, 56:64, 0] + 4, 0, 255
    )  # subtle magenta tint
    p = _save(tmp_path, rgb)
    res = check_monochrome(str(p))
    assert res.verdict == "pass_with_query"
    assert res.chroma_ratio_4 < 0.01
    assert res.chroma_cluster_max_4 < 0.01


def test_lab_relaxes_to_pass_for_localised_cast(tmp_path):
    # A tiny patch of colour (single hue) should be treated as an allowable tone
    rgb = np.full((64, 64, 3), 160, np.uint8)
    rgb[0:8, 0:8, 1] = np.clip(rgb[0:8, 0:8, 1] + 4, 0, 255)
    p = _save(tmp_path, rgb)
    res = check_monochrome(str(p))
    assert res.verdict in {"pass", "pass_with_query"}
    assert res.chroma_cluster_max_4 < 0.02


def test_chroma_cluster_highlights_local_edit(tmp_path):
    rgb = np.full((64, 64, 3), 150, np.uint8)
    for y in range(16, 40):
        for x in range(16, 48):
            if (x + y) % 2 == 0:
                rgb[y, x] = (255, 60, 60)
            else:
                rgb[y, x] = (40, 200, 220)
    p = _save(tmp_path, rgb)
    res = check_monochrome(str(p))
    assert res.verdict == "fail"
    assert res.mode == "not_mono"
    assert res.chroma_ratio_4 > 0.05
    assert res.chroma_cluster_max_4 > 0.05
    assert res.reason_summary and "Strong tint patch" in res.reason_summary
