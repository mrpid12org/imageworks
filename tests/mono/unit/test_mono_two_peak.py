import numpy as np
import cv2
from PIL import Image

from imageworks.libs.vision.mono import check_monochrome


def _save(tmp_path, rgb):
    p = tmp_path / "im.png"
    Image.fromarray(rgb).save(p)
    return p


def _toned(hue_deg: float, w=64, h=64, sat=0.2):
    """Creates a toned image with a given hue."""
    H = np.full((h, w), hue_deg / 2.0, np.float32)  # OpenCV H is [0,180)
    S = np.full((h, w), sat * 255.0, np.float32)
    V = np.linspace(80, 220, h, dtype=np.float32)[:, None].repeat(w, axis=1)
    hsv = np.stack([H, S, V], axis=-1).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def _split_toned_image(hue_highs: float, hue_shadows: float, w=64, h=64, sat=0.2):
    """Creates an image with distinct hues in highlights and shadows."""
    # Create a lightness gradient from dark to light
    L_grad = np.linspace(0, 255, h, dtype=np.uint8)[:, None].repeat(w, axis=1)

    # Create hue channels for highlights and shadows
    # Highlights (top 25% L*) will have hue_highs
    # Shadows (bottom 25% L*) will have hue_shadows
    # Midtones will blend or be neutral
    hue_channel = np.zeros((h, w), dtype=np.float32)
    sat_channel = np.full((h, w), sat * 255.0, dtype=np.float32)

    # Define lightness thresholds for highlights and shadows
    q25_L = np.percentile(L_grad, 25)
    q75_L = np.percentile(L_grad, 75)

    shadow_mask = L_grad <= q25_L
    high_mask = L_grad >= q75_L

    hue_channel[shadow_mask] = hue_shadows / 2.0
    hue_channel[high_mask] = hue_highs / 2.0

    # For midtones, let's just use a blend or a neutral hue for simplicity
    # Here, we'll just let the dominant hues from highs/shadows bleed into midtones
    # or keep midtones relatively neutral if not covered by masks.
    # A more sophisticated approach would involve smooth transitions.
    # For this test, distinct highs/shadows are key.

    hsv = np.stack(
        [hue_channel, sat_channel, L_grad.astype(np.float32)], axis=-1
    ).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def test_two_peak_pass_with_close_hues(tmp_path):
    """
    Tests that an image with two close hue peaks (e.g., yellow and orange)
    is correctly identified as a single-toned image and passes.
    """
    # Left half orange (~35°), right half yellow (~50°)
    left = _toned(35.0, w=32, sat=0.15)
    right = _toned(50.0, w=32, sat=0.15)
    rgb = np.concatenate([left, right], axis=1)
    p = _save(tmp_path, rgb)

    res = check_monochrome(str(p))

    assert res.verdict == "pass"
    assert res.mode == "toned"
    # The two hues are close enough to be merged into a single peak.
    assert len(res.top_hues_deg) == 1
    assert res.hue_peak_delta_deg is None


def test_two_peak_fail_with_distant_hues(tmp_path):
    """
    Tests that an image with two distant hue peaks (a clear split-tone)
    is correctly identified and fails.
    """
    # Left half blue (~220°), right half yellow (~60°)
    left = _toned(220.0, w=32, sat=0.18)
    right = _toned(60.0, w=32, sat=0.18)
    rgb = np.concatenate([left, right], axis=1)
    p = _save(tmp_path, rgb)

    res = check_monochrome(str(p))

    assert res.verdict == "fail"
    assert res.failure_reason == "split_toning_suspected"
    # The delta should be large
    assert res.hue_peak_delta_deg is not None and res.hue_peak_delta_deg > 90.0
    # The secondary mass should be significant
    assert res.hue_second_mass is not None and res.hue_second_mass > 0.4


def test_two_peak_pass_with_low_secondary_mass(tmp_path):
    """
    Tests that an image with a dominant primary tone and a very weak,
    distant secondary tone still passes, as the secondary tone is insignificant.
    """
    # 92% of the image is a gentle tone, 8% is a distant red
    main_tone = _toned(40.0, w=59, sat=0.15)
    secondary_tone = _toned(350.0, w=5, sat=0.15)
    rgb = np.concatenate([main_tone, secondary_tone], axis=1)
    p = _save(tmp_path, rgb)

    # Relax the hue spread tolerance to isolate the secondary_mass logic
    res = check_monochrome(str(p), toned_pass_deg=30.0, toned_query_deg=40.0)

    # Should pass (or be a query) because the second peak's mass is too small
    assert res.verdict in {"pass", "pass_with_query"}
    # The delta should be large
    assert res.hue_peak_delta_deg is not None and res.hue_peak_delta_deg > 80.0
    # The secondary mass should be very small (below the 0.1 threshold)
    assert res.hue_second_mass is not None and res.hue_second_mass < 0.1


def test_hilo_split_fail(tmp_path):
    """
    Tests that an image with distinct hues in highlights and shadows
    (e.g., warm highlights, cool shadows) is correctly identified as a split-tone
    and fails due to the delta_h_highs_shadows_deg.
    """
    # Warm highlights (e.g., orange ~30deg), cool shadows (e.g., cyan ~190deg)
    rgb = _split_toned_image(hue_highs=30.0, hue_shadows=190.0, sat=0.2)
    p = _save(tmp_path, rgb)

    res = check_monochrome(str(p))

    assert res.verdict == "fail"
    assert res.failure_reason == "split_toning_suspected"
    assert res.delta_h_highs_shadows_deg is not None
    assert res.delta_h_highs_shadows_deg > 45.0  # HILO_SPLIT_DEG
    assert res.hue_peak_delta_deg is not None  # Should also have a peak delta
    assert res.hue_second_mass is not None  # Should also have a second mass
