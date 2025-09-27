import os
import json
import time
from pathlib import Path

import numpy as np
import pytest
import cv2
from PIL import Image

from imageworks.libs.vision.mono import check_monochrome


pytestmark = pytest.mark.perf


def _write(tmp: Path, arr: np.ndarray) -> Path:
    p = tmp / f"im_{time.time_ns()}.jpg"
    Image.fromarray(arr).save(p, quality=92, subsampling=2)
    return p


def _make_neutral(w: int = 64, h: int = 64, v: int = 128) -> np.ndarray:
    return np.full((h, w, 3), v, np.uint8)


def _make_toned(
    hue_deg: float = 30.0, w: int = 64, h: int = 64, sat: float = 0.15
) -> np.ndarray:
    H = np.full((h, w), hue_deg / 2.0, np.float32)  # OpenCV H in [0,180)
    S = np.full((h, w), sat * 255.0, np.float32)
    V = np.linspace(80, 220, h, dtype=np.float32)[:, None].repeat(w, axis=1)
    hsv = np.stack([H, S, V], axis=-1).astype(np.uint8)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb


@pytest.mark.skipif(
    os.getenv("PYTEST_PERF") != "1", reason="perf tests disabled; set PYTEST_PERF=1"
)
def test_mono_throughput(tmp_path: Path):
    # Generate a small corpus
    corpus = []
    for _ in range(30):
        corpus.append(_write(tmp_path, _make_neutral()))
        corpus.append(_write(tmp_path, _make_toned(hue_deg=40.0)))
        corpus.append(_write(tmp_path, _make_toned(hue_deg=200.0)))

    durations = []
    for p in corpus:
        t0 = time.perf_counter()
        _ = check_monochrome(str(p))
        durations.append((time.perf_counter() - t0) * 1000.0)  # ms

    durations.sort()
    p50 = durations[len(durations) // 2]
    p95 = durations[int(len(durations) * 0.95) - 1]

    out = {
        "count": len(durations),
        "p50_ms": round(p50, 3),
        "p95_ms": round(p95, 3),
    }

    perf_out = os.getenv("PERF_OUT")
    if perf_out:
        Path(perf_out).write_text(json.dumps(out))

    # Soft assertion just to catch egregious regressions without being brittle
    assert p95 < 200.0, f"Mono check p95 too slow: {p95:.1f} ms"
