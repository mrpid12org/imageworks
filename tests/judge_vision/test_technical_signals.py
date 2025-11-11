import numpy as np
import pytest
from PIL import Image

from imageworks.apps.judge_vision import technical_signals
from imageworks.apps.judge_vision.technical_signals import TechnicalSignalExtractor


def create_test_image(tmp_path):
    path = tmp_path / "sample.jpg"
    arr = np.full((32, 32, 3), 128, dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)
    return path


def test_extractor_emits_musiq_metrics(monkeypatch, tmp_path):
    calls = {"musiq_variant": None}

    def fake_score_nima(image_path, flavor, use_gpu=False):
        return {"mean": 6.0 if flavor == "aesthetic" else 5.0, "std": 0.5}

    def fake_score_musiq(image_path, variant="spaq", use_gpu=False):
        calls["musiq_variant"] = variant
        return 78.9

    def fake_tonal_metrics(image_path):
        return {"mean_luminance": 0.5}

    def fake_tonal_summary(metrics):
        return "balanced tones"

    monkeypatch.setattr(
        technical_signals.aesthetic_models, "score_nima", fake_score_nima
    )
    monkeypatch.setattr(
        technical_signals.aesthetic_models, "score_musiq", fake_score_musiq
    )
    monkeypatch.setattr(
        technical_signals.tonal, "compute_tonal_metrics", fake_tonal_metrics
    )
    monkeypatch.setattr(technical_signals.tonal, "tonal_summary", fake_tonal_summary)

    extractor = TechnicalSignalExtractor(
        enable_nima=True, enable_musiq=True, use_gpu=False
    )
    image_path = create_test_image(tmp_path)

    signals = extractor.run(image_path)

    assert signals.metrics["musiq_mos"] == pytest.approx(78.9)
    assert calls["musiq_variant"] == "spaq"
    assert signals.tonal_summary == "balanced tones"
