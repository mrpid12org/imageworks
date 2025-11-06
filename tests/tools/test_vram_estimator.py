import pytest

from imageworks.tools.vram_estimator import (
    estimate_max_context_k,
    estimate_vram_gib,
)


def test_forward_estimate_simple():
    result = estimate_vram_gib(
        params_billion=8.0,
        quant="fp8",
        layers=36,
        kv_heads=8,
        head_dim=128,
        kv_dtype="fp8",
        context_k=8.0,
        batch=1,
        overhead_gib=1.5,
        vision_gib=1.0,
        frag_factor=1.10,
    )

    assert result.weights_gib == pytest.approx(7.45058, rel=1e-4)
    assert result.kv_gib == pytest.approx(0.5625, rel=1e-4)
    assert result.total_gib == pytest.approx(11.56, rel=1e-2)


def test_max_context_inverse():
    payload = estimate_max_context_k(
        total_vram_gib=12.0,
        params_billion=8.0,
        quant="fp8",
        layers=36,
        kv_heads=8,
        head_dim=128,
        kv_dtype="fp8",
        batch=1,
        overhead_gib=1.5,
        vision_gib=1.0,
        frag_factor=1.10,
        max_context_k=32.0,
    )

    assert payload["context_k"] == pytest.approx(13.6, rel=1e-2)
    assert payload["total_gib"] <= 12.0 + 1e-6
