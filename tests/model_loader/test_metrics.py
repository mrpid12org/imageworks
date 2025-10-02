from imageworks.model_loader.metrics import (
    RollingMetrics,
    PerformanceSample,
    BatchRunMetrics,
)


def test_rolling_metrics_summary():
    rm = RollingMetrics(maxlen=3)
    rm.add(PerformanceSample(ttft_ms=100.0, tokens_generated=10, duration_ms=500.0))
    rm.add(PerformanceSample(ttft_ms=120.0, tokens_generated=12, duration_ms=600.0))
    summary = rm.summary()
    assert summary["rolling_samples"] == 2
    assert summary["ttft_ms_avg"] and 100 < summary["ttft_ms_avg"] < 121
    assert summary["last_sample"]["tokens_generated"] == 12


def test_batch_run_metrics():
    bm = BatchRunMetrics(model_name="m", backend="vllm")
    # simulate two image stages
    t1 = bm.start_stage("image")
    bm.end_stage(t1)
    t2 = bm.start_stage("image")
    bm.end_stage(t2)
    bm.close_batch()
    s = bm.summary()
    assert s["model_name"] == "m"
    assert "image" in s["stages"]
    assert s["stages"]["image"]["count"] == 2
