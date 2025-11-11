from dataclasses import replace

import pytest

from imageworks.apps.judge_vision.cli import main as cli_main
from imageworks.apps.judge_vision.config import JudgeVisionConfig


@pytest.fixture
def base_config(tmp_path) -> JudgeVisionConfig:
    return JudgeVisionConfig(
        input_paths=[tmp_path],
        recursive=False,
        image_extensions=(".jpg",),
        backend="vllm",
        base_url="http://localhost:8100/v1",
        api_key="",
        timeout=30,
        max_new_tokens=256,
        temperature=0.2,
        top_p=0.9,
        model=None,
        use_registry=False,
        critique_role="judge",
        skip_preflight=True,
        dry_run=True,
        competition_id=None,
        competition_config=None,
        pairwise_rounds=None,
        pairwise_enabled=False,
        pairwise_threshold=17,
        critique_title_template="{stem}",
        critique_category=None,
        critique_notes="",
        output_jsonl=tmp_path / "judge.jsonl",
        summary_path=tmp_path / "judge.md",
        progress_path=tmp_path / "progress.json",
        enable_musiq=True,
        enable_nima=True,
        iqa_cache_path=tmp_path / "cache.jsonl",
        stage="two-pass",
        iqa_device="cpu",
    )


def test_two_pass_runs_stages_in_order(monkeypatch, base_config, tmp_path):
    cli_main._set_active_lease(None)  # reset global state
    cache_path = base_config.iqa_cache_path
    stages_seen: list[str] = []
    lease_events: list[str] = []

    class DummyLeaseClient:
        def __init__(self, base_url):
            self.base_url = base_url

        def acquire(self, owner: str, reason: str | None = None) -> str:
            lease_events.append(f"acquired:{owner}")
            return "token-123"

        def release(self, token: str) -> None:
            lease_events.append(f"released:{token}")

        def close(self):
            lease_events.append("closed")

    class DummyRunner:
        def __init__(self, config):
            self.config = config

        def run(self):
            stages_seen.append(self.config.stage)
            if self.config.stage == "iqa":
                cache_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(cli_main, "JudgeVisionRunner", DummyRunner)
    monkeypatch.setattr(cli_main, "GpuLeaseClient", DummyLeaseClient)

    cli_main._run_chained_two_pass(base_config)  # noqa: SLF001

    assert stages_seen == ["iqa", "critique"]
    assert lease_events == []


def test_two_pass_gpu_acquires_and_releases_lease(monkeypatch, base_config, tmp_path):
    config = replace(base_config, iqa_device="gpu")
    cache_path = config.iqa_cache_path
    cli_main._set_active_lease(None)
    lease_events: list[str] = []
    stages: list[str] = []

    class DummyLeaseClient:
        def __init__(self, base_url):
            self.base_url = base_url

        def acquire(self, owner: str, reason: str | None = None) -> str:
            lease_events.append(f"acquired:{owner}")
            return "token-xyz"

        def release(self, token: str) -> None:
            lease_events.append(f"released:{token}")

        def close(self):
            lease_events.append("closed")

    class DummyRunner:
        def __init__(self, cfg):
            self.config = cfg

        def run(self):
            stages.append(self.config.stage)

    def fake_container(cfg):
        stages.append(cfg.stage)
        cache_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(cli_main, "GpuLeaseClient", DummyLeaseClient)
    monkeypatch.setattr(cli_main, "JudgeVisionRunner", DummyRunner)
    monkeypatch.setattr(cli_main, "run_stage1_in_container", fake_container)

    cli_main._run_chained_two_pass(config)  # noqa: SLF001

    assert stages == ["iqa", "critique"]
    assert lease_events == [
        "acquired:judge-vision",
        "released:token-xyz",
        "closed",
    ]


def test_two_pass_requires_cache(monkeypatch, base_config):
    cli_main._set_active_lease(None)

    class NoopLeaseClient:
        def __init__(self, base_url):
            pass

        def acquire(self, owner, reason=None):
            return "token"

        def release(self, token):
            pass

        def close(self):
            pass

    class DummyRunner:
        def __init__(self, config):
            self.config = config

        def run(self):
            pass

    monkeypatch.setattr(cli_main, "JudgeVisionRunner", DummyRunner)
    monkeypatch.setattr(cli_main, "GpuLeaseClient", NoopLeaseClient)
    cache_path = base_config.iqa_cache_path
    if cache_path.exists():
        cache_path.unlink()

    with pytest.raises(FileNotFoundError):
        cli_main._run_chained_two_pass(base_config)  # noqa: SLF001


def test_two_pass_falls_back_to_cpu_when_lease_unavailable(monkeypatch, base_config):
    cli_main._set_active_lease(None)
    cache_path = base_config.iqa_cache_path
    devices: list[str] = []

    class DummyLeaseClient:
        def __init__(self, base_url):
            raise cli_main.GpuLeaseUnavailable("no endpoint")

    class DummyRunner:
        def __init__(self, config):
            self.config = config

        def run(self):
            devices.append(self.config.iqa_device)
            if self.config.stage == "iqa":
                cache_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(cli_main, "GpuLeaseClient", DummyLeaseClient)
    monkeypatch.setattr(cli_main, "JudgeVisionRunner", DummyRunner)

    cli_main._run_chained_two_pass(base_config)  # noqa: SLF001

    assert devices[0] == "cpu"


def test_run_stage1_backend_cpu_uses_local_runner(monkeypatch, base_config):
    stages: list[str] = []

    class DummyRunner:
        def __init__(self, config):
            self.config = config

        def run(self):
            stages.append(self.config.stage)

    monkeypatch.setattr(cli_main, "JudgeVisionRunner", DummyRunner)
    cli_main._run_stage1_backend(replace(base_config, stage="iqa"))  # noqa: SLF001

    assert stages == ["iqa"]


def test_run_stage1_backend_gpu_uses_container(monkeypatch, base_config):
    observed: dict[str, str] = {}

    def fake_container(config):
        observed["stage"] = config.stage

    monkeypatch.setattr(cli_main, "run_stage1_in_container", fake_container)

    cli_main._run_stage1_backend(  # noqa: SLF001
        replace(base_config, stage="iqa", iqa_device="gpu")
    )

    assert observed["stage"] == "iqa"
