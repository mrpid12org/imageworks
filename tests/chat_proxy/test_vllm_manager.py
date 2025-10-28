import asyncio
import sys

from imageworks.chat_proxy.config import ProxyConfig
from imageworks.chat_proxy.vllm_manager import VllmManager
from imageworks.model_loader.models import (
    Artifacts,
    BackendConfig,
    ChatTemplate,
    GenerationDefaults,
    PerformanceSummary,
    Probes,
    RegistryEntry,
    VersionLock,
)


def _minimal_entry(model_path: str) -> RegistryEntry:
    return RegistryEntry(
        name="test-model",
        display_name="Test Model",
        backend="vllm",
        backend_config=BackendConfig(port=24001, model_path=model_path, extra_args=[]),
        capabilities={"text": True},
        artifacts=Artifacts(aggregate_sha256="", files=[]),
        chat_template=ChatTemplate(source="embedded", path=None, sha256=None),
        version_lock=VersionLock(
            locked=False, expected_aggregate_sha256=None, last_verified=None
        ),
        performance=PerformanceSummary(
            rolling_samples=0,
            ttft_ms_avg=None,
            throughput_toks_per_s_avg=None,
            last_sample=None,
        ),
        probes=Probes(vision=None),
        profiles_placeholder=None,
        metadata={},
        generation_defaults=GenerationDefaults(),
        served_model_id=None,
        model_aliases=[],
        roles=[],
        license=None,
        source=None,
        deprecated=False,
        family=None,
        source_provider=None,
        quantization=None,
        backend_alternatives=[],
        role_priority={},
        download_format=None,
        download_location=None,
        download_path=None,
        download_size_bytes=None,
        download_files=[],
        download_directory_checksum=None,
        downloaded_at=None,
        last_accessed=None,
    )


def test_build_command_falls_back_without_uv(monkeypatch, tmp_path):
    cfg = ProxyConfig()
    manager = VllmManager(cfg)
    entry = _minimal_entry(str(tmp_path / "weights"))

    def fake_which(cmd: str):
        if cmd == "uv":
            return None
        if cmd in {"python3", "python"}:
            return "/usr/bin/python3"
        return None

    monkeypatch.setattr("imageworks.chat_proxy.vllm_manager.shutil.which", fake_which)

    command = manager._build_command(entry, "served-name", cfg.vllm_port)
    assert command[0] == sys.executable

    # Ensure we clean up async resources
    asyncio.run(manager.aclose())


def test_build_command_prefers_uv_when_available(monkeypatch, tmp_path):
    cfg = ProxyConfig()
    manager = VllmManager(cfg)
    entry = _minimal_entry(str(tmp_path / "weights"))

    def fake_which(cmd: str):
        if cmd == "uv":
            return "/opt/uv"
        return None

    monkeypatch.setattr("imageworks.chat_proxy.vllm_manager.shutil.which", fake_which)

    command = manager._build_command(entry, "served-name", cfg.vllm_port)
    assert command[:3] == ["/opt/uv", "run", "python"]

    asyncio.run(manager.aclose())
