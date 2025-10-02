from pathlib import Path

from imageworks.model_loader.hashing import update_entry_artifacts_from_download
from imageworks.model_loader.models import (
    RegistryEntry,
    BackendConfig,
    Artifacts,
    ChatTemplate,
    VersionLock,
    PerformanceSummary,
    Probes,
)


def _base_entry(tmpdir: Path) -> RegistryEntry:
    model_dir = tmpdir / "dummy_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    # create files
    (model_dir / "config.json").write_text("{}")
    (model_dir / "weights.bin").write_text("ABC")

    return RegistryEntry(
        name="dummy-download-model",
        display_name="Dummy Download Model",
        backend="unassigned",
        backend_config=BackendConfig(port=0, model_path=str(model_dir), extra_args=[]),
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
        served_model_id=None,
        model_aliases=[],
        roles=[],
        license=None,
        source=None,
        deprecated=False,
        download_format="safetensors",
        download_location="linux_wsl",
        download_path=str(model_dir),
        download_size_bytes=None,
        download_files=["config.json", "weights.bin"],
        download_directory_checksum=None,
        downloaded_at=None,
        last_accessed=None,
    )


def test_download_artifact_hashing(tmp_path: Path):
    entry = _base_entry(tmp_path)
    # compute artifacts from download list
    updated = update_entry_artifacts_from_download(entry)
    assert updated.artifacts.aggregate_sha256, "Expected aggregate hash populated"
    assert {f.path for f in updated.artifacts.files} == {"config.json", "weights.bin"}
    original_hash = updated.artifacts.aggregate_sha256

    # modify one file
    (Path(entry.download_path) / "weights.bin").write_text("ABCD")
    updated2 = update_entry_artifacts_from_download(entry)
    assert (
        updated2.artifacts.aggregate_sha256 != original_hash
    ), "Hash should change after file modification"
