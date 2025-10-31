from types import SimpleNamespace

import pytest

from imageworks.tools.model_downloader.downloader import (
    ModelDownloader,
    RepositoryMetadata,
)
from imageworks.tools.model_downloader.url_analyzer import (
    FileInfo,
    RepositoryInfo,
    AnalysisResult,
)


@pytest.fixture(autouse=True)
def _patch_registry(monkeypatch):
    monkeypatch.setattr(ModelDownloader, "_check_aria2c", lambda self: None)
    monkeypatch.setattr(
        "imageworks.model_loader.registry.load_registry",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        "imageworks.model_loader.download_adapter.load_registry",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        "imageworks.model_loader.registry.update_entries",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "imageworks.model_loader.registry.save_registry",
        lambda *args, **kwargs: None,
    )


def test_partition_files_filters_variants(monkeypatch):
    downloader = ModelDownloader()
    analysis = SimpleNamespace(
        files={
            "model_weights": [
                FileInfo(path="model-q4.gguf", size=1),
                FileInfo(path="model-q5.gguf", size=1),
            ],
            "config": [FileInfo(path="config.json", size=1)],
            "tokenizer": [],
            "optional": [],
            "large_optional": [],
        }
    )

    filtered = downloader._partition_files(analysis, weight_filter={"model-q5.gguf"})
    assert [f.path for f in filtered["weights"]] == ["model-q5.gguf"]

    # Ensure original list untouched when no filter provided
    all_files = downloader._partition_files(analysis)
    assert len(all_files["weights"]) == 2


def test_register_download_includes_support_repo(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "imageworks.tools.model_downloader.downloader.detect_format_and_quant",
        lambda target: ("gguf", "q4_k_m"),
    )

    def fake_record_download(**kwargs):
        entry = SimpleNamespace(
            metadata={},
            chat_template=SimpleNamespace(source="embedded", path=None, sha256=None),
            display_name="test",
            name="test",
            backend_config=SimpleNamespace(port=0, model_path=kwargs["path"]),
            source_provider=kwargs.get("source_provider"),
            served_model_id=kwargs.get("served_model_id"),
            download_path=kwargs["path"],
            download_format=kwargs.get("format_type"),
            download_location=kwargs.get("location"),
            download_files=kwargs.get("files") or [],
            quantization=kwargs.get("quantization"),
            roles=[],
            role_priority={},
            capabilities={},
        )
        return entry

    monkeypatch.setattr(
        "imageworks.model_loader.download_adapter.record_download", fake_record_download
    )

    downloader = ModelDownloader()
    downloader.config.linux_wsl.root = tmp_path
    downloader.config.windows_lmstudio.root = tmp_path / "lmstudio"

    target_dir = tmp_path / "bartowski" / "quant"
    target_dir.mkdir(parents=True)

    # Create dummy files to satisfy checksum + hashing
    for filename in ["model-q4.gguf", "config.json", "tokenizer.json"]:
        (target_dir / filename).write_text("stub")

    repo_meta = RepositoryMetadata(
        owner="bartowski",
        repo_name="quant",
        branch="main",
        repository_id="bartowski/quant",
        storage_repo_name="quant",
        registry_model_name="bartowski/quant",
    )

    analysis = AnalysisResult(
        repository=RepositoryInfo(owner="bartowski", repo="quant", branch="main"),
        files={},
        formats=[],
        total_size=0,
        config_content=None,
    )

    chat_template_info = {
        "has_chat_template": False,
        "has_embedded_chat_template": False,
        "external_chat_template_files": [],
        "embedded_chat_template_preview": None,
    }

    files = [
        FileInfo(path="model-q4.gguf", size=1),
        FileInfo(path="config.json", size=1),
        FileInfo(path="tokenizer.json", size=1),
    ]

    entry = downloader._register_download(
        repo_meta=repo_meta,
        primary_format="gguf",
        total_size=3,
        files=files,
        target_dir=target_dir,
        analysis=analysis,
        chat_template_info=chat_template_info,
        support_repo="dphn/Dolphin",
        selected_weights=["model-q4.gguf"],
    )

    assert entry.metadata.get("support_repository") == "dphn/Dolphin"
    assert entry.metadata.get("selected_weight_files") == ["model-q4.gguf"]
    assert "model-q4.gguf" in entry.download_files
