from imageworks.tools.model_downloader.downloader import (
    ModelDownloader,
    RepositoryMetadata,
)
from imageworks.model_loader.registry import get_entry, load_registry


class DummyAnalysis:
    class Repo:
        model_type = "causal"
        library_name = "transformers"

    repository = Repo()


def test_template_promotion_and_hash(tmp_path, monkeypatch):
    # Ensure the strict testing/demo filter doesn't skip this placeholder repo name during test
    monkeypatch.setenv("IMAGEWORKS_IMPORT_INCLUDE_TESTING", "1")
    # Create dummy target dir with a tokenizer_config.json lacking embedded template
    target = tmp_path / "modelA"
    target.mkdir()
    # External template file
    tpl = target / "chat_template.jinja"
    tpl.write_text("{{# system }}You are helpful{{/ system }}", encoding="utf-8")
    # Minimal file to satisfy format detector (mocked)
    (target / "config.json").write_text("{}")

    # Monkeypatch pieces: skip aria2c check and format detection, and registry record_download path
    monkeypatch.setattr(
        "imageworks.tools.model_downloader.downloader.ModelDownloader._check_aria2c",
        lambda self: None,
    )
    md = ModelDownloader()
    # Provide fake repo metadata
    meta = RepositoryMetadata(
        owner="o",
        repo_name="r",
        branch="main",
        repository_id="o/r",
        storage_repo_name="o__r",
        registry_model_name="o__r",
    )

    # Monkeypatch helper functions used inside _register_download flow: detect_format_and_quant -> ("gguf", None)
    monkeypatch.setattr(
        "imageworks.tools.model_downloader.downloader.detect_format_and_quant",
        lambda path: ("gguf", None),
    )

    # Call private inspection directly
    info = md._inspect_chat_templates(target)
    # Simulate register (bypass actual download path writing) by calling _register_download with minimal args
    entry = md._register_download(
        repo_meta=meta,
        primary_format="gguf",
        total_size=0,
        files=[],
        target_dir=target,
        analysis=DummyAnalysis(),
        chat_template_info=info,
    )
    # Reload registry to ensure persistence
    load_registry(force=True)
    reg_entry = get_entry(entry.name)
    assert (
        reg_entry.chat_template.path is not None
        and reg_entry.chat_template.path.endswith("chat_template.jinja")
    )
    assert reg_entry.chat_template.sha256 is not None
    # Metadata markers present
    assert reg_entry.metadata.get("primary_chat_template_file") == "chat_template.jinja"
    assert (
        reg_entry.metadata.get("primary_chat_template_sha256")
        == reg_entry.chat_template.sha256
    )
