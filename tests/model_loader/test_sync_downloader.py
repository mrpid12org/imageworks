import json
from pathlib import Path
from typer.testing import CliRunner

from imageworks.model_loader import registry
from imageworks.model_loader.cli_sync import app as sync_app

runner = CliRunner()


def _write_downloader_manifest(tmp_path: Path, files: list[str]):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    # create listed files
    for f in files:
        p = model_dir / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"content-{f}")
    manifest = {
        "model-key": {
            "model_name": "author/demo-model",
            "path": str(model_dir),
            "size_bytes": 1234,
            "format_type": "hf",
            "checksum": "abc123",
            "files": files,
        }
    }
    manifest_path = tmp_path / "models.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path, model_dir


def test_sync_downloader_creates_entry_with_hashes(tmp_path: Path, monkeypatch):
    manifest_path, model_dir = _write_downloader_manifest(
        tmp_path, ["config.json", "tokenizer.json"]
    )
    # pre-create empty registry file
    reg_path = tmp_path / "registry.json"
    reg_path.write_text("[]")
    # run sync
    res = runner.invoke(
        sync_app,
        ["sync-downloader", str(manifest_path), "--registry-path", str(reg_path)],
    )
    assert res.exit_code == 0, res.stdout
    # load registry and assert entry present
    loaded = registry.load_registry(reg_path, force=True)
    entry = loaded.get("demo-model")
    assert entry is not None
    assert entry.source is not None
    # source.files should have size and sha256
    for f in entry.source["files"]:
        assert f["size"] is not None
        assert f["sha256"] is not None


def test_sync_downloader_dry_run(tmp_path: Path):
    manifest_path, _ = _write_downloader_manifest(tmp_path, ["config.json"])
    reg_path = tmp_path / "registry.json"
    reg_path.write_text("[]")
    res = runner.invoke(
        sync_app,
        [
            "sync-downloader",
            str(manifest_path),
            "--registry-path",
            str(reg_path),
            "--dry-run",
        ],
    )
    assert res.exit_code == 0
    # registry should remain empty
    loaded_text = reg_path.read_text()
    assert loaded_text.strip() == "[]"


def test_legacy_registry_removed():
    """Legacy in-code personal tagger registry should be fully removed now.

    Import should fail. We accept either ModuleNotFoundError (fully deleted) or ImportError
    (sentinel file raising immediately). This guards against re-introduction of functionality.
    """
    import importlib  # noqa: F401 - intentional re-import in test to assert behavior

    try:
        importlib.import_module("imageworks.apps.personal_tagger.core.model_registry")
    except (ModuleNotFoundError, ImportError):
        return
    else:  # pragma: no cover - defensive
        raise AssertionError(
            "Legacy model_registry unexpectedly importable without error"
        )
