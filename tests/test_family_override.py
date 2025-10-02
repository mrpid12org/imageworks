from imageworks.model_loader.download_adapter import record_download


def test_family_override_and_served_model_id(tmp_path):
    # Create a dummy directory to simulate download path
    d = tmp_path / "dummy"
    d.mkdir()
    (d / "weights.bin").write_bytes(b"123")
    entry = record_download(
        hf_id=None,
        backend="ollama",
        format_type="gguf",
        quantization=None,
        path=str(d),
        location="linux_wsl",
        files=None,
        size_bytes=None,
        source_provider="ollama",
        roles=None,
        role_priority=None,
        family_override="custom-family-x",
        served_model_id="original:name:tag",
    )
    assert entry.family == "custom-family-x"
    assert entry.served_model_id == "original:name:tag"
    assert entry.name.startswith("custom-family-x-ollama-gguf")
