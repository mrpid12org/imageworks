from pathlib import Path

import pytest

from imageworks.model_loader.download_adapter import record_download, ImportSkipped


def test_scan_skip_testing_placeholder(tmp_path: Path):
    # Create a dummy directory to simulate a model repo path
    d = tmp_path / "owner" / "model-awq"
    d.mkdir(parents=True)
    (d / "weights.gguf").write_bytes(b"123")

    # With hf_id ending in 'model-awq', the variant name should match testing filters
    with pytest.raises(ImportSkipped):
        record_download(
            hf_id="owner/model-awq",
            backend="vllm",
            format_type="gguf",
            quantization=None,
            path=str(d),
            location="linux_wsl",
            files=None,
            size_bytes=None,
            source_provider="hf",
            roles=None,
            role_priority=None,
            family_override=None,
            served_model_id=None,
            extra_metadata=None,
            display_name=None,
        )
