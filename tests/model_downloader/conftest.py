"""
Test configuration for model downloader tests.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock
import json

from imageworks.tools.model_downloader.config import DownloaderConfig, DirectoryConfig
from imageworks.tools.model_downloader.registry import ModelRegistry


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_config(temp_dir):
    """Create a mock configuration for testing."""
    config = DownloaderConfig(
        linux_wsl=DirectoryConfig(
            root=temp_dir / "linux_wsl",
            formats=["safetensors", "pytorch", "awq", "gptq"],
            use_cases=["vllm", "transformers"],
        ),
        windows_lmstudio=DirectoryConfig(
            root=temp_dir / "windows_lmstudio",
            formats=["gguf"],
            use_cases=["lmstudio", "llama.cpp"],
            publisher_structure=True,
        ),
        registry_path=temp_dir / "registry",
        cache_path=temp_dir / "cache",
        max_connections_per_server=4,
        max_concurrent_downloads=2,
    )
    config.ensure_directories()
    return config


@pytest.fixture
def mock_registry(temp_dir):
    """Create a mock registry for testing."""
    registry_path = temp_dir / "registry" / "models.json"
    registry = ModelRegistry(registry_path)
    return registry


@pytest.fixture
def sample_hf_response():
    """Sample HuggingFace API response."""
    return {
        "id": "microsoft/DialoGPT-medium",
        "modelId": "microsoft/DialoGPT-medium",
        "pipeline_tag": "conversational",
        "library_name": "transformers",
        "tags": ["conversational", "pytorch"],
        "downloads": 12345,
        "likes": 67,
    }


@pytest.fixture
def sample_file_list():
    """Sample file list from HuggingFace repo."""
    return [
        {"type": "file", "path": "config.json", "size": 1234, "oid": "abc123"},
        {
            "type": "file",
            "path": "pytorch_model.bin",
            "size": 500000000,
            "oid": "def456",
        },
        {"type": "file", "path": "tokenizer.json", "size": 5678, "oid": "ghi789"},
        {"type": "file", "path": "README.md", "size": 2345, "oid": "jkl012"},
    ]


@pytest.fixture
def sample_config_json():
    """Sample config.json content."""
    return json.dumps(
        {
            "model_type": "gpt2",
            "architectures": ["GPT2LMHeadModel"],
            "vocab_size": 50257,
            "n_positions": 1024,
            "n_embd": 1024,
            "n_layer": 24,
            "n_head": 16,
        }
    )


@pytest.fixture
def sample_awq_config_json():
    """Sample config.json with AWQ quantization."""
    return json.dumps(
        {
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
            "quantization_config": {
                "quant_method": "awq",
                "bits": 4,
                "group_size": 128,
                "zero_point": True,
            },
        }
    )


@pytest.fixture
def mock_requests_get(monkeypatch):
    """Mock requests.get for testing without network calls."""
    responses = {}

    def mock_get(url, **kwargs):
        mock_response = Mock()

        if url in responses:
            mock_response.json.return_value = responses[url]
            mock_response.text = (
                json.dumps(responses[url])
                if isinstance(responses[url], dict)
                else responses[url]
            )
            mock_response.raise_for_status.return_value = None
            return mock_response
        else:
            mock_response.raise_for_status.side_effect = Exception(
                f"Mocked 404 for {url}"
            )
            return mock_response

    monkeypatch.setattr("requests.get", mock_get)

    def set_response(url, response):
        responses[url] = response

    return set_response
