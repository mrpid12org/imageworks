"""
Model Downloader for ImageWorks

A comprehensive tool for downloading and managing AI models across multiple formats
and directories, with support for quantized models and cross-platform compatibility.

This package provides:
- Unified model downloading from HuggingFace URLs or model names
- Automatic format detection (GGUF, AWQ, GPTQ, Safetensors, etc.)
- Smart routing to appropriate directories (WSL vs Windows LM Studio)
- Model registry for tracking and avoiding duplicates
- Integration with aria2c for high-speed parallel downloads
- Support for both interactive and programmatic usage
"""

from .downloader import ModelDownloader
from .config import DownloaderConfig
from .formats import FormatDetector
from .url_analyzer import URLAnalyzer

# Unified registry adapter (preferred for programmatic download metadata)
from imageworks.model_loader.download_adapter import (
    record_download,
    list_downloads,
    remove_download,
)


# Backwards compatibility: importing ModelRegistry now raises ImportError from registry stub.
def _deprecated_registry():  # pragma: no cover
    raise ImportError(
        "ModelRegistry is deprecated. Use record_download/list_downloads/remove_download from "
        "imageworks.model_loader.download_adapter instead."
    )


class ModelRegistry:  # type: ignore
    def __init__(self, *_, **__):  # pragma: no cover
        _deprecated_registry()


__version__ = "0.1.0"
__all__ = [
    "ModelDownloader",
    "DownloaderConfig",
    "FormatDetector",
    "URLAnalyzer",
    # adapter helpers
    "record_download",
    "list_downloads",
    "remove_download",
    # legacy placeholder (still exported to avoid AttributeError in old code)
    "ModelRegistry",
]
