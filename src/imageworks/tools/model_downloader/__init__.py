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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .downloader import ModelDownloader  # pragma: no cover
    from .config import DownloaderConfig  # pragma: no cover
    from .formats import FormatDetector  # pragma: no cover
    from .url_analyzer import URLAnalyzer  # pragma: no cover


def __getattr__(name):
    if name == "ModelDownloader":
        from .downloader import ModelDownloader as _MD

        return _MD
    if name == "DownloaderConfig":
        from .config import DownloaderConfig as _CFG

        return _CFG
    if name == "FormatDetector":
        from .formats import FormatDetector as _FD

        return _FD
    if name == "URLAnalyzer":
        from .url_analyzer import URLAnalyzer as _UA

        return _UA
    raise AttributeError(name)


def record_download(*args, **kwargs):
    from imageworks.model_loader.download_adapter import record_download as _record

    return _record(*args, **kwargs)


def list_downloads(*args, **kwargs):
    from imageworks.model_loader.download_adapter import list_downloads as _list

    return _list(*args, **kwargs)


def remove_download(*args, **kwargs):
    from imageworks.model_loader.download_adapter import remove_download as _remove

    return _remove(*args, **kwargs)


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
