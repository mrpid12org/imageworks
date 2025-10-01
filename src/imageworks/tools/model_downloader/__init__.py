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
from .registry import ModelRegistry
from .config import DownloaderConfig
from .formats import FormatDetector
from .url_analyzer import URLAnalyzer

__version__ = "0.1.0"
__all__ = [
    "ModelDownloader",
    "ModelRegistry",
    "DownloaderConfig",
    "FormatDetector",
    "URLAnalyzer",
]
