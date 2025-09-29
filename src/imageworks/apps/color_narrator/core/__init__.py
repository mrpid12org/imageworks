"""Color-Narrator core processing modules.

This package exposes the main processing primitives for VLM-guided colour
narration, including backend-agnostic VLM access, data loading, orchestration,
and metadata handling.
"""

from .vlm import VLMBackend, VLMClient, VLMRequest, VLMResponse
from .data_loader import ColorNarratorDataLoader, DataLoaderConfig, ColorNarratorItem
from .narrator import ColorNarrator, NarrationConfig, ProcessingResult
from .metadata import XMPMetadataWriter, ColorNarrationMetadata, XMPMetadataBatch

__version__ = "0.1.0"
__all__ = [
    "VLMBackend",
    "VLMClient",
    "VLMRequest",
    "VLMResponse",
    "ColorNarratorDataLoader",
    "DataLoaderConfig",
    "ColorNarratorItem",
    "ColorNarrator",
    "NarrationConfig",
    "ProcessingResult",
    "XMPMetadataWriter",
    "ColorNarrationMetadata",
    "XMPMetadataBatch",
]
