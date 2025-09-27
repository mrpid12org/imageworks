"""Color-Narrator core processing modules.

This module provides the core functionality for VLM-guided color narration,
including data loading, VLM inference, metadata handling, and orchestration.

Key modules:
- vlm: VLM inference client for Qwen2-VL-7B
- data_loader: JPEG/overlay/JSONL data loading and validation
- narrator: Main processing orchestration
- metadata: XMP metadata reading and writing
"""

from .vlm import VLMClient, VLMRequest, VLMResponse
from .data_loader import ColorNarratorDataLoader, DataLoaderConfig, ColorNarratorItem
from .narrator import ColorNarrator, NarrationConfig, ProcessingResult
from .metadata import XMPMetadataWriter, ColorNarrationMetadata, XMPMetadataBatch

__version__ = "0.1.0"
__all__ = [
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
