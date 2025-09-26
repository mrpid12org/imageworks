"""Personal tagger shared libraries.

Provides common utilities and shared functionality for personal tagger applications,
including color analysis, VLM operations, and image processing utilities.

Key modules:
- color_analysis: Color space analysis and statistical utilities
- vlm_utils: VLM inference management and prompt handling
- image_utils: Image loading, processing, and batch operations
"""

from .color_analysis import ColorAnalyzer, ColorRegionAnalyzer, ColorStatistics
from .vlm_utils import VLMPromptManager, VLMResponseParser, VLMModelManager
from .image_utils import ImageLoader, ImageProcessor, ImageComparison, ImageBatch

__version__ = "0.1.0"
__all__ = [
    "ColorAnalyzer",
    "ColorRegionAnalyzer",
    "ColorStatistics",
    "VLMPromptManager",
    "VLMResponseParser",
    "VLMModelManager",
    "ImageLoader",
    "ImageProcessor",
    "ImageComparison",
    "ImageBatch",
]
