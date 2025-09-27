"""Color-Narrator - VLM-guided color localization for monochrome images.

This module converts mono-checker color evidence into natural language statements
that describe where residual color appears and what it looks like.

Key features:
- Consumes mono-checker JSONL + overlay data (does not re-measure color)
- Uses Vision-Language Models (VLM) for natural phrasing
- Validates language output against mono-checker numeric truth
- Embeds results as XMP metadata in JPEG files
- Batch processing optimized for RTX 4080/6000 Pro

Example output: "Faint yellow-green on the zebra's mane (upper-left)."
"""

__version__ = "0.1.0"
