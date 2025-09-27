"""
Hybrid mono enhancement system.

Combines mono-checker's accurate verdicts with VLM's rich descriptions
for the best of both worlds.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional
import requests
import base64
from PIL import Image
import io

from .prompts import (
    MONO_DESCRIPTION_ENHANCEMENT_TEMPLATE,
    DEFAULT_ENHANCED_TEMPLATE,
    ENHANCED_MONO_ANALYSIS_TEMPLATE_V6,
    ENHANCED_MONO_ANALYSIS_TEMPLATE_V5,
    ENHANCED_MONO_ANALYSIS_TEMPLATE_V4,
)
from .grid_regions import ImageGridAnalyzer, GridColorRegion

logger = logging.getLogger(__name__)


@dataclass
class EnhancedMonoResult:
    """Enhanced mono result with VLM description and original verdict."""

    # Original mono-checker data
    original_verdict: str
    original_reason: str
    original_mode: str

    # VLM enhancement
    vlm_description: str
    vlm_processing_time: float
    vlm_model: str

    # Technical data
    dominant_color: str
    colorfulness: float
    chroma_max: float
    hue_std_deg: float

    # Image metadata
    title: str
    author: str
    image_path: str


class HybridMonoEnhancer:
    """Hybrid system combining mono verdicts with VLM descriptions."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "Qwen2-VL-2B-Instruct",
        max_tokens: int = 512,
        temperature: float = 0.1,
        timeout: int = 30,
        use_regions: bool = False,
        prompt_template: str = None,  # Use DEFAULT_ENHANCED_TEMPLATE if None
    ):
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.use_regions = use_regions
        self.prompt_template = prompt_template or DEFAULT_ENHANCED_TEMPLATE
        self.grid_analyzer = ImageGridAnalyzer() if use_regions else None

    def encode_image(self, image_path: Path) -> str:
        """Encode image to base64 for VLM API."""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode != "RGB":
                    img = img.convert("RGB")

                # Resize if too large (for efficiency)
                max_size = 1024
                if max(img.size) > max_size:
                    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

                # Encode to base64
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=85)
                return base64.b64encode(buffer.getvalue()).decode("utf-8")

        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {e}")
            raise

    def _format_mono_context(self, mono_data: Dict[str, Any]) -> str:
        """Format mono-checker data for VLM context."""
        verdict = mono_data.get("verdict", "unknown")
        mode = mono_data.get("mode", "unknown")
        dominant_color = mono_data.get("dominant_color", "unknown")
        dominant_hue = mono_data.get("dominant_hue_deg", 0)
        colorfulness = mono_data.get("colorfulness", 0)
        chroma_max = mono_data.get("chroma_max", 0)
        hue_std = mono_data.get("hue_std_deg", 0)

        context = f"""Mono-checker analysis: {verdict} ({mode})
Dominant color: {dominant_color} at {dominant_hue:.1f}°
Colorfulness: {colorfulness:.2f}, Max chroma: {chroma_max:.2f}
Hue variation: {hue_std:.1f}° standard deviation"""

        return context

    def _format_grid_regions(self, regions: list[GridColorRegion]) -> str:
        """Format grid regions for VLM context."""
        if not regions:
            return "No significant color detected in any 3x3 grid regions."

        region_lines = []
        for region in regions:
            pct = region.area_pct
            zone = region.tonal_zone
            region_lines.append(
                f"{region.grid_region.value}: {region.dominant_color} ({pct:.1f}% affected, {zone})"
            )

        return "\n".join(region_lines)

    def _build_enhanced_prompt(
        self, mono_data: Dict[str, Any], region_context: Optional[str] = None
    ) -> str:
        """Build enhanced prompt using configurable template with optional regions."""
        title = mono_data.get("title", "Unknown")
        author = mono_data.get("author", "Unknown")

        # Check if using legacy template (different format)
        if self.prompt_template == MONO_DESCRIPTION_ENHANCEMENT_TEMPLATE:
            return self._build_legacy_prompt(mono_data)

        # New template format
        # Technical context
        mono_context = self._format_mono_context(mono_data)

        # Region section (either populated or empty)
        region_section = ""
        if region_context:
            region_section = f"\n\nSPATIAL ANALYSIS:\n{region_context}"

        # Build prompt using selected template
        prompt = self.prompt_template.format(
            title=title,
            author=author,
            mono_context=mono_context,
            region_section=region_section,
        )

        return prompt

    def _build_legacy_prompt(self, mono_data: Dict[str, Any]) -> str:
        """Build prompt using legacy template format."""
        return self.prompt_template.format(
            title=mono_data.get("title", "Unknown"),
            author=mono_data.get("author", "Unknown"),
            dominant_color=mono_data.get("dominant_color", "unknown"),
            dominant_hue_deg=mono_data.get("dominant_hue_deg", 0.0),
            colorfulness=mono_data.get("colorfulness", 0.0),
            chroma_max=mono_data.get("chroma_max", 0.0),
        )

    # Utility methods for prompt template switching
    def set_prompt_template(self, template: str) -> None:
        """Set the prompt template for quick A/B testing."""
        self.prompt_template = template

    def use_prompt_v6(self) -> None:
        """Switch to v6 template (current default)."""
        self.set_prompt_template(ENHANCED_MONO_ANALYSIS_TEMPLATE_V6)

    def use_prompt_v5(self) -> None:
        """Switch to v5 template (location-focused)."""
        self.set_prompt_template(ENHANCED_MONO_ANALYSIS_TEMPLATE_V5)

    def use_prompt_v4(self) -> None:
        """Switch to v4 template (competition-judge style)."""
        self.set_prompt_template(ENHANCED_MONO_ANALYSIS_TEMPLATE_V4)

    def use_legacy_prompt(self) -> None:
        """Switch to legacy mono description template."""
        self.set_prompt_template(MONO_DESCRIPTION_ENHANCEMENT_TEMPLATE)

    def get_current_prompt_info(self) -> str:
        """Get information about the currently selected prompt template."""
        if self.prompt_template == ENHANCED_MONO_ANALYSIS_TEMPLATE_V6:
            return "v6 (Enhanced structured analysis)"
        elif self.prompt_template == ENHANCED_MONO_ANALYSIS_TEMPLATE_V5:
            return "v5 (Location-focused)"
        elif self.prompt_template == ENHANCED_MONO_ANALYSIS_TEMPLATE_V4:
            return "v4 (Competition judge style)"
        elif self.prompt_template == MONO_DESCRIPTION_ENHANCEMENT_TEMPLATE:
            return "Legacy (Original template)"
        else:
            return "Custom template"

    def enhance_mono_result(
        self, mono_data: Dict[str, Any], image_path: Path, use_regions: bool = False
    ) -> EnhancedMonoResult:
        """Enhance mono-checker result with improved VLM description and optional regions."""
        start_time = time.time()

        try:
            # Encode image
            image_b64 = self.encode_image(image_path)

            # Optional grid region analysis
            region_context = None
            if use_regions and self.grid_analyzer:
                try:
                    regions = self.grid_analyzer.create_grid_regions(
                        image_path, mono_data
                    )
                    if regions:
                        region_context = self._format_grid_regions(regions)
                        logger.debug(f"Grid regions: {len(regions)} regions with color")
                    else:
                        region_context = (
                            "No significant color detected in any 3x3 grid regions."
                        )
                except Exception as e:
                    logger.warning(
                        f"Grid region analysis failed: {e}, continuing without regions"
                    )
                    region_context = None

            # Build enhanced prompt
            prompt = self._build_enhanced_prompt(mono_data, region_context)

            # API request (keeping original format)
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Content-Type": "application/json"},
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_b64}"
                                    },
                                },
                            ],
                        }
                    ],
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "stream": False,
                },
                timeout=self.timeout,
            )

            if response.status_code != 200:
                raise Exception(
                    f"VLM API error: {response.status_code} - {response.text}"
                )

            result = response.json()
            vlm_description = result["choices"][0]["message"]["content"].strip()

            processing_time = time.time() - start_time

            return EnhancedMonoResult(
                original_verdict=mono_data.get("verdict", "unknown"),
                original_reason=mono_data.get("reason_summary", ""),
                original_mode=mono_data.get("mode", "unknown"),
                vlm_description=vlm_description,
                vlm_processing_time=processing_time,
                vlm_model=self.model,
                dominant_color=mono_data.get("dominant_color", "unknown"),
                colorfulness=mono_data.get("colorfulness", 0),
                chroma_max=mono_data.get("chroma_max", 0),
                hue_std_deg=mono_data.get("hue_std_deg", 0),
                title=mono_data.get("title", "Unknown"),
                author=mono_data.get("author", "Unknown"),
                image_path=str(image_path),
            )

        except Exception as e:
            logger.error(f"VLM mono enhancement failed: {e}")
            raise
