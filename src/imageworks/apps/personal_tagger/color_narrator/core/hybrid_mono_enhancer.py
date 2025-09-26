"""
Hybrid mono enhancement system.

Combines mono-checker's accurate verdicts with VLM's rich descriptions
for the best of both worlds.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any
import requests
import base64
from PIL import Image
import io

from .prompts import MONO_DESCRIPTION_ENHANCEMENT_TEMPLATE

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
        timeout: int = 120,
        max_tokens: int = 300,
        temperature: float = 0.1,
    ):
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature

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

    def enhance_mono_result(
        self, mono_data: Dict[str, Any], image_path: Path
    ) -> EnhancedMonoResult:
        """Enhance mono-checker result with VLM description."""
        start_time = time.time()

        try:
            # Encode image
            image_b64 = self.encode_image(image_path)

            # Format prompt with technical data
            top_weights_formatted = [
                f"{w:.0f}" for w in mono_data.get("top_weights", [])
            ]

            prompt_data = {
                "title": mono_data.get("title", "Unknown"),
                "author": mono_data.get("author", "Unknown"),
                "dominant_color": mono_data.get("dominant_color", "unknown"),
                "dominant_hue_deg": mono_data.get("dominant_hue_deg", 0),
                "top_colors": mono_data.get("top_colors", []),
                "top_weights": top_weights_formatted,
                "colorfulness": mono_data.get("colorfulness", 0),
                "chroma_max": mono_data.get("chroma_max", 0),
                "chroma_p95": mono_data.get("chroma_p95", 0),
                "chroma_median": mono_data.get("chroma_median", 0),
                "hue_std_deg": mono_data.get("hue_std_deg", 0),
                "hue_concentration": mono_data.get("hue_concentration", 0),
                "delta_h_highs_shadows_deg": mono_data.get(
                    "delta_h_highs_shadows_deg", 0
                ),
                "sat_median": mono_data.get("sat_median", 0),
            }

            prompt = MONO_DESCRIPTION_ENHANCEMENT_TEMPLATE.format(**prompt_data)

            # API request
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
