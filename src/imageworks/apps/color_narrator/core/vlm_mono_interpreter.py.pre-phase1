"""
VLM-based mono analysis interpreter.

Takes technical mono-checker data and image, uses VLM to generate
professional verdicts and descriptions independent of original analysis.
"""

import base64
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any
import requests
from PIL import Image
import io

from .prompts import MONO_INTERPRETATION_TEMPLATE

logger = logging.getLogger(__name__)


@dataclass
class VLMMonoResult:
    """Result from VLM mono interpretation."""

    verdict: str
    technical_reasoning: str
    visual_description: str
    professional_summary: str
    confidence_score: float
    processing_time: float
    vlm_model: str


class VLMMonoInterpreter:
    """VLM-based mono analysis interpreter."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "Qwen2-VL-2B-Instruct",
        timeout: int = 120,
        max_tokens: int = 500,
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

    def interpret_mono_data(
        self, mono_data: Dict[str, Any], image_path: Path
    ) -> VLMMonoResult:
        """Interpret mono analysis data using VLM."""
        start_time = time.time()

        try:
            # Encode image
            image_b64 = self.encode_image(image_path)

            # Format prompt with technical data (excluding verdict and descriptions)
            prompt_data = {
                "method": mono_data.get("method", "unknown"),
                "mode": mono_data.get("mode", "unknown"),
                "dominant_color": mono_data.get("dominant_color", "unknown"),
                "dominant_hue_deg": mono_data.get("dominant_hue_deg", 0),
                "top_colors": mono_data.get("top_colors", []),
                "top_weights": [f"{w:.0f}" for w in mono_data.get("top_weights", [])],
                "colorfulness": mono_data.get("colorfulness", 0),
                "chroma_max": mono_data.get("chroma_max", 0),
                "chroma_p95": mono_data.get("chroma_p95", 0),
                "chroma_p99": mono_data.get("chroma_p99", 0),
                "hue_std_deg": mono_data.get("hue_std_deg", 0),
                "hue_concentration": mono_data.get("hue_concentration", 0),
                "hue_bimodality": mono_data.get("hue_bimodality", 0),
                "mean_hue_highs_deg": mono_data.get("mean_hue_highs_deg", 0),
                "mean_hue_shadows_deg": mono_data.get("mean_hue_shadows_deg", 0),
                "delta_h_highs_shadows_deg": mono_data.get(
                    "delta_h_highs_shadows_deg", 0
                ),
                "channel_max_diff": mono_data.get("channel_max_diff", 0),
                "sat_median": mono_data.get("sat_median", 0),
            }

            prompt = MONO_INTERPRETATION_TEMPLATE.format(**prompt_data)

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
            content = result["choices"][0]["message"]["content"]

            # Parse VLM response
            verdict, technical, visual, summary = self._parse_vlm_response(content)

            processing_time = time.time() - start_time

            return VLMMonoResult(
                verdict=verdict,
                technical_reasoning=technical,
                visual_description=visual,
                professional_summary=summary,
                confidence_score=0.85,  # Could be extracted from response
                processing_time=processing_time,
                vlm_model=self.model,
            )

        except Exception as e:
            logger.error(f"VLM mono interpretation failed: {e}")
            raise

    def _parse_vlm_response(self, content: str) -> tuple[str, str, str, str]:
        """Parse structured VLM response."""
        lines = content.strip().split("\n")

        verdict = "unknown"
        technical = ""
        visual = ""
        summary = ""

        current_section = None

        for line in lines:
            line = line.strip()
            if line.startswith("VERDICT:"):
                verdict = line.replace("VERDICT:", "").strip().lower()
            elif line.startswith("TECHNICAL:"):
                technical = line.replace("TECHNICAL:", "").strip()
                current_section = "technical"
            elif line.startswith("VISUAL:"):
                visual = line.replace("VISUAL:", "").strip()
                current_section = "visual"
            elif line.startswith("SUMMARY:"):
                summary = line.replace("SUMMARY:", "").strip()
                current_section = "summary"
            elif line and current_section:
                # Continue previous section
                if current_section == "technical":
                    technical += " " + line
                elif current_section == "visual":
                    visual += " " + line
                elif current_section == "summary":
                    summary += " " + line

        return verdict, technical, visual, summary
