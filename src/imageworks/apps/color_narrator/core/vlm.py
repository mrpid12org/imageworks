"""VLM (Vision-Language Model) inference module.

Handles vLLM server communication for the default Qwen2-VL-2B model using an
OpenAI-compatible API. Provides batch processing capabilities and structured
output parsing for color narration.
"""

from typing import List, Dict, Optional, Any
from pathlib import Path
import base64
import requests
from dataclasses import dataclass


@dataclass
class VLMResponse:
    """Structured response from VLM inference."""

    description: str
    confidence: float
    color_regions: List[str]
    processing_time: float
    error: Optional[str] = None


@dataclass
class VLMRequest:
    """VLM inference request with image and analysis data."""

    image_path: Path
    overlay_path: Path
    mono_data: Dict[str, Any]
    prompt_template: str


class VLMClient:
    """Client for vLLM server with OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model_name: str = "Qwen2-VL-2B-Instruct",
        api_key: str = "EMPTY",
        timeout: int = 120,
    ):
        """Initialize VLM client.

        Args:
            base_url: vLLM server base URL
            model_name: Model identifier for API requests
            api_key: API key (EMPTY for local vLLM)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        )

    def encode_image(self, image_path: Path) -> str:
        """Encode image to base64 for API transmission."""
        try:
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode("utf-8")
        except Exception as e:
            raise RuntimeError(f"Failed to encode image {image_path}: {e}")

    def create_color_narration_prompt(
        self, mono_data: Dict[str, Any], template: Optional[str] = None
    ) -> str:
        """Create structured prompt for color narration."""
        if template is None:
            template = """You are analyzing a photograph that should be monochrome (black and white) but contains some residual color.

The image has been analyzed and found to have:
- Hue distribution: {hue_analysis}
- Chroma levels: {chroma_analysis}
- Color contamination: {contamination_level}

Please describe in natural language where you observe residual color in this image. Focus on:
1. Specific regions or objects that show color
2. The type of color cast (warm/cool, specific hues)
3. Whether the color appears intentional or accidental

Provide a concise, professional description suitable for metadata."""

            return template.format(
                hue_analysis=mono_data.get("hue_analysis", "Not available"),
                chroma_analysis=mono_data.get("chroma_analysis", "Not available"),
                contamination_level=mono_data.get("contamination_level", "Unknown"),
            )
        else:
            # Custom template - use all available data
            return template.format(**mono_data)

    def infer_single(self, request: VLMRequest) -> VLMResponse:
        """Perform single image color narration inference."""
        try:
            # Encode images
            base_image = self.encode_image(request.image_path)
            overlay_image = self.encode_image(request.overlay_path)

            # Create prompt
            prompt = self.create_color_narration_prompt(
                request.mono_data, request.prompt_template
            )

            # Build API request
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base_image}"
                                },
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{overlay_image}"
                                },
                            },
                        ],
                    }
                ],
                "max_tokens": 300,
                "temperature": 0.1,
                "stream": False,
            }

            # Make API call
            response = self.session.post(
                f"{self.base_url}/chat/completions", json=payload, timeout=self.timeout
            )

            if response.status_code != 200:
                return VLMResponse(
                    description="",
                    confidence=0.0,
                    color_regions=[],
                    processing_time=0.0,
                    error=f"API error {response.status_code}: {response.text}",
                )

            result = response.json()
            description = result["choices"][0]["message"]["content"].strip()

            # Parse structured output (basic implementation)
            color_regions = self._extract_color_regions(description)
            confidence = self._estimate_confidence(description)

            return VLMResponse(
                description=description,
                confidence=confidence,
                color_regions=color_regions,
                processing_time=result.get("usage", {}).get("total_time", 0.0),
                error=None,
            )

        except Exception as e:
            return VLMResponse(
                description="",
                confidence=0.0,
                color_regions=[],
                processing_time=0.0,
                error=f"Inference error: {str(e)}",
            )

    def infer_batch(self, requests: List[VLMRequest]) -> List[VLMResponse]:
        """Process multiple images in batch."""
        responses = []
        for request in requests:
            response = self.infer_single(request)
            responses.append(response)

            # Add small delay between requests to avoid overwhelming server
            import time

            time.sleep(0.1)

        return responses

    def _extract_color_regions(self, description: str) -> List[str]:
        """Extract color region mentions from description text."""
        # Simple keyword extraction - could be enhanced with NLP
        regions = []
        region_keywords = [
            "background",
            "foreground",
            "skin",
            "clothing",
            "hair",
            "sky",
            "water",
            "vegetation",
            "shadows",
            "highlights",
        ]

        description_lower = description.lower()
        for keyword in region_keywords:
            if keyword in description_lower:
                regions.append(keyword)

        return regions

    def _estimate_confidence(self, description: str) -> float:
        """Estimate confidence based on description characteristics."""
        # Simple confidence estimation - could be enhanced
        if "clear" in description.lower() or "obvious" in description.lower():
            return 0.9
        elif "subtle" in description.lower() or "slight" in description.lower():
            return 0.7
        elif "minimal" in description.lower() or "trace" in description.lower():
            return 0.5
        else:
            return 0.8

    def health_check(self) -> bool:
        """Check if VLM server is healthy and responding."""
        try:
            response = self.session.get(f"{self.base_url}/models", timeout=10)
            return response.status_code == 200
        except Exception:
            return False

    def infer_openai_compatible(self, request: Dict[str, Any]):
        """Direct OpenAI-compatible API call for region-based analysis."""
        try:
            # Add model name to request
            request["model"] = self.model_name

            # Set default values if not provided
            if "max_tokens" not in request:
                request["max_tokens"] = 600
            if "temperature" not in request:
                request["temperature"] = 0.1
            if "stream" not in request:
                request["stream"] = False

            # Make API call
            response = self.session.post(
                f"{self.base_url}/chat/completions", json=request, timeout=self.timeout
            )

            if response.status_code != 200:

                class ErrorResponse:
                    content = f"API error {response.status_code}: {response.text}"

                return ErrorResponse()

            result = response.json()
            content = result["choices"][0]["message"]["content"].strip()

            # Return object with content attribute for compatibility
            class CompatibleResponse:
                def __init__(self, content):
                    self.content = content

            return CompatibleResponse(content)

        except Exception as e:

            class ErrorResponse:
                content = f"Inference error: {str(e)}"

            return ErrorResponse()
