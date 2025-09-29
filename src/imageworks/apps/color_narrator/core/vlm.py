"""VLM (Vision-Language Model) inference module with pluggable backends.

Provides a high-level client that talks to OpenAI-compatible HTTP endpoints
exposed by different serving stacks (vLLM, LMDeploy, TensorRT-LLM/Triton stub).
The client keeps the existing API used by the narrator while delegating
transport details to backend-specific adapters.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Type
from abc import ABC, abstractmethod
import base64
import logging

import requests

logger = logging.getLogger(__name__)


class VLMBackend(str, Enum):
    """Supported backend identifiers."""

    VLLM = "vllm"
    LMDEPLOY = "lmdeploy"
    TRITON = "triton"


class VLMBackendError(RuntimeError):
    """Raised when a backend fails to execute an inference request."""


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


class BaseBackendClient(ABC):
    """Abstract backend adapter that communicates with a serving stack."""

    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        api_key: str = "EMPTY",
        timeout: int = 120,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        )
        self.last_error: Optional[str] = None

    @abstractmethod
    def health_check(self) -> bool:
        """Return True when the backend is reachable and ready."""

    @abstractmethod
    def chat_completions(self, payload: Dict[str, Any]) -> requests.Response:
        """Execute an OpenAI-compatible chat completion request."""

    def close(self) -> None:
        self.session.close()


class OpenAICompatibleBackend(BaseBackendClient):
    """Backend adapter for standard OpenAI-compatible HTTP endpoints."""

    MODELS_PATH = "/models"
    HEALTH_PATHS = ("/health", "/status")

    def health_check(self) -> bool:
        try:
            response = self.session.get(
                f"{self.base_url}{self.MODELS_PATH}", timeout=min(self.timeout, 10)
            )
            if response.status_code == 200:
                self.last_error = None
                return True
            self.last_error = f"HTTP {response.status_code}: {response.text[:200]}"
        except Exception as exc:  # noqa: BLE001 - surface exact backend error
            self.last_error = str(exc)
            logger.debug("OpenAI backend health check failed: %s", exc)

        # Some backends expose additional lightweight probes
        for path in self.HEALTH_PATHS:
            try:
                response = self.session.get(
                    f"{self.base_url}{path}", timeout=min(self.timeout, 5)
                )
                if response.status_code == 200:
                    self.last_error = None
                    return True
            except Exception as exc:  # noqa: BLE001
                self.last_error = str(exc)
                logger.debug("OpenAI backend health probe %s failed: %s", path, exc)

        return False

    def chat_completions(self, payload: Dict[str, Any]) -> requests.Response:
        try:
            return self.session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=self.timeout,
            )
        except Exception as exc:  # noqa: BLE001
            raise VLMBackendError(str(exc)) from exc


class LMDeployBackend(OpenAICompatibleBackend):
    """LMDeploy backend adapter.

    Uses the same OpenAI-compatible surface but keeps dedicated logging and
    future extension points.
    """

    HEALTH_PATHS = OpenAICompatibleBackend.HEALTH_PATHS + ("/v1/health",)


class TritonStubBackend(BaseBackendClient):
    """Placeholder backend for TensorRT-LLM/Triton integration."""

    STUB_MESSAGE = (
        "TensorRT-LLM backend is not yet implemented. Configure a running Triton "
        "OpenAI endpoint and update the adapter to enable full support."
    )

    def health_check(self) -> bool:
        logger.warning("%s", self.STUB_MESSAGE)
        self.last_error = self.STUB_MESSAGE
        return False

    def chat_completions(self, payload: Dict[str, Any]) -> requests.Response:
        raise VLMBackendError(self.STUB_MESSAGE)


BACKEND_REGISTRY: Dict[VLMBackend, Type[BaseBackendClient]] = {
    VLMBackend.VLLM: OpenAICompatibleBackend,
    VLMBackend.LMDEPLOY: LMDeployBackend,
    VLMBackend.TRITON: TritonStubBackend,
}


class VLMClient:
    """Facade that wraps backend-specific adapters."""

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:8000/v1",
        model_name: str = "Qwen2-VL-2B-Instruct",
        api_key: str = "EMPTY",
        timeout: int = 120,
        backend: str | VLMBackend = VLMBackend.VLLM,
        backend_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key
        self.timeout = timeout
        self.backend = (
            backend if isinstance(backend, VLMBackend) else VLMBackend(backend)
        )
        self.backend_options = backend_options or {}
        self.last_error: Optional[str] = None

        backend_cls = BACKEND_REGISTRY.get(self.backend)
        if backend_cls is None:
            raise ValueError(f"Unsupported VLM backend: {backend}")

        self._backend_client = backend_cls(
            base_url=self.base_url,
            model_name=self.model_name,
            api_key=self.api_key,
            timeout=self.timeout,
            **self.backend_options,
        )

    def encode_image(self, image_path: Path) -> str:
        """Encode image to base64 for API transmission."""
        try:
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode("utf-8")
        except Exception as exc:  # noqa: BLE001 - propagate detailed path info
            raise RuntimeError(f"Failed to encode image {image_path}: {exc}") from exc

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
        return template.format(**mono_data)

    def health_check(self) -> bool:
        """Check if the configured backend is reachable."""
        healthy = self._backend_client.health_check()
        self.last_error = self._backend_client.last_error
        return healthy

    def infer_single(self, request: VLMRequest) -> VLMResponse:
        """Perform single image color narration inference."""
        try:
            base_image = self.encode_image(request.image_path)
            overlay_image = self.encode_image(request.overlay_path)
            prompt = self.create_color_narration_prompt(
                request.mono_data, request.prompt_template
            )

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

            try:
                response = self._backend_client.chat_completions(payload)
            except VLMBackendError as exc:
                error_message = f"{self.backend.value} backend error: {exc}"
                self.last_error = error_message
                return VLMResponse(
                    description="",
                    confidence=0.0,
                    color_regions=[],
                    processing_time=0.0,
                    error=error_message,
                )

            if response.status_code != 200:
                error_message = (
                    f"API error {response.status_code}: {response.text[:200]}"
                )
                self.last_error = error_message
                return VLMResponse(
                    description="",
                    confidence=0.0,
                    color_regions=[],
                    processing_time=0.0,
                    error=error_message,
                )

            result = response.json()
            description = result["choices"][0]["message"]["content"].strip()

            color_regions = self._extract_color_regions(description)
            confidence = self._estimate_confidence(description)

            return VLMResponse(
                description=description,
                confidence=confidence,
                color_regions=color_regions,
                processing_time=result.get("usage", {}).get("total_time", 0.0),
                error=None,
            )

        except Exception as exc:  # noqa: BLE001
            error_message = f"Inference error: {exc}"
            self.last_error = error_message
            logger.debug("VLM inference failure: %s", exc, exc_info=True)
            return VLMResponse(
                description="",
                confidence=0.0,
                color_regions=[],
                processing_time=0.0,
                error=error_message,
            )

    def infer_batch(self, requests: List[VLMRequest]) -> List[VLMResponse]:
        """Process multiple images in batch."""
        responses: List[VLMResponse] = []
        for request in requests:
            responses.append(self.infer_single(request))

            import time

            time.sleep(0.1)

        return responses

    def infer_openai_compatible(self, request: Dict[str, Any]):
        """Direct OpenAI-compatible API call for region-based analysis."""
        try:
            payload = dict(request)
            payload["model"] = self.model_name
            payload.setdefault("max_tokens", 600)
            payload.setdefault("temperature", 0.1)
            payload.setdefault("stream", False)

            try:
                response = self._backend_client.chat_completions(payload)
            except VLMBackendError as exc:
                error_message = f"{self.backend.value} backend error: {exc}"
                self.last_error = error_message

                class ErrorResponse:
                    content = error_message

                return ErrorResponse()

            if response.status_code != 200:

                class ErrorResponse:
                    content = (
                        f"API error {response.status_code}: " f"{response.text[:200]}"
                    )

                self.last_error = ErrorResponse.content
                return ErrorResponse()

            result = response.json()
            content = result["choices"][0]["message"]["content"].strip()

            class CompatibleResponse:
                def __init__(self, content: str) -> None:
                    self.content = content

            return CompatibleResponse(content)

        except Exception as exc:  # noqa: BLE001

            class ErrorResponse:
                content = f"Inference error: {exc}"

            self.last_error = ErrorResponse.content
            logger.debug("OpenAI-compatible inference failed: %s", exc, exc_info=True)
            return ErrorResponse()

    def _extract_color_regions(self, description: str) -> List[str]:
        """Extract color region mentions from description text."""
        regions: List[str] = []
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
        lowered = description.lower()
        if "clear" in lowered or "obvious" in lowered:
            return 0.9
        if "subtle" in lowered or "slight" in lowered:
            return 0.7
        if "minimal" in lowered or "trace" in lowered:
            return 0.5
        return 0.8


__all__ = ["VLMBackend", "VLMClient", "VLMRequest", "VLMResponse"]
