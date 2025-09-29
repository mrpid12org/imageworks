"""Backend adapters for OpenAI-compatible VLM serving stacks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Optional, Type
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
    """LMDeploy backend adapter."""

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


def create_backend_client(
    backend: VLMBackend,
    *,
    base_url: str,
    model_name: str,
    api_key: str = "EMPTY",
    timeout: int = 120,
    backend_options: Optional[Dict[str, Any]] = None,
) -> BaseBackendClient:
    """Instantiate the backend adapter for the requested serving stack."""

    backend_cls = BACKEND_REGISTRY.get(backend)
    if backend_cls is None:
        raise ValueError(f"Unsupported VLM backend: {backend}")

    options = backend_options or {}
    return backend_cls(
        base_url=base_url,
        model_name=model_name,
        api_key=api_key,
        timeout=timeout,
        **options,
    )
