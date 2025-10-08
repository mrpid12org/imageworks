"""Embedding backends for the image similarity checker."""

from __future__ import annotations

import base64
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
from PIL import Image

from .config import SimilarityConfig

try:  # Optional dependency
    import torch
except Exception:  # pragma: no cover - dependency optional
    torch = None  # type: ignore[assignment]

try:  # Optional dependency
    import open_clip
except Exception:  # pragma: no cover - dependency optional
    open_clip = None  # type: ignore[assignment]

try:  # Optional dependency
    import requests
except Exception:  # pragma: no cover - dependency optional
    requests = None  # type: ignore[assignment]


class EmbeddingError(RuntimeError):
    """Raised when an embedding backend cannot generate a vector."""


class EmbeddingModel:
    """Base class for feature extractors."""

    def embed(self, image_path: Path) -> np.ndarray:
        raise NotImplementedError

    def batch_embed(self, image_paths: Sequence[Path]) -> Dict[Path, np.ndarray]:
        return {path: self.embed(path) for path in image_paths}

    @staticmethod
    def _normalise(vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            return vector
        return vector / norm


@dataclass
class SimpleVisionEmbedding(EmbeddingModel):
    """Deterministic visual descriptor using resized pixels + colour histograms."""

    size: int = 48
    histogram_bins: int = 24

    def embed(self, image_path: Path) -> np.ndarray:  # noqa: D401 - see base
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            resized = rgb.resize((self.size, self.size), Image.Resampling.BILINEAR)
            spatial = np.asarray(resized, dtype=np.float32) / 255.0
            spatial_vector = spatial.flatten()

            hist_components = []
            for channel in range(3):
                channel_data = spatial[:, :, channel]
                hist, _ = np.histogram(
                    channel_data, bins=self.histogram_bins, range=(0.0, 1.0), density=True
                )
                hist_components.append(hist.astype(np.float32))

            feature = np.concatenate([spatial_vector, *hist_components]).astype(np.float32)
            return self._normalise(feature)


@dataclass
class OpenClipEmbedding(EmbeddingModel):
    """Wrapper around the open_clip vision encoder."""

    model_name: str = "ViT-B-32"
    pretrained: str = "openai"
    device: str = "cpu"

    def __post_init__(self) -> None:
        if open_clip is None or torch is None:  # pragma: no cover - optional runtime
            raise EmbeddingError(
                "open_clip and torch must be installed to use the open_clip embedding backend"
            )
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(  # type: ignore[union-attr]
            self.model_name, pretrained=self.pretrained
        )
        self._model.to(self.device)
        self._model.eval()

    def embed(self, image_path: Path) -> np.ndarray:  # noqa: D401 - see base
        if open_clip is None or torch is None:  # pragma: no cover - defensive
            raise EmbeddingError("open_clip backend is unavailable")

        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            tensor = self._preprocess(rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():  # type: ignore[union-attr]
            features = self._model.encode_image(tensor)  # type: ignore[union-attr]
            features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze(0).cpu().numpy().astype(np.float32)


@dataclass
class RemoteEmbeddingClient(EmbeddingModel):
    """Call an OpenAI-compatible endpoint to retrieve embeddings."""

    base_url: str
    model: str
    api_key: str
    timeout: int = 120

    def __post_init__(self) -> None:
        if requests is None:  # pragma: no cover - optional dependency
            raise EmbeddingError("The 'requests' package is required for remote embeddings")
        self._session = requests.Session()  # type: ignore[assignment]
        self._endpoint = self.base_url.rstrip("/") + "/embeddings"

    def embed(self, image_path: Path) -> np.ndarray:  # noqa: D401 - see base
        if requests is None:  # pragma: no cover - defensive
            raise EmbeddingError("Remote embeddings unavailable (requests missing)")

        mime_type, _ = mimetypes.guess_type(image_path.name)
        with image_path.open("rb") as handle:
            payload = {
                "model": self.model,
                "input": [
                    {
                        "image": base64.b64encode(handle.read()).decode("utf-8"),
                        "mime_type": mime_type or "image/jpeg",
                    }
                ],
                "encoding_format": "float",
            }

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = self._session.post(
            self._endpoint,
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        vector = data.get("data", [{}])[0].get("embedding", [])
        array = np.asarray(vector, dtype=np.float32)
        if array.size == 0:
            raise EmbeddingError("Remote embedding endpoint returned no data")
        return self._normalise(array)

    def batch_embed(self, image_paths: Sequence[Path]) -> Dict[Path, np.ndarray]:
        # Remote batching can be expensive; fall back to default iterative behaviour.
        return super().batch_embed(image_paths)


def create_embedding_model(config: SimilarityConfig) -> EmbeddingModel:
    """Factory for embedding backends based on configuration."""

    backend = config.embedding_backend.lower()
    if backend in {"simple", "baseline"}:
        return SimpleVisionEmbedding()
    if backend in {"open_clip", "clip"}:
        return OpenClipEmbedding()
    if backend in {"remote", "openai"}:
        return RemoteEmbeddingClient(
            base_url=config.base_url,
            model=config.model,
            api_key=config.api_key,
            timeout=config.timeout,
        )
    raise EmbeddingError(f"Unknown embedding backend '{config.embedding_backend}'")
