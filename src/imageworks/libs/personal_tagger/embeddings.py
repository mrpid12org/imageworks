"""Embedding service scaffolding for keyword ranking experiments."""

from __future__ import annotations

from typing import Callable, Iterable, List, Sequence


try:  # pragma: no cover - optional dependency import
    import torch

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False

try:  # pragma: no cover - optional dependency import
    from transformers import AutoModel, AutoProcessor

    TRANSFORMERS_AVAILABLE = True
except Exception:  # pragma: no cover
    AutoModel = AutoProcessor = None  # type: ignore[assignment]
    TRANSFORMERS_AVAILABLE = False


EncoderFn = Callable[[Sequence[str]], List[List[float]]]


def _default_siglip_encoder_loader(
    model_id: str,
    *,
    device: str = "cpu",
    trust_remote_code: bool = False,
) -> EncoderFn:
    """Load a SigLIP text encoder via transformers (if available)."""

    if not (TRANSFORMERS_AVAILABLE and TORCH_AVAILABLE):
        raise RuntimeError(
            "SigLIP embeddings require transformers and torch to be installed"
        )

    processor = AutoProcessor.from_pretrained(  # type: ignore[union-attr]
        model_id, trust_remote_code=trust_remote_code
    )
    model = AutoModel.from_pretrained(  # type: ignore[union-attr]
        model_id, trust_remote_code=trust_remote_code
    ).to(device)
    model.eval()

    def _encode(texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():  # type: ignore[union-attr]
            outputs = model.get_text_features(**inputs)
            embeddings = torch.nn.functional.normalize(  # type: ignore[union-attr]
                outputs, dim=-1
            )
        return embeddings.cpu().tolist()

    return _encode


class SiglipEmbeddingService:
    """Wrapper around SigLIP-style text embeddings used for keyword ranking."""

    def __init__(
        self,
        model_id: str = "google/siglip-large-patch16-384",
        *,
        device: str = "cpu",
        trust_remote_code: bool = False,
        encoder_loader: Callable[[str], EncoderFn] | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.trust_remote_code = trust_remote_code
        self._encoder_loader = encoder_loader or (
            lambda model_id: _default_siglip_encoder_loader(
                model_id, device=self.device, trust_remote_code=self.trust_remote_code
            )
        )
        self._encoder: EncoderFn | None = None

    def _ensure_encoder(self) -> EncoderFn:
        if self._encoder is None:
            self._encoder = self._encoder_loader(self.model_id)
        return self._encoder

    def encode(self, texts: Iterable[str]) -> List[List[float]]:
        """Generate embeddings for the provided texts."""

        encoder = self._ensure_encoder()
        return encoder(list(texts))
