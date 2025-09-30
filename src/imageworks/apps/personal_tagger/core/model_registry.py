"""Structured registry of model options for the personal tagger pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class ModelVariant:
    """Represents a specific model configuration available for a stage."""

    key: str
    display_name: str
    model_id: str
    license: str
    quantisations: List[str]
    backends: List[str]
    notes: str


MODEL_REGISTRY: Dict[str, ModelVariant] = {
    "caption-qwen2.5-vl-7b-awq": ModelVariant(
        key="caption-qwen2.5-vl-7b-awq",
        display_name="Qwen2.5-VL 7B (AWQ)",
        model_id="qwen-vl/Qwen2.5-VL-7B-Instruct-AWQ",
        license="Apache 2.0",
        quantisations=["awq", "int4"],
        backends=["lmdeploy", "vllm"],
        notes="Good default captioner with JSON-friendly outputs.",
    ),
    "caption-llava-next-7b": ModelVariant(
        key="caption-llava-next-7b",
        display_name="LLaVA-NeXT 7B",
        model_id="llava-hf/llava-v1.6-mistral-7b-hf",
        license="Apache 2.0",
        quantisations=["fp16", "awq"],
        backends=["vllm"],
        notes="Detailed captions, strong OCR; tighten prompts for keywords.",
    ),
    "keywords-siglip-embed": ModelVariant(
        key="keywords-siglip-embed",
        display_name="SigLIP Large Patch16-384",
        model_id="google/siglip-large-patch16-384",
        license="Apache 2.0",
        quantisations=["int8", "onnx"],
        backends=["embedding"],
        notes="For deterministic keyword ranking using cosine similarity.",
    ),
    "description-qwen2.5-vl-32b": ModelVariant(
        key="description-qwen2.5-vl-32b",
        display_name="Qwen2.5-VL 32B",
        model_id="qwen-vl/Qwen2.5-VL-32B-Instruct",
        license="Apache 2.0",
        quantisations=["bf16"],
        backends=["lmdeploy", "vllm"],
        notes="High-quality long-form descriptions; requires >=80 GB VRAM.",
    ),
    "description-idefics2-8b": ModelVariant(
        key="description-idefics2-8b",
        display_name="Idefics2 8B",
        model_id="huggingface/idefics2-8b-instruct",
        license="Apache 2.0",
        quantisations=["fp16", "awq"],
        backends=["vllm"],
        notes="Balanced prose; works well for narrative output on 16 GB cards.",
    ),
}


def list_models(stage_prefix: Optional[str] = None) -> Iterable[ModelVariant]:
    """Yield models, optionally filtered by stage prefix."""

    if stage_prefix is None:
        return MODEL_REGISTRY.values()
    return [
        model for key, model in MODEL_REGISTRY.items() if key.startswith(stage_prefix)
    ]


def get_model(key: str) -> ModelVariant:
    """Lookup a model variant by its registry key."""

    try:
        return MODEL_REGISTRY[key]
    except KeyError as exc:  # pragma: no cover - simple guard
        raise ValueError(f"Unknown model key '{key}'") from exc
