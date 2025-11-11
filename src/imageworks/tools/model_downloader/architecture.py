"""Architecture metadata extraction for downloaded model directories."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:  # Import lazily to keep downloader fast when gguf unused.
    from gguf import GGUFReader  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    GGUFReader = None  # type: ignore

from imageworks.tools.ollama_api import OllamaClient, OllamaError
from imageworks.model_loader.simplified_naming import _extract_param_size


@dataclass
class VisionMetadata:
    image_size: Optional[int] = None
    patch_size: Optional[int] = None
    hidden_size: Optional[int] = None
    num_layers: Optional[int] = None
    projection_dim: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "projection_dim": self.projection_dim,
        }

    @property
    def is_empty(self) -> bool:
        return all(value is None for value in self.to_dict().values())


@dataclass
class ArchitectureResult:
    fields: Dict[str, Any] = field(default_factory=dict)
    field_sources: Dict[str, str] = field(default_factory=dict)
    sources: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_registry_dict(self) -> Dict[str, Any]:
        payload = dict(self.fields)
        if self.field_sources:
            payload["field_sources"] = dict(self.field_sources)
        if self.sources:
            payload["sources"] = list(dict.fromkeys(self.sources))
        if self.warnings:
            payload["warnings"] = list(dict.fromkeys(self.warnings))
        payload["collected_at"] = datetime.now(timezone.utc).isoformat()
        return payload


def _promote_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int,)):
        return int(value)
    if isinstance(value, float) and value == math.floor(value):
        return int(value)
    try:
        as_int = int(value)
    except Exception:  # noqa: BLE001
        return None
    return as_int


def _promote_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return None


def _set_field(
    result: ArchitectureResult,
    field: str,
    value: Any,
    *,
    source: str,
    transform: Optional[str] = None,
) -> None:
    if value is None:
        return
    transformed = value
    if transform == "int":
        transformed = _promote_int(value)
    elif transform == "float":
        transformed = _promote_float(value)
    elif transform == "json":
        transformed = value

    if transformed is None:
        return

    existing = result.fields.get(field)
    if existing is None:
        result.fields[field] = transformed
        result.field_sources[field] = source
        return

    # Only replace existing values if they were placeholders (None) or zeros.
    if isinstance(existing, (int, float)) and existing <= 0 and transformed:
        result.fields[field] = transformed
        result.field_sources[field] = source


def _normalize_path(path: Path) -> str:
    try:
        return str(path.resolve(strict=False))
    except Exception:  # noqa: BLE001
        return str(path)


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(data, dict):
        return data
    return None


def _extract_vision_from_config(config: Dict[str, Any]) -> VisionMetadata:
    vision_section = config.get("vision_config")
    if isinstance(vision_section, dict):
        depth = vision_section.get("depth") or vision_section.get("num_hidden_layers")
        hidden = (
            vision_section.get("hidden_size")
            or vision_section.get("vision_hidden_size")
            or vision_section.get("out_hidden_size")
        )
        image_size = vision_section.get("image_size") or vision_section.get(
            "vision_image_size"
        )
        patch_size = (
            vision_section.get("patch_size")
            or vision_section.get("patch_embed_size")
            or vision_section.get("vision_patch_size")
        )
        return VisionMetadata(
            image_size=_promote_int(image_size),
            patch_size=_promote_int(patch_size),
            hidden_size=_promote_int(hidden),
            num_layers=_promote_int(
                depth
                or vision_section.get("vision_num_hidden_layers")
                or vision_section.get("num_layers")
            ),
            projection_dim=_promote_int(
                vision_section.get("projection_dim")
                or vision_section.get("vision_projection_dim")
            ),
        )
    return VisionMetadata()


def _extract_ollama_identifier(raw_path: Optional[str]) -> Optional[str]:
    if not raw_path:
        return None
    value = raw_path.strip()
    for prefix in ("ollama://", "ollama:/", "ollama:"):
        if value.startswith(prefix):
            ident = value[len(prefix) :].strip().strip("/")
            return ident or None
    return None


def _params_billion_from_label(label: Optional[str]) -> Optional[float]:
    parsed = _extract_param_size(label)
    if not parsed:
        return None
    try:
        numeric = float(parsed.rstrip("Bb"))
    except Exception:  # noqa: BLE001
        return None
    return numeric


def _collect_from_ollama(result: ArchitectureResult, model_name: str) -> None:
    source = f"ollama:{model_name}"
    try:
        with OllamaClient() as client:
            payload = client.show_model(model_name)
    except OllamaError as exc:
        result.warnings.append(f"Ollama metadata unavailable for {model_name}: {exc}")
        return
    except Exception as exc:  # noqa: BLE001
        result.warnings.append(
            f"Ollama metadata unexpected error for {model_name}: {exc}"
        )
        return

    result.sources.append(source)
    details = payload.get("details")
    if not isinstance(details, dict):
        result.warnings.append(f"Ollama response missing details for {model_name}")
        return

    context_length = details.get("context_length") or details.get(
        "context_length_tokens"
    )
    _set_field(result, "context_length", context_length, source=source, transform="int")

    params_field = (
        details.get("parameter_size")
        or details.get("parameters")
        or details.get("parameter_size_label")
    )
    params_billion = _params_billion_from_label(params_field)
    if params_billion:
        _set_field(
            result, "params_billion", params_billion, source=source, transform="float"
        )


def _collect_from_subconfig(
    result: ArchitectureResult,
    *,
    source: str,
    payload: Dict[str, Any],
) -> None:
    mappings: Iterable[Tuple[str, Iterable[str], str]] = (
        ("num_layers", ("num_hidden_layers", "num_layers", "n_layer", "depth"), "int"),
        ("num_attention_heads", ("num_attention_heads", "n_head", "num_heads"), "int"),
        (
            "num_kv_heads",
            (
                "num_key_value_heads",
                "num_kv_heads",
                "multi_query_group_num",
                "multi_query_heads",
            ),
            "int",
        ),
        (
            "hidden_size",
            (
                "hidden_size",
                "n_embd",
                "model_dim",
                "d_model",
                "embed_dim",
                "projection_dim",
            ),
            "int",
        ),
        (
            "intermediate_size",
            (
                "intermediate_size",
                "n_inner",
                "ffn_dim",
                "mlp_hidden_size",
                "feed_forward_hidden_size",
            ),
            "int",
        ),
        (
            "max_position_embeddings",
            (
                "max_position_embeddings",
                "max_seq_len",
                "max_sequence_length",
                "max_position_embeddings_decoder",
            ),
            "int",
        ),
        (
            "context_length",
            (
                "context_length",
                "context_window",
                "max_position_embeddings",
                "max_seq_len",
                "max_sequence_length",
            ),
            "int",
        ),
    )

    for canonical, aliases, transform in mappings:
        for alias in aliases:
            if alias in payload:
                _set_field(
                    result,
                    canonical,
                    payload[alias],
                    source=source,
                    transform=transform,
                )
                break

    rope_theta = payload.get("rope_theta") or payload.get("rope_freq_base")
    _set_field(result, "rope_theta", rope_theta, source=source, transform="float")
    rope_scaling = payload.get("rope_scaling") or payload.get("rope_scaling_config")
    if rope_scaling and isinstance(rope_scaling, dict):
        result.fields.setdefault("rope_scaling", rope_scaling)
        result.field_sources["rope_scaling"] = source

    kv_precision = (
        payload.get("kv_cache_dtype")
        or payload.get("kv_cache_type")
        or payload.get("cache_dtype")
    )
    if kv_precision:
        _set_field(
            result,
            "kv_precision",
            str(kv_precision).lower(),
            source=source,
            transform="json",
        )

    vocab_size = payload.get("vocab_size")
    _set_field(result, "vocab_size", vocab_size, source=source, transform="int")

    params = payload.get("model_size_in_billions") or payload.get("params_billions")
    if params:
        _set_field(result, "params_billion", params, source=source, transform="float")


def _collect_from_hf_config(result: ArchitectureResult, config_path: Path) -> None:
    config = _load_json(config_path)
    if not config:
        result.warnings.append(
            f"Failed to parse config.json at {config_path} (ignored)."
        )
        return

    source = f"config:{_normalize_path(config_path)}"
    result.sources.append(source)

    # Inspect top-level fields.
    _collect_from_subconfig(result, source=source, payload=config)

    text_cfg = config.get("text_config")
    if isinstance(text_cfg, dict):
        _collect_from_subconfig(
            result, source=f"{source}:text_config", payload=text_cfg
        )

    decoder_cfg = config.get("decoder")
    if isinstance(decoder_cfg, dict):
        _collect_from_subconfig(result, source=f"{source}:decoder", payload=decoder_cfg)

    vision_meta = _extract_vision_from_config(config)
    if not vision_meta.is_empty:
        result.fields.setdefault("vision", vision_meta.to_dict())
        result.field_sources["vision"] = source


def _collect_from_generation_config(result: ArchitectureResult, path: Path) -> None:
    config = _load_json(path)
    if not config:
        return
    source = f"generation_config:{_normalize_path(path)}"
    if source not in result.sources:
        result.sources.append(source)
    # Generation config often includes max length hints.
    max_new_tokens = config.get("max_new_tokens")
    if max_new_tokens:
        _set_field(
            result,
            "max_generation_tokens",
            max_new_tokens,
            source=source,
            transform="int",
        )
    if "max_length" in config:
        _set_field(
            result,
            "max_length",
            config["max_length"],
            source=source,
            transform="int",
        )


def _collect_from_gguf(result: ArchitectureResult, gguf_path: Path) -> None:
    if GGUFReader is None:  # pragma: no cover - dependency optional
        result.warnings.append("gguf module unavailable; skipping GGUF inspection.")
        return

    try:
        reader = GGUFReader(str(gguf_path))
    except Exception as exc:  # noqa: BLE001 - corrupt file?
        result.warnings.append(f"Failed to read GGUF header ({gguf_path}): {exc}")
        return

    source = f"gguf:{_normalize_path(gguf_path)}"
    result.sources.append(source)

    def get(name: str) -> Any:
        try:
            return reader.get_field(name)
        except KeyError:
            return None

    mappings = (
        ("num_layers", "llama.block_count"),
        ("num_attention_heads", "llama.attention.head_count"),
        ("num_kv_heads", "llama.attention.head_count_kv"),
        ("hidden_size", "llama.embedding_length"),
        ("intermediate_size", "llama.feed_forward_length"),
        ("context_length", "llama.context_length"),
        ("rope_theta", "llama.rope.freq_base"),
    )
    for canonical, key in mappings:
        value = get(key)
        transform = "float" if canonical == "rope_theta" else "int"
        _set_field(result, canonical, value, source=source, transform=transform)

    rope_scale = get("llama.rope.freq_scale")
    if rope_scale:
        result.fields.setdefault("rope_scaling", {"freq_scale": rope_scale})
        result.field_sources["rope_scaling"] = source

    vocab_size = get("tokenizer.ggml.tokens_count") or get("llama.vocab_size")
    _set_field(result, "vocab_size", vocab_size, source=source, transform="int")


def _derive_head_dim(result: ArchitectureResult) -> None:
    if "head_dim" in result.fields:
        return
    hidden = result.fields.get("hidden_size")
    heads = result.fields.get("num_attention_heads")
    if not hidden or not heads:
        return
    try:
        if hidden % heads == 0:
            result.fields["head_dim"] = int(hidden // heads)
            result.field_sources["head_dim"] = (
                result.field_sources.get("hidden_size") or "derived"
            )
    except Exception:  # noqa: BLE001
        return


def _ensure_context_length(result: ArchitectureResult) -> None:
    if result.fields.get("context_length"):
        return
    max_pos = result.fields.get("max_position_embeddings")
    if max_pos:
        result.fields["context_length"] = max_pos
        result.field_sources["context_length"] = result.field_sources.get(
            "max_position_embeddings", "derived"
        )


def collect_architecture_metadata(
    model_dir: Path,
    *,
    raw_path: Optional[str] = None,
    served_model_id: Optional[str] = None,
    prefer_config: bool = True,
) -> ArchitectureResult:
    """Inspect a downloaded model directory and return architecture metadata."""

    result = ArchitectureResult()
    if not model_dir.exists():
        result.warnings.append(f"Model directory not found: {model_dir}")
        ollama_identifier = _extract_ollama_identifier(raw_path) or (
            served_model_id if served_model_id and served_model_id.strip() else None
        )
        if ollama_identifier:
            _collect_from_ollama(result, ollama_identifier)
        return result

    config_path = model_dir / "config.json"
    if prefer_config and config_path.exists():
        _collect_from_hf_config(result, config_path)
    else:
        # Look for nested configs (common when repo name matches top-level dir)
        for candidate in model_dir.rglob("config.json"):
            if "summaries" in candidate.parts:
                continue  # skip huggingface repo metadata
            _collect_from_hf_config(result, candidate)
            break

    gen_config = model_dir / "generation_config.json"
    if gen_config.exists():
        _collect_from_generation_config(result, gen_config)

    gguf_files = list(model_dir.glob("*.gguf"))
    if not gguf_files:
        # Some repos store weights under subdirectories (e.g., quantized/).
        gguf_files = list(model_dir.rglob("*.gguf"))
    if gguf_files:
        # Use the first GGUF header as a metadata source.
        _collect_from_gguf(result, gguf_files[0])

    _derive_head_dim(result)
    _ensure_context_length(result)

    vision = result.fields.get("vision")
    if isinstance(vision, dict) and not any(v is not None for v in vision.values()):
        result.fields.pop("vision", None)
        result.field_sources.pop("vision", None)

    return result


def infer_default_vllm_arguments(
    architecture: ArchitectureResult,
    *,
    min_context_tokens: int = 4096,
    utilization: float = 0.85,
    batch_size: int = 1,
) -> List[str]:
    """Return conservative default vLLM extra args for a new registry entry."""

    args: List[str] = []
    context_tokens = architecture.fields.get("context_length")
    if isinstance(context_tokens, int) and context_tokens > 0:
        target = max(min_context_tokens, int(context_tokens * 0.6))
        args.extend(["--max-model-len", str(target)])
    else:
        args.extend(["--max-model-len", str(min_context_tokens)])

    args.extend(["--max-num-seqs", str(batch_size)])
    args.extend(["--gpu-memory-utilization", f"{utilization:.2f}"])
    args.extend(["--tensor-parallel-size", "1"])
    return args


def merge_architecture_metadata(
    existing: Optional[Dict[str, Any]], new_payload: ArchitectureResult
) -> Dict[str, Any]:
    """Merge new architecture metadata with an existing registry payload."""

    merged = dict(existing or {})
    new_dict = new_payload.to_registry_dict()
    for key, value in new_dict.items():
        if key == "collected_at":
            merged[key] = value
            continue
        if key == "warnings":
            merged.setdefault("warnings", [])
            merged["warnings"] = list(dict.fromkeys(merged["warnings"] + value))
            continue
        if key == "sources":
            merged.setdefault("sources", [])
            merged["sources"] = list(dict.fromkeys(merged["sources"] + value))
            continue
        if key == "field_sources":
            merged_sources = dict(merged.get("field_sources") or {})
            merged_sources.update(value)
            merged["field_sources"] = merged_sources
            continue
        merged[key] = value
    return merged


__all__ = [
    "ArchitectureResult",
    "VisionMetadata",
    "collect_architecture_metadata",
    "infer_default_vllm_arguments",
    "merge_architecture_metadata",
]
