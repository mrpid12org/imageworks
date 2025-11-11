"""Streamlit page for the VRAM estimator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import streamlit as st

from imageworks.libs.hardware.gpu_detector import GPUDetector
from imageworks.model_loader.runtime_metadata import (
    RuntimeEvent,
    load_runtime_events,
)
from imageworks.tools.vram_estimator import (
    BYTES_PER_PARAM,
    KV_BYTES_ELEM,
    auto_profile,
    estimate_max_context_k,
    load_overhead_profiles,
)
from imageworks.model_loader import registry as unified_registry
from imageworks.gui.components.sidebar_footer import render_sidebar_footer

_RUNTIME_EVENTS: dict[str, RuntimeEvent] | None = None


def _estimate_vision_gib(vision: dict) -> float:
    """Heuristic VRAM estimate for a vision tower."""

    layers = vision.get("num_layers") or 24
    hidden = vision.get("hidden_size") or vision.get("projection_dim") or 1024
    image_size = vision.get("image_size") or 448
    patch = vision.get("patch_size") or 14

    try:
        tokens = (image_size / patch) ** 2 if patch else 0.0
    except Exception:  # noqa: BLE001
        tokens = 0.0

    base = 0.60
    layer_term = 0.015 * float(layers)
    hidden_term = 0.00005 * float(hidden)
    token_term = 0.0002 * float(tokens)
    total = base + layer_term + hidden_term + token_term

    return round(min(max(total, 0.5), 2.5), 2)


def _estimate_kv_gib(
    *,
    layers: int,
    kv_heads: int,
    head_dim: int,
    kv_dtype: str,
    context_tokens: float,
    batch: int,
) -> float:
    """Estimate KV cache consumption for a given context/batch."""

    kv_bytes = KV_BYTES_ELEM.get(kv_dtype.lower(), 2.0)
    tokens_active = max(context_tokens, 0.0) * max(batch, 1)
    kv_bytes_total = 2 * layers * kv_heads * head_dim * kv_bytes * tokens_active
    return kv_bytes_total / (1024**3)


def _load_registry_models() -> list[dict]:
    registry_path = Path("configs/model_registry.json")
    if not registry_path.exists():
        return []
    try:
        return json.loads(registry_path.read_text(encoding="utf-8"))
    except Exception:  # pragma: no cover - presentation only
        return []


def _auto_profile_name(profile_names: list[str]) -> Optional[str]:
    """Return the preferred profile name for the detected GPU."""

    detector = GPUDetector()
    gpus = detector.detect_gpus()
    if not gpus:
        return (
            "default"
            if "default" in profile_names
            else (profile_names[0] if profile_names else None)
        )

    gpu = gpus[0]
    name = gpu.name.lower()

    if ("4090" in name or "4080" in name) and "ada_16gb" in profile_names:
        return "ada_16gb"
    if gpu.vram_total_mb >= 80000 and "hopper_80gb" in profile_names:
        return "hopper_80gb"
    if "default" in profile_names:
        return "default"
    return profile_names[0] if profile_names else None


def _normalize_quant_label(entry: Optional[dict]) -> str:
    quant = ""
    if entry:
        quant = str(entry.get("quantization") or "").lower()
    if "fp8" in quant:
        return "fp8"
    if "bf16" in quant:
        return "bf16"
    if "fp16" in quant or "f16" in quant:
        return "fp16"
    if "q4" in quant or "int4" in quant:
        return "int4"
    if "q8" in quant or "int8" in quant:
        return "int8"
    return "bf16"


def _runtime_event_for(entry: Optional[dict]) -> Optional[RuntimeEvent]:
    global _RUNTIME_EVENTS
    if not entry:
        return None
    name = entry.get("name")
    if not name:
        return None
    if _RUNTIME_EVENTS is None:
        _RUNTIME_EVENTS = load_runtime_events()
    return _RUNTIME_EVENTS.get(name)


def _derive_estimator_defaults(entry: Optional[dict]) -> dict:
    defaults = {
        "params_billion": 8.0,
        "quant": "bf16",
        "layers": 36,
        "kv_heads": 8,
        "head_dim": 128,
        "kv_dtype": "fp16",
        "context_k": 8.0,
        "batch": 1,
        "vision_gib": None,
    }
    if not entry:
        return defaults

    metadata = entry.get("metadata") or {}
    arch = metadata.get("architecture") or {}

    params_billion = arch.get("params_billion") or metadata.get("params_billion")
    if params_billion:
        try:
            defaults["params_billion"] = float(params_billion)
        except Exception:  # noqa: BLE001
            pass

    num_layers = arch.get("num_layers")
    if isinstance(num_layers, int) and num_layers > 0:
        defaults["layers"] = num_layers

    kv_heads = arch.get("num_kv_heads") or arch.get("num_attention_heads")
    if isinstance(kv_heads, int) and kv_heads > 0:
        defaults["kv_heads"] = kv_heads

    head_dim = arch.get("head_dim")
    if isinstance(head_dim, int) and head_dim > 0:
        defaults["head_dim"] = head_dim
    else:
        hidden_size = arch.get("hidden_size")
        attn_heads = arch.get("num_attention_heads")
        if (
            isinstance(hidden_size, int)
            and hidden_size > 0
            and isinstance(attn_heads, int)
            and attn_heads > 0
            and hidden_size % attn_heads == 0
        ):
            defaults["head_dim"] = hidden_size // attn_heads

    kv_precision = arch.get("kv_precision")
    if kv_precision:
        defaults["kv_dtype"] = str(kv_precision).lower()

    context_length = arch.get("context_length") or arch.get("max_position_embeddings")
    if isinstance(context_length, int) and context_length > 0:
        defaults["context_k"] = max(1.0, round(context_length / 1024, 2))

    defaults["quant"] = _normalize_quant_label(entry)

    runtime_event = _runtime_event_for(entry)
    runtime_provided = False
    if runtime_event:
        metrics = runtime_event.payload.get("metrics") or {}
        gpu_snapshot = metrics.get("gpu_memory_gib") or {}
        measured_used = gpu_snapshot.get("used")
        if measured_used:
            quant_key = defaults["quant"]
            bytes_per_param = BYTES_PER_PARAM.get(
                quant_key, BYTES_PER_PARAM.get("bf16", 2.0)
            )
            weights_gib = defaults["params_billion"] * 1e9 * bytes_per_param / (1024**3)
            runtime_tokens = metrics.get("runtime_context_tokens")
            if isinstance(runtime_tokens, (int, float)) and runtime_tokens > 0:
                measured_context_tokens = float(runtime_tokens)
            else:
                measured_context_tokens = defaults.get("context_k", 1.0) * 1024
            batch_hint = int(
                metrics.get("runtime_batch")
                or metrics.get("configured_max_num_seqs")
                or 1
            )
            kv_gib_runtime = _estimate_kv_gib(
                layers=defaults["layers"],
                kv_heads=defaults["kv_heads"],
                head_dim=defaults["head_dim"],
                kv_dtype=defaults["kv_dtype"],
                context_tokens=measured_context_tokens,
                batch=batch_hint,
            )
            residual = max(measured_used - weights_gib - kv_gib_runtime - 0.5, 0.0)
            defaults["vision_gib"] = round(residual, 3)
            defaults["runtime_snapshot"] = {
                "gpu_used_gib": round(measured_used, 3),
                "weights_gib": round(weights_gib, 3),
                "kv_gib": round(kv_gib_runtime, 3),
                "timestamp": runtime_event.timestamp,
            }
            runtime_provided = True

            runtime_context_tokens = metrics.get("runtime_context_tokens")
            if (
                isinstance(runtime_context_tokens, (int, float))
                and runtime_context_tokens > 0
            ):
                defaults["context_k"] = max(
                    1.0, round(float(runtime_context_tokens) / 1024.0, 2)
                )
            runtime_batch = metrics.get("runtime_batch") or metrics.get(
                "configured_max_num_seqs"
            )
            if isinstance(runtime_batch, (int, float)) and runtime_batch > 0:
                defaults["batch"] = int(runtime_batch)

            defaults["overhead_gib"] = 0.5

    vision_meta = arch.get("vision")
    if not runtime_provided and isinstance(vision_meta, dict) and vision_meta:
        defaults["vision_gib"] = _estimate_vision_gib(vision_meta)
    return defaults


def _default_vram_budget():
    detector = GPUDetector()
    gpus = detector.detect_gpus()
    if not gpus:
        return 16.0, None
    primary = gpus[0]
    total_gib = primary.vram_total_mb / 1024
    return round(total_gib * 0.9, 2), primary


def _initialize_form_state(prefix: str, entry: Optional[dict]) -> tuple[dict, bool]:
    defaults = _derive_estimator_defaults(entry)
    selection_key = f"{prefix}_selected_model"
    current_id = (entry or {}).get("name") or "(manual)"
    previous_id = st.session_state.get(selection_key)
    selection_changed = previous_id != current_id
    st.session_state[selection_key] = current_id

    for field, value in defaults.items():
        state_key = f"{prefix}_{field}"
        if selection_changed:
            if value is not None:
                st.session_state[state_key] = value
            else:
                st.session_state.pop(state_key, None)
        else:
            st.session_state.setdefault(state_key, value)

    return defaults, selection_changed


def _read_flag(extra_args: list[str], flag: str) -> Optional[str]:
    token = f"--{flag}"
    for idx, arg in enumerate(extra_args):
        if arg == token:
            if idx + 1 < len(extra_args):
                value = extra_args[idx + 1]
                if not value.startswith("--"):
                    return value
            return None
        if arg.startswith(f"{token}="):
            return arg.split("=", 1)[1]
    return None


def _write_flag(extra_args: list[str], flag: str, value: str) -> list[str]:
    token = f"--{flag}"
    for idx, arg in enumerate(extra_args):
        if arg == token:
            if idx + 1 < len(extra_args) and not extra_args[idx + 1].startswith("--"):
                extra_args[idx + 1] = value
            else:
                extra_args.insert(idx + 1, value)
            return extra_args
        if arg.startswith(f"{token}="):
            extra_args[idx] = f"{token}={value}"
            return extra_args
    extra_args.extend([token, value])
    return extra_args


def _apply_estimator_to_registry(
    entry_dict: dict, *, context_tokens: int, batch: int, kv_dtype: str
) -> None:
    """Update registry extra args using estimator results."""

    registry = unified_registry.load_registry(force=True)
    entry_name = entry_dict.get("name")
    if not entry_name:
        raise RuntimeError("Selected entry missing registry name.")
    entry_obj = registry.get(entry_name)
    if entry_obj is None:
        raise RuntimeError(f"Registry entry '{entry_name}' not found.")

    extra_args = list(entry_obj.backend_config.extra_args or [])
    if entry_obj.backend == "vllm":
        extra_args = _write_flag(extra_args, "max-model-len", str(context_tokens))
        extra_args = _write_flag(extra_args, "max-num-seqs", str(max(batch, 1)))
        current_gpu_mem = _read_flag(extra_args, "gpu-memory-utilization") or "0.90"
        extra_args = _write_flag(
            extra_args, "gpu-memory-utilization", str(current_gpu_mem)
        )
        if kv_dtype and kv_dtype != "auto":
            extra_args = _write_flag(extra_args, "kv-cache-dtype", kv_dtype)
        entry_obj.backend_config.extra_args = extra_args
    elif entry_obj.backend == "ollama":
        extra_args = _write_flag(extra_args, "num-ctx", str(context_tokens))
        entry_obj.backend_config.extra_args = extra_args
    else:
        raise RuntimeError(
            f"Backend '{entry_obj.backend}' not supported for auto-apply."
        )

    if context_tokens > 0:
        entry_obj.generation_defaults.context_window = context_tokens

    unified_registry.update_entries([entry_obj], save=True)


def _render_gpu_overview() -> None:
    detector = GPUDetector()
    gpus = detector.detect_gpus()
    if not gpus:
        st.info("No NVIDIA GPU detected (or `nvidia-smi` unavailable).")
        return

    st.markdown("### Detected GPUs")
    for gpu in gpus:
        st.write(
            f"- GPU {gpu.index}: **{gpu.name}** — total VRAM: "
            f"{gpu.vram_total_mb / 1024:.1f} GiB (free {gpu.vram_free_mb / 1024:.1f} GiB)"
        )


def _render_inverse_estimator():
    st.subheader("Max Context Solver")
    profiles = load_overhead_profiles()
    profile_names = sorted(profiles)

    if profile_names:
        auto_default = _auto_profile_name(profile_names)
        if "inverse_profile" not in st.session_state:
            st.session_state["inverse_profile"] = (
                auto_default if auto_default in profile_names else profile_names[0]
            )

    cols = st.columns(2)
    with cols[0]:
        selected_profile = st.selectbox(
            "Profile for inverse run",
            profile_names,
            key="inverse_profile",
        )
    with cols[1]:
        auto_button = st.button("Auto-detect for inverse", key="inverse_auto")

    profile_data = profiles[selected_profile]
    if auto_button:
        auto_name = _auto_profile_name(profile_names)
        if auto_name and auto_name in profile_names:
            st.session_state["inverse_profile"] = auto_name
            selected_profile = auto_name
            profile_data = profiles[auto_name]
            st.success(f"Applied auto-detected profile '{auto_name}'.")
        else:
            auto = auto_profile(None)
            profile_data = auto
            st.success("Applied auto-detected profile.")

    model_options = _load_registry_models()
    model_names = ["(manual entry)"] + [
        entry.get("display_name", entry.get("name", "unknown"))
        for entry in model_options
    ]

    selected_model = st.selectbox("Model (optional)", model_names, index=0)
    selected_entry = None
    if selected_model != "(manual entry)":
        try:
            selected_index = model_names.index(selected_model) - 1
            if selected_index >= 0:
                selected_entry = model_options[selected_index]
        except ValueError:
            selected_entry = None

    defaults, selection_changed = _initialize_form_state("inverse", selected_entry)

    if selected_entry:
        architecture = (selected_entry.get("metadata") or {}).get("architecture") or {}
        with st.expander("Discovered architecture", expanded=False):
            st.json({k: v for k, v in architecture.items() if k not in {"sources"}})
        runtime_info = defaults.get("runtime_snapshot")
        if runtime_info:
            st.caption(
                f"Runtime GPU snapshot: {runtime_info['gpu_used_gib']:.2f} GiB "
                f"(logged {runtime_info['timestamp']}). "
                "Vision/overhead defaults incorporate this measurement."
            )

    primary_budget, gpu_info = _default_vram_budget()
    st.session_state.setdefault("inverse_total_vram", primary_budget)
    total_vram = st.number_input(
        "VRAM budget (GiB)", min_value=1.0, step=0.5, key="inverse_total_vram"
    )
    if gpu_info:
        st.caption(
            f"Detected {gpu_info.name} with {gpu_info.vram_total_mb / 1024:.1f} GiB total; using 90% ({primary_budget:.2f} GiB) as default budget."
        )

    params_key = "inverse_params_billion"
    layers_key = "inverse_layers"
    heads_key = "inverse_kv_heads"
    head_dim_key = "inverse_head_dim"
    batch_key = "inverse_batch"
    overhead_key = "inverse_overhead_gib"
    vision_key = "inverse_vision_gib"
    frag_key = "inverse_frag"
    bound_key = "inverse_bound"
    st.session_state.setdefault(params_key, defaults.get("params_billion", 8.0))
    st.session_state.setdefault(layers_key, defaults.get("layers", 32))
    st.session_state.setdefault(heads_key, defaults.get("kv_heads", 8))
    st.session_state.setdefault(head_dim_key, defaults.get("head_dim", 128))
    st.session_state.setdefault(batch_key, int(defaults.get("batch", 1)))
    overhead_default = defaults.get("overhead_gib")
    if overhead_default is None:
        overhead_default = float(profile_data.get("overhead_gib", 1.5))
    st.session_state.setdefault(overhead_key, overhead_default)
    st.session_state.setdefault(
        vision_key,
        defaults.get("vision_gib", float(profile_data.get("vision_gib", 0.0))),
    )
    if st.session_state[vision_key] is None:
        st.session_state[vision_key] = float(profile_data.get("vision_gib", 0.0))
    st.session_state.setdefault(frag_key, float(profile_data.get("frag_factor", 1.10)))
    if st.session_state[frag_key] is None:
        st.session_state[frag_key] = float(profile_data.get("frag_factor", 1.10))
    st.session_state.setdefault(bound_key, 64.0)

    params_billion = st.number_input(
        "Parameters (billions)", min_value=0.1, step=0.1, key=params_key
    )
    quant_key = "inverse_quant"
    quant_options = list(BYTES_PER_PARAM.keys())
    st.session_state.setdefault(
        quant_key,
        defaults.get("quant", quant_options[0]),
    )
    quant = st.selectbox("Quantisation", quant_options, key=quant_key)
    layers = st.number_input("Decoder layers", min_value=1, step=1, key=layers_key)
    kv_heads = st.number_input("KV heads per layer", min_value=1, step=1, key=heads_key)
    head_dim = st.number_input("Head dimension", min_value=16, step=8, key=head_dim_key)
    kv_dtype_key = "inverse_kv_dtype"
    st.session_state.setdefault(
        kv_dtype_key, defaults.get("kv_dtype", list(KV_BYTES_ELEM.keys())[0])
    )
    kv_dtype = st.selectbox(
        "KV precision", list(KV_BYTES_ELEM.keys()), key=kv_dtype_key
    )
    batch = st.number_input("Batch size", min_value=1, step=1, key=batch_key)
    if st.session_state[overhead_key] is None:
        st.session_state[overhead_key] = float(profile_data.get("overhead_gib", 1.5))
    overhead_gib = st.number_input("Overhead GiB", step=0.1, key=overhead_key)
    if st.session_state[vision_key] is None:
        st.session_state[vision_key] = float(profile_data.get("vision_gib", 0.0))
    vision_gib = st.number_input("Vision tower GiB", step=0.1, key=vision_key)
    frag_factor = st.number_input("Fragmentation factor", step=0.01, key=frag_key)
    max_context_k = st.number_input(
        "Upper bound (k tokens)", min_value=1.0, step=1.0, key=bound_key
    )

    def _solve_context() -> dict:
        return estimate_max_context_k(
            total_vram_gib=total_vram,
            params_billion=params_billion,
            quant=quant,
            layers=int(layers),
            kv_heads=int(kv_heads),
            head_dim=int(head_dim),
            kv_dtype=kv_dtype,
            batch=int(batch),
            overhead_gib=overhead_gib,
            vision_gib=vision_gib,
            frag_factor=frag_factor,
            max_context_k=max_context_k,
        )

    payload = _solve_context()

    metrics = st.columns(3)
    metrics[0].metric("Context (k tokens)", f"{payload['context_k']:.2f}")
    metrics[1].metric("Context (tokens)", f"{payload['context_tokens']:.0f}")
    metrics[2].metric("Estimated total (GiB)", f"{payload['total_gib']:.2f}")

    st.json(payload)

    if st.button("Recalculate with current inputs"):
        payload = _solve_context()
        st.json(payload)


def main():
    st.set_page_config(page_title="⚡ VRAM Estimator", layout="wide")
    st.title("⚡ VRAM Estimator")
    st.write(
        "Estimate VRAM requirements for vLLM deployments or solve for the maximum achievable context window "
        "at a given VRAM budget. Values use the ImageWorks estimator engine."
    )
    with st.sidebar:
        render_sidebar_footer()

    with st.expander("GPU Overview", expanded=False):
        _render_gpu_overview()

    _render_inverse_estimator()


if __name__ == "__main__":  # pragma: no cover
    main()
