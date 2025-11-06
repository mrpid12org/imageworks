"""Streamlit page for the VRAM estimator."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from imageworks.libs.hardware.gpu_detector import GPUDetector
from imageworks.tools.vram_estimator import (
    BYTES_PER_PARAM,
    KV_BYTES_ELEM,
    auto_profile,
    estimate_max_context_k,
    estimate_vram_gib,
    load_overhead_profiles,
)


def _load_registry_models() -> list[dict]:
    registry_path = Path("configs/model_registry.json")
    if not registry_path.exists():
        return []
    try:
        return json.loads(registry_path.read_text(encoding="utf-8"))
    except Exception:  # pragma: no cover - presentation only
        return []


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


def _render_forward_estimator():
    st.subheader("Forward Estimate")
    profiles = load_overhead_profiles()
    profile_names = sorted(profiles)

    cols = st.columns(2)
    with cols[0]:
        selected_profile = st.selectbox(
            "Overhead profile",
            profile_names,
            index=profile_names.index("default") if "default" in profile_names else 0,
        )
    with cols[1]:
        auto_button = st.button(
            "Auto-detect profile",
            help="Use detected GPU type to choose overhead settings.",
        )

    profile_data = profiles[selected_profile]
    if auto_button:
        auto = auto_profile(None)
        profile_data = auto
        st.success("Applied auto-detected profile.")

    model_options = _load_registry_models()
    model_names = ["(manual entry)"] + [
        entry.get("display_name", entry.get("name", "unknown"))
        for entry in model_options
    ]

    with st.form("vram_estimate"):
        selected_model = st.selectbox("Model (optional)", model_names, index=0)
        params_billion = st.number_input(
            "Parameters (billions)", value=8.0, min_value=0.1, step=0.1
        )
        quant = st.selectbox("Quantisation", list(BYTES_PER_PARAM.keys()), index=2)
        layers = st.number_input("Decoder layers", value=36, min_value=1, step=1)
        kv_heads = st.number_input("KV heads per layer", value=8, min_value=1, step=1)
        head_dim = st.number_input("Head dimension", value=128, min_value=16, step=8)
        kv_dtype = st.selectbox("KV precision", list(KV_BYTES_ELEM.keys()), index=2)
        context_k = st.number_input(
            "Context window (k tokens)", value=8.0, min_value=0.25, step=0.25
        )
        batch = st.number_input("Batch size", value=1, min_value=1, step=1)

        overhead_gib = st.number_input(
            "Overhead GiB", value=float(profile_data.get("overhead_gib", 1.5)), step=0.1
        )
        vision_gib = st.number_input(
            "Vision tower GiB",
            value=float(profile_data.get("vision_gib", 0.0)),
            step=0.1,
        )
        frag_factor = st.number_input(
            "Fragmentation factor",
            value=float(profile_data.get("frag_factor", 1.10)),
            step=0.01,
        )

        submitted = st.form_submit_button("Estimate VRAM")
        if submitted:
            if selected_model != "(manual entry)":
                st.info(
                    "Registry metadata is informational only; quantitative inputs are manual."
                )

            estimate = estimate_vram_gib(
                params_billion=params_billion,
                quant=quant,
                layers=int(layers),
                kv_heads=int(kv_heads),
                head_dim=int(head_dim),
                kv_dtype=kv_dtype,
                context_k=context_k,
                batch=int(batch),
                overhead_gib=overhead_gib,
                vision_gib=vision_gib,
                frag_factor=frag_factor,
            )

            metrics = st.columns(3)
            metrics[0].metric("Weights (GiB)", f"{estimate.weights_gib:.2f}")
            metrics[1].metric("KV Cache (GiB)", f"{estimate.kv_gib:.2f}")
            metrics[2].metric("Total (GiB)", f"{estimate.total_gib:.2f}")

            st.json(estimate.to_dict())


def _render_inverse_estimator():
    st.subheader("Inverse Estimate (Max Context)")
    profiles = load_overhead_profiles()
    profile_names = sorted(profiles)

    cols = st.columns(2)
    with cols[0]:
        selected_profile = st.selectbox(
            "Profile for inverse run",
            profile_names,
            index=profile_names.index("default") if "default" in profile_names else 0,
            key="inverse_profile",
        )
    with cols[1]:
        auto_button = st.button("Auto-detect for inverse", key="inverse_auto")

    profile_data = profiles[selected_profile]
    if auto_button:
        auto = auto_profile(None)
        profile_data = auto
        st.success("Applied auto-detected profile.")

    with st.form("vram_inverse"):
        total_vram = st.number_input(
            "VRAM budget (GiB)", value=16.0, min_value=1.0, step=0.5
        )
        params_billion = st.number_input(
            "Parameters (billions)",
            value=8.0,
            min_value=0.1,
            step=0.1,
            key="inverse_params",
        )
        quant = st.selectbox(
            "Quantisation", list(BYTES_PER_PARAM.keys()), index=2, key="inverse_quant"
        )
        layers = st.number_input(
            "Decoder layers", value=36, min_value=1, step=1, key="inverse_layers"
        )
        kv_heads = st.number_input(
            "KV heads per layer", value=8, min_value=1, step=1, key="inverse_heads"
        )
        head_dim = st.number_input(
            "Head dimension", value=128, min_value=16, step=8, key="inverse_head_dim"
        )
        kv_dtype = st.selectbox(
            "KV precision", list(KV_BYTES_ELEM.keys()), index=2, key="inverse_kv_dtype"
        )
        batch = st.number_input(
            "Batch size", value=1, min_value=1, step=1, key="inverse_batch"
        )

        overhead_gib = st.number_input(
            "Overhead GiB",
            value=float(profile_data.get("overhead_gib", 1.5)),
            step=0.1,
            key="inverse_overhead",
        )
        vision_gib = st.number_input(
            "Vision tower GiB",
            value=float(profile_data.get("vision_gib", 0.0)),
            step=0.1,
            key="inverse_vision",
        )
        frag_factor = st.number_input(
            "Fragmentation factor",
            value=float(profile_data.get("frag_factor", 1.10)),
            step=0.01,
            key="inverse_frag",
        )
        max_context_k = st.number_input(
            "Upper bound (k tokens)",
            value=64.0,
            min_value=1.0,
            step=1.0,
            key="inverse_bound",
        )

        submitted = st.form_submit_button("Calculate Max Context")
        if submitted:
            payload = estimate_max_context_k(
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

            metrics = st.columns(3)
            metrics[0].metric("Context (k tokens)", f"{payload['context_k']:.2f}")
            metrics[1].metric("Context (tokens)", f"{payload['context_tokens']:.0f}")
            metrics[2].metric("Estimated total (GiB)", f"{payload['total_gib']:.2f}")

            st.json(payload)


def main():
    st.title("⚡ VRAM Estimator")
    st.write(
        "Estimate VRAM requirements for vLLM deployments or solve for the maximum achievable context window "
        "at a given VRAM budget. Values use the ImageWorks estimator engine."
    )

    with st.expander("GPU Overview", expanded=False):
        _render_gpu_overview()

    _render_forward_estimator()
    st.divider()
    _render_inverse_estimator()


if __name__ == "__main__":  # pragma: no cover
    main()
