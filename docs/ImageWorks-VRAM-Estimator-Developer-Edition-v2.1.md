# 🧠 ImageWorks VRAM Estimator — Developer Edition v2.1 (Streamlit Integrated)
**Unified Architecture, API, and GUI Specification**
Supersedes all earlier drafts (2024–2025).

---

## 1. Overview

This edition merges the **estimator engine**, **CLI/REST interfaces**, and a **Streamlit GUI** for seamless integration into the existing ImageWorks tool ecosystem.
It is the definitive source for VRAM planning, supporting FP8 / INT8 / INT4 / FP4 / FP16 / BF16 quantisations.

---

## 2. Objectives

- Accurate VRAM prediction for vLLM models (language and vision-language).
- Support both **forward** and **inverse** (max context for given VRAM) estimation.
- Streamlit-based GUI (no new web frameworks).
- Auto-detect GPU overheads from local profiles.
- Allow registry-driven model selection.

---

## 3. Directory Layout

```
src/
└── imageworks/
    ├── tools/
    │   ├── vram_estimator.py         ← quantitative engine (Section 6)
    │   ├── overhead_profiles.json     ← optional per-GPU calibration
    │   └── ...
    ├── cli/
    │   └── vram_estimate.py          ← CLI
    ├── services/
    │   └── vram_estimator_api.py     ← FastAPI (optional)
    └── streamlit_app/
        └── pages/
            └── VRAM_Estimator.py     ← GUI (Section 8)
```

---

## 4. Quantisation Coverage

| Quant Type | Bytes / Param | Notes |
|-------------|---------------|-------|
| FP16 / BF16 | 2.0 | Baseline |
| FP8 (W8A8) | 1.0 (+5–10%) | Ada/Hopper native |
| INT8 | 1.0 (+0–10%) | GPTQ-INT8 or similar |
| INT4 (AWQ/GPTQ) | 0.55–0.6 | Includes scale metadata |
| FP4 (Experimental) | 0.5 | Use override |
| Custom | user-defined | CLI/GUI/API override |

---

## 5. VRAM Components

| Component | Description |
|------------|-------------|
| **Weights** | static param memory |
| **KV Cache** | scales with layers × heads × tokens × batch |
| **Overhead** | driver, workspace, prefill, fragmentation |
| **Vision Tower** | fixed ~1 GiB |

**KV Cache Formula:**

\[
\text{KV}_{GiB} = \frac{2 × L × H_{kv} × d_{head} × \text{bytes(elem)} × (\text{context} × \text{batch})}{1024^3}
\]

---

## 6. Core Estimator (Python Engine)

```python
BYTES_PER_PARAM = {
    "fp16": 2.0, "bf16": 2.0,
    "fp8": 1.0, "int8": 1.0,
    "int4": 0.55, "fp4": 0.50,
}

KV_BYTES_ELEM = {"fp16": 2, "bf16": 2, "fp8": 1}

def estimate_vram_gib(
    params_billion=8.0,
    quant="fp8", L=36, H_kv=8, d_head=128,
    kv_dtype="fp8", context_k=8, batch=1,
    overhead_gib=1.5, vision_gib=1.0,
    frag_factor=1.10,
    bytes_per_param_override=None,
    kv_bytes_override=None,
):
    """Forward estimator: returns detailed VRAM breakdown (GiB)."""
    bpp = BYTES_PER_PARAM.get(quant, 1.0) if bytes_per_param_override is None else bytes_per_param_override
    kvb = KV_BYTES_ELEM.get(kv_dtype, 2) if kv_bytes_override is None else kv_bytes_override

    weights_gib = (params_billion * 1e9 * bpp) / (1024 ** 3)
    tokens_active = context_k * 1024 * batch
    kv_gib = (2 * L * H_kv * d_head * kvb * tokens_active) / (1024 ** 3)
    total_gib = (weights_gib + kv_gib + overhead_gib + vision_gib) * frag_factor

    return {
        "weights_gib": weights_gib,
        "kv_gib": kv_gib,
        "overhead_plus_vision_gib": overhead_gib + vision_gib,
        "frag_factor": frag_factor,
        "total_gib": total_gib,
        "params": {
            "quant": quant, "L": L, "H_kv": H_kv, "d_head": d_head,
            "kv_dtype": kv_dtype, "kv_bytes": kvb,
            "context_k": context_k, "batch": batch
        }
    }
```

---

## 7. Streamlit GUI Integration

See the full implementation details in the main response — `pages/VRAM_Estimator.py` includes:
- automatic registry model selection
- auto-loaded GPU overhead profile
- JSON output + download
- integrated forward/inverse estimates

---

## 8. Validation & CI

| Stage | Description |
|--------|--------------|
| Unit | Verify estimator math (weights/KV/overhead). |
| Integration | Compare Streamlit + CLI outputs to `nvidia-smi`. |
| Regression | Maintain ≤ ± 3 % drift across driver versions. |

---

## 9. Summary

This **v2.1 edition** integrates:
- The quantitative estimator (forward/inverse)
- Profiling-aware overheads
- CLI, REST, and Streamlit interfaces
- Automatic registry + GPU detection

It is now the authoritative ImageWorks VRAM Estimator specification for both **developers and GUI users**.

---

### References
- vLLM Docs (Paged Attention, KV caching)
- NVIDIA Transformer Engine FP8 overview
- Qwen-VL and Gemma-2 config structures
- ImageWorks internal model registry schema
