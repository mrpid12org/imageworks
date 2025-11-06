# VRAM Estimator Reference

The VRAM estimator predicts the GPU memory requirements for vLLM deployments and
solves the inverse problem of finding the maximum context window that fits in a
given VRAM budget. It provides:

- a reusable Python engine (forward + inverse estimators),
- a Typer-based CLI (`imageworks-vram-estimator`),
- a Streamlit GUI page inside the ImageWorks control centre, and
- pluggable overhead profiles for different GPU families.

This document summarises the behaviour. For design background see
[`ImageWorks-VRAM-Estimator-Developer-Edition-v2.1.md`](../ImageWorks-VRAM-Estimator-Developer-Edition-v2.1.md).

## Components

| Layer | Module | Purpose |
| --- | --- | --- |
| Engine | `imageworks.tools.vram_estimator.estimate_vram_gib` | Forward estimate returning detailed weights, KV cache, and total GiB usage. |
| Engine | `imageworks.tools.vram_estimator.estimate_max_context_k` | Inverse estimation using binary search to find the highest context window for a VRAM budget. |
| CLI | `imageworks-vram-estimator` | Typer app with `estimate`, `max-context`, and `profiles` commands. |
| Streamlit | `⚡ VRAM Estimator` page (`src/imageworks/gui/pages/9_⚡_VRAM_Estimator.py`) | Interactive form for forward/inverse calculations and GPU detection. |
| Data | `src/imageworks/tools/overhead_profiles.json` | Overhead presets (Ada 16 GB, Hopper 80 GB, dual-GPU studio, etc.). |

## Estimation Model

The estimator breaks VRAM consumption into:

- **Weights:** parameters × bytes-per-param (supporting FP16/BF16/FP8/INT8/INT4/FP4).
- **KV cache:** 2 (keys+values) × layers × KV heads × head dim × tokens × bytes-per-element.
- **Overhead:** runtime buffers, attention workspace, CUDA driver, etc.
- **Vision tower:** optional extra GiB for multimodal models.
- **Fragmentation factor:** multiplier to cover allocator fragmentation and throttling.

Formulae and defaults are described fully in Section 5 of the design spec linked
above.

## CLI Usage

```
imageworks-vram-estimator estimate --params-billion 8 \
  --quant fp8 --layers 36 --kv-heads 8 --head-dim 128 \
  --kv-dtype fp8 --context-k 8 --batch 1 --profile ada_16gb

imageworks-vram-estimator max-context --total-vram-gib 12 \
  --params-billion 8 --quant fp8 --layers 36 --kv-heads 8 --head-dim 128 \
  --kv-dtype fp8 --batch 1 --profile ada_16gb

imageworks-vram-estimator profiles
```

The CLI auto-detects the first available GPU (via `GPUDetector`) to choose a
sensible overhead profile when none is provided. Use `--json` to emit machine
readable output.

## Streamlit Page

The GUI tab (⚡ VRAM Estimator) exposes the same engine with:

- GPU detection summary (`nvidia-smi` via `GPUDetector`).
- Forward estimate form with manual or registry-assisted inputs.
- Inverse estimate form to target specific VRAM budgets.
- Profile selector and on-demand auto detection.
- JSON output for copy/paste.

## Integration Notes

- Overhead profiles may be extended by editing
  `src/imageworks/tools/overhead_profiles.json`. Project packaging includes this
  asset by default.
- The estimator is agnostic to registry metadata—context length, layers, and
  head sizes are user supplied. Automating model introspection would require
  enriching the registry schema.
- Unit tests live in `tests/tools/test_vram_estimator.py`.
