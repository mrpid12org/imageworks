# VRAM Estimator Runbook

Operational guide for the ImageWorks VRAM estimator (CLI + GUI).

---

## 1. Prerequisites

- Python environment with ImageWorks installed (`uv pip install -e .` or inside
  the dev container).
- Optional: NVIDIA GPU + `nvidia-smi` for automatic profile detection.

---

## 2. CLI Workflow

### Inspect available profiles

```
uv run imageworks-vram-estimator profiles
```

Shows overhead/fragmentation presets (default, ada\_16gb, hopper\_80gb, studio\_dual).

### Forward estimate (VRAM required for given context)

```
uv run imageworks-vram-estimator estimate \
  --params-billion 8 \
  --quant fp8 --layers 36 --kv-heads 8 --head-dim 128 \
  --kv-dtype fp8 --context-k 8 --batch 1 \
  --profile ada_16gb
```

- `context-k` is measured in thousands (8 ⇒ 8192 tokens).
- Use `--json` to emit machine-readable output.
- Omit `--profile` to auto-detect based on the first available GPU.

### Inverse estimate (max context for VRAM budget)

```
uv run imageworks-vram-estimator max-context \
  --total-vram-gib 12 \
  --params-billion 8 \
  --quant fp8 --layers 36 --kv-heads 8 --head-dim 128 \
  --kv-dtype fp8 --batch 1 \
  --profile ada_16gb
```

Returns the largest context length (in k tokens) that fits within 12 GiB.

---

## 3. Streamlit GUI (⚡ VRAM Estimator)

1. Launch the GUI control centre:
   ```
   ./scripts/launch_gui.sh
   ```
2. Navigate to the **⚡ VRAM Estimator** tab.
3. Optional: expand “GPU Overview” to confirm detected hardware.
4. Use the **Forward Estimate** form to enter model parameters and context.
   - Select an overhead profile or press “Auto-detect profile”.
   - Choosing a registry model now auto-prefills layers, heads, KV precision, and
     context length from `metadata.architecture`.
   - Click **Estimate VRAM** to view weights, KV cache, totals, and JSON output.
   - Press **Apply to Model Settings** to write the current context + batch back into
     the registry (vLLM `--max-model-len` / `--max-num-seqs`, Ollama `--num-ctx`).
5. Use the **Inverse Estimate** form to find the maximum context for a specified
   VRAM budget.
   - Adjust `batch`, `layers`, and other fields as needed.
   - Results include both the context in tokens and the estimated total GiB.

---

## 4. Overhead Profiles

Profiles live in `src/imageworks/tools/overhead_profiles.json`. Add or adjust
entries to reflect calibrated measurements (e.g., custom fragmentation or vision
tower overheads). Package builds include this asset automatically.

---

## 5. Troubleshooting

| Symptom | Resolution |
| --- | --- |
| CLI reports unknown profile | Run `imageworks-vram-estimator profiles` to confirm available names. |
| GUI doesn’t show GPU info | Ensure `nvidia-smi` is available and the container/user has GPU access. |
| Estimates look too high/low | Adjust `overhead_gib`, `vision_gib`, or `frag_factor` manually based on empirical measurements. |
| Need programmatic usage | Import `estimate_vram_gib` / `estimate_max_context_k` from `imageworks.tools.vram_estimator`. |

---

## 6. Related Material

- [VRAM Estimator Reference](../reference/vram-estimator.md)
- [Developer Edition v2.1 Specification](../ImageWorks-VRAM-Estimator-Developer-Edition-v2.1.md)
