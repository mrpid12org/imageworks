# Measuring and Optimising FP8 Quantization Performance for Qwen3‑VL‑8B‑Instruct‑Abliterated

This document provides guidance for timing and optimising the FP8 conversion of **`Qwen3‑VL‑8B‑Instruct‑abliterated`** on your system:

- **Environment:** WSL 2 Ubuntu on Windows
- **GPU:** RTX 4080 (16 GB VRAM, Ada Lovelace)
- **CPU:** Intel i9‑13900K (24 cores / 32 threads)
- **RAM:** 64 GB
- **Virtual Env:** Python 3.10+ virtual environment (UV)

---

## 1  Measure End‑to‑End Duration

Use `/usr/bin/time` for precise wall‑clock, CPU, and memory statistics.

```bash
/usr/bin/time -v python fp8_qwen3vl_abliterated.py 2>&1 | tee fp8_quant.log
```

### Live GPU Monitoring
```bash
nvidia-smi dmon -s pucvmet
```

### Optional Disk / I/O Insight
```bash
iostat -xm 2
```

---

## 2  Add a Per‑Layer Progress Logger

Insert this into **`fp8_qwen3vl_abliterated.py`** before calling `oneshot()`:

```python
import time, os, psutil, torch

def log(msg):
    t = time.strftime("%H:%M:%S")
    print(f"[{t}] {msg}", flush=True)

def layerwise_timing(model):
    log("Beginning per‑layer dry‑run timing…")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            start = time.perf_counter()
            dummy = torch.randn(1, module.in_features, device="cpu", dtype=torch.float32)
            try:
                _ = module(dummy)
            except Exception:
                pass
            dt = time.perf_counter() - start
            log(f"{name:80s}  {dt*1000:8.2f} ms")
    log("Layerwise timing complete.")

# Before oneshot():
from transformers import AutoModelForVision2Seq
model = AutoModelForVision2Seq.from_pretrained(SOURCE_MODEL, trust_remote_code=True)
layerwise_timing(model)
del model
```

This reports per‑layer inference times before quantisation so you can identify heavy modules (I/O vs compute vs serialization).
Use it for profiling; it adds negligible total runtime.

---

## 3  Understand What Controls Duration

FP8 quantisation is *data‑free*—no training—so runtime is dominated by:

| Factor | Description | Optimisation |
|:-------|:-------------|:--------------|
| **Disk I/O** | Reading BF16 weights & writing FP8 safetensors | Store both source & output on NVMe; avoid network drives |
| **CPU serialization** | Tensor packing, safetensor writing | Run in WSL Ubuntu 22.04+ with all cores enabled; minimise background processes |
| **GPU availability** | Occasional CUDA kernel acceleration | Close other GPU jobs; `CUDA_VISIBLE_DEVICES=0` |
| **Python overhead** | Import / graph build | Keep environment lean; warm HF cache |
| **Shard count** | Many tiny shards = more I/O overhead | Keep default or consolidate afterwards |

---

## 4  Environment and Thread Settings (WSL Ubuntu)

```bash
# Thread counts (use physical cores)
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Fast Hugging Face cache on NVMe
export HF_HOME=/mnt/fast_nvme/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME

# Pin GPU
export CUDA_VISIBLE_DEVICES=0
```

Ensure WSL has at least 16 logical processors exposed and the NVIDIA driver with CUDA 12.x support (`nvidia-smi` should report the GPU correctly inside WSL).

---

## 5  Smoke Test First

To validate setup before a long run:

1. Edit the script temporarily:
   ```python
   max_seq_length = 512
   OUTPUT_DIR = "/tmp/fp8_test"
   ```
2. Run once; confirm the small FP8 model loads in Transformers.
3. Revert to your full settings (`max_seq_length=2048`, final output path).

---

## 6  Post‑Conversion Validation and Serving

After successful quantisation:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve /models/qwen3-vl-8b-abliterated-fp8   --trust-remote-code   --max-model-len 2048   --max-num-seqs 1   --gpu-memory-utilization 0.92
```

- **Reduce** `--max-model-len` (1536 → 1024) if OOM.
- **Limit** image size ≈ 512 px max.
- Expect smooth FP8 performance (~2× faster & ~½ VRAM vs BF16).

---

## 7  Summary Checklist

| Step | Command | Verified |
|:-----|:---------|:---------|
| Time whole run | `/usr/bin/time -v python fp8_qwen3vl_abliterated.py` | ☐ |
| Monitor GPU | `nvidia‑smi dmon ‑s pucvmet` | ☐ |
| Layer profiling | `layerwise_timing(model)` | ☐ |
| Tune threads & I/O | `OMP_NUM_THREADS / HF_HOME` | ☐ |
| Smoke test | `max_seq_length = 512` | ☐ |
| Serve via vLLM | `vllm serve …` | ☐ |

---

### Notes for Your System (WSL Ubuntu + RTX 4080 + i9‑13900K)

- FP8 conversion is **I/O and serialization bound**, not compute‑bound; NVMe speed dominates.
- WSL 2 performance is close to native Linux for this workload.
- You can parallelise safetensor writes using `torch.set_num_threads()` if CPU bound.
- Add antivirus exclusions for your model directory to prevent Windows Defender I/O throttling.

---

**Purpose:** gives you an accurate timing baseline and identifies any slow modules or environment bottlenecks during FP8 quantisation under WSL Ubuntu.
