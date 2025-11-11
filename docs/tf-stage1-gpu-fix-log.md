# TensorFlow Stage 1 GPU Fix - Implementation Log

**Date:** November 10, 2025
**Objective:** Fix GPU detection and enable TensorFlow GPU acceleration for Judge Vision Stage 1 (NIMA/MUSIQ models)

---

## Initial Problem Analysis

### Original Issues (from tf-stage1-gpu-diagnostics.md)
1. `.venv-tf` directory persistence causing CPU TensorFlow to shadow GPU version
2. Multiple container orchestration attempts with increasing dependency conflicts
3. Trying to run entire Judge Vision pipeline in container → dependency hell

### Root Cause Discovery
The fundamental mistake was **trying to run too much in the container**. Only the TensorFlow GPU inference needs containerization - everything else (tonal analysis, mono detection, compliance checking) should stay on the host.

---

## Architecture Redesign

### New Approach: Microservice-Style Containerized Inference

**Key Insight:** Separate what needs GPU from what doesn't.

#### What Runs in Container (GPU)
- NIMA aesthetic scoring
- NIMA technical scoring
- MUSIQ quality scoring

#### What Runs on Host (CPU)
- All Judge Vision orchestration
- Tonal metrics calculation
- Monochrome detection
- Contrast/saturation/edge detection
- VLM critique generation

### Implementation Pattern
```
Host (Judge Vision)
  → Detects GPU mode requested
  → Sends image bytes to long-lived TensorFlow IQA service (Docker container)
  → Service returns JSON {nima_aesthetic, nima_technical, musiq_spaq}
  → Host caches results and continues processing
  → After Stage 1, host POSTs /shutdown so the service frees the GPU
```

---

## Changes Made

### 1. Fixed Import Dependency Chain

**Problem:** Container script imported Judge Vision package → triggered `__init__.py` → imported modules needing cv2 (opencv) → crash

**Chain of Doom:**
```
from imageworks.apps.judge_vision.tf_inference_service import run_inference
  ↓
judge_vision/__init__.py executes
  ↓
from .technical_signals import TechnicalSignalExtractor
  ↓
from imageworks.libs.vision import tonal
  ↓
import cv2  ❌ ModuleNotFoundError
```

**Solution A:** Make `vision/__init__.py` graceful with missing cv2
```python
# src/imageworks/libs/vision/__init__.py
try:
    from .mono import check_monochrome, MonoResult
except ImportError:
    # cv2 not available (container environment)
    pass
```

**Solution B:** Run inference script directly, bypass package imports
```python
# Old: Import as module (triggers __init__.py)
python3 -c "from imageworks.apps.judge_vision.tf_inference_service import run_inference"

# New: Run as script (no package initialization)
python3 /path/to/tf_inference_service.py image.jpg true
```

**Files Modified:**
- `src/imageworks/libs/vision/__init__.py` - Added try/except around mono import
- `src/imageworks/apps/judge_vision/tf_container_wrapper.py` - Changed to execute script directly

---

### 2. Fixed Python 3.10 Compatibility

**Problem:** TensorFlow container uses Python 3.10, which doesn't have `tomllib` (added in 3.11)

**Solution:** Add fallback to `tomli` backport
```python
try:
    import tomllib
except ImportError:
    import tomli as tomllib
```

**Files Modified:**
- `src/imageworks/apps/judge_vision/competition.py`
- `src/imageworks/tools/model_downloader/config.py`

---

### 3. Fixed Function Name Mismatches

**Problem:** `tf_inference_service.py` called wrong function names from `aesthetic_models.py`

**Wrong:**
```python
aesthetic_models.nima_aesthetic_score(...)  # Doesn't exist
aesthetic_models.musiq_score(...)  # Doesn't exist
```

**Correct:**
```python
aesthetic_models.score_nima(img_path, flavor="aesthetic", use_gpu=True)
aesthetic_models.score_nima(img_path, flavor="technical", use_gpu=True)
aesthetic_models.score_musiq(img_path, variant="spaq", use_gpu=True)
```

**Files Modified:**
- `src/imageworks/apps/judge_vision/tf_inference_service.py`

---

### 4. Fixed Numpy Version Conflicts

**Problem:** Installing `tensorflow-hub` via pip upgraded numpy to 2.x, breaking TensorFlow 2.15 (compiled with numpy 1.x)

**Error:**
```
numpy.dtype size changed, may indicate binary incompatibility.
Expected 96 from C header, got 88 from PyObject
```

**Solution:** Pin numpy<2.0 during package installation
```bash
pip install -q 'numpy<2.0' tomli tensorflow-hub
```

**Files Modified:**
- `src/imageworks/apps/judge_vision/tf_container_wrapper.py` - Updated install command

---

### 5. Fixed Infinite Recursion

**Problem:** `aesthetic_models.py` detects GPU mode → calls container wrapper → container imports aesthetic_models → detects GPU mode → tries to spawn another container → infinite loop

**Solution:** Add environment variable flag to detect when already inside container
```python
# In container wrapper - set flag
"-e", "JUDGE_VISION_INSIDE_CONTAINER=1"

# In aesthetic_models.py - check flag
inside_container = os.environ.get("JUDGE_VISION_INSIDE_CONTAINER") == "1"
if use_gpu and not inside_container and os.environ.get("JUDGE_VISION_USE_TF_CONTAINER", "1") == "1":
    # Use container
```

**Files Modified:**
- `src/imageworks/apps/judge_vision/tf_container_wrapper.py` - Set flag
- `src/imageworks/libs/vision/aesthetic_models.py` - Check flag in `score_nima()` and `score_musiq()`

---

### 6. Fixed TensorFlow Hub Cache Configuration

**Problem:** Container wasn't finding pre-downloaded MUSIQ model weights, might try to download from internet

**Solution:** Explicitly set TFHUB cache environment variables to point to mounted volume
```python
musiq_cache = "/root/ai-models/weights/judge-iqa/musiq/tfhub-cache"
cmd = [
    ...
    "-e", f"TFHUB_CACHE_DIR={musiq_cache}",
    "-e", f"TFHUB_CACHE={musiq_cache}",
    ...
]
```

**Files Modified:**
- `src/imageworks/apps/judge_vision/tf_container_wrapper.py` - Added TFHUB env vars

---

### 7. Updated CLI and GUI Integration

**Problem:** Old code still had container orchestration logic (run_stage1_in_container, terminate_active_container)

**Solution:** Simplified to always run locally - TensorFlow inference happens automatically in containers when needed

**Files Modified:**
- `src/imageworks/apps/judge_vision/cli/main.py` - Removed old container orchestration
- `src/imageworks/gui/pages/9_⚖️_Judge_Vision.py` - Updated help text

---

### 8. Added long-lived TensorFlow IQA service + management scripts

**Problem:** Spawning a Docker container per image was slow and held root-owned state; we needed a single lightweight
service that could stay up during Stage 1 and expose a simple HTTP API.

**Solution:**
- Added `scripts/start_tf_iqa_service.sh` / `scripts/stop_tf_iqa_service.sh` to launch `judge-tf-iqa` with the project and
  model volumes mounted.
- Exposed `/health`, `/infer`, and `/shutdown` endpoints in `tf_inference_service.py` so the host can post JPEG bytes
  and retrieve JSON scores without importing the rest of Judge Vision.
- Updated `tf_container_wrapper.py` to try the HTTP endpoint first, then fall back to a one-off `docker run` when needed.

**Files Modified:**
- `scripts/start_tf_iqa_service.sh`, `scripts/stop_tf_iqa_service.sh`
- `src/imageworks/apps/judge_vision/tf_inference_service.py`
- `src/imageworks/apps/judge_vision/tf_container_wrapper.py`

---

### 9. Automated service shutdown after Stage 1

**Problem:** Leaving the TensorFlow container running kept ~2 GB of VRAM allocated, preventing vLLM from restarting for Stage 2.

**Solution:**
- Added `/shutdown` endpoint and `shutdown_tf_service()` helper.
- The CLI now calls the helper after every Stage 1 run (two-pass or IQA-only) unless `JUDGE_VISION_TF_AUTO_SHUTDOWN=0`.

**Files Modified:**
- `src/imageworks/apps/judge_vision/tf_inference_service.py`
- `src/imageworks/apps/judge_vision/tf_container_wrapper.py`
- `src/imageworks/apps/judge_vision/cli/main.py`

---

### 10. Normalised MUSIQ inputs to RGB JPEG

**Problem:** MUSIQ expects 3-channel JPEG bytes; grayscale or 16-bit TIFF/PNG inputs caused reshape failures.

**Solution:** Use Pillow to open each image, convert to RGB8, and send the encoded JPEG bytes to TensorFlow. Non-image
sidecars now fail fast without crashing the service.

**Files Modified:**
- `src/imageworks/libs/vision/aesthetic_models.py`

---

## What Works Now ✅

1. **IQA service stays resident:** `judge-tf-iqa` loads MUSIQ/NIMA once and handles all requests over HTTP.
2. **Automatic shutdown:** Stage 1 calls `/shutdown` so the GPU is freed before Stage 2 reloads vLLM.
3. **Package Installation:** Container installs only the missing deps (`numpy<2`, `tensorflow-hub`, `tomli`) and reuses NVIDIA’s TensorFlow build.
4. **No Recursion:** Environment flags prevent the container from re-spawning itself; fallback to local CPU runs still works.
5. **Model Cache Mounted:** TFHub cache and model weights are mounted read-only so no extra downloads occur.
6. **RGB-normalised inputs:** MUSIQ now receives guaranteed RGB8 JPEG bytes, so grayscale/16-bit assets no longer crash the model.

---

## What Still Needs Attention ⚠️

1. **Service health integration** – Surface `/health` status (and restart failures) in the GUI so users see when the IQA service is offline.
2. **Remote service support** – Document and test pointing `JUDGE_VISION_TF_SERVICE_URL` at a remote GPU host for multi-user setups.
3. **Resilience tests** – Add integration tests that exercise the HTTP fallback + auto-shutdown path, including when Docker is missing or the container exits early.

---

## Testing Commands

### Test the IQA Service
```bash
# Start the service
scripts/start_tf_iqa_service.sh

# Health check
curl -s http://127.0.0.1:5105/health | jq

# Run an inference
curl -s -X POST http://127.0.0.1:5105/infer \
  -H 'Content-Type: application/json' \
  -d '{"image_path": "/path/to/image.jpg", "use_gpu": true}' | jq

# Shut it down when finished
curl -s -X POST http://127.0.0.1:5105/shutdown
```

### Fallback one-off container (for debugging)
```bash
TEST_IMG="/path/to/test/image.jpg"
docker run --rm --gpus all \
  -v "$PWD:$PWD" \
  -v "$HOME/ai-models:/root/ai-models" \
  -v "$(dirname "$TEST_IMG"):$(dirname "$TEST_IMG")" \
  -w "$PWD" \
  -e PYTHONPATH=$PWD/src \
  -e IMAGEWORKS_MODEL_ROOT=/root/ai-models \
  -e TFHUB_CACHE_DIR=/root/ai-models/weights/judge-iqa/musiq/tfhub-cache \
  -e TFHUB_CACHE=/root/ai-models/weights/judge-iqa/musiq/tfhub-cache \
  -e TF_CPP_MIN_LOG_LEVEL=2 \
  -e JUDGE_VISION_INSIDE_CONTAINER=1 \
  nvcr.io/nvidia/tensorflow:24.02-tf2-py3 \
  bash -c "pip install -q 'numpy<2.0' tensorflow-hub tomli && \
           python3 $PWD/src/imageworks/apps/judge_vision/tf_inference_service.py run '$TEST_IMG'"
```

### Test from GUI
1. Launch GUI: `./scripts/launch_gui.sh`
2. Navigate to Judge Vision page
3. Select IQA Device: **GPU** (starts/stops the service automatically)
4. Run Stage 1 (or Two-pass)
5. Check logs: `tail -f logs/judge_vision.log`

---

## Environment Variables

### Host Side
- `JUDGE_VISION_TF_SERVICE_URL` – Override the default `http://127.0.0.1:5105` endpoint (useful for remote GPUs).
- `JUDGE_VISION_TF_PORT` / `JUDGE_VISION_TF_CONTAINER` / `JUDGE_VISION_TF_IMAGE` – customise the service name, port, and Docker image.
- `JUDGE_VISION_TF_AUTO_SHUTDOWN=0` – keep the service running after Stage 1 (default `1`, meaning shut down automatically).
- `JUDGE_VISION_USE_TF_SERVICE=0` – skip the HTTP call entirely and fall back to per-request `docker run` (mainly for debugging).
- `JUDGE_VISION_USE_TF_CONTAINER=0` – force CPU TensorFlow even when `--iqa-device gpu` is selected.
- `IMAGEWORKS_MODEL_ROOT` – location of the shared `weights/` tree.

### Container Side (set automatically by the scripts)
- `PYTHONPATH=/workspace/src` – so `imageworks` imports resolve inside the service container.
- `IMAGEWORKS_MODEL_ROOT=/root/ai-models` – matches the mounted host directory.
- `TFHUB_CACHE_DIR` / `TFHUB_CACHE` – point at `/root/ai-models/weights/judge-iqa/musiq/tfhub-cache`.
- `TF_CPP_MIN_LOG_LEVEL=2` – keep TensorFlow quiet.
- `JUDGE_VISION_INSIDE_CONTAINER=1` – stop recursion if `aesthetic_models` executes in-container.

---

## File Summary

### New Files Created
- `src/imageworks/apps/judge_vision/tf_inference_service.py` - Minimal TensorFlow inference HTTP service (runs in container)
- `src/imageworks/apps/judge_vision/tf_container_wrapper.py` - Host wrapper for HTTP calls + docker fallback
- `scripts/start_tf_iqa_service.sh` / `scripts/stop_tf_iqa_service.sh` - Manage the long-running service container

### Files Modified
- `src/imageworks/libs/vision/__init__.py` - Graceful cv2 import failure
- `src/imageworks/libs/vision/aesthetic_models.py` - Container detection, recursion prevention, RGB JPEG normalisation
- `src/imageworks/apps/judge_vision/technical_signals.py` - Import restructuring (previous attempt)
- `src/imageworks/apps/judge_vision/competition.py` - Python 3.10 compatibility
- `src/imageworks/apps/judge_vision/cli/main.py` - Removed old container orchestration + auto shutdown of TF service
- `src/imageworks/gui/pages/9_⚖️_Judge_Vision.py` - Updated help text
- `src/imageworks/tools/model_downloader/config.py` - Python 3.10 compatibility

---

## Next Steps

1. Hook the IQA service health status into the GUI so users can restart it without diving into logs.
2. Document/test remote service URLs (`JUDGE_VISION_TF_SERVICE_URL`) for multi-user GPU hosts.
3. Add automated tests that exercise HTTP inference, docker fallback, and auto-shutdown flows.

---

## Lessons Learned

1. **Don't over-containerize** - Only containerize what absolutely needs the specialized environment
2. **Watch for import side effects** - Python package `__init__.py` files execute on any import
3. **Pin dependencies** - Especially in mixed environments (container + host)
4. **Version matching is critical** - cuDNN, CUDA, TensorFlow versions must align
5. **Test incrementally** - Verify each layer works before adding the next
6. **Environment variables are your friend** - Use them for feature flags and configuration
7. **Recursion prevention** - Always have a base case when code can call itself indirectly

---

## Resources

- **NVIDIA TensorFlow Containers:** https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow
- **TensorFlow GPU Guide:** https://www.tensorflow.org/install/gpu
- **cuDNN Compatibility:** https://developer.nvidia.com/cudnn
- **Original Diagnostic Notes:** `/docs/tf-stage1-gpu-diagnostics.md`
