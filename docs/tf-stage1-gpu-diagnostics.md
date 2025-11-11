# Judge Vision Stage 1 GPU Diagnostics (Nov 9 → 10, 2025)

**STATUS: FIXED (Nov 10, 2025) - Architecture Changed**

## Final Solution: Containerize Only TensorFlow Inference

After multiple attempts to run the entire Stage 1 pipeline in a container, we realized the correct approach is much simpler:

**Only the TensorFlow inference (NIMA/MUSIQ) needs to run in the container.**

### New Architecture

1. **Judge Vision runs on the host** with all existing dependencies working normally
2. **When IQA scores are needed**, `aesthetic_models.py` calls a containerized inference service
3. **Container receives**: image path
4. **Container returns**: JSON with scores `{nima_aesthetic, nima_technical, musiq_spaq}`
5. **Host continues** with the rest of Judge Vision processing

### Files Created

- `src/imageworks/apps/judge_vision/tf_inference_service.py` - Minimal inference script that runs inside container
- `src/imageworks/apps/judge_vision/tf_container_wrapper.py` - Host-side wrapper that calls the container
- `src/imageworks/libs/vision/aesthetic_models.py` - Modified to auto-detect GPU mode and call container

### How It Works

When `score_nima()` or `score_musiq()` is called with `use_gpu=True`:
1. If the long-lived TensorFlow service (`judge-tf-iqa`) is reachable, send the image bytes over HTTP (`/infer`).
2. The container loads MUSIQ/NIMA once, runs the request, and replies with JSON `{nima_aesthetic, nima_technical, musiq_spaq}`.
3. The host caches responses per image so subsequent calls reuse the same payload.
4. If the service is unavailable, `tf_container_wrapper` falls back to a one-off `docker run` using the same inference script.
5. After Stage 1 completes, the CLI posts to `/shutdown` so the service exits and releases the GPU before Stage 2 reloads vLLM.

The helper scripts (`scripts/start_tf_iqa_service.sh` / `scripts/stop_tf_iqa_service.sh`) wrap all of this so you can
manually start/stop the container when debugging, but the CLI/GUI will automatically call the shutdown endpoint.

### Benefits

- ✅ No complex dependency management in container
- ✅ Host environment stays clean and working
- ✅ Container only does one thing: TensorFlow inference
- ✅ Fallback to local TensorFlow if container fails
- ✅ Easy to debug - just JSON in/out

---

## Previous Attempts (For Reference)

Multiple issues were encountered trying to run all of Stage 1 in the container:

> **Legacy Note:** The `stage1_container.py` flow described below has been retired in favour of the
> long-lived HTTP service. The details remain here purely for historical context.

### `/home/stewa/code/imageworks/src/imageworks/apps/judge_vision/stage1_container.py`

1. **Removed persistent venv directories**: Deleted `.venv-tf` and `.tf-backend` using `docker run --rm -v "$PWD":/work busybox rm -rf /work/.venv-tf /work/.tf-backend`

2. **Eliminated all pip installs except tomli**: The container already has all needed packages (TensorFlow, PIL, numpy, scipy, etc.). Only install `tomli` (Python 3.10 backport of tomllib)

3. **Set PYTHONPATH correctly**: Export `PYTHONPATH={PROJECT_ROOT}/src:${PYTHONPATH}` to ensure imageworks modules are found

4. **Added GPU verification**: Check TensorFlow GPU detection before running Stage 1

### `/home/stewa/code/imageworks/src/imageworks/apps/judge_vision/competition.py`

Added fallback import for Python 3.10 compatibility:
```python
try:
    import tomllib
except ImportError:
    import tomli as tomllib  # Python 3.10 backport
```

## Verification

After the fix, logs confirm:
- ✅ `GPU devices: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`
- ✅ No PIL import errors
- ✅ No module import errors

---

## Original Diagnostic Information (For Reference)

This note originally captured everything we observed while trying to run the Judge Vision deterministic IQA stage (TensorFlow) on the RTX 4080. It explains the current behaviour, where to find the relevant logs, changes already made, and what we still need to investigate next time.

---

## What We Tried

1. **GPU leasing now self-heals**
   Each run acquires the chat-proxy GPU lease (`logs/judge_vision.log` entries like `GPU lease granted (token=…)`, `Running Stage 1 inside TensorFlow container (GPU)`), and releases it when Stage 1 exits. If a run is interrupted (e.g., GUI sends SIGTERM) and the lease doesn’t return, the CLI will now query `/v1/gpu/status` and auto-force-release any Judge Vision lease older than `JUDGE_VISION_GPU_SELF_LEASE_MAX_AGE` seconds (default 240 s). The chat proxy also enforces a global `GPU_LEASE_STALE_MAX_AGE` (default 900 s) and exposes `/v1/gpu/force_release` for manual recovery.

2. **TensorFlow container launches successfully**
   We start `nvcr.io/nvidia/tensorflow:24.02-tf2-py3` via `docker run --gpus all …`. Logs show the standard TensorFlow banner followed by our progress/two-pass messages.

3. **Added automated diagnostics**
   `src/imageworks/apps/judge_vision/stage1_container.py` now spawns a background thread that logs every ~10 s:
   - `nvidia-smi --query-compute-apps=pid,process_name,used_memory`
   - `docker ps --filter ancestor=nvcr.io/nvidia/tensorflow:24.02-tf2-py3`

   Look for `[tf-stage1][diag] …` lines near the timestamps of interest.

4. **Logged TensorFlow device discovery**
   `imageworks.libs.vision.aesthetic_models._require_tensorflow` now prints either `TensorFlow detected GPU device(s): …` or `TensorFlow GPU device list is empty; using CPU execution`. Every run so far logs the warning (GPU list empty).

5. **Confirmed base image sees the GPU**
   Running the container manually:
   ```bash
   docker run --rm --gpus all nvcr.io/nvidia/tensorflow:24.02-tf2-py3 \
     python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   ```
   outputs `[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`. So the NVIDIA image is CUDA-capable when we don’t layer extra tooling on top.

6. **Diagnosed uv/venv interaction**
   Stage 1 historically created a per-run `.venv-tf` with its own Python/`tensorflow` wheel. That causes two issues:
   - `_imaging` import errors (because the wheel lacks the compiled extensions that live in `/usr/local/lib/python3.10/dist-packages`).
   - GPU disappearance (because the CPU-only `tensorflow` wheel from PyPI shadows NVIDIA’s build).

   We simplified `_container_script` so that `uv sync` uses the container’s stock Python (`UV_PYTHON=python3`, `UV_PYTHON_DOWNLOADS=never`) instead of downloading CPython 3.12. The `.venv-tf` folder still exists (owned by root because it lives inside the container), but uv now layers packages over NVIDIA’s interpreter rather than replacing it.

---

## Key Log Locations

| Path | Purpose |
|------|---------|
| `logs/judge_vision.log` | Main run log. Search for `GPU lease`, `[tf-stage1][diag]`, or `TensorFlow GPU device list` entries. |
| `logs/judge_vision.log` near `nvidia-smi apps` | Shows whether CUDA processes were present while Stage 1 was running. |
| `logs/judge_vision.log` near `TensorFlow GPU device list` | Confirms whether TensorFlow enumerated `/physical_device:GPU:0`. |
| `docker logs imageworks-chat-proxy` | Not directly part of Stage 1 GPU, but useful if lease/restart behaviour needs to be cross‑checked. |

---

## Current Behaviour (00:00:22 UTC run)

```
2025-11-10 00:00:22 [WARNING] imageworks.libs.vision.aesthetic_models:
    TensorFlow GPU device list is empty; using CPU execution.
[tf-stage1][diag] nvidia-smi apps: <no output>
[tf-stage1][diag] docker ps: <container running>
```

Despite pulling the correct container, TensorFlow inside Judge Vision still fails to see the GPU, so all IQA work continues on CPU. The GPU diagnostics confirm that no CUDA context is opened during the run.

---

## Theories / Why This Is Still Failing

1. **Residual `.venv-tf` state**
   Even though we now tell `uv` to use the system Python, the existing `.venv-tf` directory was created earlier (owned by root, with Python 3.12). Because we mount the repo into the container, the venv persists between runs and may still override the container interpreter. We need to wipe `.venv-tf/` (and its lock file) so that future runs let uv create a fresh environment with the container’s interpreter.

2. **uv’s site‑customizations**
   If `.venv-tf` was created with `--system-site-packages`, pip/uv might still be installing wheels that override `/usr/local/lib/python3.10/dist-packages`. Removing the venv and letting uv rebuild it without system packages may help, or we may have to add `PYTHONPATH=/usr/local/lib/python3.10/dist-packages:$PYTHONPATH` back once the environment is clean.

3. **nvidia-container-toolkit inside WSL**
   Manual docker tests show GPU visibility is fine outside our tooling, so we don’t suspect the driver stack. However, it’s worth confirming that `docker info | grep -i runtime` lists `nvidia`, and that our Stage 1 launch continues to pass `--gpus all`.

---

## Next Steps

1. **Reset `.venv-tf` cache**
   - From the host, delete the `.venv-tf` directory under the repo (`sudo rm -rf .venv-tf .tf-backend/bin/uv*` if root ownership blocks removal, or delete via `docker run -v "$PWD":/work busybox rm -rf /work/.venv-tf`).
   - Re-run Stage 1 so uv recreates the environment using the container’s Python 3.10 + CUDA build.

2. **Verify device detection immediately after reset**
   - Check `logs/judge_vision.log` for a line like `TensorFlow detected GPU device(s): /physical_device:GPU:0`.
   - Inspect the `[tf-stage1][diag] nvidia-smi apps:` entries to confirm a PID shows up while IQA is running.

3. **If GPUs still missing**
   - Add logging of `which python` and `python --version` inside `_container_script` to verify the interpreter path.
   - Consider skipping uv entirely (run `python -m pip install -r …` inside the container) to remove any wrapper layers.
   - As a sanity check, `docker run --rm --gpus all nvcr.io/nvidia/tensorflow:24.02-tf2-py3 bash -lc "python - <<'PY' …"` to ensure mounted volumes don’t block device nodes.

4. **Once GPU visible**
   - Re-run IQA only (Stage 1) and observe the runtime vs. prior CPU runs to confirm we’re benefiting from acceleration.

Feel free to pick up from here in the next session; the active components now live in
`src/imageworks/apps/judge_vision/tf_inference_service.py`, `tf_container_wrapper.py`, and
`src/imageworks/libs/vision/aesthetic_models.py` (RGB encoding + HTTP plumbing). The `stage1_container.py`
notes above are retained only for historical reference.***
