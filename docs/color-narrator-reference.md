# Color Narrator Reference

> See [AI Models and Prompting](ai-models-and-prompting.md) for a project-wide view of model selection, experiments, and prompting strategies.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Installation and Setup](#installation-and-setup)
4. [Usage](#usage)
5. [Configuration](#configuration)
6. [API Reference](#api-reference)
7. [Development](#development)
8. [Troubleshooting](#troubleshooting)

## Overview

Color Narrator produces competition-ready descriptions of residual colour found in supposedly monochrome images. It consumes outputs from the Mono workflow, calls a local vision-language model, and writes structured metadata directly into the originating JPEG files (falling back to JSON sidecars only when XMP tooling is unavailable).

### Key Capabilities
- **VLM Integration** – Pluggable OpenAI-compatible backends (LMDeploy + Qwen2.5-VL-7B-AWQ by default, vLLM retained for smaller checkpoints, Triton stub for future work).
- **Mono Workflow Context** – Reads contamination metrics, overlays, and metadata created by the Mono checker.
- **Metadata Authoring** – Persists findings as structured XMP (or sidecar JSON during development).
- **Batch Processing** – Streams large folders, reporting progress and error details.
- **CUDA Acceleration** – Optimised for NVIDIA RTX-class GPUs; falls back to CPU with reduced throughput.

### Workflow
```
Mono analysis → Color Narrator VLM prompts → Metadata serialization
```

## Architecture

Color Narrator lives in `src/imageworks/apps/color_narrator/` and is composed of the following modules:

| Component | Location | Responsibilities |
|-----------|----------|------------------|
| CLI entry point | `cli/main.py` | Typer application exposing `narrate`, `validate`, and helper commands. |
| API service *(future)* | `api/` | Placeholder package reserved for a FastAPI automation layer. |
| VLM client | `core/vlm.py` | Wraps OpenAI-compatible HTTP calls, handles retries, and normalises responses. |
| Data loader | `core/data_loader.py` | Validates image/overlay/JSONL triplets and applies contamination filters. |
| Narrator orchestrator | `core/narrator.py` | Coordinates batches, logging, error handling, and persistence. |
| Metadata manager | `core/metadata.py` | Embeds narration metadata in JPEG XMP (with a JSON fallback when XMP is unavailable). |
| Hybrid mono enhancer | `core/hybrid_mono_enhancer.py` | Experimental enhancer that blends mono stats with fresh VLM descriptions. |

Shared utilities in `src/imageworks/libs/personal_tagger/` provide reusable colour analysis, VLM prompt management, and image helpers for both Color Narrator and the forthcoming Personal Tagger.

## Installation and Setup

### Prerequisites
- Python 3.9+
- CUDA 12.8+ with compatible NVIDIA drivers (recommended for GPU execution)
- OpenAI-compatible VLM server: vLLM `>=0.4` (default) or LMDeploy `>=0.5`
- `uv` for environment management (see `docs/dev-env/ide-setup-wsl-vscode.md`)

### Install Project Dependencies
```bash
uv sync
```

### Launch the VLM Server

#### Option A: LMDeploy (default Qwen2.5-VL-7B-AWQ)
```bash
uv run python scripts/start_lmdeploy_server.py --eager

# Confirm the server is available
curl http://localhost:24001/v1/models
```

The helper script resolves the local AWQ weights under `$IMAGEWORKS_MODEL_ROOT` (default `~/ai-models/weights/Qwen2.5-VL-7B-Instruct-AWQ`) and enables eager mode to stay within a 16 GB GPU budget. Adjust ports or batch sizes inside `scripts/start_lmdeploy_server.py` as needed.

#### Option B: vLLM (legacy Qwen2-VL-2B)
```bash
uv run python scripts/start_vllm_server.py

# Health check for the vLLM OpenAI endpoint
curl http://localhost:8000/v1/models
```

vLLM remains handy for lightweight experimentation or when GPU memory is tight; point it at `Qwen2-VL-2B-Instruct` or other supported safetensors under `$IMAGEWORKS_MODEL_ROOT`.

```bash
# Example: serve an AWQ quantised checkpoint with TurboMind backend on port 24001
uv run lmdeploy serve api_server \
  "$IMAGEWORKS_MODEL_ROOT/Qwen2.5-VL-7B-Instruct-AWQ" \
  --model-name Qwen2.5-VL-7B-AWQ \
  --backend turbomind \
  --server-name 0.0.0.0 \
  --server-port 24001 \
  --vision-max-batch-size 1 \
  --eager-mode

curl http://localhost:24001/v1/models
```

For GPUs with ≥48 GB VRAM you can point either script at larger Qwen2/VL checkpoints; remember to update `pyproject.toml` so the CLI points to the matching model name. The narrator CLI now uses these paths by default—pass `--debug` if you want to fall back to the bundled sample assets.

## Usage

### Narrate Residual Colour
```bash
uv run imageworks-color-narrator narrate \
  --images "${INPUT}/competition_images" \
  --overlays "${INPUT}/overlays" \
  --mono-jsonl ./outputs/results/mono_results.jsonl \
  --summary ./outputs/summaries/narrate_summary.md
```

The `narrate` command:

- Processes only **fail** and **pass_with_query** mono verdicts (pure passes are skipped automatically).
- Embeds narration metadata directly into the JPEG (libxmp), falling back to JSON sidecars if XMP tooling is absent.
- Writes a human-readable `narrate_summary.md` and mirrors key counts in the terminal.
- Accepts `--prompt <id>` (numbered templates) and `--list-prompts` to support quick A/B testing. Prompts that support spatial guidance can be paired with `--regions` to include 3×3 grid hints when available.
- Supports `--dry-run` (skip writes) and `--debug`. When `--debug` is supplied without explicit paths the command automatically uses `tests/shared/sample_production_images` (images and overlays together) and `tests/shared/sample_production_mono_json_output/production_sample.jsonl` for quicker iteration.

### Validate Existing Narrations
```bash
uv run imageworks-color-narrator validate \
  --images ./competition_images \
  --mono-jsonl ./outputs/results/mono_results.jsonl
```

`enhance-mono` has been folded into `narrate`; the legacy command now displays a guidance message.

## Configuration

Runtime defaults live in `pyproject.toml`:

```toml
[tool.imageworks.color_narrator]
vlm_backend = "lmdeploy"
vlm_base_url = "http://localhost:24001/v1"
vlm_model = "Qwen2.5-VL-7B-AWQ"
vlm_timeout = 120
vlm_max_tokens = 300
vlm_temperature = 0.1
vlm_vllm_base_url = "http://localhost:8000/v1"
vlm_vllm_model = "Qwen2-VL-2B-Instruct"
vlm_vllm_api_key = "EMPTY"
vlm_lmdeploy_base_url = "http://localhost:24001/v1"
vlm_lmdeploy_model = "Qwen2.5-VL-7B-AWQ"
vlm_lmdeploy_api_key = "EMPTY"
vlm_lmdeploy_eager = true
vlm_triton_base_url = "http://localhost:9000/v1"
vlm_triton_model = "Qwen2-VL-2B-Instruct"

default_batch_size = 4
min_contamination_level = 0.1
require_overlays = true
max_concurrent_requests = 4

default_images_dir = "/mnt/d/Proper Photos/photos/ccc competition images"
default_overlays_dir = "/mnt/d/Proper Photos/photos/ccc competition images"
default_mono_jsonl = "outputs/results/mono_results.jsonl"

backup_original_files = true
overwrite_existing_metadata = false
metadata_version = "1.0"

chroma_threshold = 5.0
min_region_size = 100
high_chroma_threshold = 15.0

debug_save_intermediate = false
debug_output_dir = "outputs/debug"
log_level = "INFO"
```

- `vlm_backend` accepts `vllm`, `lmdeploy`, or `triton` (stub). Backend-specific overrides follow the pattern `vlm_<backend>_*`.
- Environment overrides still work, e.g. `IMAGEWORKS_COLOR_NARRATOR__VLM_BACKEND=lmdeploy`.

Override settings via CLI flags, environment variables (`IMAGEWORKS_COLOR_NARRATOR__*`), or by editing the configuration block. Any missing path defaults fall back to the mono configuration (`[tool.imageworks.mono]`), so pointing both commands at the same competition import tree requires minimal CLI arguments.

> **Note:** `chroma_threshold`, `min_region_size`, and `high_chroma_threshold` are placeholders for future experiments. The current narrator defers to mono-checker output when deciding what to describe.

## API Reference

Color Narrator exposes a FastAPI service that mirrors the CLI. Example request payload for `/narrate`:

```json
{
  "images": ["/data/competition/IMG_0001.jpg"],
  "overlays": ["/data/competition/IMG_0001_overlay.png"],
  "mono_jsonl": "/data/mono_results.jsonl",
  "options": {
    "min_contamination_level": 0.1,
    "require_overlays": true
  }
}
```

The service relays requests to the VLM client and returns structured narration results, including confidence scores and metadata actions. Refer to `src/imageworks/apps/color_narrator/api/` for route definitions and Pydantic schemas.

## Development

### Local Checks
```bash
uv run ruff check src/imageworks/apps/color_narrator
uv run black src/imageworks/apps/color_narrator
uv run mypy src/imageworks/apps/color_narrator
uv run pytest tests/color_narrator
```

### Prompt Templates
`src/imageworks/libs/personal_tagger/vlm_utils.py` exposes `VLMPromptManager` and `VLMPromptTemplate` for registering custom prompts. Example:

```python
from imageworks.libs.personal_tagger.vlm_utils import VLMPromptManager, VLMPromptTemplate

manager = VLMPromptManager()
manager.register_template(
    VLMPromptTemplate(
        name="custom_analysis",
        template="Analyze this image for: {focus_area}\n\nContext: {context}",
        required_params=["focus_area"],
        optional_params=["context"],
        description="Custom analysis template",
    )
)
```

### Test Data
Integration tests rely on assets under `tests/shared/`. Keep personally identifiable imagery out of the repository and rely on synthetic or licensed examples.

## Troubleshooting

| Symptom | Likely Cause | Suggested Fix |
|---------|--------------|---------------|
| `VLM backend '<name>' is not available` | Backend offline or wrong base URL/model config | Verify with `curl <base_url>/models` and check `vlm_<backend>_*` settings.
| CUDA out-of-memory errors | Batch too large or GPU under-provisioned | Reduce `default_batch_size`, limit concurrent requests, or host on a larger GPU.
| Overlay files reported as missing | File naming mismatch or optional overlays disabled | Ensure overlay suffix matches expectations or set `require_overlays = false`.
| Metadata not written | Insufficient permissions or backup failures | Check backup directory, confirm `backup_original_files` is writable, and run with `--debug` for verbose logs.
| Slow throughput | Large images or vLLM load | Downscale with `target_size`, adjust vLLM `--gpu-memory-utilization`, or serialise processing with `--max-concurrent-requests 1`.

### Health Check Snippet
```python
from imageworks.apps.color_narrator.core.vlm import VLMClient

client = VLMClient(
    base_url="http://localhost:23333/v1",
    model_name="tQwen2.5-VL-7B-Instruct",
    backend="lmdeploy",
)

if client.health_check():
    print(f"✅ {client.backend.value} backend is healthy")
else:
    print(f"❌ {client.backend.value} backend is not responding: {client.last_error}")
```

### Performance Timing Example
```python
from imageworks.apps.color_narrator.core.narrator import ColorNarrator
import time

narrator = ColorNarrator()
start = time.time()
results = narrator.process_all()
elapsed = time.time() - start

print(f"Processed {len(results)} items in {elapsed:.1f}s")
```

For additional architectural context, review `docs/spec/imageworks-colour-narrator-specification.md` and the Mono documentation referenced earlier.
