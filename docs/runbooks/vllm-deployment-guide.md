# vLLM Deployment Guide

This guide captures the practical steps and lessons from hosting vision-language models (Qwen2 / Qwen2.5 / LLaVA families) with [vLLM](https://vllm.ai/) for internal tooling (Color Narrator, Personal Tagger, etc.).

## Table of Contents

1. [Environment Requirements](#environment-requirements)
2. [Model Selection](#model-selection)
3. [Server Setup](#server-setup)
4. [Operational Playbook](#operational-playbook)
5. [Troubleshooting](#troubleshooting)
6. [Further Reading](#further-reading)

## Environment Requirements

- Ubuntu 22.04+ or WSL2 with CUDA pass-through
- NVIDIA driver supporting **CUDA 12.8** or later
- Python tooling managed via `uv`
- `vllm` `>=0.4` installed in the project environment (`uv sync` handles this) **or**
  build the dedicated executor image via `Dockerfile.vllm` (used by `docker-compose.chat-proxy.yml`).
- Leave a buffer of GPU memory when running inside Docker. The chat proxy’s
  orchestrator defaults to `CHAT_PROXY_VLLM_GPU_MEMORY_UTILIZATION=0.75`; lower
  it further if you are tight on VRAM (e.g., 16 GB cards running AWQ models),
  and cap the maximum sequence length with `CHAT_PROXY_VLLM_MAX_MODEL_LEN`
  (commonly `8192`) to avoid huge cache allocations during startup.
- Network access for model downloads (unless weights are pre-cached)

## Model Selection

| Model | VRAM Footprint (16 GB GPU) | Quant Support | Status / Notes |
|-------|----------------------------|---------------|----------------|
| `Qwen2-VL-2B-Instruct` | ~11 GB runtime | Native fp16 | ✅ Lightweight fallback |
| `Qwen2.5-VL-7B-Instruct-AWQ` | ~15–16 GB (AWQ marlin) | ✅ AWQ | ✅ Primary (fits 16 GB) |
| `Qwen2-VL-7B-Instruct` (fp16) | >20 GB | n/a | ❌ OOM on 16 GB |
| `LLaVA-1.5-7B` (fp16) | >20 GB | n/a | ❌ OOM (use AWQ) |
| `LLaVA-1.5-7B-AWQ` | ~15–16 GB | ✅ AWQ | ⚠ Vision 500 errors w/ generic template (see Troubleshooting) |
| GGUF vision variants | n/a | n/a | ❌ Not supported by vLLM multimodal yet |

**Recommendation (16 GB GPU):** Prefer `Qwen2.5-VL-7B-Instruct-AWQ` for best quality within memory limits. If unavailable, fall back to `Qwen2-VL-2B-Instruct`.

### Why AWQ?
AWQ (Activation-aware Weight Quantization) retains quality near fp16 while reducing memory. vLLM auto-detects and activates optimized marlin kernels (`awq_marlin`) when serving AWQ quantised Qwen2.5 weights.

## Server Setup

### 1. Prepare the Environment
```bash
uv run pip install --upgrade vllm
uv run python -c "import vllm, torch; print(vllm.__version__)"
```

### 2. Download Model Weights
Ensure your shared weights directory is defined (the dev env guide exports
`IMAGEWORKS_MODEL_ROOT=$HOME/ai-models/weights`). The commands below assume that
variable is available in your shell.

```bash
uv run python -m vllm.entrypoints.openai.pull \
  --model Qwen2-VL-2B-Instruct
```

### 3. Launch the Server (Examples)

> ⚠️ Since Stage 1 shipped, the recommended approach is to run `docker compose -f docker-compose.chat-proxy.yml up -d vllm-executor`
> and let the chat proxy talk to `http://imageworks-vllm:8600/admin/activate`. The direct `vllm serve` commands below remain useful
> for debugging or bespoke deployments but are no longer wired directly into the proxy container.

#### a. Qwen2.5-VL-7B-Instruct-AWQ (16 GB card)
```bash
MODEL_ROOT=${IMAGEWORKS_MODEL_ROOT:-$HOME/ai-models/weights}
MODEL_DIR="$MODEL_ROOT/existing/Qwen2.5-VL-7B-Instruct-AWQ"  # local directory layout
SERVE_NAME=qwen2.5-vl-7b-awq
nohup uv run vllm serve "$MODEL_DIR" \
  --served-model-name "$SERVE_NAME" \
  --host 0.0.0.0 --port 24001 \
  --trust-remote-code \
  --max-model-len 4096 \
  --enforce-eager \
  > vllm_llava_server.log 2>&1 &
```

Key flags:
- `--served-model-name` sets the identifier clients must supply in the `model` field.
- `--enforce-eager` avoids CUDA graph capture quirks on some AWQ kernels (slightly higher latency, better stability during experimentation).

#### b. Qwen2-VL-2B-Instruct (fallback)
```bash
nohup uv run vllm serve "$IMAGEWORKS_MODEL_ROOT/Qwen2-VL-2B-Instruct" \
  --served-model-name qwen2-vl-2b \
  --host 0.0.0.0 --port 24001 \
  --trust-remote-code \
  --max-model-len 4096 \
  > vllm_qwen2_2b.log 2>&1 &
```

#### c. LLaVA (when experimenting)
If using LLaVA 1.5 AWQ you may need to provide an explicit chat template. The start script will auto-select a packaged template for common LLaVA/Vicuna models; you can also pass an explicit path with `--chat-template`.
```bash
nohup uv run vllm serve "$IMAGEWORKS_MODEL_ROOT/Qwen2-VL-2B-Instruct" \
  --served-model-name Qwen2-VL-2B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --trust-remote-code \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.8 \
  --max-num-seqs 16 \
  > vllm_server.log 2>&1 &
```

### 4. Validate the Deployment
```bash
curl http://localhost:8000/v1/models
uv run python - <<'PY'
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
print(client.models.list())
PY
```

## Operational Playbook

- **Process Supervision** – Wrap the launch command with `systemd`, `supervisord`, or `tmux` for durability.
- **Logging** – Tail `vllm_server.log` for request stats; raise `--log-level` to `DEBUG` for investigations.
- **Health Checks** – Reuse `VLMClient.health_check()` (see `core/vlm.py`) or add an HTTP probe hitting `/v1/models`.
- **Scaling Requests** – Tune `--max-num-seqs` and `max_concurrent_requests` (client-side) to balance throughput and latency.
- **Upgrades** – Validate new vLLM releases in a separate environment; compatibility across minor versions is good but not guaranteed.
- **Admin service** – The `imageworks-vllm` container exposes `/admin/activate`, `/admin/deactivate`, and `/admin/state`. The chat proxy calls these endpoints before forwarding any Stage 2 traffic, so keep this container healthy even if you host the inference workers elsewhere.

### Networking from Docker / WSL

When the chat proxy runs inside Docker (bridge network) or WSL2, loopback addresses inside the container do **not** refer to the
host operating system. Configure the registry entry for vLLM (and other OpenAI-compatible backends) with an explicit
`backend_config.host` or `backend_config.base_url` so the proxy can reach the server:

```json
{
  "name": "qwen2.5-vl-7b-instruct_(BF16)",
  "backend": "vllm",
  "backend_config": {
    "host": "host.docker.internal",
    "port": 8000,
    "base_url": "http://host.docker.internal:8000/v1"
  },
  "served_model_id": "qwen2.5-vl-7b-instruct"
}
```

- Use `host.docker.internal` for Windows/macOS Docker Desktop and recent WSL2 builds. For Linux hosts, substitute the actual LAN
  address or the gateway IP advertised by Docker (`ip route | grep default`).
- `backend_config.host` is sufficient for on-host deployments (the proxy will append `/v1`). Specify `base_url` only when the
  backend is exposed behind an alternate path or TLS terminator.
- In the containerized proxy stack, Ollama entries default to
  `http://imageworks-ollama:11434`. Override `IMAGEWORKS_OLLAMA_HOST` if the
  daemon resides outside that service (for example, a host-level install).

## Troubleshooting

| Symptom | Diagnosis | Mitigation |
|---------|-----------|------------|
| `CUDA out of memory` | Model too large or concurrent load too high | Use AWQ variant, lower `--max-num-seqs`, reduce context length, or upgrade GPU. |
| `ValueError: GGUF format is not supported` | Attempting to use GGUF | Use safetensors HF repos (AWQ / fp16). |
| 400 error: "default chat template no longer allowed" | Transformers >=4.44 requires explicit template metadata | Supply `--chat-template` pointing to the correct Jinja template OR upgrade to a model bundle containing `chat_template.json`. |
| 500 on multimodal (LLaVA 1.5 AWQ) but text works | Template + OpenAI vision format mismatch | Try removing custom template, or adjust payload to legacy single string with `<image>` placeholder; prefer Qwen2.5 AWQ for stability. |
| Server exits when shell closes | Process not daemonized | Use `nohup`, `systemd`, `supervisord`, or `tmux`. |
| Slow first request | Kernel warm-up / CUDA graph compilation | Ignore; subsequent requests are faster. |
| `429 Too Many Requests` | Client concurrency saturation | Lower client parallelism or bump server `--max-num-seqs`. |
| Vision requests very slow | Large base64 images | Downscale client-side (e.g. max 1024px) before embedding in data URI. |

### Served Model Names
Historically the Personal Tagger passed explicit model identifiers via `--caption-model`, `--keyword-model`, `--description-model`. These flags are now deprecated in normal workflows.

Preferred path:
```bash
uv run imageworks-personal-tagger run \
  -i ./images \
  --use-registry \
  --caption-role caption \
  --keyword-role keywords \
  --description-role description
```

If you change the served model (e.g. `qwen2.5-vl-7b-awq` → `qwen2.5-vl`) update the layered registry (`configs/model_registry.curated.json` / `.discovered.json`) and re-lock hashes. The merged snapshot (`model_registry.json`) is regenerated automatically. Only use legacy explicit flags for rapid experiments; they will be removed after the migration window.

### Preflight Checks
The Personal Tagger now performs an optional preflight (enabled by default) that validates:
1. `/v1/models` reachable
2. Text-only chat returns 200
3. Vision (1x1 PNG) chat returns 200

Skip with `--skip-preflight` if running against a high-latency remote endpoint.

## Appendix: Unified Registry Glossary (Condensed)
Field | Meaning
----- | -------
`name` | Logical registry key for selection.
`backend` | Serving implementation (vLLM/LMDeploy/etc.).
`served_model_id` | Name advertised by backend (must match `/v1/models`).
`backend_config.model_path` | Filesystem location of model weights.
`roles[]` | Functional purposes (e.g. `caption`, `description`).
`capabilities` | Modalities; vLLM multimodal builds require `vision=true`.
`version_lock` | Hash drift control (`verify --lock`).
`artifacts.aggregate_sha256` | Deterministic snapshot identity.
`model_aliases[]` | Fallback identifiers for discovery/preflight.
`deprecated` | Still usable but excluded from default role resolution.

Use role-based selection in higher-level apps; use explicit `--served-model-name` only for low-level server tuning or experiments.

## Further Reading

- [AI Models and Prompting Guide](../guides/ai-models-and-prompting.md)
- [Color Narrator Reference](color-narrator-reference.md)
- [vLLM Documentation](https://docs.vllm.ai/)
- [Transformers Chat Templates](https://huggingface.co/docs/transformers/main/chat_templating)
