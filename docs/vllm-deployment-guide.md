# vLLM Deployment Guide

This guide captures the practical steps and lessons from hosting Qwen2 vision-language models for Color Narrator using [vLLM](https://vllm.ai/).

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
- `vllm` `>=0.4` installed in the project environment (`uv sync` handles this)
- Network access for model downloads (unless weights are pre-cached)

## Model Selection

| Model | VRAM Footprint (16 GB GPU) | Status |
|-------|----------------------------|--------|
| `Qwen2-VL-2B-Instruct` | ~11 GB total (weights + activations + CUDA graphs) | ✅ Recommended fallback |
| `Qwen2-VL-7B-Instruct` | >20 GB | ❌ Out-of-memory on 16 GB GPUs |
| Quantised GGUF variants | N/A | ❌ Unsupported for vision models in vLLM `0.10.x` |

**Recommendation:** Use `Qwen2-VL-2B-Instruct` when you need a lightweight vLLM backend; otherwise serve the default LMDeploy AWQ model.

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

### 3. Launch the Server
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

## Troubleshooting

| Symptom | Diagnosis | Mitigation |
|---------|-----------|------------|
| `CUDA out of memory` | Model too large or concurrent load too high | Drop batch size, reduce `--max-num-seqs`, or migrate to a larger GPU. |
| `ValueError: GGUF format is not supported` | Attempting to use quantised GGUF weights | Use native Hugging Face safetensors; GGUF VLMs remain unsupported in vLLM. |
| Server exits when shell closes | Process tied to login session | Use `nohup`, a process manager, or run under `systemd`. |
| Slow first request | CUDA graph compilation on cold start | Allow the warm-up request to finish; subsequent calls run at normal speed. |
| `429 Too Many Requests` | Client saturates concurrency | Lower `max_concurrent_requests` in Color Narrator or raise server limits. |

## Further Reading

- [AI Models and Prompting Guide](ai-models-and-prompting.md)
- [Color Narrator Reference](color-narrator-reference.md)
- [vLLM Documentation](https://docs.vllm.ai/)
