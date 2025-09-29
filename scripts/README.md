# Scripts Directory

This directory contains utility scripts for development and deployment.

## Files

- `start_lmdeploy_server.py` – launches the default LMDeploy backend (Qwen2.5-VL-7B-AWQ)
- `start_vllm_server.py` – legacy helper for the Qwen2-VL-2B vLLM server

## Usage

```bash
# Start default LMDeploy backend
python scripts/start_lmdeploy_server.py --eager

# Start fallback vLLM backend
python scripts/start_vllm_server.py
```

For production deployments, see the deployment documentation in `docs/`.
