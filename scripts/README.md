# Scripts Directory

This directory contains utility scripts for development and deployment.

## Files

- `start_vllm_server.py` - VLM server startup script for development
  - Configures and starts the Qwen2-VL-2B-Instruct server
  - Used for local development and testing
  - See docs for production deployment guidelines

## Usage

```bash
# Start VLM server for development
python scripts/start_vllm_server.py
```

For production deployments, see the deployment documentation in `docs/`.
