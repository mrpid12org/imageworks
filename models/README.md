# Model Directory Setup

This directory contains VLM models used by the color narrator system. **Models are not included in git** due to their large size (27GB+ total).

## Required Models

The system supports the following models:

### Qwen2-VL-2B-Instruct (Recommended for development)
- **Size**: ~4.2GB
- **Memory**: Uses ~10.88GB VRAM total
- **Speed**: Sub-1 second inference
- **Download**:
  ```bash
  uv run huggingface-cli download Qwen/Qwen2-VL-2B-Instruct --local-dir models/Qwen2-VL-2B-Instruct
  ```

### Qwen2-VL-7B-Instruct (Higher quality)
- **Size**: ~15GB
- **Memory**: Higher VRAM requirements
- **Download**:
  ```bash
  uv run huggingface-cli download Qwen/Qwen2-VL-7B-Instruct --local-dir models/Qwen2-VL-7B-Instruct
  ```

## Directory Structure

```
models/
├── Qwen2-VL-2B-Instruct/       # Recommended model (not in git)
├── Qwen2-VL-7B-Instruct/       # Larger model (not in git)
└── README.md                    # This file (in git)
```

## vLLM Server Setup

Start the vLLM server with:

```bash
# For 2B model (recommended)
./start_vllm_server.py --model Qwen2-VL-2B-Instruct --port 8000

# For 7B model (if you have enough VRAM)
./start_vllm_server.py --model Qwen2-VL-7B-Instruct --port 8000
```

## Git Exclusion

Models are excluded from git via `.gitignore`:
- `models/` - All model directories
- `*.safetensors`, `*.gguf`, `*.pt`, `*.bin` - Model weight files

This prevents accidentally committing large binary files to the repository.
