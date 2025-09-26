# vLLM Deployment Guide for Color-Narrator

## Overview
This guide documents the process and learnings from deploying vLLM with Qwen2-VL vision models for the Color-Narrator system. Based on real deployment experience with RTX 4080 16GB and CUDA 12.9.

## Key Learnings & Gotchas

### 1. Memory Constraints (Critical)
**Problem**: Initial attempt with Qwen2-VL-7B-Instruct (13.6GB) failed with CUDA OOM errors.
```bash
# FAILS - 7B model too large for 16GB VRAM
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB
```

**Solution**: Switch to Qwen2-VL-2B-Instruct (4.2GB) which fits comfortably:
- Model weights: 4.15 GiB
- Total GPU usage: ~11 GiB (fits in 16GB with room for overhead)
- Performance: Still excellent for color description tasks

**Memory Rule**: For vision models, allocate ~2.5x model size in VRAM for safety.

### 2. GGUF Quantization Limitation (Major Discovery)
**Problem**: Attempted GGUF quantization (Q6_K) to reduce memory usage.
```bash
# FAILS - GGUF not supported for vision models yet
ValueError: GGUF format is not supported for vision models in vLLM
```

**Status**: GGUF quantization for vision models is not yet supported in vLLM v0.10.2. This is a known limitation.

**Workaround**: Use smaller base models (2B instead of 7B) rather than quantization.

### 3. Server Startup & Stability
**Problem**: Server would shut down unexpectedly or fail to stay running.

**Solutions**:
```bash
# Wrong - server dies when terminal closes
vllm serve ./models/Qwen2-VL-2B-Instruct

# Right - persistent background server
nohup uv run vllm serve ./models/Qwen2-VL-2B-Instruct \
  --served-model-name Qwen2-VL-2B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --trust-remote-code \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.8 \
  > vllm_server.log 2>&1 &
```

**Key Points**:
- Use `nohup` and `&` for background execution
- Must use `uv run` prefix to access the correct Python environment
- Redirect output to log file for debugging

### 4. Configuration Parameters
**Optimal settings for RTX 4080 16GB with 2B model**:
```bash
--gpu-memory-utilization 0.8    # Use 80% of VRAM
--max-model-len 4096            # Sufficient for image+text context
--trust-remote-code             # Required for Qwen2-VL
--host 0.0.0.0                  # Accept connections from any interface
```

**Avoid**:
- `--disable-log-requests` (makes debugging harder)
- `--max-num-seqs 2` (unnecessary with good VRAM)
- Values > 0.9 for `gpu-memory-utilization` (risks OOM)

## Deployment Process

### 1. Install Dependencies
```bash
# Install vLLM with CUDA support
uv add "vllm[cuda]>=0.10.0"

# Verify installation
uv run python -c "import vllm; print('vLLM version:', vllm.__version__)"
```

### 2. Download Model
```bash
# Download 2B model (recommended for 16GB VRAM)
uv run huggingface-cli download Qwen/Qwen2-VL-2B-Instruct \
  --local-dir ./models/Qwen2-VL-2B-Instruct

# Verify download
du -sh ./models/Qwen2-VL-2B-Instruct/  # Should be ~4.2GB
```

### 3. Start Server
```bash
# Start persistent server
cd /path/to/project
nohup uv run vllm serve ./models/Qwen2-VL-2B-Instruct \
  --served-model-name Qwen2-VL-2B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --trust-remote-code \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.8 \
  > vllm_server.log 2>&1 &

echo "Server PID: $!"  # Note the process ID
```

### 4. Verify Deployment
```bash
# Wait for initialization (takes ~60 seconds)
sleep 60

# Test API endpoint
curl -s http://localhost:8000/v1/models | python3 -m json.tool

# Should return:
{
    "object": "list",
    "data": [
        {
            "id": "Qwen2-VL-2B-Instruct",
            "object": "model",
            ...
        }
    ]
}
```

### 5. Update Configuration
Update `pyproject.toml`:
```toml
[tool.imageworks.color_narrator]
vlm_model = "Qwen2-VL-2B-Instruct"  # Changed from 7B to 2B
vlm_base_url = "http://localhost:8000/v1"
```

## Performance Benchmarks

### Model Comparison (RTX 4080 16GB)
| Model | Size | VRAM Usage | Status | Quality |
|-------|------|------------|---------|---------|
| Qwen2-VL-7B-Instruct | 13.6GB | >16GB | ❌ OOM | N/A |
| Qwen2.5-VL-7B-Q6K-GGUF | 7.6GB | N/A | ❌ Not Supported | N/A |
| Qwen2-VL-2B-Instruct | 4.2GB | ~11GB | ✅ Works | Excellent |

### Startup Times
- Model loading: ~0.7 seconds
- CUDA graph compilation: ~3 seconds
- Total initialization: ~60 seconds

### Inference Performance
- Simple color descriptions: ~2-3 seconds
- Complex scene analysis: ~5-7 seconds
- Concurrent requests: Up to 4 (max_concurrent_requests)

## Troubleshooting

### Server Won't Start
```bash
# Check if port is in use
lsof -i :8000

# Check logs
tail -f vllm_server.log

# Kill existing server
pkill -f vllm
```

### CUDA Out of Memory
```bash
# Check GPU usage
nvidia-smi

# Reduce memory utilization
--gpu-memory-utilization 0.7

# Use smaller model
# Switch from 7B to 2B variant
```

### Connection Refused
```bash
# Ensure server is running
ps aux | grep vllm

# Check if API is responding
curl http://localhost:8000/health

# Verify firewall/networking
netstat -tlnp | grep 8000
```

## Integration Testing

### Test Color-Narrator Integration
```bash
# Run with real vLLM server
cd /path/to/project
uv run imageworks-color-narrator narrate \
  -i test_color_narrator/images \
  -o test_color_narrator/overlays \
  -j test_color_narrator/test_mono_results.jsonl \
  --debug

# Validate results
uv run imageworks-color-narrator validate \
  -i test_color_narrator/images \
  -j test_color_narrator/test_mono_results.jsonl
```

Expected output:
```
🎨 Color-Narrator - Narrate command
🤖 VLM: Qwen2-VL-2B-Instruct at http://localhost:8000/v1
📷 Processing: [filename]
🤖 Calling VLM for color description...
📝 VLM Response: [Natural language color description]
💾 Writing metadata to image...
✅ Metadata written successfully
```

## Hardware Requirements

### Minimum
- GPU: 8GB VRAM (for 2B model)
- RAM: 16GB system RAM
- Storage: 10GB free space for models

### Recommended
- GPU: 16GB+ VRAM (RTX 4080/4090, RTX A4000+)
- RAM: 32GB system RAM
- Storage: 50GB+ for multiple models
- CUDA: 12.0+ with compatible drivers

### Tested Configurations
- **RTX 4080 16GB + CUDA 12.9**: ✅ Works perfectly with 2B model
- **RTX 6000 Pro 48GB + CUDA 12.8**: ✅ Should handle 7B+ models

## Production Deployment Notes

### Security
- Change `--host 0.0.0.0` to `--host 127.0.0.1` for localhost-only access
- Consider reverse proxy (nginx) for production
- Monitor GPU temperature and usage

### Monitoring
```bash
# Monitor server logs
tail -f vllm_server.log

# Monitor GPU usage
watch nvidia-smi

# Monitor system resources
htop
```

### Backup Strategy
- Keep multiple model versions in `./models/`
- Backup working vLLM server configuration
- Document model-specific settings

This deployment guide should prevent future deployment pain and help others avoid the same pitfalls we encountered!
