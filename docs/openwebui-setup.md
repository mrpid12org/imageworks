## OpenWebUI Integration Setup

This guide describes running OpenWebUI alongside the ImageWorks Chat Proxy.

### 1. Prerequisites
* Docker Desktop (or Linux Docker Engine) with NVIDIA Container Toolkit for GPU acceleration.
* Chat proxy running locally (default): `imageworks-chat-proxy` on port `8100`.

### 2. Start OpenWebUI via Docker Compose

```
docker compose -f docker-compose.openwebui.yml up -d
```

Then open: http://localhost:3000

### 3. Configure OpenWebUI
In OpenWebUI settings (Admin or Global Settings) set:
* API Base URL: `http://host.docker.internal:8100/v1` (or `http://localhost:8100/v1` if using `network_mode: host`)
* API Key: leave blank or `EMPTY` if UI requires a token field.

Duplicate prevention when using the Chat Proxy:
- Disable the Ollama provider so models aren’t double-listed from both the proxy and Ollama.
- In `docker-compose.openwebui.yml`, ensure:
	- `USE_OLLAMA_DOCKER=false`
	- No `OLLAMA_BASE_URL` is provided
- If you later re-enable Ollama, consider filtering backends in the proxy or keep only one provider active in OpenWebUI.

### 4. GPU Acceleration
The compose file uses `device_requests` with `capabilities: [gpu]` for NVIDIA. Ensure:
* `nvidia-smi` works on host.
* `docker run --gpus all nvidia/cuda:12.2.0-base nvidia-smi` succeeds before launching OpenWebUI.

If GPUs are not detected inside the container, confirm the NVIDIA Container Toolkit installation and that Docker Desktop WSL integration includes your distro.

### 5. Networking Notes
* `host.docker.internal` works on Docker Desktop. On native Linux you may need to replace with the host IP or use `network_mode: host` (Linux only).
* For external LAN access: adjust compose port mapping (e.g. `0.0.0.0:3000:8080`). Secure with reverse proxy + auth if exposing beyond trusted LAN.

### 6. Environment Variables (Chat Proxy)
| Variable | Purpose | Default |
|----------|---------|---------|
| CHAT_PROXY_HOST | Bind host | 127.0.0.1 |
| CHAT_PROXY_PORT | Bind port | 8100 |
| CHAT_PROXY_ENABLE_METRICS | Enable metrics endpoint | false |
| CHAT_PROXY_LOG_PATH | JSONL log path | logs/chat_proxy.jsonl |
| CHAT_PROXY_MAX_LOG_BYTES | Rotate threshold | 25MB |
| CHAT_PROXY_BACKEND_TIMEOUT_MS | Upstream request timeout | 120000 |
| CHAT_PROXY_STREAM_IDLE_TIMEOUT_MS | Abort idle stream | 60000 |
| CHAT_PROXY_AUTOSTART_ENABLED | Enable autostart | false |
| CHAT_PROXY_AUTOSTART_MAP | JSON/map of model->command | (none) |
| CHAT_PROXY_REQUIRE_TEMPLATE | Enforce template presence | true |
| CHAT_PROXY_MAX_IMAGE_BYTES | Max decoded image bytes | 6000000 |
| CHAT_PROXY_DISABLE_TOOL_NORMALIZATION | Skip tool normalization | false |
| CHAT_PROXY_LOG_PROMPTS | Log raw prompts | false |
| CHAT_PROXY_SCHEMA_VERSION | Schema version tag | 1 |

### 7. Updating OpenWebUI
```
docker compose -f docker-compose.openwebui.yml pull openwebui
docker compose -f docker-compose.openwebui.yml up -d
```

### 8. Future Pinning
Replace `:latest` with a specific version tag once you settle on a stable baseline.

### 9. Troubleshooting
| Symptom | Possible Cause | Resolution |
|---------|----------------|-----------|
| 404 from chat proxy | Wrong API Base URL | Verify `http://host.docker.internal:8100/v1/models` reachable |
| No streaming in UI | SSE blocked or proxy disabled streaming | Ensure `stream=true` is passed (OpenWebUI default) |
| Tool calls not surfaced | Backend emitted legacy function_call | Leave normalization enabled (default) |
| Large image failure 413 | Exceeds `CHAT_PROXY_MAX_IMAGE_BYTES` | Increase env var or reduce image size |
| Autostart never triggers | AUTOSTART disabled | Set `CHAT_PROXY_AUTOSTART_ENABLED=1` & provide map |

### 10. Autostart Map Example
```
export CHAT_PROXY_AUTOSTART_ENABLED=1
export CHAT_PROXY_AUTOSTART_MAP='{"llava-13b-q4_k_m":{"command":["ollama","run","llava:13b-q4_k_m"]}}'
```

### 11. Security Reminder
Auth is not yet implemented (Phase 1). Keep bindings on localhost or implement a reverse proxy with basic auth before wide exposure.
