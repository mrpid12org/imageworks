## OpenWebUI Integration Setup

Run OpenWebUI alongside the ImageWorks Chat Proxy with docker-compose. The proxy is containerized and auto-starts so OpenWebUI always sees a healthy endpoint at `http://chat-proxy:8100/v1`.

### 1. Prerequisites
* Docker (Linux or Docker Desktop) and NVIDIA Container Toolkit for GPU acceleration (optional).

### 2. Start services

```
docker compose -f docker-compose.openwebui.yml build chat-proxy
docker compose -f docker-compose.openwebui.yml up -d
```

Open: http://localhost:3000

### 3. OpenWebUI configuration (via compose env)
The compose file sets the base URL and disables the Ollama provider by default:
- `OPENAI_API_BASE_URL=http://chat-proxy:8100/v1`
- `OPENAI_API_KEY=EMPTY`
- `ENABLE_OLLAMA_API=false`
- `OLLAMA_BASE_URLS=` and `OLLAMA_API_CONFIGS=` left blank
- `RESET_CONFIG_ON_START=true` ensures env values override stale DB settings

The proxy container bundles `vllm[vision]` so the single-port orchestrator can
launch models internally. GPU access (`gpus: all`) is required; ensure the
NVIDIA Container Toolkit is installed. Weight directories are bind-mounted at
the same absolute paths used on the host (`/home/you/ai-models/weights`) so
registry entries continue to resolve correctly.

Single-port orchestration is enabled by default
(`CHAT_PROXY_VLLM_SINGLE_PORT=1`). Switching models from OpenWebUI stops the
running vLLM process, starts the requested entry, and resumes the conversation
once `/v1/health` reports ready. You can pre-warm or stop models manually with
`uv run imageworks-loader activate-model <logical_name>` or
`uv run imageworks-loader activate-model --stop`. Set
`CHAT_PROXY_VLLM_SINGLE_PORT=0` if you prefer to manage vLLM outside the
container. On 16 GB GPUs, consider lowering
`CHAT_PROXY_VLLM_GPU_MEMORY_UTILIZATION` (e.g., to `0.7`) and overriding
`CHAT_PROXY_VLLM_MAX_MODEL_LEN` (e.g., `8192`) so the orchestrator fits larger
models without long initialization or OOM retries.
You typically don’t need to edit settings inside the UI; the environment config is applied on container start.

### 4. Installed-only model listing
The proxy excludes non-installed entries by default. If it runs in Docker, it must be able to see your HF weights paths. Two options:
1) Mount your HF weights at the same absolute path inside the proxy container, e.g.:
```
services:
	chat-proxy:
		volumes:
			- /home/you/ai-models/weights:/home/you/ai-models/weights:ro
```
2) Or relax filtering by setting `CHAT_PROXY_INCLUDE_NON_INSTALLED=1`.

### 5. GPU Acceleration
Both `chat-proxy` and `openwebui` request GPU access (`gpus: all`). Ensure:
* `nvidia-smi` works on the host
* Docker sees GPUs (`docker run --gpus all nvidia/cuda:12.8.0-base nvidia-smi`)

### 6. Networking notes
* `openwebui` reaches the proxy by service name `chat-proxy`; no host networking required.
* To expose OpenWebUI to your LAN, keep the default port mapping `3000:8080` and secure access as needed.

### 7. Proxy environment quick reference
See `docs/reference/chat-proxy.md` for a full table. Common toggles:
- `CHAT_PROXY_SUPPRESS_DECORATIONS=1` (default)
- `CHAT_PROXY_INCLUDE_NON_INSTALLED=0` (default)
- `CHAT_PROXY_ENABLE_METRICS=0` (optional)
- `CHAT_PROXY_VLLM_SINGLE_PORT=1` (default single active vLLM instance)

### 8. Troubleshooting
| Symptom | Likely Cause | Fix |
|--------|--------------|-----|
| Models list is empty in OpenWebUI | Proxy not running / not healthy | `docker compose ps` and check `chat-proxy` is healthy |
| Only Ollama models appear | Ollama provider still enabled in UI DB | Ensure compose has `ENABLE_OLLAMA_API=false` and `RESET_CONFIG_ON_START=true`; restart container |
| Fewer (HF) models than expected | Proxy can’t see your HF paths | Add a volume mount at the same absolute path or set `CHAT_PROXY_INCLUDE_NON_INSTALLED=1` |
| 404 from proxy | Wrong base URL | Use `http://chat-proxy:8100/v1` inside compose; `http://127.0.0.1:8100/v1` on host |

### 9. Updating OpenWebUI
```
docker compose -f docker-compose.openwebui.yml pull openwebui
docker compose -f docker-compose.openwebui.yml up -d
```

### 10. Future Pinning
Replace `:latest` with a specific version tag once you settle on a stable baseline.

### 11. Autostart Map Example
```
export CHAT_PROXY_AUTOSTART_ENABLED=1
export CHAT_PROXY_AUTOSTART_MAP='{"llava-13b-q4_k_m":{"command":["ollama","run","llava:13b-q4_k_m"]}}'
```

### 12. Security Reminder
Auth is not yet implemented (Phase 1). Keep bindings on localhost or implement a reverse proxy with basic auth before wide exposure.
