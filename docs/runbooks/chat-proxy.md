# Chat Proxy Runbook

Operate the OpenAI-compatible chat proxy that fronts ImageWorks registry models
for UI clients such as OpenWebUI.

## 1. Configure registry and environment
- Ensure `configs/model_registry.json` is up to date via `uv run imageworks-loader
  list`. Logical entries must include backend configuration and version locks.【F:src/imageworks/model_loader/registry.py†L1-L200】
- Export proxy environment variables or define them in your process manager. Key
  settings: `CHAT_PROXY_HOST`, `CHAT_PROXY_PORT`, `CHAT_PROXY_LOOPBACK_ALIAS`,
  and `CHAT_PROXY_VLLM_SINGLE_PORT` (default `1`).【F:src/imageworks/chat_proxy/app.py†L22-L140】

## 2. Launch the proxy
```bash
uv run imageworks-chat-proxy
```
- Startup registers FastAPI routes, loads the layered registry, and initialises
  the single-port vLLM orchestrator if enabled.【F:src/imageworks/chat_proxy/app.py†L142-L280】
- Logs stream to `CHAT_PROXY_LOG_PATH` (default `logs/chat_proxy.jsonl`); rotate
  by adjusting `CHAT_PROXY_MAX_LOG_BYTES`.【F:src/imageworks/chat_proxy/logging_utils.py†L1-L160】

## 3. Validate endpoints
- Health check: `curl http://127.0.0.1:8100/v1/health`
- Model list: `curl http://127.0.0.1:8100/v1/models | jq`
- Chat completion:
  ```bash
  curl http://127.0.0.1:8100/v1/chat/completions \
    -H 'Authorization: Bearer EMPTY' \
    -H 'Content-Type: application/json' \
    -d '{"model":"qwen2", "messages":[{"role":"user","content":"ping"}]}'
  ```
  Tool and vision payloads are normalised according to `normalization.py` before
  forwarding to the backend.【F:src/imageworks/chat_proxy/normalization.py†L1-L149】

## 4. Coordinate GPU backends (vLLM + Ollama)
- `docker-compose.chat-proxy.yml` now defines two services:
  - `chat-proxy`: the API gateway (exposes :8100, launches vLLM when required)
  - `imageworks-ollama`: GPU-enabled Ollama runtime (:11434)
- The proxy keeps the services in sync. When a request targets a different backend
  than the previous one, it proactively frees resources before forwarding:
  - Switching from Ollama → vLLM triggers an unload of any running GGUF
    checkpoints via `ollama ps`/`api/stop` so the GPU is clear.【F:src/imageworks/chat_proxy/ollama_manager.py†L1-L80】【F:src/imageworks/chat_proxy/forwarder.py†L356-L390】
  - Switching from vLLM → Ollama deactivates the orchestrated vLLM worker before
    checking Ollama health, ensuring the GGUF runner has room to start.【F:src/imageworks/chat_proxy/forwarder.py†L356-L390】
- Ollama-backed registry entries now default to `http://imageworks-ollama:11434`
  via `IMAGEWORKS_OLLAMA_HOST`, so no loopback alias is required. Update legacy
  entries that still target `127.0.0.1` to benefit from the automatic VRAM
  coordination.
- To seed or update GGUF models, exec into the container:
  ```bash
  docker exec -it imageworks-ollama ollama pull mistral:instruct
  docker exec -it imageworks-ollama ollama ps
  ```
  Models live under `ollama-data/` in the repo root, mounted at `/root/.ollama`.
- From the GUI, the Backends tab now exposes a “Restart Chat Proxy” button that wraps
  `docker restart imageworks-chat-proxy`, giving you a one-click way to apply registry
  updates or clear GPU memory without leaving the dashboard.【F:src/imageworks/gui/pages/2_🎯_Models.py†L1878-L1908】

## 5. Manage vLLM instances
- Activate or stop vLLM models using the loader CLI:
  ```bash
  uv run imageworks-loader activate-model <logical_name>
  uv run imageworks-loader activate-model --stop
  uv run imageworks-loader current-model
  ```
  These commands update the orchestrator state file referenced by
  `CHAT_PROXY_VLLM_STATE_PATH`.【F:src/imageworks/model_loader/cli_sync.py†L202-L270】
- The proxy auto-switches models on incoming requests when `CHAT_PROXY_VLLM_SINGLE_PORT=1`.

## 6. Integrate with OpenWebUI
- Use `docker-compose.openwebui.yml` as a template. Mount model directories into
  the proxy container so installed-only filtering works, or set
  `CHAT_PROXY_INCLUDE_NON_INSTALLED=1` to expose logical entries.【F:docs/reference/chat-proxy.md†L1-L120】
- Configure OpenWebUI with `OPENAI_API_BASE_URL=http://chat-proxy:8100/v1` and
  `OPENAI_API_KEY=EMPTY`.

## 7. Troubleshooting
| Symptom | Checks |
| --- | --- |
| Proxy lists no models | Ensure registry files are mounted/readable and that `CHAT_PROXY_INCLUDE_NON_INSTALLED` matches your deployment. Verify via `uv run imageworks-loader list`.【F:src/imageworks/chat_proxy/models.py†L1-L200】 |
| `CapabilityError` responses | The selected logical model lacks requested features (e.g. vision). Update registry roles or choose a compatible entry.【F:src/imageworks/chat_proxy/profile_manager.py†L1-L200】 |
| Autostart fails for vLLM | Confirm `CHAT_PROXY_AUTOSTART_ENABLED=1` and commands in `CHAT_PROXY_AUTOSTART_MAP`. For orchestrated vLLM models, ensure GPU access inside the container and valid base weights.【F:src/imageworks/chat_proxy/vllm_manager.py†L1-L220】 |
| Vision chats exceed context | Adjust `CHAT_PROXY_VISION_TRUNCATE_HISTORY` or keep defaults to drop history for image requests, avoiding max-token errors.【F:src/imageworks/chat_proxy/forwarder.py†L126-L178】【F:src/imageworks/chat_proxy/config.py†L68-L123】 |

Rotate logs regularly and monitor `/v1/metrics` (when enabled) for latency trends.
