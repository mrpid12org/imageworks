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

## 4. Manage vLLM instances
- Activate or stop vLLM models using the loader CLI:
  ```bash
  uv run imageworks-loader activate-model <logical_name>
  uv run imageworks-loader activate-model --stop
  uv run imageworks-loader current-model
  ```
  These commands update the orchestrator state file referenced by
  `CHAT_PROXY_VLLM_STATE_PATH`.【F:src/imageworks/model_loader/cli_sync.py†L202-L270】
- The proxy auto-switches models on incoming requests when `CHAT_PROXY_VLLM_SINGLE_PORT=1`.

## 5. Integrate with OpenWebUI
- Use `docker-compose.openwebui.yml` as a template. Mount model directories into
  the proxy container so installed-only filtering works, or set
  `CHAT_PROXY_INCLUDE_NON_INSTALLED=1` to expose logical entries.【F:docs/reference/chat-proxy.md†L1-L120】
- Configure OpenWebUI with `OPENAI_API_BASE_URL=http://chat-proxy:8100/v1` and
  `OPENAI_API_KEY=EMPTY`.

## 6. Troubleshooting
| Symptom | Checks |
| --- | --- |
| Proxy lists no models | Ensure registry files are mounted/readable and that `CHAT_PROXY_INCLUDE_NON_INSTALLED` matches your deployment. Verify via `uv run imageworks-loader list`.【F:src/imageworks/chat_proxy/models.py†L1-L200】 |
| `CapabilityError` responses | The selected logical model lacks requested features (e.g. vision). Update registry roles or choose a compatible entry.【F:src/imageworks/chat_proxy/profile_manager.py†L1-L200】 |
| Autostart fails for vLLM | Confirm `CHAT_PROXY_AUTOSTART_ENABLED=1` and commands in `CHAT_PROXY_AUTOSTART_MAP`. For orchestrated vLLM models, ensure GPU access inside the container and valid base weights.【F:src/imageworks/chat_proxy/vllm_manager.py†L1-L220】 |
| Vision chats exceed context | Adjust `CHAT_PROXY_VISION_TRUNCATE_HISTORY` or keep defaults to drop history for image requests, avoiding max-token errors.【F:src/imageworks/chat_proxy/forwarder.py†L126-L178】【F:src/imageworks/chat_proxy/config.py†L68-L123】 |

Rotate logs regularly and monitor `/v1/metrics` (when enabled) for latency trends.
