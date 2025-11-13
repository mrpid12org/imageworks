# Chat Proxy Runbook

This runbook describes the day-to-day operations for the ImageWorks chat proxy service across CLI and GUI entry points.

---
## 1. Prerequisites & Preflight

1. Ensure the deterministic model registry is up to date: `uv run imageworks-models list` should succeed.
2. Confirm the companion containers are running (or ready to start via Compose):
   - `imageworks-vllm` (built from `Dockerfile.vllm`, exposes `http://imageworks-vllm:8600`).
   - `imageworks-ollama` (`ollama/ollama:latest`, exposes `http://imageworks-ollama:11434`).
   - `imageworks-tf-iqa` (Stage 1 service for Judge Vision).
3. Validate configuration overrides by running `uv run python -c "from imageworks.chat_proxy.config import ProxyConfig; print(ProxyConfig.load())"` and checking host/port/log paths. Pay attention to `CHAT_PROXY_VLLM_REMOTE_URL` and `CHAT_PROXY_OLLAMA_BASE_URL`.
4. GUI users should start the proxy **before** launching Streamlit; otherwise the GUI will show backend unavailable banners. When using docker compose, start `chat-proxy` and `vllm-executor` together: `docker compose -f docker-compose.chat-proxy.yml up -d chat-proxy vllm-executor`.

---
## 2. Starting & Stopping the Service

### 2.1 CLI workflow
#### Local dev (bare-metal / uv)
1. Activate environment (if not using `uv` wrappers).
2. Launch service: `uv run imageworks-chat-proxy`.
3. Optional custom host/port/base URL overrides:
   ```bash
   CHAT_PROXY_HOST=0.0.0.0 \
   CHAT_PROXY_PORT=9000 \
   CHAT_PROXY_VLLM_REMOTE_URL=http://127.0.0.1:8600 \
   CHAT_PROXY_OLLAMA_BASE_URL=http://127.0.0.1:11434 \
   uvicorn imageworks.chat_proxy.app:app
   ```
4. Stop with `Ctrl+C`. The proxy automatically relinquishes GPU leases, tells vLLM to unload via the remote admin service, and drops Ollama models via keep-alive requests.

#### Compose deployment (recommended)
1. Build the stack once: `docker compose -f docker-compose.chat-proxy.yml build chat-proxy vllm-executor tf-iqa-service`.
2. Bring services up: `docker compose -f docker-compose.chat-proxy.yml up -d chat-proxy`.
   - This starts `imageworks-chat-proxy`, ensures `imageworks-vllm` (admin API on port 8600) is healthy, and keeps `imageworks-tf-iqa` warm for Stage 1 IQA.
3. Stop or restart: `docker compose -f docker-compose.chat-proxy.yml down` or `... restart chat-proxy`.
4. Inspect logs: `docker logs -f imageworks-chat-proxy` (proxy), `docker logs -f imageworks-vllm` (admin service), `docker logs -f imageworks-ollama` (keep-alive unload status).

### 2.2 GUI workflow
1. Open **Settings → Backends**.
2. Use **Start Chat Proxy** button (executes the same CLI command via supervisor script).
3. Monitor status pill; it polls `/v1/healthz` until the proxy is reachable.
4. Use **Stop Chat Proxy** to terminate gracefully (sends `SIGINT`).

---
## 3. Validating Availability

1. API: `curl http://127.0.0.1:8100/v1/models | jq length` should return non-zero.
2. VLLM admin: `curl http://localhost:8600/health` should return `{"status":"ok"}`; `/admin/state` lists the active logical model when Stage 2 is running.
3. GUI: Dashboard “Active model” card should show either *None* or the active vLLM logical name, with latency sparkline populated.
4. Logs: tail `logs/chat_proxy.jsonl` – new entries should appear for each GUI prompt. Keep an eye on `[ollama-manager]` lines; the proxy now logs when keep-alive unloads succeed or if the GPU lease waits on VRAM.

---
## 4. Managing Backends via Proxy

### 4.1 Activate a vLLM model
- CLI: `uv run imageworks-models activate-model qwen2-7b-instruct`.
- GUI: Models page → “Activate via Proxy” on desired row.
- Behind the scenes the proxy calls `imageworks-vllm`’s admin service (`/admin/activate`) which launches the requested served model on port `24001`.
- Verify: `uv run imageworks-models current-model`, `curl http://localhost:8600/admin/state`, or GUI Settings status chip.

### 4.2 Switch Ollama tag
- Update registry entry `served_model_id` or use `ollama run <tag>` to prewarm.
- The proxy now frees VRAM by issuing a dummy `/api/generate` + `/api/chat` request with `keep_alive=0`; no `/api/stop` call is attempted because the endpoint never existed. Logs show `[ollama-manager] keep_alive` lines when unload completes.

### 4.3 Disable autostart temporarily
- CLI: `CHAT_PROXY_AUTOSTART_ENABLED=0 uv run imageworks-chat-proxy`.
- GUI: Settings → Backends → toggle **Autostart** switch (writes env override to `.env.local`).

---
## 5. Registry Hot Reload & Profile Control

1. Edit `configs/model_registry.curated.json` or use downloader tooling to refresh `model_registry.discovered.json`.
2. Proxy detects file modification and reloads automatically. To force reload, run `uv run python -c "from imageworks.model_loader.registry import load_registry; load_registry(force=True)"`.
3. Deployments: use GUI Settings → Profiles to switch between `production`, `staging`, or `testing`. This rewrites `ProfileManager` state and updates the `/v1/models` filter instantly.

---
## 6. Incident Response

| Issue | Action |
|-------|--------|
| API returns 502/504 | Check `logs/chat_proxy.jsonl` for `err_backend_unavailable`; confirm `imageworks-vllm` is healthy via `/health` and restart affected containers. |
| GUI shows “Template required” | Ensure chat template exists in `src/imageworks/chat_templates`, or set `CHAT_PROXY_REQUIRE_TEMPLATE=0` while triaging. |
| Stale model list | Run `uv run imageworks-models list` to ensure registry loads, then restart proxy. |
| Memory pressure | Reduce `CHAT_PROXY_VLLM_GPU_MEMORY_UTILIZATION` (now enforced inside `imageworks-vllm`) or unload heavy models via `/admin/deactivate`. |
| VRAM not freed before Judge Vision Stage 2 | Tail `imageworks-chat-proxy` logs for `[ollama-manager]` entries; a failing keep-alive unload means Ollama is holding a model—restart `imageworks-ollama` or set `OLLAMA_KEEP_ALIVE=0`. |
| Log file at rotation limit | Increase `CHAT_PROXY_MAX_LOG_BYTES` or enable external log shipping. |

---
## 7. Change Management

1. Capture configuration diffs (`env` overrides, registry updates) in change ticket.
2. Test new backend combination by running `uv run imageworks-chat-proxy` in staging and sending a sample prompt via `curl`.
3. Update GUI documentation if new roles or deployment profiles are added.
4. Archive JSONL logs before redeployments if audit retention is required.

---
## 8. Post-Deployment Checklist

- [ ] `/v1/models` lists expected logical models.
- [ ] GUI Dashboard shows healthy heartbeat and no warning banners.
- [ ] Autostart triggers correctly when first request arrives (check `_staging/autostart/*.log`).
- [ ] Metrics endpoint `/metrics` (if enabled) scrapes without error.
- [ ] Log volume within retention window.
