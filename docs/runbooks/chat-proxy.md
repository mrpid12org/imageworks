# Chat Proxy Runbook

This runbook describes the day-to-day operations for the ImageWorks chat proxy service across CLI and GUI entry points.

---
## 1. Prerequisites & Preflight

1. Ensure the deterministic model registry is up to date: `uv run imageworks-models list` should succeed.
2. Confirm GPU backends are reachable (vLLM, Ollama) using `nvidia-smi` or `curl http://127.0.0.1:11434/api/tags`.
3. Validate configuration overrides by running `uv run python -c "from imageworks.chat_proxy.config import ProxyConfig; print(ProxyConfig.load())"` and checking host/port/log paths.
4. GUI users should start the proxy **before** launching Streamlit; otherwise the GUI will show backend unavailable banners.

---
## 2. Starting & Stopping the Service

### 2.1 CLI workflow
1. Activate environment (if not using `uv` wrappers).
2. Launch service: `uv run imageworks-chat-proxy`.
3. Optional custom host/port: `CHAT_PROXY_HOST=0.0.0.0 CHAT_PROXY_PORT=9000 uvicorn imageworks.chat_proxy.app:app`.
4. Stop with `Ctrl+C`. For graceful backend shutdown (vLLM single-port) run `uv run imageworks-models activate-model --stop` after proxy exits.

### 2.2 GUI workflow
1. Open **Settings → Backends**.
2. Use **Start Chat Proxy** button (executes the same CLI command via supervisor script).
3. Monitor status pill; it polls `/v1/healthz` until the proxy is reachable.
4. Use **Stop Chat Proxy** to terminate gracefully (sends `SIGINT`).

---
## 3. Validating Availability

1. API: `curl http://127.0.0.1:8100/v1/models | jq length` should return non-zero.
2. GUI: Dashboard “Active model” card should show either *None* or the active vLLM logical name, with latency sparkline populated.
3. Logs: tail `logs/chat_proxy.jsonl` – new entries should appear for each GUI prompt.

---
## 4. Managing Backends via Proxy

### 4.1 Activate a vLLM model
- CLI: `uv run imageworks-models activate-model qwen2-7b-instruct`.
- GUI: Models page → “Activate via Proxy” on desired row.
- Verify: `uv run imageworks-models current-model` or Settings page status chip.

### 4.2 Switch Ollama tag
- Update registry entry `served_model_id` or use `ollama run <tag>` to prewarm.
- Proxy autostart map `ollama:<tag>` ensures `AutostartManager` loads the desired model when requests arrive.

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
| API returns 502/504 | Check `logs/chat_proxy.jsonl` for `err_backend_unavailable`; confirm backend URL and restart service. |
| GUI shows “Template required” | Ensure chat template exists in `src/imageworks/chat_templates`, or set `CHAT_PROXY_REQUIRE_TEMPLATE=0` while triaging. |
| Stale model list | Run `uv run imageworks-models list` to ensure registry loads, then restart proxy. |
| Memory pressure | Reduce `CHAT_PROXY_VLLM_GPU_MEMORY_UTILIZATION` or stop heavy model before launching another. |
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

