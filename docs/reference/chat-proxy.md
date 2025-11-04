# Chat Proxy Operations Guide

The chat proxy exposes a FastAPI layer that normalises requests from GUI clients and external tools into deterministic calls to ImageWorks backends (vLLM, LMDeploy/Ollama via registry). It orchestrates single-port model activation, handles prompt/response sanitation, records telemetry, and enforces deployment profiles.

---
## 1. Capability Summary

| Area | Details |
|------|---------|
| Multi-backend routing | Supports vLLM single-port activation, Ollama tag switching, and generic OpenAI-compatible HTTP relay via `ChatForwarder`. |
| Model registry integration | Reads from layered registry (`model_loader.registry`) with hot reload when curated/discovered layers change; honours profile-based whitelists and testing filters. |
| Autostart orchestration | `AutostartManager` can start/stop services (vLLM, Ollama) on demand using `CHAT_PROXY_AUTOSTART_MAP` definitions with configurable grace periods. |
| Payload normalisation | Enforces template requirements, strips images for text-only models, down-samples oversize base64 payloads, harmonises tool calls, and truncates history for vision/reasoning models. |
| Telemetry & logging | Streams JSONL conversations (`logs/chat_proxy.jsonl`), aggregates latency/cost metrics, and optionally exposes Prometheus-style counters. |
| Deployment profiles | `ProfileManager` + `RoleSelector` enforce named deployment profiles (production, staging, testing) and role-to-model mappings surfaced to the GUI. |
| Health & discovery | `/v1/models` enumerates install state, capabilities, active status, and served identifiers; `/v1/debug/registry` dumps cached registry names to aid troubleshooting. |

---
## 2. Architecture & Components

1. **FastAPI app (`chat_proxy.app`)** – wires configuration, managers, and routers; reloads registry on disk change.
2. **`ChatForwarder`** – central request pipeline: resolves registry entry, ensures backend availability, prepares backend-specific payloads, streams responses, and records metrics/logs.
3. **Managers**
   - `VllmManager` manages single-port vLLM lifecycle using state persisted in `_staging/active_vllm.json`.
   - `OllamaManager` issues load/unload commands to Ollama REST endpoints.
   - `AutostartManager` interprets autostart map and coordinates service activation.
4. **Policy helpers** – `capabilities.supports_vision/reasoning`, `normalization.normalize_response`, and `role_selector.RoleSelector` enforce runtime rules.
5. **Metrics + logging** – `metrics.MetricsAggregator` collects per-request samples; `logging_utils.JsonlLogger` rotates JSONL logs according to `max_log_bytes`.
6. **Configuration** – `ProxyConfig.load()` merges environment overrides for host/port, logging, timeouts, autostart, history truncation, and Ollama/vLLM tunables.

Sequence:
```
Client → FastAPI endpoint (OpenAI schema) → ChatForwarder
       → Registry lookup + policy checks
       → Backend activation (autostart) → HTTP streaming to backend
       → Response normalisation → Client stream & metrics/log capture
```

---
## 3. Interfaces

### 3.1 CLI service launcher
- Entrypoint: `uv run imageworks-chat-proxy` (`chat_proxy.app:main`).
- Flags: rely on environment variables (see §4). FastAPI can also be run via `uvicorn imageworks.chat_proxy.app:app --host 0.0.0.0 --port 8100` for custom hosting.

### 3.2 HTTP API
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/v1/chat/completions` | POST | OpenAI-compatible chat completions (primary entry). |
| `/v1/models` | GET | Lists filtered registry entries with availability flags, served IDs, active status, and capability hints. |
| `/v1/debug/registry` | GET | Raw cache dump of registry names/display names (restricted to debug use). |
| `/v1/healthz` | GET | Basic process health (if mounted via `autostart`). |

### 3.3 GUI integration
- **Settings → Backends**: toggles autostart, inspects registry entries, and surfaces active vLLM state via `/v1/models` & `/v1/autostart/status` helper calls.
- **Dashboard**: displays current active model, proxy uptime, and error counters using metrics aggregated by `MetricsAggregator`.
- **Models Page**: uses `/v1/models` for inventory and surfaces “Activate via proxy” actions that call `imageworks-models activate-model` behind the scenes.

---
## 4. Configuration & Environment

Key environment variables consumed by `ProxyConfig`:
```
CHAT_PROXY_HOST=0.0.0.0
CHAT_PROXY_PORT=8100
CHAT_PROXY_LOG_PATH=/var/log/imageworks/chat_proxy.jsonl
CHAT_PROXY_MAX_LOG_BYTES=50000000
CHAT_PROXY_BACKEND_TIMEOUT_MS=180000
CHAT_PROXY_STREAM_IDLE_TIMEOUT_MS=90000
CHAT_PROXY_AUTOSTART_ENABLED=1
CHAT_PROXY_AUTOSTART_MAP=vllm:qlora-prod,ollama:color-narrator
CHAT_PROXY_REQUIRE_TEMPLATE=1
CHAT_PROXY_MAX_IMAGE_BYTES=8000000
CHAT_PROXY_LOG_PROMPTS=1
CHAT_PROXY_INCLUDE_NON_INSTALLED=0
CHAT_PROXY_LOOPBACK_ALIAS=imageworks-chat
CHAT_PROXY_VLLM_SINGLE_PORT=1
CHAT_PROXY_VLLM_PORT=24001
CHAT_PROXY_VLLM_STATE_PATH=/srv/imageworks/_staging/active_vllm.json
CHAT_PROXY_VLLM_GPU_MEMORY_UTILIZATION=0.85
CHAT_PROXY_VLLM_MAX_MODEL_LEN=8000
CHAT_PROXY_VISION_KEEP_LAST_N_TURNS=1
CHAT_PROXY_REASONING_TRUNCATE_HISTORY=1
CHAT_PROXY_REASONING_KEEP_LAST_N_TURNS=2
CHAT_PROXY_OLLAMA_BASE_URL=http://ollama.internal:11434
```
Autostart map format: comma-separated `<backend>=<service>` definitions referencing `autostart.AutostartManager` adapters (vLLM, Ollama, shell commands).

Other knobs:
- `CHAT_PROXY_ENABLE_METRICS=1` exposes Prometheus at `/metrics`.
- `CHAT_PROXY_SUPPRESS_DECORATIONS=0` preserves backend-specific metadata for debugging.
- `CHAT_PROXY_INCLUDE_NON_INSTALLED=1` reveals registry entries whose artifacts are not yet on disk (useful for planning).

---
## 5. Data & Artifacts

- **Logs**: JSONL transcripts at `logs/chat_proxy.jsonl` (rotates at `max_log_bytes`). Each entry includes request metadata, model, backend latency, token counts, and errors.
- **vLLM state**: `_staging/active_vllm.json` tracks the active logical model, PID, and port for single-port orchestration.
- **Autostart runfiles**: Temporary PID and readiness files under `_staging/autostart` (created on demand).
- **Metrics**: In-memory `MetricsAggregator` retains recent samples; optional exporter pushes to `/metrics`.

---
## 6. Operational Considerations

- **History truncation**: Vision models drop previous turns unless `CHAT_PROXY_VISION_KEEP_LAST_N_TURNS > 0`; reasoning models default to 1 prior turn.
- **Template enforcement**: If `require_template` is enabled, chat requests must resolve to a chat template in `chat_templates/`; missing templates yield `err_template_required`.
- **Image gating**: Payloads with embedded images are size-checked; oversize attachments trigger `err_payload_too_large` (HTTP 413).
- **Role-aware selection**: `RoleSelector` ensures only models flagged for the active deployment profile (production/testing) are exposed via `/v1/models`.
- **Autostart**: When a request targets a vLLM model and no instance is active, the proxy invokes `VllmManager.activate()` and waits up to `vllm_start_timeout_s` before failing with `err_model_start_timeout`.

---
## 7. Troubleshooting Quick Reference

| Symptom | Likely Cause | Mitigation |
|---------|--------------|------------|
| `404 model not found` | Registry cache stale or profile excludes model. | Trigger reload (`touch configs/model_registry.discovered.json`), verify profile, or hit `/v1/debug/registry`.
| `409 capability mismatch` | Client requested vision but model lacks `capabilities.vision`. | Choose a vision-capable logical name or adjust GUI role mapping.
| vLLM never activates | Autostart disabled or incorrect `CHAT_PROXY_AUTOSTART_MAP`. | Enable autostart, verify service command, check `_staging/autostart/*.log`.
| Streams cut abruptly | `stream_idle_timeout_ms` too low for slow backends. | Increase timeout and restart proxy.
| GUI shows “non installed” | File path from registry missing. | Run `imageworks-download normalize-formats --rebuild` or adjust registry entry.

---
## 8. Related Tooling

- `imageworks-models` CLI provides manual registry inspection and vLLM control (see dedicated guide).
- GUI Settings and Dashboard pages call into the proxy for backend health; keep proxy running before launching Streamlit (`uv run imageworks-gui`).
- Integration tests use `CHAT_PROXY_INCLUDE_TEST_MODELS=1` to surface synthetic fixtures.

