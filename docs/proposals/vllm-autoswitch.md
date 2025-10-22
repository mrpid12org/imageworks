# Proposal: Single-Port vLLM Autoswitch Orchestration

## 1. Background & Motivation

OpenWebUI is being used as the day-to-day chat client. With Ollama or LM Studio, switching models is seamless: selecting a new model tears down the old backend and brings the new one up on the same port. Today the ImageWorks stack (chat proxy + loader + vLLM launch scripts) requires manual steps:

- vLLM instances are launched manually (or via the autostart map) but remain running until explicitly stopped.
- Each vLLM registry entry has its own port. Attempting to autostart two different models on a single-GPU system causes conflicts (404 Backend unavailable).
- Switching models often requires editing the registry/autostart map or terminating processes by hand.

Goal: deliver a “single active vLLM model” experience that automatically unloads the currently running model and loads the requested one, reusing the same (configurable) port. This should feel LM Studio-esque while remaining compatible with existing components.

## 2. Scope

### In Scope
- Introduce an orchestrator that manages a single vLLM process:
  - Stops any running vLLM instance.
  - Starts the requested model via our existing launch script (`scripts/start_vllm_server.py`) using a canonical port.
  - Persists active model metadata so the chat proxy knows which backend is currently running.
- Update the chat proxy to trigger the orchestrator whenever a new model is requested and no matching vLLM process is active.
- Add CLI tooling to manually switch the active vLLM model (mirror the orchestrator behaviour).
- Update autostart logic to delegate to the orchestrator instead of shelling directly to `start_vllm_server.py`.
- Registry fields (`served_model_id`, `backend_config.port`) stay authoritative, but only the active model’s port must be bound at any one time.

### Out of Scope
- Concurrent multi-model vLLM serving. (Future: allow disabling the controller to return to the current multi-port behaviour.)
- Managing Ollama processes – Ollama already handles on-demand switching internally.
- Changes to OpenWebUI itself – we continue to expose an OpenAI-compatible endpoint.

## 3. Requirements

### Functional
1. **Single active vLLM**: On a single machine, at most one vLLM instance is running. When the user (or proxy) requests a different vLLM-backed model:
   - The orchestrator detects the change.
   - Gracefully stops the existing vLLM process.
   - Starts the new model on the canonical port (default: `24001`, configurable).
   - Updates internal state with the active model name and PID.
2. **Autostart integration**:
   - `CHAT_PROXY_AUTOSTART_ENABLED=1` still functions, but the map only needs to list logical names. When autostart fires, it calls the orchestrator with the logical name instead of executing the raw command.
   - Autostart does not spawn additional vLLM processes; it always reuses the orchestrator.
3. **Manual control hook**:
   - Add `imageworks-loader activate-model <logical_name>` CLI command. It invokes the orchestrator and returns when the model is live.
   - Provide `--stop` or `activate-model none` to shut down the active vLLM instance.
4. **State tracking**:
   - Maintain an `active_vllm.json` (or similar) under `_staging` (or another configurable path) with details: logical name, served model ID, PID, port, start timestamp.
   - Orchestrator relies on this file to decide whether a model is already active.
5. **Health check**:
   - After starting a new model, wait for `/v1/health` to return 200 before completing the switch. If health fails (e.g., OOM), rollback (no model running) and surface an error to the caller/proxy.
6. **Chat proxy behaviour**:
   - When `/v1/chat/completions` hits a vLLM-backed model:
     - If the active model matches, forward the request immediately.
     - If different, invoke the orchestrator and wait for completion (with timeout) before forwarding.
     - Stream errors gracefully to the client if the orchestrator fails.

### Non-Functional
- **Graceful shutdown**: Allow vLLM some time to terminate. Send SIGTERM, wait, then SIGKILL if necessary.
- **Timeouts & retries**: Configurable start timeout (default 120 seconds) before declaring failure.
- **Locking**: Prevent concurrent orchestrator invocations (e.g., two simultaneous requests for different models) via a file lock / `asyncio.Lock`.
- **Logging**: Record transitions (stop/start, failure reasons) in `logs/chat_proxy.jsonl` and optionally to stdout.
- **Config overrides**:
  - `CHAT_PROXY_VLLM_SINGLE_PORT=1` to enable the orchestrator. Default: enabled for single-port setups.
  - `CHAT_PROXY_VLLM_PORT` for the canonical port (default 24001).
  - `CHAT_PROXY_VLLM_HEALTH_TIMEOUT` etc.
- **Backwards compatibility**: Provide an escape hatch (`CHAT_PROXY_VLLM_SINGLE_PORT=0`) to revert to the existing multi-port/multi-process behaviour (no autostop).

## 4. Design Overview

### Components & Interactions
1. **Orchestrator module** (new): `src/imageworks/chat_proxy/vllm_manager.py`
   - `activate(logical_name: str) -> ActiveState`
   - `deactivate()`
   - Handles process management using `scripts/start_vllm_server.py` (`--registry-name` flag).
   - Reads/writes `active_vllm.json`.
2. **Chat Proxy integration** (`forwarder.py`):
   - Before contacting a vLLM backend, ensure orchestrator active state matches.
   - If orchestrator reports failure, return 424 with a structured error.
3. **Autostart layer** (`autostart.py`):
   - Instead of running commands from `CHAT_PROXY_AUTOSTART_MAP`, call `activate(model_name)` when needed.
   - Map can become optional; default to all vLLM entries call the orchestrator.
4. **CLI** (`imageworks-loader`):
   - `activate-model`, `stop-vllm`, maybe `current-model`.
5. **Registry**:
   - No schema changes. `served_model_id` remains the actual server ID; the orchestrator must pass `entry.served_model_id` to the launch script.
6. **Configuration**:
   - New env vars for orchestrator (single port toggle, port, timeouts, state file path).
7. **Tests**:
   - Unit tests for the orchestrator (mocking subprocess, simulating success/failure).
   - Proxy tests to ensure switching works and errors propagate.
   - CLI tests for activate/stop commands.

### Failure Modes & Handling
- **Launch failure (OOM, missing weights)**: orchestrator returns error -> proxy emits `backend_unavailable` with hint message. Active state becomes “none”.
- **Kill failure**: warn but proceed to start new model (after force kill). If kill fails repeatedly, escalate (error to caller).
- **Concurrent activations**: lock ensures only one activation happens at a time; subsequent requests await the lock.
- **Manual kill outside orchestrator**: health probe fails; orchestrator notices vLLM process gone and treats as “inactive”, re-launches on next request.

## 5. Work Plan

1. **Design & scaffolding**
   - Define `ActiveVllmState` dataclass (logical name, served id, port, pid, started_at).
   - Implement state file read/write utilities with locking.
   - Implement orchestrator start/stop (with health checks & logging).
2. **Proxy integration**
   - Inject orchestrator into `ChatForwarder`.
   - Modify autostart manager to use orchestrator.
3. **CLI commands**
   - `imageworks-loader activate-model <logical_name>`
   - `imageworks-loader deactivate-model`
   - `imageworks-loader current-model`
4. **Config plumbing**
   - Extend `ProxyConfig` with orchestrator settings.
   - Update docker-compose defaults (single port, new env vars).
5. **Docs**
   - Update `docs/runbooks/openwebui-setup.md` with the new behaviour.
   - Mention fallback mode (set `CHAT_PROXY_VLLM_SINGLE_PORT=0` to restore manual multi-port behaviour).
6. **Testing**
   - Unit/integration tests for orchestrator logic (using `pytest` and `anyio` for async).
   - Simulate switching via CLI and proxy tests.
   - Manual QA checklist (switching between models, failure scenarios, fallback).
7. **Rollout**
   - Ensure existing deployments with multiple models concurrently can opt out (default to single-port for local setup; consider leaving off by default for shared servers if necessary).

## 6. Open Questions
- Do we need to support more than one canonical port (e.g., separating text-only vs vision models)? (Initial answer: no; keep it simple.)
- Should we integrate with a process supervisor (systemd) instead of direct subprocess management? (Probably overkill for local/dev use.)
- Can Personal Tagger reuse the orchestrator or does it still need multiple simultaneous models? (If it needs concurrency, gate this behaviour behind an env flag and disable it when the tagger is active.)

## 7. Summary

Implementing a single active vLLM controller will make OpenWebUI behave much closer to LM Studio: selecting a model automatically restarts the backend on a shared port, without manual registry edits or process wrangling. The change is contained: a new orchestrator module, proxy/autostart integration, and a CLI wrapper. With appropriate configuration toggles, we retain flexibility for future multi-GPU setups while dramatically simplifying the default workflow.
