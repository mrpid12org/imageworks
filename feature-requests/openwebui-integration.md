# OpenWebUI Integration Feature Request

Status: Requirements Clarified (Phase 1 Implementation NOT STARTED – awaiting final approval).

---
## 1. Headline
Provide a unified, backend-agnostic OpenAI-compatible access layer so OpenWebUI can chat (text, vision, tools) with any model registered in ImageWorks (initially Ollama, later vLLM) without building a custom web UI.

## 2. Free-Form Notes (Requester Input Summary)
- Must reuse OpenWebUI exactly; no bespoke web frontend.
- User wants to select any downloaded model (registry-managed) inside OpenWebUI.
- Start with Ollama-backed models; add vLLM ASAP (immediately if low incremental work).
- Vision-capable conversations where model supports it.
- Tool-use (function calling) passthrough where model supports it.
- Browser-based usage (local network acceptable). ***i will eventually want to access this from outside my local network
- Avoid duplicating capability already in OpenWebUI; leverage its chat, history, tools, streaming.

---
## 3. Options Considered
### Option B – Chat Proxy (Loader-Side Facade)
A new FastAPI endpoint in the existing model loader service that:
1. Lists logical models (from registry) in an OpenAI-compatible `/v1/models` response shape.
	- Clarification: "OpenAI-compatible" here means we mirror the de-facto OpenAI API schema that OpenWebUI (and many generic LLM clients) already speak (e.g. `{ "data": [ {"id": "model-name", "object": "model" } ] }`).
	- This is distinct from "OpenAPI" (the specification framework). We adopt the OpenAI wire format to avoid custom adapters inside OpenWebUI.
2. Accepts chat requests at a unified path (e.g. `/v1/chat/completions` or custom `/v1/chat`) with a `model` field equal to the logical name.
3. Resolves logical name → backend descriptor (endpoint URL + internal served model id) via registry.
4. Forwards the request to the actual backend (vLLM / Ollama) rewriting `model` only if the logical registry name differs from the backend's served model identifier.
	- If the backend already accepts the provided logical name unchanged, no rewrite occurs. The proxy's key value is resolving a human/registry-friendly label to the correct backend base URL and served identifier.
5. Returns backend response (optionally normalizing minor schema differences).
6. (Optional) Streams tokens back (Server-Sent Events or chunked) transparently.

Pros:
- Minimal code surface; quick to deliver.
- Shields OpenWebUI from backend heterogeneity.
- Central spot to enforce version locks, capabilities, future role selection.
- Easy to extend to more backends later.

Cons:
- OpenWebUI only sees the unified proxy model list (can’t directly display extra registry metadata unless we augment response).
- Advanced registry-specific metrics (hash lock, performance) not visible unless added to custom fields.

### Option D – Full OpenWebUI Plug-In / Extension
A custom plugin inside OpenWebUI that:
1. Calls ImageWorks loader API (`/v1/models`, `/v1/select`) to populate dynamic model dropdown with capabilities, status badges (vision, locked, etc.).
2. Intercepts model selection events and sets appropriate backend `base_url` + `model` internally.
3. Optionally adds panels (metrics, probes) and actions (verify, vision probe) via UI.

Pros:
- Rich UI integration (capabilities, performance, lock status visible).
- Potential to expose roles vs models directly in UI.

Cons:
- Higher maintenance (track OpenWebUI internal APIs & upgrade cadence).
- Slower initial delivery.
- Requires learning OpenWebUI extension points (time cost).

### Core Difference
Option B treats OpenWebUI as a generic OpenAI client pointed at one base URL; Option D modifies/extends OpenWebUI to become registry-aware. Functionality overlap exists; difference is depth of UI integration and maintenance cost.

---
## 4. Explicit Requirements (Verbatim Normalized)
1. Use OpenWebUI (no custom UI build).
2. Select any downloaded registry model inside the UI.
3. Support Ollama backend first; add vLLM (preferably immediate if trivial).
4. Vision chat support when model supports vision.
5. Tool/function calling support when model supports it.
6. Works in a browser (local deployment acceptable).
7. Leverage existing OpenWebUI features (history, streaming, tool UI) instead of reimplementing.

---
## 5. Inferred / Implicit Requirements (To Confirm)
- Unified model list must include both Ollama and vLLM variants with consistent naming.
- Need mapping: logical registry name vs backend served model name.
	- Comment noted: existing downloader "display name" + quantization string likely sufficient for initial UX; we will surface quantization explicitly (e.g. suffix or separate metadata field) so the user can distinguish `q4_k_m` vs `fp16`.
- Proxy may need streaming support to fully exploit OpenWebUI UX.
- Tool call schema consistency (OpenAI `tool_calls`) may require normalization.
- Vision payloads (base64 images) must pass through untouched.
- Authentication: none for local Phase 1; future external exposure will introduce API key or session auth (placeholder requirement added to Phase 2/3 roadmap).
- Concurrency: at least N simultaneous chats without queueing (specify N).
- Error semantics: structured JSON with meaningful codes (404 not found, 409 lock violation, 424 backend not ready) vs plain 500.
- Potential desire to later filter models (text-only vs vision) in UI list.
- Logging/auditing of proxied requests (maybe optional toggle).
- Support for future additional backends (Triton, custom GGUF) without UI changes.

---
## 6. Clarifying Questions & Answers (Locked)
All initial clarification questions have been answered. Streaming Day 1 confirmed. Conditional tool schema normalization approved. Auto-launch hook limited to starting existing models (no implicit downloads).

1. Primary Intent: Quick unified endpoint first (Option B). Future deeper metadata (Option D) optional.
**Answer:** optoion B
2. Model Naming UX: Show registry names (tweaked to mirror downloader display name + quant). We'll include quantization token explicitly.
**Answer:** prefer display_name-quant
3. Streaming: (Explanation) Real-time streaming = incremental token responses (Server-Sent Events or chunked) rather than waiting for full completion.
	- Proposal: Enable streaming in Phase 1 (low incremental effort with pass-through) so OpenWebUI retains full UX (progressive tokens, tool call early exposure). If you prefer deferral, specify.
	- Action Needed: Confirm: Do you want streaming enabled Day 1? (Default assumption: YES.)
    **Answer:** YEs, streaming day 1
4. Tool / Function Calling: Requirement: proxy must transparently forward tool invocation fields (`tools`, `tool_choice`, and return `tool_calls`).
	- Enhancement: Add a `capabilities.tools=true` flag or metadata inference at import time (detect models known to support tool/function calling) — Phase 2.
5. Vision Payload: Primary: pasted screenshots & attached JPEGs. Strategy: accept OpenAI-compatible multi-part message entries (`{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}`). Optional optimization: future image resizing/compression pre-forward.
6. Security: Phase 1: none (trusted LAN). Phase 2: introduce simple API key or session-based auth (e.g., header `X-API-Key` or cookie session). Requirement added to roadmap.
7. Concurrency: Target 1–2 concurrent human users. Performance target: optimize for low overhead (<50ms additional proxy latency). TTFT measured but not gated.
8. Failure Behavior: Use OpenAI-style error container: `{ "error": { "type": "backend_unavailable", "code": 424, "message": "...", "hint": "..." } }`.
9. Metrics Visibility: Track TTFT & tokens/sec in proxy instrumentation and optionally expose via a non-standard field (e.g. `extensions.performance`).
	- "Lock status" refers to version hash locking of model artifacts for reproducibility — not critical to show in OpenWebUI Phase 1; omit.
10. Role Abstraction: Defer. Only raw model selection Phase 1.
11. Future Backends: Prepare abstraction for additional backend identifiers (e.g. `triton`) with pluggable resolver; design proxy to treat backend generically.
12. Logging: Phase 1 minimal rolling file or JSONL log with timestamp, user (implicit 'local'), model, token counts, TTFT.
	- Retention: 30 days local rotation. Future (external users): per-user segregation & access control.
    **Answer:** acceptable
13. Model Lifecycle: Add an auto-launch capability: if backend endpoint fails health check, invoke a model start script (e.g. vLLM launcher or Ollama pull) before retry.
	- Phase 1 Scope Decision: Basic attempt (one retry) vs fully managed process supervisor (Phase 2). Proposal: Phase 1 = optional lazy start hook (configurable). Confirm preference.
    **Answer:** invoke a model start scipt should only ever mean trying to spin up a backend and load an existing model  so we would never need 'ollama pull' as this tries to download a model.  we would instead use 'ollama run'.  this should work, as the only models we are exposing to the user to select in the openwebui browser window are the ones that are in the registry and present on disk
14. Timeline: Proceed with Phase 1 ASAP (proxy + streaming + vision + basic logging). Future phases scheduled opportunistically.
15. Tool Schema Normalization:
	- Explanation: Some backends may return tool/function call results in slightly different JSON (naming or nesting). Normalizing ensures OpenWebUI always sees the canonical OpenAI `tool_calls` array with entries `{id, type: "function", function: {name, arguments}}`.
	- Trade-off: Normalization adds a translation layer (maintenance) but prevents UI feature breakage; pass-through is simpler but may cause tools tab to fail for certain models.
	- Proposal: Implement light validation + conditional normalization (only if divergence detected). Requires small schema mapper component.
	- Action Needed: Confirm acceptance of conditional normalization approach.
    **Answer:** go for light normalisation

---
## 7. Initial Draft Plan (PRELIMINARY – will refine after answers)

### Goal
Deliver a unified OpenAI-compatible interface enabling OpenWebUI to access all registry-managed models (Ollama + vLLM) with vision & tool support.

### Why It Matters
Centralizes governance (locking, selection, capabilities) while leveraging existing best-in-class chat UI, accelerating experimentation and reducing bespoke UI maintenance.

### Scope (In)
- Unified model enumeration.
- Logical name → backend resolution.
- Chat (non-stream + optional streaming) proxy.
- Vision payload passthrough.
- Tool call passthrough (no transformation initially unless required).

### Scope (Out – Initial)
- Deep OpenWebUI plugin UI panels (unless Option D chosen post-answers).
- Role abstraction.
- Auth (if confirmed local-only) — subject to answer.
- Advanced metrics dashboards.

### Must-Have Requirements (Phase 1 Final)
1. Unified OpenAI-compatible proxy exposes all registry-backed models (Ollama + vLLM) at `/v1/models`.
2. Model `id` format: `display_name-quant` (quant omitted if none) – consistent with downloader naming semantics.
3. Chat endpoint (`/v1/chat/completions`) supports:
	- Text-only messages
	- Vision messages (OpenAI-compatible `image_url` entries with data URI)
	- Tool calling fields passthrough (`tools`, `tool_choice`) and returns normalized `tool_calls` when needed.
4. Streaming (SSE or chunked) enabled Day 1 for incremental token delivery.
5. Error envelope always OpenAI style with `error.type`, `error.code`, `error.message`, optional `hint`.
6. Auto-launch (optional) attempts to start a stopped backend using a configured command (never performs downloads; `ollama run <model>` allowed, not `ollama pull`).
7. Performance instrumentation: capture `ttft_ms`, `tokens_generated`, `duration_ms`, `tokens_per_second` internally.
8. Logging: JSONL file (30‑day rotation) with core metrics and model identifiers.
9. Local LAN deployment requires no auth; design leaves hook for future API key layer.
10. vLLM support included from the start if present in registry.

### Acceptance Criteria (Phase 1)
| ID | Scenario | Action | Expected Result |
|----|----------|--------|-----------------|
| AC1 | List mixed backends | GET `/v1/models` | JSON includes at least one `backend="ollama"` and one `backend="vllm"` (if vLLM models registered) with `id` matching `display_name-quant`. |
| AC2 | Text chat basic | POST `/v1/chat/completions` (no images) | 200; `choices[0].message.content` non-empty. |
| AC3 | Streaming tokens | POST with streaming enabled | Receive incremental chunks ending with final completion; aggregated content matches non-stream response semantics. |
| AC4 | Vision message | POST with base64 image data URI | 200; assistant response references visual content (heuristic non-empty). |
| AC5 | Tool call passthrough | POST including `tools` & prompt triggering tool | Response contains `choices[0].message.tool_calls[]` (normalized format). |
| AC6 | Unknown model | POST with `model="does-not-exist"` | 404 with error envelope & `error.type="model_not_found"`. |
| AC7 | Backend down w/ auto-launch | Stop backend; request chat | Proxy attempts start; on success returns 200; on failure returns 424 `backend_unavailable` with hint. |
| AC8 | Logging record | Any successful chat | Log line includes logical model, backend model, ttft_ms, tokens_per_second. |
| AC9 | Performance metrics endpoint (if enabled) | GET `/v1/metrics` | Returns rolling aggregates (count ≥1 after first chat). |
| AC10 | Non-vision model image attempt | Send vision payload to text-only model | 409 with `error.type="capability_mismatch"`. |
| AC11 | Parallel streaming load | Run 5 simultaneous streamed chats | All complete; no deadlocks; average added proxy latency <50ms per request (informational metric). |
| AC12 | Oversize image rejection (if limit enabled) | Image > configured max bytes | 413 with `error.type="payload_too_large"` (or test skipped if limit disabled). |
| AC13 | Autostart single spawn | Two concurrent chats to stopped model | Exactly one autostart attempt logged; both eventually succeed or one fails with backend_unavailable if startup fails. |
| AC14 | Disable normalization flag | Set env `CHAT_PROXY_DISABLE_TOOL_NORMALIZATION=true` and trigger legacy `function_call` | Response preserves original `function_call` (no synthesized tool_calls array). |
| AC15 | Backend counters metrics | GET `/v1/metrics` after mixed backend usage | JSON includes per-backend counters (e.g., `requests_by_backend.ollama >=1`). |
| AC16 | Model start timeout distinct | Force autostart that never becomes healthy | 424 with `error.type="model_start_timeout"`. |

### Design Sketch (Phase 1 Final – Option B)
- Endpoints (Phase 1):
	- `GET /v1/models` (OpenAI-compatible) – returns logical (display) model list with quant + capability tags (extensions.allowed_modalities, extensions.quantization).
	- `POST /v1/chat/completions` – Accepts standard OpenAI payload; resolves `model` logical name; rewrites if necessary; forwards.
	- `GET /v1/health` – Simple readiness (proxy itself OK + counts of reachable backends).
	- (Optional) `GET /v1/metrics` – Aggregate recent TTFT / throughput (non-standard; JSON).
- Streaming: SSE / chunked pass-through implemented Day 1 (unless user objects).
- Error Mapping: Use OpenAI style `error` envelope with added `hint` field.
- Auto-Launch Hook: Optional pluggable command executed on first resolution failure (config file mapping model/backend → start script).
- Logging: Append JSON lines: timestamp, user, logical_model, backend_model, tokens_in/out, ttft_ms, duration_ms, vision=true/false, tool_calls=n.

### Task Breakdown (Phase 1 Execution Plan)
| Task | Size (est) | Notes |
|------|------------|-------|
| Requirements refinement after answers | XS | Update doc |
| Proxy model list endpoint | XS | Simple transform |
| Logical resolution + forward (non-stream) | S | Core logic |
| Vision passthrough test harness | S | Local base64 image test |
| Tool call passthrough validation | S/M | Depends on available model |
| Streaming support (SSE) | M | Implement Day 1 (assumed required) |
| Auto-launch hook (basic) | S | Optional spawn + retry once |
| Tool schema validator/normalizer | S/M | Conditional normalization layer |
| Quantization display enrichment | XS | Add quant token to model id |
| Performance instrumentation | S | Measure ttft & throughput |
| Error contract + mapping | S | Standard structure |
| Documentation update | XS | Usage guide |
| (Optional) Logging / audit layer | S | If required |

### Risks (Updated)
- Backend discrepancy in tool call format → transformation complexity.
- Large image payload performance / memory usage.
- Future plugin pivot causing rework if we start with proxy only.

### Mitigations
- Keep code modular (separate resolver + forwarder).
- Add capability flags in model list early to avoid later schema break.
- Provide extension field for future plugin consumption.

### Savepoint
Design locked pending your review of this refined spec. No implementation begun.

---
## 10. Phase Roadmap

| Phase | Focus | Key Additions | Out-of-Scope |
|-------|-------|---------------|--------------|
| 1 | Core proxy | Models list, chat (text/vision/tools), streaming, logging, optional auto-launch | Auth, advanced metrics UI, roles |
| 2 | Hardening & Auth | API key/session auth, per-user log segregation, tool capability inference, metrics endpoint stabilization | Plugin UI, external multi-tenant features |
| 3 | Enhanced UX / Optional Plugin | OpenWebUI plugin (registry metadata, performance surfacing), role-based selection | Federation, multi-region |
| 4 | Advanced Ops | Process supervision integration, percentile metrics, autoscaling hooks | Full MLOps orchestration |

---
## 11. Open Decisions / Confirmations
| Code | Topic | Current Proposal | Needs Approval? |
|------|-------|------------------|-----------------|
| OD1 | Streaming transport | SSE (text/event-stream) pass-through | If alternative (chunked) preferred, say so |
**Answer:** streaming, SSE
| OD2 | Auto-launch retries | Single attempt + backoff (e.g. 5s) then fail | Confirm acceptable |
**Answer:**acceptable
| OD3 | Metrics endpoint | Optional `/v1/metrics` behind config flag | Confirm include in Phase 1 |
**Answer:** acceptable
| OD4 | Log location | `logs/chat_proxy.log` rotating (size or time) | Confirm path & rotation strategy |
**Answer:** accceptable
| OD5 | Quant display | `id=display_name-quant`, also expose `extensions.quantization` | Confirm naming style |
**Answer:** acceptable

Please annotate with **Approve** / alternative suggestions before we proceed.

---
## 8. Next Steps
1. Provide inline **Answer:** sections to questions (Section 6).
2. We refine plan → finalize spec.
3. Choose path (Proxy vs Plugin or phased approach).
4. Implement Phase 1 tasks.

---
## 9. Appendix (If Needed Later)
Will add: response schema examples, error format contract, migration considerations.

---
*End of Draft. Edit below with answers before we proceed.*

---
## 12. Detailed Implementation Plan (Final Pre-Coding Blueprint)

This section freezes the concrete implementation approach for Phase 1 (proxy) and the agreed downloader enhancements so work can resume even if chat context is lost.

### 12.1 Module & File Layout

```
src/imageworks/chat_proxy/
  __init__.py
  app.py                 # FastAPI app factory & route registration
  config.py              # Environment/config parsing (enable_metrics, log paths, autostart)
  models.py              # Pydantic request/response schemas (OpenAI-compatible + extensions)
  resolver.py            # Logical model → backend resolution (registry lookup + health)
  forwarder.py           # Request adaptation & streaming/non-stream forwarding
  normalization.py       # Tool call normalization & response shape adjustments
  metrics.py             # Rolling metrics aggregator (TTFT, throughput)
  logging_utils.py       # Chat logging (JSONL) + rotation (size or time-based)
  autostart.py           # Optional backend startup logic (ollama run, vLLM script)
  errors.py              # Central error classes → HTTP + OpenAI error envelope
tests/chat_proxy/
  test_models_endpoint.py
  test_chat_basic.py
  test_chat_streaming.py
  test_tool_normalization.py
  test_vision_guardrails.py
  test_error_mapping.py
```

CLI entrypoint: add `imageworks-chat-proxy = imageworks.chat_proxy.app:main` in `pyproject.toml`.

### 12.2 Configuration Inputs
| Variable | Purpose | Default |
|----------|---------|---------|
| CHAT_PROXY_HOST | Bind host | 127.0.0.1 (must explicitly set to 0.0.0.0 for LAN) |
| CHAT_PROXY_PORT | Bind port | 8100 |
| CHAT_PROXY_ENABLE_METRICS | Enable `/v1/metrics` | false |
| CHAT_PROXY_LOG_PATH | JSONL log path | logs/chat_proxy.jsonl |
| CHAT_PROXY_MAX_LOG_BYTES | Rotate threshold | 25_000_000 |
| CHAT_PROXY_BACKEND_TIMEOUT_MS | Per-request timeout | 120000 |
| CHAT_PROXY_STREAM_IDLE_TIMEOUT_MS | Abort stream if no data | 60000 |
| CHAT_PROXY_AUTOSTART_ENABLED | Allow autostart | false |
| CHAT_PROXY_AUTOSTART_MAP | JSON or path to JSON mapping model→start command | (none) |
| CHAT_PROXY_REQUIRE_TEMPLATE | Enforce chat template presence | true |
| CHAT_PROXY_MAX_IMAGE_BYTES | Max accepted base64 image decoded size | 6_000_000 |
| CHAT_PROXY_DISABLE_TOOL_NORMALIZATION | Skip normalization if true | false |
| CHAT_PROXY_LOG_PROMPTS | Log user/assistant text content | false (only metadata) |
| CHAT_PROXY_SCHEMA_VERSION | Internal schema version tag | 1 |

### 12.3 `/v1/models` Endpoint Logic
1. Load registry (layered logic already present).
2. Include all entries whose artifacts exist on disk (or are resolvable via backend listing) regardless of backend immediate health; do NOT prune for current availability (autostart may recover). No expensive health checks here.
3. Build `id = display_name + ('-' + quantization if quantization)`.
4. Include `object="model"`, and under `extensions`:
	- `backend`: (ollama|vllm|lmdeploy|unassigned?)
	- `quantization`: quant or null
	- `modalities`: ["text"], ["text","vision"] if capability indicates vision
	- `has_chat_template`: metadata flag if present
	- `templates`: array of detected template relative paths (may be >1)
	- `primary_template`: first detected template (heuristic; not a guarantee of backend usage)
	- `schema_version`: mirrors `CHAT_PROXY_SCHEMA_VERSION`
5. Return `{ "object": "list", "data": [ ... ] }`.

### 12.4 `/v1/chat/completions` Request Handling Flow
1. Parse payload into Pydantic `ChatCompletionRequest` (mirror OpenAI: model, messages, tools, tool_choice, temperature, stream, etc.). Unknown fields preserved in `extra` for pass-through.
2. Validate logical model id exists.
3. Capability checks:
	- If any message contains image part and model lacks `vision` capability → raise capability_mismatch (409).
	- If `CHAT_PROXY_REQUIRE_TEMPLATE` true and registry metadata `has_chat_template` false → error type `template_required` 409 with hint (suggest redownload with optional templates or supply manual `--chat-template`).
4. Resolve backend target (resolver): returns backend type, base_url, served_model_id, autostart_command (optional).
5. If backend unreachable (connection refused / timeout):
	- If autostart enabled for this model and not already starting (per-model async lock), invoke autostart command (single attempt) then wait up to 5s and re-probe.
	- If still unreachable after grace → 424 `model_start_timeout` if we initiated autostart; else 424 `backend_unavailable` (include hint explaining autostart may be disabled or misconfigured).
6. Adapt request:
	- Rewrite `model` to backend served id if different.
	- Preserve messages order, do not inject system wrapper (backend template handles formatting).
	- Leave tools/tool_choice untouched.
7. Forward:
	- Streaming: open async client (httpx) with `stream=True`, re-chunk SSE tokens to client unchanged except optional post-hook to accumulate metrics.
	- Non-stream: single request/response.
8. Capture metrics:
	- TTFT: timestamp at first streamed delta (or whole response arrival).
	- Token counts: parse `usage` if backend supplies; if not available in streaming, approximate via counting whitespace-delimited tokens (fallback, flagged `estimated=true`).
	- Increment per-backend request counters (total + streaming vs non-stream) in memory.
9. Tool normalization (post):
	- Inspect `choices[].message`: If legacy `function_call` present, wrap into `tool_calls` array with synthetic id (`call_<n>`). Remove deprecated field from outer message unless passthrough is explicitly required.
	- Ensure each tool call has `id`, `type="function"`, `function={name, arguments}` (arguments JSON string – ensure string type).
10. Error mapping: convert any backend HTTP errors or exceptions into standardized envelope.
11. Log JSON line.
12. Return final body (or stream) to caller.

### 12.5 Streaming Implementation Details
Transport: SSE (`text/event-stream`). Idle timeout enforced by `CHAT_PROXY_STREAM_IDLE_TIMEOUT_MS` (abort + emit timeout error if no upstream data). Downstream client disconnect triggers immediate upstream cancellation.
Algorithm:
1. Initiate upstream request with `stream=True`.
2. For each upstream SSE `data:` line:
	- Pass through unmodified for latency.
	- On first non-empty delta: record TTFT.
	- Append content fragments to buffer for final metrics.
3. Intercept `[DONE]` sentinel → finalize metrics, emit synthetic internal log event.
Fallback: If upstream is chunked JSON (Ollama style), adapt to SSE by wrapping each chunk as `data: {json}\n\n`.

### 12.6 Metrics Aggregation
In-memory ring buffer (size N=500 recent requests) + counters:
Stored per entry: model, backend, ttft_ms, tokens_out, duration_ms, tokens_per_second, timestamp, stream(bool), estimated_counts(bool).
Per-backend counters: total_requests, streaming_requests.
`/v1/metrics` (if enabled):
```
{
	"uptime_seconds": <float>,
	"rolling": {
		"count": X,
		"avg_ttft_ms": ..., "p95_ttft_ms": ..., "avg_tokens_per_second": ...
	},
	"requests_by_backend": {"ollama": {...}, "vllm": {...}},
	"schema_version": 1
}
```
No persistence Phase 1 (logs cover historical). All approximations flagged via `estimated_counts` ratio if any.

### 12.7 Logging Format (One Line per Completion / Stream)
```
{
  "ts": "2025-10-03T12:34:56.789Z",
  "model_logical": "llava-13b-q4_k_m",
  "backend": "vllm",
  "model_backend_id": "Qwen2-VL-2B-Instruct",
  "stream": true,
  "vision": true,
  "tool_calls": 1,
  "ttft_ms": 842,
  "tokens_out": 215,
  "duration_ms": 5312,
  "tokens_per_second": 40.5,
	"estimated_counts": false,
	"backend_request_index": 42,
	"streaming": true,
  "status": 200,
  "error_type": null
}
```
Rotation: when file exceeds `CHAT_PROXY_MAX_LOG_BYTES` rename with timestamp suffix, start new file, prune >30 days old.

### 12.8 Error Envelope Contract
```
HTTP 4xx/5xx
{
  "error": {
	"type": "model_not_found|backend_unavailable|model_start_timeout|capability_mismatch|template_required|payload_too_large|timeout|upstream_error",
	 "code": 404/409/424/504/502,...,
	 "message": "Human readable",
	 "hint": "Actionable suggestion (optional)",
	 "details": {"backend_status": "..."} (optional)
  }
}
```

### 12.9 Tool Call Normalization Logic (Definitive)
Input variants handled:
1. Legacy single `function_call` object at `message.function_call` → wrap into `tool_calls=[{id, type:"function", function:{name, arguments}}]`.
2. `tool_calls` present but missing `id` → synthesize deterministic id: `call_<index>`.
3. `arguments` as dict → `json.dumps` to string.
4. Non-function tool types (future) pass-through if already conforming.
5. Remove now-normalized `function_call` field to avoid ambiguity.
Guarantee: Output always has `message.tool_calls` array or none; never both `function_call` and `tool_calls`.
If `CHAT_PROXY_DISABLE_TOOL_NORMALIZATION=true` then logic is bypassed and original payload (including `function_call`) is forwarded (AC14).

### 12.10 Autostart Strategy
Config map example (JSON):
```
{
  "llava-13b-q4_k_m": {
	 "backend": "vllm",
	 "command": ["python", "scripts/start_vllm_server.py", "--model", "/path/to/weights", "--served-model-name", "llava-13b-q4_k_m"]
  },
  "phi3-mini": {"backend": "ollama", "command": ["ollama", "run", "phi3:mini"]}
}
```
Execution:
1. On first resolution failure (connection refused), look up logical id in map.
2. Spawn command (subprocess, detached) with stdout/stderr redirected to `logs/autostart/<model>.log`.
3. Wait 5s → re-probe. If success, continue; else raise `backend_unavailable`.
No retries beyond 1 attempt Phase 1.

### 12.11 Security Considerations (Phase 1)
- Bind to 127.0.0.1 by default; user must opt-in to broader exposure by setting `CHAT_PROXY_HOST=0.0.0.0` (log a warning recommending auth when doing so even though Phase 1 lacks auth layer).
- Clearly document absence of auth.
- Add warning log if host is public (not RFC1918) and `CHAT_PROXY_REQUIRE_TEMPLATE=false` (indicates permissive state).

### 12.12 Dependency Notes
Reuse existing `httpx` if present (otherwise add). Avoid heavy new deps. Streaming handled via `httpx.AsyncClient.stream`.

### 12.13 Testing Strategy (Phase 1)
Unit tests (pytest):
1. Models list transformation uses fixture registry entry with/without quantization.
2. Tool normalization cases: legacy function_call, missing id, dict arguments.
3. Vision rejection for text-only model (image part triggers 409).
4. Error mapping: simulate backend 404, connection error → maps to backend_unavailable.
5. Streaming: mock upstream SSE sequence & verify pass-through and metric capture.
6. Autostart: mock subprocess + failing first probe + succeeding second.
7. Concurrency: simulate 5 parallel streaming requests (ensure no shared-state corruption).
8. Oversize image: supply payload > limit (expect 413) if limit enabled.
9. Disable normalization flag: ensure legacy `function_call` passes unchanged.
10. Distinct error on startup timeout vs plain backend unreachable.

### 12.14 Future Hooks (Not Implemented Now)
- Auth middleware insertion point after request parsing.
- Rate limiting bucket keyed by IP/API key.
- Template hash enforcement (store & verify hash each request if stricter reproducibility desired).

---
## 13. Downloader Enhancements Plan (Chat Template Reliability)

Objective: Ensure external chat template artefacts needed for serving are consistently downloaded *without* requiring `--include-optional` for common cases, and improve detection fidelity.

### 13.1 Current Gaps Recap
- Sidecar templates (chat_template.json / *.jinja) categorized as optional; skipped unless `--include-optional`.
- Detection only scans top-level directory.
- No hashing of template file(s).
- Non-standard names (e.g., `prompt_format.j2`) may be missed.

### 13.2 Enhancements (Phase 1 Adjacent / Low Risk)
1. Promote Recognized Templates: After `_partition_files`, scan optional lists for filenames matching regex:
	- `(?i)(chat[_-]?template\.(json|jinja|j2))`
	- Any file ending `.jinja` or `.j2`
	- Files containing `chat_template` substring
	Move matches (<2MB size) to required_files.
2. Recursive Detection: Modify `_inspect_chat_templates` to walk directory (depth unlimited or configurable) collecting candidate files (size <2MB, contains `{{` & `}}`).
3. Hashing: Compute SHA256 for each external template stored under `metadata.chat_template_hashes = [{"file": name, "sha256": hash}]`.
4. Metadata Flag: Add `metadata.template_detection_version = 1` for future migrations.
5. CLI Display: If multiple external templates found, list top 3 and show count. First detected becomes `primary_template` reference in registry metadata (heuristic only; not enforced for backend usage).

### 13.3 Non-Goals (Deferred)
- Automatic selection between multiple templates (leave to user/backend script).
- Validating Jinja syntax beyond brace presence.
- Inferring conversation style or safety tags from template content.

### 13.4 Failure / Edge Handling
- If hashing fails (IOError), continue without hash for that file (log warning).
- If recursive search finds >20 candidate files, cap list to avoid pathological repos.

### 13.5 Pseudocode Insert (Downloader `_partition_files` Post-Processing)
```
promoted = []
for f in list(optional_files):
	 name = f.path.lower()
	 if is_template_candidate(name) and f.size < 2_000_000:
		  optional_files.remove(f)
		  required_files.append(f)
		  promoted.append(f.path)
if promoted:
	 self._log(f"💬 Promoted template files to required: {promoted}")
```

Helper `is_template_candidate(name)` implements regex / heuristics above.

### 13.6 `_inspect_chat_templates` Adjustments
```
for path in target_dir.rglob('*'):
	 if path.is_file() and path.stat().st_size < 2_000_000:
		  lower = path.name.lower()
		  if template_name_match(lower):
				head = path.read_text(errors='ignore')[:400]
				if '{{' in head and '}}' in head:
					 template_files.append(path.relative_to(target_dir).as_posix())
					 hashes[path.relative_to(target_dir).as_posix()] = sha256_of(path)
```
Store `hashes` under metadata as described.

### 13.7 Migration Consideration
On first run after upgrade, existing entries without `template_detection_version` remain unchanged until a re-download or explicit `verify --refresh-templates` (future command) triggers re-scan (deferred).

### 13.8 Risks & Mitigations
| Risk | Mitigation |
|------|------------|
| False positives (unrelated .jinja) | Size limit + brace presence heuristic |
| Performance on large trees | Early size/name filters; short-circuit after 20 files |
| Hash drift after manual edits | Future `verify` subcommand could recompute and warn |

---
## 14. Implementation Ordering
1. Commit downloader enhancements (promote + recursive detect + hashing) with unit tests.
2. Commit chat proxy scaffold (endpoints + minimal forwarder) behind feature flag.
3. Add streaming + metrics + logging.
4. Add tool normalization & vision guardrails.
5. Integrate autostart.
6. Add metrics endpoint & finalize docs.
7. Quality gates (tests, lint) & finalize.

---
## 15. Done Definition (Phase 1)
- All AC1–AC16 passing (tests where automatable; manual smoke for vision/tool if needed).
- Downloader promotes templates & registry metadata reflects new hash fields.
- Proxy returns correct model list & handles streaming without memory leaks.
- Documentation updated (usage + limitations + security note).
- Reproducible startup via CLI entrypoint.

---
## 16. Post-Phase 1 Backlog (Captured for Future)
- Auth layer (API key) & rate limiting.
- Template hash enforcement & mismatch warnings.
- Per-user log segmentation & retention policy config.
- Extended OpenWebUI plugin for rich metadata.
- Adaptive concurrency / queue metrics.

---
End of Implementation Blueprint.

---
## 17. Design Rationale (Why We Chose This Path)

This section captures the reasoning behind major decisions so a future engineer can evaluate trade-offs or revisit assumptions without reconstructing prior context.

### 17.1 Proxy (Option B) vs Full Plugin (Option D)
| Chosen | Reason | Alternative Rejected | Revisit When |
|--------|--------|----------------------|--------------|
| Proxy-first (Option B) | Fastest path to value; leverages stable OpenAI wire protocol; minimal coupling to OpenWebUI internals | Full plugin requires learning & tracking OpenWebUI extension APIs; higher maintenance | If richer UI (capabilities, performance dashboards) inside OpenWebUI becomes critical or plugin API stabilizes further |

### 17.2 OpenAI-Compatible API Shape
Reason: De facto interoperability layer (OpenWebUI + generic clients). Avoids bespoke adapters. Alternative (custom REST) would necessitate an OpenWebUI fork or plugin sooner.

### 17.3 SSE Streaming Day 1
Reason: OpenWebUI UX noticeably degraded without streaming (latency perception, tool call surfacing). SSE is widely supported and backend (vLLM/Ollama) already stream; minimal incremental complexity. Alternative (batch only) trades small development savings for poorer user experience.

### 17.4 Tool Call Normalization (Light, Conditional)
Reason: Prevent subtle UI breakage when backend emits legacy `function_call` vs `tool_calls`. Conditional approach avoids unnecessary transformations on already-compliant payloads. Alternative (pass-through only) would push complexity into each consumer; alternative (heavy normalization always) adds overhead and risk of accidental semantic alteration.

### 17.5 Enforcing Presence of Chat Templates
Reason: Recent transformer releases disallow implicit “default” templates; missing template leads to confusing backend 400 errors. Early proactive guard yields clearer remediation hint. Alternative (laissez-faire) would reduce initial checks but increase downstream debugging churn.

### 17.6 Template Handling Strategy (Defer Formatting to Backend)
Reason: Avoid re-implementing evolving templating semantics (Jinja variations, system vs user role patterns). Backend already battle-tested for formatting. Alternative (proxy-level templating) increases duplication and divergence risk.

### 17.7 Downloader Enhancements (Promotion + Recursive Detection + Hashing)
Reason: Reliability of prompt formatting depends on template presence. Minimal code changes significantly decrease support burden (“why does model error on 400?”). Hashes future-proof for reproducibility and integrity warnings. Alternative (documentation-only guidance) relies on user diligence and yields inconsistent environments.

### 17.8 Autostart (Single Attempt, No Downloads)
Reason: Keeps scope constrained: start already-installed models only; avoids conflating provisioning with routing. Prevents surprise network pulls. Alternative (full supervisor or multi-retry) introduces complexity (state machine, exponential backoff) not yet justified (
low concurrency, early phase).

### 17.9 Metrics In-Memory (Rolling Window) + JSONL Logs
Reason: Simplicity & low operational overhead. Files provide long-term raw data; memory buffer powers quick UI or health probes. Alternative (database / Prometheus exporter) adds infra cost prematurely. Revisit if multi-instance scaling or external dashboards prioritized.

### 17.10 Error Envelope with `hint`
Reason: Speeds troubleshooting; reduces context needed from logs. Alternative (strict OpenAI error only) misses opportunity for guided remediation (e.g., “re-download with --include-optional”).

### 17.11 Quantization in Model ID (display_name-quant)
Reason: Disambiguation in UI critical when multiple variants differ only by quantization; embedding directly in id avoids secondary column reliance. Alternative (separate metadata field only) risks user picking unintended precision.

### 17.12 Minimal Dependency Footprint
Reason: Lower supply chain risk and environment friction. Chose `httpx` (if not already present) for async streaming; avoided heavier frameworks (no Celery, no Redis) until concurrency demands prove need.

### 17.13 Internal Abstractions (resolver / forwarder / normalization separated)
Reason: Facilitates future Option D pivot; each concern independently testable. Alternative (monolithic endpoint logic) yields faster initial coding but hampers evolution (adding Triton backend, adding auth middleware, etc.).

### 17.14 Capability Guard (Vision)
Reason: Early rejection localizes mismatch error to proxy with clear message vs undefined backend behavior (some backends silently ignore image parts or return vague errors). Alternative (let backend fail) yields inconsistent user experiences.

### 17.15 Deferred Items (Auth, Plugin UI, Multi-Template Selection)
Reason: Premature implementation would delay critical path (functional proxy). Deferral chosen where user value < implementation cost now. Documented explicitly to prevent assumption of oversight.

### 17.16 Risks & Monitor Points
| Area | Risk | Mitigation / Trigger to Revisit |
|------|------|----------------------------------|
| Normalization | Unexpected backend schema change | Add schema version detection; fallback passthrough |
| Autostart | Race conditions on rapid concurrent requests | Introduce per-model async lock (implemented) and throttle if repeated startup failures observed |
| Template detection | False negatives in nested dirs | Expand recursion depth + caching if support tickets recur |
| Metrics accuracy | Estimated tokens degrade reliability | Integrate backend token usage endpoints once stable |
| Logging growth | Disk usage over long runs | Add size-based pruning + compression in Phase 2 |

### 17.17 When to Reconsider Major Decisions
| Trigger | Decision to Reassess |
|---------|----------------------|
| Need rich per-model metadata in UI | Build Option D plugin or hybrid metadata endpoint consumed by OpenWebUI custom panel |
| Multi-tenant / external exposure | Introduce API keys, rate limits, request signing |
| High concurrency (>20 concurrent streams) | Add async connection pool tuning, backpressure, queue metrics |
| Need reproducible audits | Persist metrics to structured store (e.g., SQLite/Parquet) |
| Frequent template hash mismatches | Enforce hash verification at request time |

---
End of Rationale Section.
