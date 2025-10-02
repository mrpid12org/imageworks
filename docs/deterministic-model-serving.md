# Deterministic Model Serving Specification

## 0. Context & High‑Level Rationale (vLLM + Ollama Hybrid)
We evaluated multiple backends (vLLM, LMDeploy, Ollama, llama.cpp direct / custom FastAPI) under goals of: deterministic model selection, reproducibility, multimodal (vision) support, throughput, low operational friction, and future extensibility (profiles, performance comparisons). A hybrid decision (vLLM + Ollama for now; GGUF custom server optional later) maximizes strengths while containing complexity.

### Why Keep vLLM
- High throughput & batching: Continuous batching + PagedAttention excels for concurrent chat & long context.
- Strong AWQ + FP16/BF16 support: Production-grade scheduling for GPU-bound multi-user scenarios.
- Multimodal readiness: Clear roadmap for broader multimodal coverage (already works with some vision LLaVA/Qwen variants when projector supported).
- OpenAI API compatibility: Minimizes adaptation cost for existing tools / tagger pipelines.
- Fine-grained control of KV cache & memory tuning.

### vLLM Trade-offs
- Larger steady-state GPU memory overhead (scheduler + cache structures).
- Slower cold start vs lightweight runners.
- Less convenient ad‑hoc model experimentation (needs manual download & launch script vs `ollama run`).
- Limited native GGUF / exotic quant format support.

### Why Add Ollama
- Frictionless experimentation: One-line pulls for many community models (especially GGUF quantized variants).
- Simple lifecycle: Lazy load/unload; fast local iteration.
- Strong GGUF ecosystem coverage (small VRAM-friendly variants) complementary to AWQ.
- Integrated modelfile build process for custom system prompts or templates.
- Lower ops burden for casual evaluation, side-by-side comparisons.

### Ollama Trade-offs
- Lower peak throughput; less advanced batching / scheduling.
- Less granular performance tuning (KV cache, scheduler knobs limited).
- Vision support uneven / emerging depending on model family.
- Performance metrics less exposed (need wrapper instrumentation).

### Hybrid Justification
- vLLM handles production-grade, high-throughput, vision-capable AWQ/FP16 models.
- Ollama supplies rapid prototyping, GGUF lightweight variants, convenient model digestion for comparison.
- Deterministic registry + loader abstracts differences; clients see a single contract.
- Future: Can introduce a native llama.cpp (GGUF) vision server or consolidate if one backend becomes dominant.

## 1. High-Level Requirements
MUST:
- Deterministic model selection: Client specifies exact logical model name; no implicit routing / substitution.
- Explicit failure on missing model or backend error (no fallback in Phase 1).
- Reproducibility: Version locking via file digests + backend descriptor; detection of drift.
- Support both vLLM and Ollama concurrently under unified registry schema.
- Manual vision probe utility to validate multimodal readiness.
- Capture performance metrics: time-to-first-token (TTFT) & tokens/sec (streaming throughput) per model.
- Fixed ports (static assignment per backend instance) for predictable integration.
- Chat template handling (internal or external .jinja) recorded in registry metadata.

SHOULD (Phase 1 or 1.5 if low risk):
- Rolling performance stats persisted (EMA / sliding window) for TTFT & throughput.
- Structured error payloads (no stack trace leakage) on selection/load errors.
- Registry migration script idempotent & safe.

FUTURE (Deferred but design accommodates):
- Profiles (logical role aggregates referencing explicit model names).
- Fallback lists (opt-in) for degraded continuity.
- Automated scheduled probes & drift revalidation.
- GPU topology awareness / multi-GPU placement.
- Latency percentiles & quality probe metrics aggregation.

## 2. Non-Functional Requirements
- Determinism: Same input registry snapshot + same model artifact hashes → identical selection outcome.
- Observability: Structured JSON logs for selection, load, probe, performance samples.
- Extensibility: Schema forward-compatible with profiles & fallback without destructive migrations.
- Minimal startup overhead: Loader lazily starts model processes on first selection (unless prewarm requested).
- Security: No remote code execution via model configs; only whitelisted command templates for each backend.

## 3. Open Questions & Resolutions
1. Hashing scope → Hash critical files (weights, tokenizer.*, config.json, mm_projector.*, chat template(s)). Aggregate SHA256 stored. Per-file hashes optional list.
2. Hashing frequency → On registry insertion + explicit verify command; lazy verify (checksum missing or mismatch) before first load.
3. Vision probe cadence → Manual only (CLI or API trigger). No automatic schedule in Phase 1.
4. Probe data stored? → Yes: last_probe (timestamp), probe_version (semantic), vision_ok (bool), notes, sample_latency_ms.
5. Fallback semantics → Not implemented in Phase 1. Missing model = immediate failure.
6. Profiles → Deferred. Reserve nullable JSON column/field profile_placeholders to avoid later schema churn.
7. Ports → Fixed mapping defined in configuration (e.g., vLLM base 8001+, Ollama API assumed default 11434, custom GGUF 8100 if introduced). Stored in registry entry.
8. Failure handling when model unavailable → Immediate error (HTTP 409 or 424). No substitution.
9. Performance metrics granularity → Per-request sample; maintain rolling window of last N (default 50) with aggregated average & P50; design to add P95 later.
10. API surface stability → Versioned under /v1/ with explicit schema docs; additive changes only in Phase 1.

## 4. Data Model (Registry Schema)
Logical model registry entry (YAML or JSON persisted plus optional DB/JSONL):
```
{
  "name": "llava-1.5-7b-awq",           // stable logical identifier (client uses this)
  "backend": "vllm" | "ollama" | "gguf" | "lmdeploy",
  "backend_config": {                    // backend-specific launch params (ports, max_tokens, quantization)
    "port": 8001,
    "model_path": "/abs/path/...",
    "extra_args": ["--max-model-len=4096"]
  },
  "capabilities": {                      // explicit modal capabilities
    "text": true,
    "vision": true,
    "audio": false
  },
  "artifacts": {                         // reproducibility
    "aggregate_sha256": "...",
    "files": [ {"path": "config.json", "sha256": "..."}, ... ]
  },
  "chat_template": {
    "source": "embedded" | "external",
    "path": ".../template.jinja",
    "sha256": "..."
  },
  "version_lock": {                      // version pin & drift detection
    "locked": true,
    "expected_aggregate_sha256": "...",
    "last_verified": "2025-10-02T12:00:00Z"
  },
  "performance": {
    "rolling_samples": 37,
    "ttft_ms_avg": 180,
    "throughput_toks_per_s_avg": 72.5,
    "last_sample": {
      "ttft_ms": 175,
      "tokens_generated": 256,
      "duration_ms": 3400
    }
  },
  "probes": {
    "vision": {
      "vision_ok": true,
      "timestamp": "2025-10-02T11:55:00Z",
      "probe_version": "vision-probe-1",
      "latency_ms": 950,
      "notes": "basic caption successful"
    }
  },
  "profiles_placeholder": null,          // reserved for future profile mapping
  "metadata": {                          // free-form additions
    "notes": "Initial import"
  }
}
```

## 5. Loader Service Architecture
Components:
- Registry Provider: Loads registry JSON/YAML file(s) → in-memory index keyed by name.
- Model Process Manager: Launches backend processes on demand (if not already up) using sanitized command templates.
- Health & Readiness Checker: For vLLM uses /health or sample completion; for Ollama uses /api/tags or quick generate with small prompt.
- Performance Instrumentor: Wraps streaming responses; timestamps first token & completion; updates rolling metrics.
- Vision Probe Executor: CLI/API that performs image prompt test and records result.
- Hash Verifier: Computes aggregate hash as needed; compares to lock; raises error if mismatch when locked.

Control Flow (select_model):
1. Lookup registry entry.
2. If version_lock.locked: verify (unless recently verified within TTL, e.g., 10 min or manual-only Phase 1 → manual only for now).
3. Ensure backend process started (spawn if absent).
4. Run readiness check (timeout configurable, default 30s).
5. Return connection descriptor: { endpoint_url, model_name (backend internal if different), capabilities, tokenization_hints }.

## 6. Backend Adapters
- vLLM Adapter: Start via Python script (`start_vllm_server.py`) with fixed port; supports chat template override arg. Readiness: HTTP GET to /health or small completion.
- Ollama Adapter: Assume daemon already running at fixed port (config). Ensure model pulled; if not present, error (no implicit pull when locked). Use `ollama show <model>` for readiness (or /api/show).
- GGUF Adapter (Optional Phase ≥1.5): If reintroduced, would wrap llama.cpp directly; current plan uses Ollama for GGUF (no custom script maintained).

## 7. API Surface (Phase 1)
Endpoints (HTTP + Python API parity):
- GET /v1/models → list logical models with subset of metadata (capabilities, perf summary, lock state, vision_ok).
- POST /v1/select {"name": "llava-1.5-7b-awq"} → returns descriptor {endpoint, backend, internal_model_id, capabilities}.
- POST /v1/probe/vision {"name": "llava-1.5-7b-awq", "image_url"/"image_b64"} → triggers manual vision probe; returns result & updates registry.
- POST /v1/verify {"name": "..."} → recompute hashes; if locked & mismatch return 409.
- GET /v1/models/{name}/metrics → performance metrics snapshot.

Errors:
- 404 model not found.
- 409 version lock violation.
- 424 dependency / backend not ready (no fallback attempted).

## 8. Probing & Metrics
Vision Probe: Minimal prompt (e.g., "Describe this image succinctly.") with 1x1 or small test image. Success if non-empty text and no error. Latency captured (request → completion). Manual only.
Performance Metrics:
- Instrument streaming: record t0 (request dispatch), t_first (first token event), t_done (final token). TTFT = t_first - t0. Throughput = (tokens_generated - 1)/(t_done - t_first) * 1000.
- Store rolling window (deque up to N). Recompute averages on update.
- Expose via metrics endpoint + aggregated in list models for quick comparison.
Design allows future: percentile computation; merging with external quality metrics.

## 9. Version Lock & Hashing
- On add/import: compute aggregate hash (sorted concatenation of per-file SHA256, then SHA256 of that string).
- version_lock.locked=true → selection requires hash equality; mismatch triggers 409 error with details (expected vs actual) — no auto-correction.
- Manual verify command recomputes and updates last_verified when consistent.
- Drift policy (Phase 1): Fail hard; future phases might allow soft warning mode.

## 10. Implementation Phases
P0 (Core Determinism):
- Schema extension (registry file format change) + migration utility.
- Loader service skeleton (list/select, process launch, readiness).
- Hash computation & lock enforcement (manual verify only).
- Performance instrumentation (TTFT, throughput) for vLLM & Ollama.
- Vision probe CLI/API (manual trigger, stored result).

P1 (Hardening & Docs):
- Structured error handling & logging.
- Metrics endpoint + rolling window statistics.
- Documentation of contract & examples.

Deferred (Planned):
- Profiles (requires registry extension; placeholder already present).
- Fallback lists & substitution semantics (disabled until explicit opt-in design complete).
- Automated scheduled probes.
- Percentile & historical metrics persistence.

## 11. Future Extensions / Design Allowances
Profiles Placeholder: `profiles_placeholder` reserved to store mapping structures later without migration.
Fallback Path: Add `fallback_candidates: ["modelA", "modelB"]` list later; current logic simply rejects if name unavailable.
Multi-GPU: Potential `placement` field (gpu_ids array) — ignored now.
Advanced Probes: Additional semantic quality metrics (noun recall, adjective density) can be nested under `probes.vision.metrics`.

## 12. Operational Considerations
- Fixed Ports Config: Central config file `model_backends.yaml` mapping backend → base port; registry entries must align.
- Concurrency: Rely on backend internal scheduling (vLLM) + stateless loader endpoint.
- Restart Strategy: Loader will detect dead process and re-launch only on explicit select; no auto-heal background loop in P0.
- Logging: JSON lines; fields: timestamp, event_type (select|launch|probe|performance_sample|verify), model, backend, latency_ms, ttft_ms, tokens_per_s.

## 13. Security & Validation
- Sanitize launch args (whitelist known flags) to avoid arbitrary injection from registry file.
- Read-only hash verification; never executes model-provided scripts.
- Optional allow-list for model root directories.

## 14. Testing Strategy (Phase 1)
- Unit: hash computation deterministic, registry parsing, rolling metrics math.
- Integration: spin ephemeral vLLM with a tiny model (or mock) → select → stream completion → metrics recorded.
- Vision probe mock (inject test image) verifying success path & registry update.
- Lock violation test (tamper file) → expect 409.

## 15. Risks & Mitigations
- Risk: Long cold start harming first TTFT metric. Mitigation: Mark first sample distinctly; optionally exclude from rolling stats (future flag).
- Risk: Registry corruption. Mitigation: Validate schema on load, refuse start if invalid; backup previous file before write.
- Risk: Ollama digest changes silently. Mitigation: Include reported model manifest digest in artifacts hash set when possible.

## 16. Minimal CLI / API Commands (Illustrative)
- `imageworks-models list`
- `imageworks-models select llava-1.5-7b-awq`
- `imageworks-models probe-vision llava-1.5-7b-awq tests/assets/sample.jpg`
- `imageworks-models verify llava-1.5-7b-awq`
- `imageworks-models metrics llava-1.5-7b-awq`

## 17. Summary
This specification formalizes a deterministic, reproducible, performance-aware hybrid model serving architecture leveraging vLLM for production multimodal throughput and Ollama for rapid GGUF experimentation. Phase 1 (P0 + P1 above) delivers deterministic selection, version locking, manual vision probing, and performance metrics (TTFT & throughput), deliberately omitting fallback and profiles while preserving forward compatibility.

## 18. Knock-on Changes to Existing Modules
Adopting deterministic model serving requires coordinated updates across existing Imageworks applications to remove implicit backend assumptions and introduce explicit registry-driven selection.

### 18.1 Personal Tagger
Current State:
- Directly configures `backend`, `base_url`, and per-stage model names (`caption_model`, `keyword_model`, `description_model`).
- Constructs backend clients via `create_backend_client` using an enum (no awareness of registry hashes / capabilities).
- Uses a single OpenAI-compatible POST `/chat/completions` (non-streaming) so TTFT cannot be measured internally.

Required Changes (Phase 1 Alignment):
1. Configuration:
  - Introduce new CLI flags / config keys: `--caption-registry-model`, `--keyword-registry-model`, `--description-registry-model` (or a single `--registry-model` applied to all when unspecified individually).
  - Deprecate (but temporarily support) legacy `--backend` / `--base-url`; emit warning if used and bypass loader only if explicitly forced (escape hatch until removal in later phase).
2. Selection Flow:
  - Before creating `OpenAIChatClient`, call Loader API `POST /v1/select` with the logical registry name; obtain `{ endpoint, internal_model_id, capabilities }`.
  - Validate `capabilities.vision == true`; if false, abort early with actionable error.
3. Client Construction:
  - Replace direct enum-based backend creation with a simplified HTTP client that only needs the returned `endpoint` and `internal_model_id` (the loader guarantees compatibility).
  - Retain legacy path for test `FakeInferenceEngine` (no change needed beyond ignoring loader).
4. Performance Metrics:
  - (Optional early improvement) Switch to streaming mode when `--enable-metrics` flag set: capture first chunk arrival for TTFT; forward full text after completion for unchanged downstream parsing.
  - Post per-stage metrics to Loader optional endpoint (future) or rely on Loader instrumentation when requests are proxied (future Phase ≥1.5). For Phase 1 simply record local debug log entries to aid manual comparison.
5. Error Handling:
  - Standardize on surfacing loader errors (404, 409, 424) as structured messages prefixed with `loader_error:` in record notes.
6. Tests:
  - Add unit test ensuring selection rejects a non-vision model when vision is required.
  - Add regression test for lock violation scenario (simulate by tampering registry hash and expecting 409 upon select before inference).

Deferred / Prepared Hooks:
- Placeholders for future profile usage: allow `--prompt-profile` to map to a set of logical registry model names (not implemented yet, just design-friendly variable naming).

### 18.2 Color Narrator
Current State:
- `NarrationConfig` includes `vlm_base_url`, `vlm_model`, and `vlm_backend` referencing LMDeploy/vLLM directly.
- Health check logic directly probes backend endpoints.

Required Changes:
1. Configuration:
  - Replace `vlm_base_url`, `vlm_model`, `vlm_backend` with single `vlm_registry_model` (logical name). Keep legacy fields temporarily; if they are present and `vlm_registry_model` absent, perform legacy mode with deprecation warning.
2. Initialization:
  - On startup call Loader `select_model` for `vlm_registry_model`; receive endpoint + internal model id; store in `self.vlm_client`.
  - Remove direct backend enum usage; the loader enforces capability and lock (vision requirement must be validated).
3. Health Check:
  - Replace direct `VLMClient.health_check()` call with loader-mediated readiness (Loader `select` already ensures readiness). Optionally still probe with a minimal completion if `--extra-health-check` flag provided.
4. Performance Metadata:
  - Capture overall narration inference time per image already done; extend to log (debug) TTFT & tokens/sec if streaming mode becomes enabled (future).
5. Vision Probe Integration:
  - Optionally call Loader vision probe before batch if `--verify-vision` flag set. For Phase 1, manual CLI remains sufficient; narrator just trusts selection.
6. Tests:
  - Update integration test (if present) to mock Loader select response rather than direct backend health.

### 18.3 Shared VLM Library (`imageworks.libs.vlm`)
Changes:
1. Add new backend enum value or refactor so application code does NOT require enumerating underlying backend; instead treat everything as generic OpenAI-compatible once Loader selection succeeds.
2. Provide a thin `SelectedModelDescriptor` dataclass: `{ logical_name, internal_model_id, endpoint, capabilities }` used by both narrator and personal tagger.
3. Introduce utility `select_or_fail(logical_name: str, require_vision: bool = False)` that wraps Loader HTTP call and validates capabilities.
4. Keep existing `create_backend_client` but mark it as deprecated for direct application usage (internal tests & legacy path only).

### 18.4 Registry & Loader Additions to Support Modules
1. Loader `select` accepts optional `require_capabilities` array (e.g., `["vision"]`) returning 409 if not satisfied.
2. Loader response includes simplified `tokenization_hints` so downstream can adjust max token counts without probing.
3. Provide small Python helper `imageworks.model_selection` with cached select results to reduce repeated HTTP calls within a single process run.

### 18.5 Migration & Transitional Strategy
Steps:
1. Implement Loader & registry schema (P0).
2. Introduce helper abstraction layer (`model_selection` module).
3. Migrate Color Narrator to use loader (smallest surface area change).
4. Migrate Personal Tagger (more moving parts: multiple stages/models).
5. Deprecate direct backend flags; emit warnings for one release.
6. Remove deprecated paths once profiles/fallback features land (future).

### 18.6 Risk Mitigation
- Dual Path Complexity: Keep legacy code paths isolated behind `if legacy_mode:` blocks with clear TODO removal comments.
- Streaming Introduction Regression: Introduce streaming behind feature flag to avoid altering existing output parsing unexpectedly.
- Capability Drift: Loader enforces capability; add a CI test that enumerates registry entries and ensures `capabilities.vision` true for models used by vision-dependent modules.

### 18.7 Summary
The knock-on changes centralize model selection in the Loader, eliminate repeated backend-specific logic, and prepare modules for future profiles/fallback without disruptive refactors. Early adoption focuses on minimal invasive changes while establishing strong determinism guarantees.

## 19. Repository & Directory Convention Requirements
All deterministic serving components (registry files, loader service, tests, logs, metrics artifacts) must comply with existing Imageworks repository conventions to ensure consistency and discoverability.

### 19.1 Existing Conventions Recap
- `src/imageworks/` – Python package code only (no ad-hoc data dumps).
- `tests/` – All unit & integration tests; mirrors package sub-structure where practical.
- `logs/` – Runtime log outputs (rotated or timestamped JSONL/text files).
- `outputs/` – User-facing result artifacts (JSONL run outputs, summaries, metrics exports).
- `configs/` – Static configuration samples, templates, or baseline YAML/JSON used to bootstrap state.
- `models/` – (Presently minimal) holds README; large model weights are external (`~/ai-models/weights/`).

### 19.2 Deterministic Serving Additions
1. Registry Storage:
  - Primary registry file: `configs/model_registry.json` (or `.yaml`).
  - A backup snapshot written on every successful mutation: `configs/model_registry.<UTCDate>-<hash>.json` (optional Phase ≥1.5).
2. Loader Service Code:
  - Module path: `src/imageworks/model_loader/` (package namespace reserved for registry, hashing, selection API, process manager, adapters, and metrics).
3. Metrics & Probes:
  - Rolling in-memory; optional export command writes to `outputs/metrics/<model>/<date>.jsonl` when invoked.
  - Vision probe artifacts (sample probe requests/responses) stored under `outputs/probes/<model>/vision_<timestamp>.json` for auditability (manual only Phase 1).
4. Logs:
  - Loader runtime logs: `logs/loader.log` (plain or JSON lines). If JSON lines, use `event_type` key for filtering.
  - Per-model launch logs optional: `logs/model_launch_<model>.log` (captured stdout/stderr from first start). Avoid continuous duplication; truncate or rotate if exceeding size threshold (future improvement).
5. Temporary / PID Files:
  - Avoid scattering; if needed (e.g., tracking spawned process PIDs), place under `logs/pids/` or manage in-memory only.
6. Test Data / Fixtures:
  - Synthetic registry fixtures: `tests/fixtures/registry/` (small JSON/YAML examples for hashing & lock tests).
  - Probe test images: reuse existing `tests/test_images/` rather than adding new directories.
7. CLI Entry Points:
  - Commands added via `pyproject.toml` under `[project.scripts]` prefixed `imageworks-models-` (e.g., `imageworks-models-select`, `imageworks-models-probe`). A unified multi-command Typer app (`imageworks-models`) may wrap them.
8. Documentation:
  - This spec already under `docs/`. Any quickstart for deterministic loader: `docs/deterministic-loader-quickstart.md` (optional enhancement) linking from main README if added.

### 19.3 Rationale / Avoided Anti-Patterns
- Keep registry immutable during runtime except through explicit commands (prevents hidden state in `src/`).
- Do not write performance metrics into `configs/` (reserved for inputs, not time-series data).
- No large binary artifacts inside repo-managed `models/` to avoid bloat; store only references & metadata hashes.
- Use structured naming for logs & metrics to enable simple glob queries (`logs/*loader*.log`, `outputs/metrics/*/*.jsonl`).

### 19.4 Open Questions / Future (Not Blocking Phase 1)
- Rotation policy for `loader.log` (size vs time-based). Default: manual or rely on external logrotate until internal implemented.
- Whether to persist a consolidated `outputs/metrics/summary.json` aggregating all models (Phase ≥1.5).
- Optional SQLite backing store under `outputs/registry_state.db` for richer queries; plain JSON is sufficient for now.

### 19.5 Acceptance Checklist (Phase 1)
- [ ] Registry file exists at `configs/model_registry.json` and validates against schema.
- [ ] Loader does not write outside `logs/` or `outputs/`.
- [ ] Tests for hashing, lock violation, selection error live under `tests/model_loader/`.
- [ ] Vision probe output stored under `outputs/probes/<model>/` when executed.
- [ ] README and/or deployment guide references new registry location.

## 20. Updated Summary
All deterministic serving elements adhere to the repository’s established directory responsibilities, ensuring reproducibility metadata and operational logs are easy to locate while avoiding repo clutter. This alignment reduces friction for contributors, simplifies CI inclusion of new tests, and keeps future extensions (profiles, fallback, advanced metrics) straightforward to layer in without structural rework.
