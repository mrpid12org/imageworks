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
### 4.1 Variant Naming Convention
Logical model ("variant") names follow a normalized, hyphen‑joined pattern to ensure stability, discoverability, and ergonomic filtering:

```
<family>-<backend>-<format>-<quant>
```

Only the first two components are strictly required (`family`, `backend`). `format` and `quant` are included when they disambiguate materially different artifacts. Examples:

| Scenario | Input Source (HF ID / Context) | Derived Variant Name |
|----------|--------------------------------|----------------------|
| AWQ quantized vision model (vLLM) | liuhaotian/llava-v1.5-13b (AWQ) | llava-v1.5-13b-vllm-awq |
| FP16 vision model (vLLM) | liuhaotian/llava-v1.5-13b (fp16) | llava-v1.5-13b-vllm-fp16 |
| GGUF quant (Ollama / llama.cpp) | TheBloke/Mistral-7B-Instruct-v0.2-GGUF (Q4_K_M) | mistral-7b-instruct-v0.2-ollama-gguf-q4_k_m |
| Plain instruct (no quant) | mistralai/Mistral-7B-Instruct-v0.2 (fp16) | mistral-7b-instruct-v0.2-vllm-fp16 |
| Embedding model | google/siglip-base-patch16-256 | siglip-base-patch16-256-vllm (format omitted if single) |

Normalization Rules:
1. `family` is derived from the HuggingFace repo tail (optionally with branch suffix) lower‑cased; characters `/`, `_`, spaces, and `@` become `-`; repeated `-` collapsed.
2. `backend` is one of: `vllm`, `ollama`, `lmdeploy`, `gguf` (future backends may append). For Ollama backed GGUF we still use `ollama` as backend; the `format` captures `gguf`.
3. `format` (optional) denotes storage/serving format or weight packaging: examples `awq`, `gguf`, `fp16`, `bf16`, `safetensors`. Omit when no ambiguity (e.g., single canonical form) OR when already implied by quant (e.g., `gguf` present).
4. `quant` (optional) is a quantization spec or tier, lower‑cased, preserving inner underscores (e.g., `q4_k_m`, `int4`, `awq`, `gptq`). If quant string equals `format` it is not duplicated.
5. Components that are `None` / empty are skipped; no trailing hyphens.
6. Final name must be <= 80 chars; if longer, middle segments may be abbreviated (future enhancement; not yet implemented in adapter).

Collision Handling:
- If two downloads would produce the same name (e.g. same family/backend/format/quant) the second updates the existing entry rather than creating a duplicate. Distinguish by adding an explicit quant or format if both variants are required concurrently.

Rationale:
- Encourages ergonomic shell filtering (`grep -F '-awq'`, `grep vllm`), consistent table alignment, and deterministic reverse mapping from variant → source.
- Minimizes surprises when scripting environment variable names or log parsing.

Adapter Behavior:
- The download adapter infers `family` and constructs the name automatically; manual overrides are currently not exposed (future `--name-override` may be added if needed).
- Re-download with different format/quant updates the same entry unless the naming changes; to preserve both, download into a distinct format/quant combination.

Cross‑Reference:
- See `../reference/model-downloader.md` for CLI examples (`remove`, `verify`, `list-roles`) operating on these variant names.

---
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

### 9.1 Sync & Source File Hashes (New)
`sync-downloader` now imports downloader manifests and enriches each entry's `source.files` with `size` and (optionally) per-file `sha256` (controlled by `--include-file-hashes/--no-include-file-hashes`). This supplements (not replaces) the reproducibility `artifacts` block. The `artifacts` block is intended for the *subset of critical runtime files* whose aggregate hash drives version locking, whereas `source.files` provides provenance & inventory metadata.

Recommended practice:
1. Run `sync-downloader` after adding new models (this seeds `source`).
2. Inspect and prune unnecessary large binary files from the future hashing scope if you later migrate to full recursive hashing (keeps aggregate stable and fast).
3. Run `verify <name>` to compute `artifacts.*`.
4. Lock with `verify <name> --lock` (or separate `lock` command if using the multi-command CLI) once satisfied.

Rationale for split: `source.files` may contain dozens/hundreds of weight shards or auxiliary files; locking all recursively can be expensive. We start with targeted essential files and can graduate to full-directory hashing per model as needed (configurable via future `include_patterns`).

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

## 21. Implementation Deep-Dive Addendum (2025-10-02)
This addendum captures the detailed analysis and recommended initial implementation slices so the actionable plan is preserved outside of transient discussion.

### 21.1 Codebase Review Highlights
- Personal Tagger currently selects backends via raw config fields: `backend`, `base_url`, and individual model names (`caption_model`, `keyword_model`, `description_model`). No deterministic registry indirection or capability validation layer exists yet.
- Both Personal Tagger (`OpenAIInferenceEngine`) and Color Narrator (`VLMClient`) invoke `create_backend_client` directly—clear seams for replacing with a loader selection abstraction.
- Inference requests are non‑streaming (`"stream": False`) everywhere; TTFT (time to first token) cannot be measured yet. We will design metrics infra now so streaming can be introduced behind a flag without structural upheaval.
- No existing `configs/model_registry.json`; schema introduction is greenfield. A lightweight JSON file plus dataclass models is sufficient for Phase 1; pydantic is optional.
- Backend abstraction (`VLMBackend` enum) currently enforces explicit enumeration (vllm, lmdeploy, triton stub). Loader integration should reduce the need for application code to know the backend type directly.

### 21.2 Recommended Phase 1 (P0) Implementation Slice
1. Registry + hashing (read-only + verify CLI) introduced first—no process management side effects initially.
2. Loader service skeleton (Python module API only): selection + minimal process manager stubs.
3. Opt-in adaptation path for Personal Tagger (least invasive, keeps legacy path).
4. Vision probe CLI (manual use) storing results to `outputs/probes/`.
5. Metrics rolling window structure (TTFT placeholder, throughput later once streaming added).
6. After stabilization, migrate Color Narrator similarly.

### 21.3 Registry Schema (Initial Fields)
Mandatory:
- name, backend, backend_config (port, model_path, extra_args[])
- capabilities {text, vision, audio}
- artifacts.aggregate_sha256 (may be empty until first verify)
- version_lock {locked: bool, expected_aggregate_sha256?, last_verified?}
Optional / Deferred but scaffolded:
- chat_template {source, path, sha256}

### 21.7 Phase 1 Implementation Status (Code Landed)
The following components have been fully implemented in the repository (2025‑10‑02 snapshot):

- Registry core: `configs/model_registry.json` + dataclass models (`model_loader/models.py`).
- Registry load/save/update: `model_loader/registry.py` with atomic write (`save_registry`).
- Hashing & lock enforcement: `model_loader/hashing.py` (`verify_model`, `VersionLockViolation`).
- Selection service: `model_loader/service.py` (`select_model`, capability guard + endpoint synthesis).
- Metrics primitives: `model_loader/metrics.py` (`RollingMetrics`, `BatchRunMetrics`, `StageTiming`).
- Vision probe: `model_loader/probe.py` (manual, records vision probe outcome in registry entry).
- FastAPI HTTP API: `model_loader/api.py` exposing list/select/verify/probe/metrics.
- Typer CLI: `model_loader/cli.py` with commands: list, select, verify, probe-vision, metrics, lock/unlock.
- Application integration:
  * Personal Tagger: deterministic selection for caption/keyword/description stages pre-config; batch metrics persisted to `outputs/metrics/personal_tagger_batch_metrics.json`.
  * Color Narrator: deterministic selection via `--vlm-registry-model`; batch metrics persisted to `outputs/metrics/color_narrator_batch_metrics.json`.

### 21.8 CLI Reference (Implemented)
`imageworks-models list` → Lists logical models (name, backend, vision capability, lock state).

`imageworks-models select <name>` → Returns JSON descriptor: `{endpoint, backend, internal_model_id, capabilities}`.

`imageworks-models verify <name>` → Recomputes artifact hashes, updates `artifacts.files` & `aggregate_sha256`; if locked and mismatch → non‑zero exit (VersionLockViolation message).

`imageworks-models probe-vision <name> <image_path>` → Runs vision probe, updates `probes.vision` block.

`imageworks-models metrics <name>` → Dumps rolling performance summary (placeholder until streaming metrics wired).

`imageworks-models lock <name>` / `unlock <name>` → Toggles `version_lock.locked`. When locking with empty `expected_aggregate_sha256`, next successful verify seeds the expected hash.

### 21.9 HTTP API Reference (FastAPI)
Base: `/v1` (served via `imageworks-models-api`).

- `GET /v1/models` → `[ { name, backend, capabilities, version_lock, performance, probes } ]` (subset fields).
- `POST /v1/select {"name": "..."}` → `{ endpoint, backend, internal_model_id, capabilities }` (HTTP 404 if missing, 409 if lock mismatch & design later, currently raised in verify path only, capability error 400/409 depending on configuration).
- `POST /v1/verify {"name": "..."}` → `{ status: "ok", aggregate_sha256, last_verified }` or 409 on lock violation.
- `POST /v1/probe/vision {"name": "...", "image_b64"|"image_url": "..."}` → `{ vision_ok, latency_ms, timestamp, notes }`.
- `GET /v1/models/{name}/metrics` → Performance summary from rolling window (empty placeholder if no samples yet).

Error Semantics (current): simple JSON `{ "error": "message" }` with HTTP status codes (404, 409). Future improvements: structured `code` field.

### 21.10 Version Lock Workflow (Operational)
1. Add / edit registry entry with `version_lock.locked = false`.
2. Run `imageworks-models verify <name>` to compute initial hashes.
3. Enable lock: `imageworks-models lock <name>` (sets `locked=true`; if `expected_aggregate_sha256` empty it will be populated on next verify).
4. Re-run `verify` (or allow selection flows to optionally call verify preflight in future) → if mismatch with expected, command exits with lock violation, and selection SHOULD be avoided until resolved.
5. To intentionally update artifacts: unlock → verify (new hash) → lock → verify (seeds new expectation).

### 21.11 Batch Metrics Persistence
Both Personal Tagger and Color Narrator now emit aggregated per-run metrics:
- Personal Tagger: `outputs/metrics/personal_tagger_batch_metrics.json`
- Color Narrator: `outputs/metrics/color_narrator_batch_metrics.json`

Schema:
```
{
  "history": [ { "model_name": str, "backend": str, "batch_total_seconds": float, "stages": { "image": {"count": int, "avg_seconds": float, ... } }, "timestamp": ISO8601, "model_load_seconds": null|float } ],
  "last": { ... duplicate of most recent summary }
}
```
Stage granularity is currently limited to `image` items; future additions may include `download`, `preprocess`, `inference`, `metadata_write` if instrumentation is expanded inside the processing loops (low-risk additive change).

### 21.12 Test Coverage Snapshot
Added tests under `tests/model_loader/`:
- `test_registry.py`: registry load success, duplicate detection, missing entry, selection capability enforcement.
- `test_hashing.py`: verify unlocked path, lock violation, initial lock seeding of expected hash.
- `test_metrics.py`: RollingMetrics aggregation math, BatchRunMetrics stage aggregates.
- `test_cli.py`: Typer CLI smoke tests for list, select, verify.

Planned (future enhancements): streaming metrics sampling tests once streaming integration introduced; probe vision test with synthetic image (manual path currently untested in CI due to external dependencies).

### 21.13 Known Gaps / Deferred Items (Post‑Phase 1)
- Streaming integration (TTFT real measurement) not yet wired; RollingMetrics presently unused in app flows.
- Automated probe scheduling & historical probe retention.
- Registry backup snapshot rotation & integrity checksum.
- Profiles & fallback semantics.
- Extended stage timing (multi-phase segmentation) and percentile metrics.

### 21.14 Usage Quickstart (Practical Example)
1. Inspect models: `imageworks-models list`
2. Verify hashes (after placing model artifacts): `imageworks-models verify llava-1.5-7b-awq`
3. Lock model: `imageworks-models lock llava-1.5-7b-awq` then re-verify.
4. Run personal tagger with deterministic models: `imageworks-personal-tagger run --caption-registry-model llava-1.5-7b-awq ...`
5. Run color narrator: `imageworks-color-narrator narrate --vlm-registry-model llava-1.5-7b-awq -i <images> -o <overlays> -j <mono.jsonl>`
6. Review batch metrics: `cat outputs/metrics/personal_tagger_batch_metrics.json`.

### 21.14.1 Ollama (GGUF) Vision Model Quickstart
Experimental support added for an `ollama` backend entry assuming Ollama provides OpenAI-compatible `/v1` endpoints.

1. Install & start Ollama:
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ollama serve
  ```
2. Pull a vision model (example – adjust tag if repository layout changes):
  ```bash
  ollama pull qwen2.5-vl:7b-q4
  ```
  (If an exact tag does not exist, use an available Qwen2.5-VL 7B 4‑bit variant or build a Modelfile referencing a local GGUF.)
3. Registry entry (already added in repo) named `qwen2.5-vl-7b-gguf-q4` targets port `11434`.
4. Run personal tagger deterministically:
  ```bash
  uv run imageworks-personal-tagger run \
    --input-dir /path/to/images \
    --dry-run \
    --use-loader \
    --caption-registry-model qwen2.5-vl-7b-gguf-q4 \
    --keyword-registry-model qwen2.5-vl-7b-gguf-q4 \
    --description-registry-model qwen2.5-vl-7b-gguf-q4
  ```
5. First run uses unlocked hash (no artifacts recorded). To lock later, stage critical files in a managed folder, update `model_path`, then:
  ```bash
  imageworks-models verify qwen2.5-vl-7b-gguf-q4
  imageworks-models lock qwen2.5-vl-7b-gguf-q4
  ```

Notes:
- Endpoint assumption: Ollama exposes `/v1/chat/completions`. If not available, add or run an OpenAI proxy; otherwise selection will succeed but inference may fail.
- Streaming still disabled; TTFT metrics remain placeholder.
- For consistent comparisons with AWQ vLLM model, keep prompt & token settings identical.

### 21.15 Running Log (Appended)
2025-10-02: Integrated deterministic selection into both personal tagger and color narrator; added batch metrics persistence and initial automated test suite.
- performance {rolling_samples, ttft_ms_avg, throughput_toks_per_s_avg, last_sample{...}}
- probes {vision {...}}
- profiles_placeholder (null)
- metadata {notes}

### 21.4 Hashing Utility Design
Functions:
- `compute_file_hash(path: Path) -> str`
- `compute_aggregate_hash(file_hashes: list[str]) -> str` (SHA256 of sorted concatenated per-file hashes)
- `collect_artifact_hashes(entry) -> list[(relative_path, sha256)]`
Edge Cases:
- Missing file → raise explicit error (fail fast)
- Empty file list → aggregate hash of empty string (documented)

### 21.5 Loader Service Skeleton
Modules (under `src/imageworks/model_loader/`):
- `models.py`: dataclasses: RegistryEntry, BackendConfig, SelectedModel, PerformanceSnapshot, VisionProbeResult.
- `registry.py`: load/validate JSON file → in-memory dict keyed by name.
- `hashing.py`: utilities above.
- `process_manager.py`: stubs `ensure_started(entry)` (vLLM stub logs intent; Ollama noop with readiness check placeholder).
- `metrics.py`: rolling window structure.
- `service.py`: `select_model(name, require_capabilities=None)` orchestrating: lookup → (future) hash verify (manual now) → process ensure → return descriptor.
Contract of `SelectedModel`:
```
{
  logical_name: str,
  endpoint_url: str,  # e.g. http://localhost:<port>/v1
  internal_model_id: str,  # may equal logical name
  backend: str,
  capabilities: dict
}
```

### 21.6 Performance Metrics Design
Classes:
- `PerformanceSample(ttft_ms: Optional[float], tokens_generated: int, duration_ms: float)`
- `RollingMetrics(maxlen=50)` with `add(sample)` and `summary()` ignoring None TTFT values.
Future accommodation: flag to exclude first-sample cold start from averages.

### 21.7 Vision Probe CLI
### 21.7.1 Batch & Stage Metrics Extension (Added 2025-10-02)
In addition to per-request TTFT/throughput, a lightweight batch/run metrics facility was introduced:
- `BatchRunMetrics`: captures model load time, total batch duration, and per-stage aggregated stats (caption/keyword/description or others).
- `StageTiming`: start/end wrapper enabling multiple occurrences of the same stage.
Export structure example:
```
{
  "model_name": "llava-1.5-7b-awq",
  "backend": "vllm",
  "model_load_seconds": 4.21,
  "batch_total_seconds": 123.45,
  "stages": {
    "caption": {"count": 50, "total_seconds": 40.2, "avg_seconds": 0.80, "min_seconds": 0.65, "max_seconds": 1.10},
    "keyword": {"count": 50, ...},
    "description": {"count": 50, ...}
  }
}
```
Integration Guidance:
1. Instantiate once per tagger run.
2. Record model load (when loader triggers a cold start).
3. Wrap each stage inference call with `start_stage` / `end_stage`.
4. At end call `close_batch()` and persist summary JSON alongside existing run artifacts (e.g. `outputs/results/personal_tagger_batch_metrics.json`).
5. (Optional) Append per-batch line to a rolling `outputs/metrics/<model>/batches.jsonl` for longitudinal analysis.

Command: `imageworks-models probe-vision <logical_name> <image_path>`
Flow:
1. Load registry entry.
2. (Future) verify hash if locked.
3. Ensure backend running.
4. Send minimal prompt with embedded image.
5. Determine success: HTTP 200 and non-empty text.
6. Record latency (request→completion) and store JSON at `outputs/probes/<model>/vision_<timestamp>.json`.
7. Update in-memory registry and optionally persist `probes.vision` block back to file.
Failure modes: missing model (404), capability mismatch (409-like), backend not ready (424-like), inference error.

### 21.8 Personal Tagger Adaptation Plan (Incremental)
New CLI flags:
- `--caption-registry-model`, `--keyword-registry-model`, `--description-registry-model`
- `--use-loader` (boolean, auto-enabled if any registry model arg supplied)
Steps:
1. Early in run, if loader mode, call `select_model` for each logical name (deduplicate if same).
2. Validate `vision` capability for all selected entries.
3. Override `base_url` from returned endpoint and model names from `internal_model_id` before instantiating `OpenAIChatClient`.
4. Record note `selected_via_loader` in `PersonalTaggerRecord.notes`.
5. Metrics hook: capture total stage duration for now; placeholder for future TTFT once streaming enabled via `--stream` flag.
Legacy Support:
- If legacy backend flags provided without loader flags, continue old path but emit deprecation warning.

### 21.9 Color Narrator Adaptation Plan
Analogous single flag: `--vlm-registry-model` (or config key). If present:
- Perform selection (require vision capability) → set `base_url` and `model_name` before creating `VLMClient`.
- Deprecate direct `vlm_backend`, `vlm_base_url`, `vlm_model` (warn if used with new flag absent).

### 21.10 Streaming & Metrics Future Hook
Add `--stream` flag in both modules (no-op initially). When implemented:
- Use streaming API to timestamp first token (TTFT) and final token, compute throughput = (tokens_generated-1)/(t_done - t_first)*1000.
- Feed `RollingMetrics` with real samples.

### 21.11 Testing Strategy (Concrete Additions)
- `tests/model_loader/test_registry_load.py`: load & validate sample registry.
- `tests/model_loader/test_hashing.py`: deterministic hash outputs (fixed tiny temp files).
- `tests/model_loader/test_selection_capabilities.py`: capability mismatch raises error.
- `tests/personal_tagger/test_loader_integration.py`: mock selection returns endpoint; ensures config override applied.
- (Optional) `tests/model_loader/test_metrics.py`: rolling average computations.

### 21.12 Sequencing & Risk Mitigation
Order:
1. Registry + hashing + models (isolated, testable).
2. Loader service (pure Python API) + metrics skeleton.
3. Personal Tagger opt-in adaptation.
4. Vision probe CLI.
5. Color Narrator migration.
6. (Later) Streaming enablement & HTTP microservice facade.
Risks & Mitigations:
- Cold start outlier: future flag to exclude first sample.
- Dual path complexity: isolate legacy logic under clear `if legacy_mode:` with TODO removal comments.
- Capability drift: loader enforces; add CI test enumerating required vision models.

### 21.13 Immediate Next Actions (Actionable Checklist)
- [ ] Create `configs/model_registry.json` sample with one vLLM model.
- [ ] Implement `model_loader` package skeleton (models, registry, hashing, service stubs).
- [ ] Add `RollingMetrics` and unit tests.
- [ ] Introduce selection helper module (`imageworks/model_selection.py`).
- [ ] Wire Personal Tagger optional loader mode + flags.
- [ ] Draft vision probe CLI (non-streaming).

### 21.14 Summary
This plan establishes a minimally invasive deterministic selection path while preserving existing workflows. By landing the registry + loader skeleton first, downstream modules migrate incrementally, reducing risk and enabling early feedback before streaming and full metrics are introduced.

### 21.15 Role-Based Selection (Cross-Reference)
Personal Tagger now supports dynamic role-based resolution via `--use-registry` and role flags (`--caption-role`, `--keyword-role`, `--description-role`). Each role maps to the first non-deprecated registry entry advertising that role and required capabilities (vision). This indirection allows changing recommended models centrally by editing `configs/model_registry.json` without modifying deployment scripts. See `../domains/personal-tagger/model-registry.md` section 11 for full details.

### 21.15 Implemented CLI Flags & Utilities (Running Log)
Implemented (Phase 1 skeleton):
- Personal Tagger: `--use-loader`, `--caption-registry-model`, `--keyword-registry-model`, `--description-registry-model` (currently resolve and log selections; runtime config mutation staged for full integration step).
- Color Narrator: `--vlm-registry-model` (overrides backend/base-url/model when selection succeeds).
- Vision Probe: `run_vision_probe(model_name, image_path)` persists JSON under `outputs/probes/<model>/`.
- Metrics: `RollingMetrics` for per-request samples (TTFT placeholder) and `BatchRunMetrics` for batch/stage aggregation including model load time.
Pending wiring:
- Update runners to emit batch metrics JSON automatically.
- Mutate tagger runtime config with selected endpoint/model IDs (currently only logged).
- Expose a CLI wrapper for `run_vision_probe` (future `imageworks-models probe-vision`).
