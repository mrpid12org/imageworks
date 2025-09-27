# Implementation Plan: Pluggable VLM Backends

## Goal
Create a reusable VLM client abstraction that supports vLLM, LMDeploy, and TensorRT-LLM/Triton backends via the OpenAI-compatible API, so any ImageWorks module can switch backends through configuration.

## Why it matters
- Lets us pick the best backend for production throughput vs. experimentation.
- Shields downstream modules from model-serving churn.
- Establishes a shared serving layer for narrator, personal tagger, and future tooling.

## Scope
- Refactor VLM client into a backend-agnostic interface.
- Implement backend selector (config + CLI).
- Provide concrete clients for vLLM (existing), LMDeploy (new), and a minimal TensorRT-LLM/Triton stub.
- Update tests/documentation.
- No changes to downstream business logic beyond selecting the backend.

## Requirements & Acceptance
- **R1:** Backend chosen via `pyproject.toml` and CLI flag (default vLLM).
  **AC1:** CLI run with `--vlm-backend lmdeploy` succeeds on sample assets.
- **R2:** LMDeploy client implements OpenAI-compatible health + inference.
  **AC2:** Mock LMDeploy test passes using our harness.
- **R3:** TensorRT-LLM/Triton stub supports health checks and emits helpful guidance.
  **AC3:** `--vlm-backend triton` logs stub warning and exits with actionable message.
- **R4:** Factory returns the correct client type, logging backend name.
  **AC4:** Unit test verifies factory mapping and CLI outputs backend info.
- **R5:** Docs updated with setup steps for each backend.
  **AC5:** README/guide include backend table and example commands.

## Interfaces & Data
- **Inputs:**
  - `pyproject.toml` keys (e.g., `vlm_backend`, `vlm_base_url`, `vlm_model`, `vlm_timeout`).
  - CLI flags: `--vlm-backend`, `--vlm-base-url`, `--vlm-model`, `--vlm-timeout`.
- **Outputs:** Existing narration outputs unchanged; log line `🤖 VLM: <backend>/<model>` updated to show backend.
- **Contracts:**
  - Backend enum (e.g., `VLMBackend = Enum("vllm", "lmdeploy", "triton")`).
  - Abstract client interface: `health_check() -> bool`, `infer_single(request: VLMRequest) -> VLMResponse`.
  - OpenAI message payload unchanged (images + prompt).

## Design Sketch
1. Introduce `VLMBackend` enum + factory in `core.vlm`.
2. Refactor existing `VLMClient` into `BaseVLMClient`; move current logic to `VLLMClient`.
3. Implement `LMDeployClient` using requests.Session with OpenAI format (reuse base helpers).
4. Implement `TritonClient` with: health check (HTTP GET), inference placeholder raising informative error (TODO for future real support).
5. Update CLI/narrator to instantiate via factory, log backend, pass through config defaults.
6. Provide config validation + helpful errors (unknown backend, missing URL).

## Tasks (with size)
1. **Backend plumbing:** Add enum, config parsing, CLI flag, logging  S.
2. **Client refactor:** Extract base class, move vLLM implementation  M.
3. **LMDeploy client:** Implement health/infer, reuse base request builder  M.
4. **Triton stub:** Health check + informative error placeholder  S.
5. **Tests:** Unit tests for factory + clients, integration with mock LMDeploy server  M.
6. **Docs:** Update README, color-narrator reference, and configuration docs  S.

## Risks & Mitigations
- **API incompatibility:** LMDeploy/Triton may diverge from OpenAI spec → keep request builder centralized; add integration test with mock server.
- **Regression in narrator:** Ensure default remains vLLM; run existing narrator tests.
- **Incomplete Triton support:** Document stub status; plan follow-up once real endpoint accessible.

## Savepoint / Resume
- **Current status:** Plan drafted; implementation pending.
- **Next action:** Branch off main (`git checkout -b feature/vlm-backends`) and implement Task 1 (backend plumbing).
- **Branch:** _feature/vlm-backends_ (to be created).
- **Files in flight:** None yet.
- **Open questions:** Confirm Triton endpoint expectations, credential handling, and whether to centralize request session configuration.
