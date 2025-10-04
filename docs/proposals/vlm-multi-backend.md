# Feature Request: Pluggable VLM Backends for ImageWorks Suite

## 0) TL;DR (1–2 sentences)
Enable ImageWorks modules (starting with Color Narrator) to switch between multiple OpenAI-compatible VLM backends—vLLM, LMDeploy, and TensorRT-LLM/Triton—without changing business logic.

---

## 1) Why / Outcome
- **Why now:** Need flexibility to choose fast vs. experimental serving stacks, cope with model architecture churn, and share the same backend across future ImageWorks tools.
- **Definition of success:** Narrator (and other modules) connect to any configured backend via OpenAI-style endpoints, with no code changes beyond config/flags.

---

## 2) Scope
- **In scope:** VLM client refactor, backend selection via config/CLI, vLLM + LMDeploy implementation, TensorRT-LLM/Triton stub, tests, docs.
- **Out of scope:** Model training, deployment automation, aggressive performance tuning.

---

## 3) Requirements
### 3.1 Functional
- [ ] F1 — Backend chosen via config/CLI; defaults to vLLM.
- [ ] F2 — LMDeploy backend implemented with OpenAI-compatible API.
- [ ] F3 — TensorRT-LLM/Triton stub (health + infer placeholder).
- [ ] F4 — CLI surfaces backend selection and handles connection errors gracefully.
- [ ] F5 — Tests cover backend selection and routing.

### 3.2 Non-functional (brief)
- **Performance:** Must support RTX 4080 (~15 GB VRAM) and RTX 6000 Pro (96 GB), CUDA ≥ 12.8; baseline throughput comparable to existing vLLM path.
- **Reliability / failure handling:** Clear errors when backend unreachable; fallback instructions.
- **Compatibility:** OpenAI-compatible protocol; reusable across ImageWorks modules.
- **Persistence / metadata:** Existing outputs (XMP/JSON) unchanged.

**Acceptance Criteria (testable)**
- [ ] AC1 — Setting `vlm_backend=lmdeploy` processes sample images successfully.
- [ ] AC2 — `--vlm-backend triton` logs stub warning and fails with actionable message if server absent.
- [ ] AC3 — Unit test ensures backend factory returns correct client classes.
- [ ] AC4 — CLI help documents backend options/default.
- [ ] AC5 — Docs explain setup and usage per backend.

---

## 4) Effort & Priority (quick gut check)
- **Effort guess:** Medium (~2–3 dev days).
- **Priority / sequencing notes:** Needed before expanding narrator/personal tagger experimentation; unblock benchmarking different serving stacks.

---

## 5) Open Questions / Notes
- How detailed should the initial Triton/TensorRT integration be (stub vs. full support)?
- Where to store backend-specific credentials/configs (pyproject vs. env)?

---

## 6) Agent Helper Prompt
“Read this issue. Ask me to clarify intent, dependencies, and fit with existing ImageWorks components before proposing solutions. Then suggest any missing requirements or risks.”

---

## 7) Links (optional)
- **Specs / docs:** TBD
- **Related issues / PRs:** —
- **External refs:** NVIDIA TensorRT-LLM docs, LMDeploy OpenAI proxy docs
