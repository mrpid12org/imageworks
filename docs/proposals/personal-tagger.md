# Feature Request: Personal Tagger Module

## 0) TL;DR (1–2 sentences)
Automate keyword tagging, accessibility captions, and rich descriptive text for personal Lightroom libraries, writing results into image metadata (JPEG) or XMP sidecars (RAW/TIFF).

---

## 1) Why / Outcome
- **Why now:** Manual tagging/captioning across thousands of personal images is time-consuming; a consistent automated assistant improves searchability and accessibility.
- **Definition of success:** For each processed image, Lightroom-visible keywords, caption, and long-form description are written to metadata, with a JSON summary for auditing.

---

## 2) Scope
- **In scope:** Keyword generation (~15 unique tags), short caption (1–2 sentences), long descriptive prompt/inference text; metadata writing for JPEG vs. RAW/TIFF; CLI summaries + JSON log; reuse of existing models/prompts where sensible.
- **Out of scope:** Competition pipeline integration, GUI tooling, Lightroom plugin development, large-scale training of new models.

---

## 3) Requirements
### 3.1 Functional
- [ ] F1 — Accept personal library images (RAW/TIFF/JPEG) and emit keywords (~15), caption, and descriptive text.
- [ ] F2 — Write metadata to files: embed in JPEG, XMP sidecar for RAW/TIFF (Lightroom-readable fields).
- [ ] F3 — Output terminal summary and JSON report mirroring other ImageWorks modules.
- [ ] F4 — Allow configuration of model/backend, prompt set, and output directories separate from competition defaults.
- [ ] F5 — Reuse / adapt prompting and specialised models from existing prototype where appropriate.

### 3.2 Non-functional (brief)
- **Performance:** Operate on RTX 4080 (15 GB) through RTX 6000 Pro (96 GB), CUDA ≥ 12.8; batch-friendly but responsive for single image runs.
- **Reliability / failure handling:** Skip/write warnings when metadata write fails; preserve original files (optionally back up).
- **Compatibility:** Runs locally (WSL), handles ORF, CR2/CR3, TIFF, JPEG with companion sidecars; adheres to Lightroom keyword/caption fields.
- **Persistence / metadata:** No separate DB; metadata + JSON stored alongside images or configured output path.

**Acceptance Criteria (testable)**
- [ ] AC1 — Running the CLI on a sample mixed-format set writes keywords/caption/description visible in Lightroom.
- [ ] AC2 — RAW/TIFF files receive XMP sidecars with correct fields; JPEGs embed metadata in-place.
- [ ] AC3 — JSON output lists generated text and metadata paths for each image.
- [ ] AC4 — Switching prompts/models via config works without code edits (vLLM/LMDeploy backend reuse).
- [ ] AC5 — Prototype prompts/models integrated or documented, matching quality expectations.

---

## 4) Effort & Priority (quick gut check)
- **Effort guess:** Large (multi-stage development: ingestion, inference, metadata writing, testing).
- **Priority / sequencing notes:** Enables personal library automation; may precede more advanced Lightroom tooling.

---

## 5) Open Questions / Notes
- Which exact models (caption/tag/description) from the prototype should become defaults?
- How to manage model-specific configuration (e.g., tag taxonomy limits)?
- Do we need batch scheduling/throttling for large personal collections?

---

## 6) Agent Helper Prompt
"Read this feature request. Ask me concise questions about metadata fields, model preferences, and library sizes before proposing designs. Then suggest missing requirements or risks."

---

## 7) Links (optional)
- **Prototype repo:** https://github.com/mrpid12org/photo-tools
- **Specs / docs:** TBD
- **Related issues / PRs:** —
