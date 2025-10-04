# Feature Request: Competition Image Similarity Checker

## 0) TL;DR (1–2 sentences)
Detect when a newly submitted competition photo closely matches any historic submission, flagging exact repeats and near-duplicates (burst variants, toning tweaks) before other checks run.

---

## 1) Why / Outcome
- **Why now:** Manual review of decade-long competition archives is impractical; organisers need automation to enforce the "one submission per image" rule.
- **Definition of success:** Each new entry produces a similarity report in Markdown/JSONL highlighting the top prior matches with clear scores and reasoning.

---

## 2) Scope
- **In scope:** Similarity engine, CLI/Markdown/JSONL reporting, reuse of Lightroom image folders, integration entry-point before mono checker.
- **Out of scope:** GUI tooling, long-term storage/indexing service, automated disqualification workflows.

---

## 3) Requirements
### 3.1 Functional
- [ ] F1 — Analyse a candidate image against the competition library and emit ranked similarity results.
- [ ] F2 — Support multiple similarity modes (exact match, near-duplicate in time, tonal/processing variants).
- [ ] F3 — Produce human-readable Markdown + terminal summary + machine-readable JSONL akin to mono/narrator outputs.
- [ ] F4 — Integrate with existing folder defaults / Lightroom workflow; usable as first step in competition pipeline.
- [ ] F5 — Configurable thresholds to label results as PASS / QUERY / FAIL.

### 3.2 Non-functional (brief)
- **Performance:** Handle batch submissions on RTX 4080 (15 GB) up to RTX 6000 Pro (96 GB) with CUDA ≥ 12.8; reasonable latency for single-image checks.
- **Reliability / failure handling:** Graceful handling of missing historical assets or corrupt files; log actionable errors.
- **Compatibility:** Runs in existing ImageWorks environment; may mix deterministic vision + AI/VLM models.
- **Persistence / metadata:** Outputs stored alongside other ImageWorks results (JSONL + Markdown); no new database required.

**Acceptance Criteria (testable)**
- [ ] AC1 — CLI run on sample set generates Markdown + JSONL with top-N matches and similarity labels.
- [ ] AC2 — Exact duplicate images are flagged as FAIL with similarity score above configured threshold.
- [ ] AC3 — Tonal/burst variants appear as QUERY with descriptive rationale.
- [ ] AC4 — Integration pipeline can call similarity checker before mono and receive pass/fail status.
- [ ] AC5 — Fallback behaviour documented when GPU resources unavailable (e.g., CPU path or clear error).

---

## 4) Effort & Priority (quick gut check)
- **Effort guess:** Large (multiple dev-days; includes research/spike on similarity approach).
- **Priority / sequencing notes:** Precedes mono/narrator in submission workflow; unlocks automated compliance checks.

---

## 5) Open Questions / Notes
- Which similarity technique balances accuracy vs. speed (feature embeddings vs. perceptual hash vs. VLM)?
- How many historic images need to be scanned per run; should we pre-index embeddings?
- Where to cache embeddings/features for future re-use?

---

## 6) Agent Helper Prompt
"Read this feature request. Ask me concise questions about intended similarity metrics, data volumes, and how results should integrate with the submission pipeline before proposing designs. Then suggest missing requirements or risks."

---

## 7) Links (optional)
- **Specs / docs:** TBD
- **Related issues / PRs:** —
- **External refs:** perceptual hashing, CLIP / image embedding search guides
