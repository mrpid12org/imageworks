# Implementation Plan: Competition Image Similarity Checker

## Goal
Automatically flag new competition entries that are identical or near-duplicates of historical submissions, producing Markdown/JSONL reports and pass/query/fail status.

## Why it matters
- Enforces the "one submission per image" rule without manual trawling through archives.
- Provides early warning before running heavier analyses (mono, narrator).
- Builds a reusable similarity backbone for future ImageWorks modules.

## Scope
- Similarity analysis engine (feature extraction, comparison, thresholds).
- CLI and report generation matching ImageWorks style.
- Configurable integration point before mono checker.
- Reuse existing Lightroom folder conventions.

## Requirements & Acceptance
- **R1:** Compare candidate image against competition library, returning ranked matches.
  **AC1:** CLI run outputs Markdown + JSONL listing top matches with scores.
- **R2:** Detect exact duplicates and near duplicates (burst variants, tonal edits, new crops).
  **AC2:** Exact match flagged as FAIL; tonal variant flagged as QUERY in sample suite.
- **R3:** Configurable thresholds + labels (pass/query/fail).
  **AC3:** Adjusting threshold affects classification without code changes.
- **R4:** Integration hook for pipeline (e.g., call from CLI before mono).
  **AC4:** Pipeline demo shows similarity check invoked first, returning status code for next step.
- **R5:** Operates on RTX 4080 / RTX 6000 Pro with CUDA ≥ 12.8; clear error if GPU unavailable.
  **AC5:** Fallback message recorded in logs when resources insufficient.

## Interfaces & Data
- **Inputs:**
  - Candidate image path(s) (CLI arg or config default).
  - Historical library root (existing competition directories).
  - Optional cached embedding/index store (future extension).
- **Outputs:**
  - `outputs/results/similarity_results.jsonl` (configurable path).
  - `outputs/summaries/similarity_summary.md` + terminal summary.
  - Exit code / status for pipeline (e.g., 0 pass, 10 query, 20 fail).
- **Contracts:** JSONL record per candidate: `{ "candidate": str, "match": str, "score": float, "label": "pass|query|fail", "notes": str }`.

## Design Sketch
- Use image embedding model (e.g., CLIP or existing VLM) to compute features; fallback to perceptual hash for quick wins.
- Optionally cache embeddings for historical library to speed repeated runs (store in JSON/NPZ).
- Compute similarity (cosine distance or hash difference) and apply configurable thresholds for PASS/QUERY/FAIL.
- Generate reports (Markdown table, JSONL records, console summary).
- Provide CLI command `imageworks-similarity check` (or integrate into main CLI) with config fallback paths.

## Tasks & Sizes
1. **T1 (S):** Define CLI command, config keys, and output layout.
2. **T2 (M):** Research/select embedding/hash approach; prototype on sample images.
3. **T3 (M):** Implement feature extraction + similarity scoring module with caching support.
4. **T4 (S):** Implement thresholding + classification logic (pass/query/fail).
5. **T5 (S):** Generate Markdown/JSONL/console reports.
6. **T6 (M):** Integration tests using sample production images (exact match, tonal variant).
7. **T7 (S):** Documentation update (README, workflow guides) and usage examples.

## Risks & Mitigations
- **Embedding accuracy vs. speed:** Start with CLIP embeddings + optional perceptual hash fallback; allow configuration.
- **Large library performance:** Introduce caching + chunked search; consider FAISS/Faiss-lite later.
- **GPU availability:** Provide CPU fallback or clear error; document requirements.

## Savepoint
- **Current status:** Requirements + plan captured; implementation not started.
- **Next action:** Create branch `feature/image-similarity-checker` and spike embedding approach (T2).
- **Branch:** _feature/image-similarity-checker_ (to create).
- **Files in flight:** None yet.
- **Open questions:** Decide on embedding model, storage format for cached features, and CLI naming.
