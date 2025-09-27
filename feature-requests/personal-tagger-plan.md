# Implementation Plan: Personal Tagger Module

## Goal
Deliver a CLI module that generates Lightroom-compatible keywords (~15), accessibility captions, and rich descriptive text for personal library images, writing results directly to metadata or XMP sidecars alongside JSON audit logs.

## Why it matters
- Saves hours of manual tagging/captioning across large personal archives.
- Provides consistent descriptions for accessibility and future AI workflows (text-to-image prompts).
- Reuses shared model infrastructure, showcasing flexible backend support.

## Scope
- Ingest RAW (ORF/CR2/CR3), TIFF, and JPEG files from user-configured directories.
- Run specialised caption/tag/description inference (leveraging prototype prompts/models).
- Write outputs to metadata (JPEG) or XMP sidecars (RAW/TIFF).
- Emit console summary + JSON report; no GUI integration yet.

## Requirements & Acceptance
- **R1:** Process mixed-format images, generating keywords, caption, and descriptive text per file.
  **AC1:** Sample run produces all three text outputs for each image in JSON.
- **R2:** Persist metadata in Lightroom-visible fields (keywords, IPTC caption, etc.).
  **AC2:** Lightroom reads generated metadata for RAW/TIFF via sidecar and JPEG in-place edits.
- **R3:** Support configurable model/backend (vLLM/LMDeploy/etc.) and prompt selection.
  **AC3:** Changing config swaps prompt/model without code changes.
- **R4:** Integrate prototype prompts/models as defaults; allow extension.
  **AC4:** Default outputs match quality of prototype for provided test set.
- **R5:** Provide dry-run mode and logging for failures/backups.
  **AC5:** Dry-run writes JSON only; metadata unchanged.

## Interfaces & Data
- **Inputs:**
  - Config/CLI: source directories, output paths, backend/model selection, prompt template.
  - Image files: ORF, CR2/CR3, TIFF, JPEG (with optional embedded or sidecar JPEG).
- **Outputs:**
  - Metadata updates: JPEG embedded IPTC/XMP keywords/caption/description; RAW/TIFF `.xmp` sidecars with same fields.
  - JSON log (e.g., `outputs/results/personal_tagger.jsonl`).
  - Console summary (counts, sample previews).
- **Contracts:** JSON record `{ "image": path, "keywords": [..], "caption": str, "description": str, "metadata_written": bool, "notes": str }`.

## Design Sketch
- Build ingestion layer reusing ImageWorks data loader patterns but geared to personal directories.
- Use shared VLM backend abstraction (post multi-backend work) with prompts from prototype repo.
- Implement model orchestration to gather keywords (e.g., tag-specific prompt), caption, and long description sequentially or via single multi-output prompt.
- Metadata writer: extend existing XMP writer (reuse from narrator) for RAW/TIFF; leverage JPEG IPTC/XMP embedding helper.
- CLI command (`imageworks-personal-tagger`) with config defaults; dry-run option.
- JSON logger + console summary akin to narrator/mono.

## Tasks & Rough Sizes
1. **T1 (S):** Define CLI/config schema (directories, backend, dry-run, prompt selections).
2. **T2 (M):** Implement ingestion + data loader for mixed file types with optional embedded previews.
3. **T3 (M):** Port/adapt prototype prompts/models; integrate with backend abstraction.
4. **T4 (M):** Implement inference pipeline (keywords/caption/description) with configurable prompts.
5. **T5 (M):** Extend metadata writer for JPEG in-place + RAW/TIFF XMP sidecars; include backups.
6. **T6 (S):** Generate JSON logging and console/Markdown summary (optional markdown if needed later).
7. **T7 (M):** Tests: unit (prompt selection, metadata writes), integration on sample personal library set.
8. **T8 (S):** Documentation (README section, workflow guide) referencing prototype learnings.

## Risks & Mitigations
- **Metadata field mismatch:** Confirm Lightroom field mappings (keywords, IPTC caption, description) with small test set; provide config for custom fields.
- **Model quality variance:** Keep prompts configurable; support fallback to deterministic tagging if VLM unavailable.
- **Large batch performance:** Provide batching + rate-limit options; allow dry-run for quick audits.

## Savepoint
- **Current status:** Requirements captured; implementation pending.
- **Next action:** Branch `feature/personal-tagger` and set up CLI/config skeleton (T1).
- **Branch:** _feature/personal-tagger_ (to create).
- **Files in flight:** None yet.
- **Open questions:** Exact metadata fields/lightroom mapping, which prototype models to bundle, caching for repeated runs.
