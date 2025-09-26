# Photo Tools Requirements Specification

_Version 1.0 – consolidated and approved_

## Table of Contents
- [1. Scope and Objectives](#1-scope-and-objectives)
  - [1.1 Mono Checker (CCC)](#11-mono-checker-ccc)
  - [1.2 Personal Photo-Library Tagger](#12-personal-photo-library-tagger)
- [2. Source Volumes and Formats](#2-source-volumes-and-formats)
- [3. High-Level Architecture](#3-high-level-architecture)
- [4. Functional Requirements](#4-functional-requirements)
  - [4.1 Common Ingest](#41-common-ingest)
  - [4.2 Mono Checker](#42-mono-checker)
  - [4.3 Personal Photo-Library Tagger](#43-personal-photo-library-tagger)
- [5. Web Interfaces](#5-web-interfaces)
- [6. Development & Testing Facilities](#6-development--testing-facilities)
- [7. Non-Functional Requirements](#7-non-functional-requirements)
- [8. Project Phasing & Priorities](#8-project-phasing--priorities)
- [9. Acceptance Criteria](#9-acceptance-criteria)
- [Appendix A – System Context & Goals](#appendix-a--system-context--goals)
- [Appendix B – Container & Component Architecture](#appendix-b--container--component-architecture)

## 1. Scope and Objectives

Two related but distinct projects share common technology.

### 1.1 Mono Checker (CCC)
Tools to assist Cambridge Camera Club (CCC) competition officers in validating entries.

Goals:
- True monochrome check: determine if an uploaded image is genuinely single-hue (neutral or toned).
- Near-duplicate detection: flag if a submitted image has been entered before or trivially altered.
- Phase 3 goal: run near-duplicate checks directly against the live CCC database without requiring a GPU on CCC servers.

### 1.2 Personal Photo-Library Tagger
Tools to enrich and organise a personal Lightroom library.

Goals:
- Generate concise, non-destructive keywords (tags).
- Produce optional one-sentence alt-text captions and longer descriptive text.
- Find near-duplicates for burst pruning or variant edits.
- Provide semantic similarity search (“find images like this” or “find images matching this text”).

## 2. Source Volumes and Formats

### Competition
- ~200 images per competition, ~10 competitions per year, plus ~8 special-interest categories (~10k images/year).
- Historical archive: 50–100k JPEG images, 1920 × 1200 px, sRGB.
- All images supplied as JPEG with minimal metadata.

### Personal Library
- ~60k images today, growing 3–5k/year.
- Formats: Olympus/Canon RAW (ORF, CR2/CR3, ARW, DNG—including DxO/linear DNG with full RGB), full-resolution JPEG sidecars, <10% TIFF (8/16-bit).
- Lightroom previews and cache files exist but originals remain the authoritative source.

## 3. High-Level Architecture

Two projects share a common ingest and preprocessing library.

```
ingest & preprocessing → analysis engines → metadata writer
                 ↑
            SQLite/FAISS
                 ↑
     Web UI / Lightroom Plugin
```

- **Ingest & Pre-process**
  - Read JPEG, TIFF, or RAW.
  - Prefer embedded JPEG or Lightroom preview; if absent, perform a fast demosaic (never persist a new derivative).
  - Convert in memory to an inference view: sRGB, longest edge 640 px.
  - Cache in a size-capped, content-addressed transient store (auto-evicted, no permanent resized files).

- **Analysis Engines**
  - Tagger/VLM (RAM, wd-vit, LLaVA, Qwen-VL, Florence-2 etc.).
  - Caption/description generation.
  - pHash/dHash for perceptual hashing.
  - Semantic embeddings (OpenCLIP or DINOv2) stored in FAISS.
  - Monochrome detector: channel-diff + hue-variance.

- **Metadata Writer**
  - Uses ExifTool or equivalent.
  - Never alters colour profiles of master files.
  - RAW: XMP sidecars; JPEG/TIFF: in-file XMP/IPTC.
  - Custom namespace `XMP:AutoTag:*` for internal fields.

- **Databases**
  - SQLite: images, proposals, run metadata, failures.
  - FAISS: semantic embedding index.
  - No persistent storage of resized images.

## 4. Functional Requirements

### 4.1 Common Ingest
- Accept JPEG, TIFF (8/16-bit), RAW + sidecar.
- Honour embedded ICC profile; do not change colour profiles in originals.
- Create transient inference view for analysis; optionally use Lightroom previews if accessible.

### 4.2 Mono Checker

**Phase 1 – True Monochrome**
- Analyse newly downloaded CCC submissions.
- Verdict per image:
  - Pass – channel max-diff ≤ 2 (8-bit).
  - Pass-with-Query – hue variance slightly above threshold.
  - Fail – clearly multi-colour.

**Phase 2 – Local Near-Duplicate Search**
- Build FAISS index of historical CCC images (one-time job).
- For new submissions: pHash prefilter → FAISS query (cosine ≥ 0.97) → optional SIFT/ORB verification.

**Phase 3 – Live CCC Integration (future)**
- Precompute pHash + embeddings server-side so CPU-only queries suffice.
- Provide API for CCC site to check new uploads against stored hashes.

### 4.3 Personal Photo-Library Tagger
- **Bulk Tagging**
  - Produce 10–15 keywords, each 1–3 words, lower-case.
  - Avoid plurals or near-synonyms (use lemmatiser or synonym DB).
  - Never overwrite existing user tags.

- **Captions & Long Descriptions**
  - Optional one-sentence alt-text (≤ 25 words).
  - Optional long description (≤ 250 words) in Cooper-Hewitt style.

- **Similarity Search**
  - Mode A: “find images like this one.”
  - Mode B: “find images matching this text description.”
  - Results shown inside Lightroom (plugin preferred).

- **Near-Duplicate Finder**
  - Support burst pruning or different edits of the same shot.

## 5. Web Interfaces

### 5.1 Shared Web UI
- Tabs:
  - Duplicates Review – query vs candidates, similarity metrics, accept/reject.
  - Tag QA – image + existing vs proposed tags, accept/edit/reject.
  - Monochrome Check – verdict and histograms.
  - Failed Images – decode/profile errors, retry.
- Keyboard-centric hotkeys for rapid review.

### 5.2 Lightroom Plugin (Phase 5)
- Allows:
  - Selecting images or collections.
  - Running tagging/caption jobs.
  - Previewing proposals, editing, committing or rejecting.
  - Triggering similarity searches and showing results in Lightroom’s grid.

## 6. Development & Testing Facilities

- **DB-only proposal storage**
  - Generated tags/captions stored only in SQLite until approved.
  - Nothing written to XMP until commit.

- **Commit/Revert logic**
  - Track `existing_keywords`, `added_keywords`, `run_id`.
  - Commit: add only `added_keywords`.
  - Revert: remove only `added_keywords` (never touch pre-existing tags).

- **Prompt/Model Test Harness**
  - CLI + Web UI to run experiments with different prompts/models.
  - Metrics: concision, relevance, inference time.

- No file-level backups needed during development; the master Lightroom library remains the source of truth.

## 7. Non-Functional Requirements

- **Performance**
  - Mono check: ≥ 500 images/min on RTX 4080.
  - Bulk tagging: 60k images in < 4 h using fast tagger.
  - Duplicate search: p95 < 100 ms per query against 100k images.

- **Portability & Deployment**
  - Windows 11 with WSL2 Ubuntu recommended; Docker optional.
  - GPU auto-detection; support 4-/8-bit quantised models when VRAM is tight.

- **Privacy & Safety**
  - Processing is local by default; cloud use requires explicit opt-in.

- **Logging & Retention**
  - Minimal CSV log for CCC mono/duplicate outcomes if required.
  - No long-term audit trail unless requested later.

## 8. Project Phasing & Priorities

| Phase | Deliverable | Target User |
|-------|-------------|-------------|
| P1 (Must) | CCC True-Monochrome checker + Web UI (including Failed Images tab) | Competition judges |
| P2 (Must) | Personal bulk tagging + optional captions | Personal |
| P3 (Should) | Personal long descriptions | Personal |
| P4 (Should) | CCC near-duplicate detection (local archive) | Competition judges |
| P5 (Could) | Lightroom plugin for tagging, captions, similarity search | Personal |
| P6 (Could) | CCC live-site duplicate integration (GPU-free) | Competition judges |

## 9. Acceptance Criteria

- Mono check accuracy: ≤ 1% false negatives, ≤ 2% false positives on labelled set.
- Tagging: ≥ 90% of images yield 10–15 non-repetitive tags, median length ≤ 2 words.
- Bulk run of 60k images completes in < 4 h on RTX 4080.
- Duplicate recall ≥ 95% at false-positive rate ≤ 2% using cosine ≥ 0.97 + pHash filter.
- Successful Lightroom round-trip: keywords/captions visible without corrupting colour profile or existing metadata.

---

## Appendix A – System Context & Goals

_Version 0.2 (draft)_

### Purpose
Imageworks is a local-first application for:
- Competition checking: verify monochrome entries and detect near-duplicates.
- Personal library tagging: generate concise tags, optional captions/long descriptions, and support similarity searches for the Lightroom library.

The emphasis is on privacy (local processing), speed, reproducibility, and a path to CPU-only operation on CCC infrastructure.

### Primary Actors

| Actor | Role |
|-------|------|
| Operator (Stewa) | Runs CLI/Web UI, reviews results, configures models and thresholds. |
| Competition Officers / Judges | (Later) run qualification checks before judging on CCC systems. |
| External Systems | Lightroom Classic (catalogue & XMP), CCC competition site/database (later integration). |

### External Environment
- Host: Windows 11 with WSL2 Ubuntu for primary execution; optional Docker.
- Hardware (local): RTX 4080 now; RTX 6000 Pro later.
- Hardware (CCC): typical server without a GPU (assume CPU-only).
- Data sources:
  - Personal Lightroom library (~60k images; RAW+JPEG+TIFF; +3–5k/year).
  - CCC entries (~10k/year) and historical archive (~50–100k JPEGs).

### System Context (technology-neutral)

```
┌────────────────────────────┐
│  CCC Website / DB          │  (future integration)
└────────────┬───────────────┘
             │
             ▼
┌──────────────┐    ┌────────────────────────────────┐
│ Lightroom    │    │ IMAGEWORKS                      │
│ Catalogue    │───▶│ - CLI / Web UI                 │
│ & Files      │    │ - Ingestion & Analysis         │
└──────────────┘    │ - Local persistence layer (*)  │
                    └────────────────────────────────┘
                             ▲
                             │
                      ┌──────┴───────┐
                      │   Operator   │
                      └──────────────┘

(*) Local persistence layer = storage for job state, proposals, and similarity indexes. Specific technologies are not decided here.
```

### Key Data Flows – Personal Library
1. **Selection** – operator selects a subset in Lightroom (rating/flag/collection) or folders directly; selection is passed to Imageworks.
2. **Ingestion & Preprocessing** – build inference view per image (sRGB, resized in memory; no permanent derivatives). RAW sources follow preference order: embedded JPEG preview, full-resolution sidecar JPEG, fast demosaic fallback.
3. **Analysis** – tag/caption generation; optional long description; similarity or near-duplicate search on demand.
4. **Review & Commit** – operator reviews proposed tags/captions (no in-file changes yet). On commit, write to XMP/IPTC while preserving existing tags.

### Key Data Flows – Competition Entries
1. **Acquisition** – operator downloads current competition JPEGs and summary sheet.
2. **Qualification** – true-monochrome check → pass / pass-with-query / fail; near-duplicate screening against local archive (initially).
3. **CCC Integration (later)** – new entries checked on CCC systems using pre-computed metadata/indexes; must operate CPU-only.

#### Monochrome Checker – Interpretation Guide

The checker labels each JPEG as `PASS`, `PASS (query)`, or `FAIL`. It measures LAB chroma and hue variation to decide whether colour is both strong and multi-hued (FIAP/PSA definition of colour) or simply a single tint.

For a detailed, step-by-step explanation of the pass/fail logic, including thresholds and edge cases, please see the standalone [Monochrome Technical Documentation](../monochrome-technical.md).

This sequence mirrors the competition definition: pure greyscale and single-hue toning pass; dual-hue or colour-on-grey mixtures fail; borderline/low-footprint cases require human review.

**Key metrics in the summary**

 Key metrics in the summary:
- **Chroma (C*)** - distance from neutral in the LAB plane (`C* = sqrt(a*^2 + b*^2)`). Small values mean almost no tint; higher values flag stronger colour.
- **Hue spread** - circular standard deviation of hue in degrees. Think “how wide is the tint fan?” Small spreads mean a single tint; large spreads mean multiple hue families, which is the first warning sign of split-toning.
- **Chroma percentiles** - C* max and C*99 show how saturated the brightest regions are. Anything above about 3 is plainly visible.
- **Chroma footprint (pct > C*2 / pct > C*4)** - share of pixels whose chroma exceeds 2 or 4 units. The C*2 percentage counts faint colour; the C*4 percentage counts clearly visible colour. Together they tell us whether colour is confined to speckles or covers meaningful areas of the frame.
- **Largest chroma cluster** - largest smoothed connected region above C*2/C*4 (after a small morphological close), reported as the share of high-chroma pixels within that region. A large cluster highlights concentrated edits (e.g., colour introduced by cloning or selective adjustments) even when the overall footprint is small.
- **Hue drift** - how the average hue shifts as brightness changes (degrees per L*). This reveals when shadows and highlights carry different colours: low drift means the tint stays consistent, while high drift (especially with a wide hue spread) is the hallmark of split-toning or false-colour rendering.

By default the checker only forces a FAIL when the strong-colour footprint exceeds roughly 10 % of the frame (`lab_fail_c4_ratio`) or any single cluster crosses 8 % (`lab_fail_c4_cluster`) **and** the hue spread sits outside the toned limit. These guard rails, along with the uniform-toned exception above, can be tuned per session: `uv run imageworks-mono check --lab-fail-c4-ratio 0.05` for example. Use `uv run imageworks-mono --help` to see every CLI switch and default.

Visual aids:
- `lab_chroma` overlays mark where chroma magnitude is high, even for faint casts.
- `lab_residual` overlays colour the hue families so opposing tones become obvious.
- Overlay files store the fail or query summary in their metadata description so Lightroom shows the reason alongside the preview.

Infrared entries:
- False-colour IR captures often contain two complementary lobes even when the print looks mono. Treat them like any other query: inspect overlays and use B&W/Color Grading toggles in Lightroom. You can raise the hard-fail thresholds temporarily (`--lab-fail-c4-ratio`) if you want more leniency for a dedicated IR judging session.

### Non-Functional Goals
- Local-only by default; explicit opt-in for any network use.
- Performance targets (local workstation):
  - Mono check throughput suitable for 200+ images per batch.
  - Initial bulk tagging of ~50k images within a few hours (with GPU when available).
- Portability: Windows 11 + WSL2; optional Docker.
- Reproducibility: record model/version, thresholds, and config with each run.
- CCC deployment constraint: CPU-only operation on CCC servers for monochrome and near-duplicate checks in later phases.

### Open Questions
1. Lightroom selection hand-off: preferred mechanism—plugin invocation, exported list of file paths, or both?
2. RAW choice policy: default order among (embedded JPEG / sidecar JPEG / full RAW) and genre-specific overrides.
3. Local persistence layer: data to store (proposals, runs, embeddings, hashes, config), expected size, backup/restore needs—before choosing a storage technology.
4. CCC integration path: batch export from CCC vs live API; update cadence for pre-computed indexes.
5. Accessibility & review UX: minimal Web UI vs Lightroom-native plugin, and timeline for each.

If this version aligns with intent, the next artefact will be the Container & Component Architecture (v0.1): still solution-agnostic where possible but concrete enough to drive later tech choices.

---

## Appendix B – Container & Component Architecture

_Version 0.3 (draft)_

### Overview
Two applications share a common core, plus a separate Tagging/Description Service and a Prompt & Model Lab to support experimentation and future web integration.

```
+--------------------------------------------------------------+
|                            IMAGEWORKS                        |
|--------------------------------------------------------------|
| Shared Core Services                                         |
|  • Image Ingestion & Pre-processing                          |
|  • Colour/Mono Analysis                                      |
|  • Perceptual Hashing & Embedding Generation                 |
|  • Similarity Matching                                       |
|  • Metadata Writer (XMP/IPTC)                                |
|  • Persistence Layer (jobs, proposals, configs, indexes)     |
|--------------------------------------------------------------|
| Applications & Auxiliary Services                            |
|  ┌──────────────────┐ ┌─────────────────────────┐ ┌─────┐   |
|  │ Competition      │ │ Personal Tagging        │ │Lab │   |
|  │ Checker (WebUI)  │ │ & Similarity (CLI/Web)  │ │(R&D)│   |
|  └──────────────────┘ └─────────────────────────┘ └─────┘   |
+--------------------------------------------------------------+
```

### Containers (deployable units)

| Container | Purpose | Deployment Notes |
|-----------|---------|------------------|
| Web UI / API – Mono Checker | Minimal browser UI for CCC officers/admins to run mono checks and near-duplicate screening on manually downloaded images (short/medium term). | Local machine initially; later can be hosted on CCC infrastructure. No CLI expected for CCC users. |
| Web UI / API – Personal Review | Optional browser UI for reviewing proposed tags/captions, similarity/duplicate results, and failures. | Local workstation; complements CLI. |
| CLI (Personal only) | Power-user entry point for bulk runs, automation, scripting (tag propose/commit/revert, build/search indexes). | Local WSL/Ubuntu. Not aimed at CCC officers. |
| Core Processing Engine | Library/workers used by Web UIs and CLI: ingestion, mono, hashing/embeddings, similarity matcher, XMP write. | Scales from CPU-only to GPU. |
| Tagging/Description Service (separated) | Dedicated service boundary for generating tags, short captions, long descriptions; enables rapid model iteration and web integration. | Runs locally; can be replaced with cloud/offline variant later. |
| Prompt & Model Lab | Environment for running experiments, comparing prompts/models, capturing metrics, storing proposals. | Local workstation or isolated staging box. |
| Persistence Layer | SQLite/FAISS-based storage for images, proposals, embeddings, run metadata, configs, failures. | Shared by services; pluggable backend later. |

### Collaborations (logical view)

| Interaction | Participants | Notes |
|-------------|--------------|-------|
| Competition Web UI ↔ Core Engine | Requests monochrome verdicts, pHash lookup, duplicate candidates; displays results to CCC officers. | Requires CPU-only fallback. |
| Personal CLI/Web ↔ Tagging Service | Issues tagging/caption jobs, retrieves proposals, commits to metadata writer. | Supports batching and asynchronous jobs. |
| Prompt & Model Lab ↔ Persistence | Writes experimental runs, captures metrics, exports summaries. | Keeps outputs DB-only until approved. |
| CLI (Personal) ↔ Services | Bulk runs, automation, scripting; not intended for CCC officers. |  |

### Special Design Emphases
1. **Web-first for CCC** – CCC officers should not need a CLI; initial UI is minimal with long-term integration into the competition site.
2. **Separated Tagging/Description Service** – Provides a clean boundary to swap models/prompts without changing ingestion or UI layers.
3. **Prompt & Model Lab (Test Harness)** – Enables rapid prompt/model experimentation, side-by-side comparisons, and per-run documentation. Stores DB-only proposals until committed.
4. **CPU-only Deployment Path** – Every CCC-side component (mono check, hashing, embeddings, index search) must run without a GPU for server deployment.

### Deployment Scenarios

| Scenario | Components | Notes |
|----------|------------|-------|
| **A) Local Personal Use** | Personal Web UI (optional), CLI, Core Engine, Tagging Service, Persistence. | GPU acceleration when available; Lightroom provides selections now, plugin later. |
| **B) CCC Officer Workstation (short/med term)** | Competition Web UI, Core Engine (CPU ok), Tagging Service (optional), Persistence. | Input: manually downloaded entries; output: mono verdicts and duplicate candidates. |
| **C) CCC Server (endgame)** | Integrated Checker Service (CPU-only), Persistence, minimal Admin UI. | Input: direct from CCC site/database; must handle typical batch sizes promptly. |

### Updated Open Design Questions
1. Lightroom selection – exported file list vs plugin hand-off, and desired UX.
2. R&D Lab scope – metrics (precision/recall, concision, latency) and comparison modes (per-image A/B, corpus summary).
3. Persistence abstraction – minimal contract required so Web UI/CLI can operate while allowing store replacement later.
4. CCC integration contract – expected data from CCC (file handles, IDs, metadata) and return payload (verdicts, duplicate links).
5. Access control (later) – if CCC server hosts the checker, what is the simplest admin authentication model?
