# Imageworks Color-Narrator Specification

_Version 1.0 – Personal Tagger: VLM-guided color localization_

## Table of Contents

- [1. Purpose and Overview](#1-purpose-and-overview)
- [2. Scope](#2-scope)
- [3. Position in Imageworks](#3-position-in-imageworks)
- [4. Functional Requirements](#4-functional-requirements)
- [5. Non-Functional Requirements](#5-non-functional-requirements)
- [6. Data Contract](#6-data-contract)
- [7. Processing Flow](#7-processing-flow)
- [8. VLM Backend](#8-vlm-backend)
- [9. Configuration](#9-configuration)
- [10. Directory Structure](#10-directory-structure)
- [11. Environment and Dependencies](#11-environment-and-dependencies)
- [12. Acceptance Criteria](#12-acceptance-criteria)
- [13. Risk Management](#13-risk-management)
- [14. Development Milestones](#14-development-milestones)
- [15. Repository Structure](#15-repository-structure)

## 1. Purpose and Overview

Color-Narrator converts **mono-checker** color evidence into **short, accurate natural-language statements** that say *where* residual color appears and *what it looks like*.

Example: "Faint yellow-green on the zebra's mane (upper-left)."

### Key Principles

- CN **does not** re-measure color. It **consumes** mono-checker metrics + overlays.
- CN uses a **vision-enabled LLM (VLM)** for natural phrasing; numeric truth stays with mono-checker.
- **Batch-first** (50–100 JPEGs), **JPEG-only Phase 1**, **embedded XMP** (no sidecars).

## 2. Scope

### Phase 1 (In Scope)

- **Inputs**: Competition **JPEG** originals, mono-checker **JSONL**, and overlay PNGs (`lab_residual`, `lab_chroma`).
- **Outputs**: 1–3 concise sentences per image; **embedded** XMP in the JPEG; small narration JSON for audit.
- **Operation**: Headless batch on **RTX 4080**; scalable to **RTX 6000 Pro**.

### Later Phases (Out of Scope)

- **Phase 2**: TIFF ingestion/support (same logic).
- **Phase 3**: RAW ingestion (operate on rendered previews; metadata strategy TBC).
- **Optional**: VLM-assisted boxes/masks; judge PDF/HTML reports.

## 3. Position in Imageworks

- Lives under `apps/color_narrator/`.
- Runs **after** mono-checker has written its JSONL + overlays.
- Reads from the **competition Lightroom directory layout** (no LR DB access).

## 4. Functional Requirements

- **F1**: Read JPEG + JSONL + overlays from configured folders.
- **F2**: Focus on the **largest color clusters** provided by mono-checker (new `regions[]` array).
- **F3**: Prompt a VLM (triptych panels and/or region crops) for **1–3** concise findings.
- **F4**: **Validate** color words against mono-checker hue/chroma; drop/correct mismatches.
- **F5**: Write sentences + tags into the **JPEG XMP** (no sidecars).
- **F6**: CLI + Python API; batch mode; progress + resumability.
- **F7**: Safe fallback if `regions[]` absent (location-agnostic sentence).

## 5. Non-Functional Requirements

- **Performance**: ~≤10 min per 50 JPEGs on RTX 4080 with 7B-class VLM and batching.
- **Scalability**: Profile for RTX 6000 Pro (optionally larger model).
- **Isolation**: Independent uv env (`.venv-cn`) and optional-deps group separate from mono-checker.
- **Reproducibility**: Pinned model revision; versioned prompts.

## 6. Data Contract

Add to each JSONL record:

```json
"regions": [
  {
    "level": "c4",                 // mask threshold used (e.g., C*ab > 4)
    "area_pct": 4.92,              // % of frame area
    "bbox_xywh": [x, y, w, h],     // in original image pixels
    "centroid_norm": [0.18, 0.22], // 0..1 (x,y)
    "mean_hue_deg": 88.0,
    "hue_name": "yellow-green",    // small, fixed lexicon
    "mean_cab": 5.1                // mean chroma for the component
  }
  // …sorted by area desc until ≥90% cumulative or max N=3
]
```

No new color maths; this simply serializes structure already implicit in mono-checker (used today for cluster metrics and overlays).

**Back-compatibility**: If regions[] is missing, CN can approximate top blobs from overlay PNGs as a fallback (precision is lower; still usable).

## 7. Processing Flow

### Phase 1 Process

1. **Load batch**: Resolve JPEGs, matching overlays, and JSONL records from configured folders.

2. **Select regions**: Choose top components by area_pct (cover ≥90% of colored pixels; cap at 3).

3. **Prepare prompt**:
   - **Triptych**: Original (A) + lab_residual (B, "where/which hue") + lab_chroma (C, "how strong"), with a tiny legend; or
   - **Crops**: per-region bbox ± 20% padding plus a small full-image thumbnail.

4. **Call VLM** (batched): Ask for 1–3 concise findings (object/part if evident, location phrase, color family).

5. **Validate**:
   - Map color words → hue bands; require overlap with mean_hue_deg (region if present, else global dominant hue(s)).
   - Map mean_cab to strength adjectives via bins (e.g., very faint / faint / noticeable / strong).
   - Normalize location to cardinal / rule-of-thirds using centroid_norm. Drop any sentence failing color validation.

6. **Write outputs**:
   - **JPEG XMP**: Append sentences to XMP-xmp:UserComment (≤3 lines); add compact tags to XMP-dc:Subject (e.g., mono:leak:yellow-green, mono:loc:upper-left, mono:strength:faint).
   - **Narration JSON** (for audit/repro): sentences + backing (regions used, model, prompt version, validation result).

## 8. VLM Backend

### Model Selection
**Production Model**: Qwen2.5-VL-7B-AWQ (LMDeploy)
- **Reason**: Fits within RTX 4080 16GB VRAM constraints (10.88GB total usage)
- **Performance**: Sub-1 second inference for color description tasks
- **Quality**: Excellent for color contamination detection and location description

**Alternative Tested**: Qwen2-VL-7B-Instruct
- **Status**: Failed due to CUDA OOM errors on 16GB VRAM
- **Model Size**: 13.6GB (too large for current hardware)

## 9. Configuration

Configuration is maintained in `pyproject.toml` (not YAML), mirroring the mono-checker approach:

```toml
[tool.imageworks.color_narrator]
# I/O (competition Lightroom layout)
images_dir         = "/PATH/SerialPDI_2025-26/R1/originals"
overlays_dir       = "/PATH/SerialPDI_2025-26/R1/overlays"
mono_results_jsonl = "/PATH/SerialPDI_2025-26/R1/analysis/mono_results.jsonl"

# VLM back-end (RTX 4080 default; later swap endpoint on 6000 Pro)
backend            = "openai"                         # generic OpenAI-style client
endpoint_url       = "http://127.0.0.1:8000/v1"
model              = "Qwen2.5-VL-7B-AWQ"
batch_size         = 6
panel_long_edge_px = 1280

# Narration & validation
max_regions        = 3
sentence_cap       = 3
enforce_hue_check  = true
strength_bins      = [1.5, 3.0, 6.0]                  # very faint / faint / noticeable / strong

# JPEG metadata (Phase 1 policy)
embed_xmp          = true                              # write into the JPEG (no sidecars)
use_sidecars       = false
```

## 10. Directory Structure

Configurable directories (defaults shown), matching current exports:

```
…/originals/<stem>.jpg
…/overlays/<stem>_lab_residual.png
…/overlays/<stem>_lab_chroma.png
…/analysis/mono_results.jsonl
```

CN never queries the Lightroom catalog; it follows the folder layout.

## 11. Environment and Dependencies

Keep mono-checker and CN completely separate while sharing the same repo.

### Optional Dependency Groups

In `pyproject.toml`:

```toml
[project.optional-dependencies]
mono = [
  # mono-checker deps (leave exactly as-is today)
]
color-narrator = [
  "torch>=2.3",          # install CUDA wheel via the proper index-url
  "vllm>=0.5.0",         # dev OpenAI-style server for 4080
  "httpx>=0.27",         # API client
  "piexif>=1.1.3"        # or use ExifTool shell-out per your current pattern
]
```

### Virtual Environment Setup

Two uv virtual environments (side-by-side):

```bash
.venv-mono/   # mono-checker's known-good env
.venv-cn/     # Color-Narrator's env (ML stack, models, etc.)
```

### Bootstrap (WSL)

```bash
# MONO (unchanged)
uv venv .venv-mono
source .venv-mono/bin/activate
uv pip install -e ".[mono]"
deactivate

# CN (new, isolated)
uv venv .venv-cn
source .venv-cn/bin/activate
uv pip install -e ".[color-narrator]"
# Then install CUDA-enabled torch & start your local OpenAI-style VLM endpoint.
deactivate
```

Use VS Code's interpreter selector to point each workspace/task to the appropriate venv. uv's parallel resolver/downloader yields faster installs than plain pip.

## 12. Acceptance Criteria

- **Factuality**: 0 incorrect color families after validation across the batch (sentences that disagree with mono-checker are suppressed or corrected).
- **Usefulness**: On a curated set (incl. edge cases/"QUERY" images), ≥90% reviewer agreement that the top sentence correctly identifies where color is.
- **Performance**: ≤~10 minutes per 50 JPEGs on RTX 4080 using a 7B VLM with batching (target).
- **Metadata**: XMP round-trips and appears in Lightroom; no sidecars.
- **Robustness**: If overlays or regions[] are missing, CN still emits a safe, location-agnostic sentence (no hallucinated locations).

## 13. Risk Management

- **VLM hallucination** → Strict post-validation against mono-checker hue/chroma; conservative editing; drop non-conforming lines.
- **Overlay/JSON missing** → Fallback narration with lower confidence flag; never invent bounding boxes/locations.
- **Environment drift** → uv extras + separate venvs prevent dependency clashes with mono-checker.
- **Throughput** → Batching, panel downscaling, concise prompts; 6000 Pro profile for heavier models.

## 14. Development Milestones

1. **Spec sign-off** (this document).
2. **Design sign-off** (component boundaries, prompt templates, hue-word table, XMP field map).
3. **Prototype** (orchestrator, adapter, validator, writer; minimal serving).
4. **Batch evaluation** on QUERY/toned examples; refine prompts & bins.
5. **(Optional) 6000 Pro profile** (Triton/TensorRT-LLM; larger model).

## 15. Repository Structure

```
imageworks/
├─ apps/
│  ├─ mono_checker/                       # existing (unchanged)
│  └─ color_narrator/                     # CN source
├─ docs/
│  ├─ spec/
│  │  └─ imageworks-colour-narrator-specification.md   # THIS FILE
│  ├─ design/
│  │  └─ imageworks-colour-narrator-design.md          # design doc (separate)
│  ├─ dev-env/
│  │  └─ ide-setup-wsl-vscode.md                      # existing
│  └─ reference/                                      # (optional: hue tables, prompt versions)
├─ tests/
│  └─ color_narrator/                                 # (future) tests
├─ data/
│  ├─ samples/                                        # (optional) QA fixtures
│  └─ analysis/                                       # (optional) local outputs
├─ pyproject.toml                                     # add [tool.imageworks.color_narrator]
├─ .venv-mono/                                        # uv env for mono-checker
└─ .venv-cn/                                          # uv env for Color-Narrator
```

## References

- Imageworks overall spec and mono-checker documentation (color analysis, overlays, JSON fields).
- IDE/dev environment: WSL + VS Code + uv conventions (paths, caches, pre-commit).
- Competition round outputs: mono-checker PASS/QUERY summaries and per-image notes (basis for CN's language and validation).

This specification intentionally avoids external library/API references so it can live entirely within the repo and remain stable over time.
