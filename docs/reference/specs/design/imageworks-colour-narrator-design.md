# Imageworks Color-Narrator Design Document

_Version 1.0 – Personal Tagger: VLM-guided color localization_

## Table of Contents

- [1. Overview](#1-overview)
- [2. Design Tenets](#2-design-tenets)
- [3. Runtime Architecture](#3-runtime-architecture)
- [4. Input and Directory Conventions](#4-input-and-directory-conventions)
- [5. Data Contract](#5-data-contract)
- [6. Configuration](#6-configuration)
- [7. Component Design](#7│  ├─ mono_checker/                        # existing (unchanged)
│  └─ color_narrator/                     # CN sourcemponent-design)
- [8. CLI and Python API](#8-cli-and-python-api)
- [9. Environment and Dependencies](#9-environment-and-dependencies)
- [10. Logging and Observability](#10-logging-and-observability)
- [11. Performance Plan](#11-performance-plan)
- [12. Testing Strategy](#12-testing-strategy)
- [13. Failure Modes and Recovery](#13-failure-modes-and-recovery)
- [14. Security and Safety](#14-security-and-safety)
- [15. Development Milestones](#15-development-milestones)
- [16. Repository Structure](#16-repository-structure)
- [17. Open Decisions](#17-open-decisions)

## 1. Overview

Color-Narrator converts **mono-checker** color evidence into **concise, plain-English** statements that say *where* residual color appears and *what it looks like*. CN **does not** re-compute color; it consumes mono-checker's JSON + overlays and uses a **vision-language model (VLM)** for natural phrasing, then validates the language against mono-checker's numbers. Phase 1 targets **competition JPEGs** and writes **embedded XMP** (no sidecars). The module is **uv-managed** and **environment-isolated** from mono-checker.

> **Implementation status:** The shipping build standardises on **Qwen2.5-VL-7B-AWQ** served via LMDeploy. vLLM + Qwen2-VL-2B remains available as a low-VRAM fallback.

## 2. Design Tenets

1. **Single source of truth** for color → **mono-checker** (Lab masks, clusters, overlays, JSON metrics).
2. **Language generation** → **VLM**, never trusted blindly; **post-validated** against mono-checker hue/chroma.
3. **Batch-first**, **headless**, **idempotent**; safe to re-run on a batch.
4. **Isolation & reproducibility** → separate **uv** extra requires and separate **venv** (`.venv-cn`) so mono-checker stays pristine.
5. **No Lightroom DB access**; only the known **competition folder layout**.

## 3. Runtime Architecture

```
CLI/API
└─ Batch Orchestrator (queues, progress, retries)
   ├─ Data Loader (JPEG, overlays, JSONL index)
   ├─ Region Selector (prefer mono regions[]; overlay fallback)
   ├─ Prompt Packer (triptych and/or region crops)
   ├─ VLM Adapter (OpenAI-style; vLLM now, Triton later)
   ├─ Validator (hue & strength checks; location normalisation)
   └─ Writer (JPEG XMP embed + narration JSON)
```

### Backend Options

- **Phase 1**: OpenAI-style endpoint serving **Qwen2-VL-7B** (via **vLLM multimodal**) on **RTX 4080**.
- **Upgrade (Phase 1.5/2)**: Switch endpoint to **Triton/TensorRT-LLM** on **RTX 6000 Pro**, optionally larger model (e.g., Qwen2.5-VL-72B). Adapter signature does not change.

## 4. Input and Directory Conventions

CN is configured via `pyproject.toml` (`[tool.imageworks.color_narrator]`), mirroring mono-checker's approach.

### File Structure

- **Original JPEG**: `…/originals/<stem>.jpg`
- **Overlays** (produced by mono-checker):
  - `…/overlays/<stem>_lab_residual.png` (hue direction; where/which color)
  - `…/overlays/<stem>_lab_chroma.png` (intensity; how strong)
- **Analysis JSONL**: `…/analysis/mono_results.jsonl`

Each line is a JSON record keyed by original filename; CN expects mono-checker to add a lightweight **`regions[]`** array (see §5).

**File key (`stem`)**: `stem := filename without extension` (must match across JPEG/overlays/JSON). CN will log any missing counterpart and proceed (with fallbacks where possible).

## 5. Data Contract

Mono-checker adds a `regions[]` field to each record:

```json
"regions": [
  {
    "level": "c4",                   // which mask/threshold (e.g., C*ab > 4)
    "area_pct": 4.92,                // % of total image pixels
    "bbox_xywh": [x, y, w, h],       // pixel coords in original image space
    "centroid_norm": [0.18, 0.22],   // normalised 0..1 (x,y)
    "mean_hue_deg": 88.0,            // degrees on Lab hue circle
    "hue_name": "yellow-green",      // compact lexicon label
    "mean_cab": 5.1                  // mean chroma of the component
  }
  // …sorted by area desc until ≥90% cumulative colored area or N=3
]
```

### Notes

This serializes structure mono-checker already derives for "largest cluster" metrics and overlays; no new color maths is introduced.

If regions[] is absent, CN can fallback to extracting coarse blobs from lab_chroma.png (threshold→morphology→connected components); this is used only to localize narrative and never to decide pass/fail.

## 6. Configuration

```toml
[tool.imageworks.color_narrator]
# I/O (competition Lightroom layout)
images_dir         = "/PATH/SerialPDI_2025-26/R1/originals"
overlays_dir       = "/PATH/SerialPDI_2025-26/R1/overlays"
mono_results_jsonl = "/PATH/SerialPDI_2025-26/R1/analysis/mono_results.jsonl"

# VLM back-end (OpenAI-style; RTX 4080 now; swap endpoint on 6000 Pro later)
backend            = "openai"
endpoint_url       = "http://127.0.0.1:8000/v1"
model              = "Qwen/Qwen2-VL-7B-Instruct"

# Batching & panels
batch_size         = 6                      # tune for VRAM
panel_long_edge_px = 1280                   # triptych/crop resize

# Narration & validation
max_regions        = 3
sentence_cap       = 3
enforce_hue_check  = true
strength_bins      = [1.5, 3.0, 6.0]        # very faint / faint / noticeable / strong

# JPEG metadata (Phase 1 policy)
embed_xmp          = true                   # write into the JPEG (no sidecars)
use_sidecars       = false
```

## 7. Component Design

### 7.1 Data Loader

- Build an index from mono_results.jsonl: {stem → record} (O(1) lookup).
- Resolve paths for JPEG and overlay PNGs from stem.
- Dry-run mode reports missing artifacts before inference (no partial surprises).
- Optional prefetch and decode caching to amortize I/O (threaded pool).

#### Edge Cases

- **Overlay(s) missing** → set overlay_mode = none and later use direct-photo narration with stricter validation.
- **Record missing** → skip with warning; continue batch.

### 7.2 Region Selector

**Preferred**: Use record["regions"] sorted by area_pct. Keep the top-K until ≥90% cumulative colored area or K = max_regions.

**Fallback** (no regions[]):

1. Load lab_chroma.png (grayscale or alpha magnitude), normalize to [0,1].
2. Auto-threshold by Otsu or a small fixed ε (e.g., top 2–4% brightest) to capture genuinely colored areas.
3. Morphological open (3×3) then close (3×3); remove components with area_pct < 0.02%.
4. Compute bbox, centroid_norm. For hue in fallback, use record's global dominant_hue_deg or hue_modes if present.

### 7.3 Prompt Packer

CN supports two evidence presentations to the VLM:

#### Triptych (default when overlays exist)

A single image with three panes:

- **A**: Original JPEG (downscaled to panel_long_edge_px)
- **B**: lab_residual (where/which hue)
- **C**: lab_chroma (how strong; brighter = stronger)

A tiny overlay legend is rendered at the bottom ("B = where (hue), C = strength (brighter = stronger)").

#### Region Crop(s)

For each selected region: crop bbox with ±20% context (clamped to bounds) + a thumbnail of the full frame in a corner for orientation.

#### System Instruction

"You are auditing a monochrome competition photo. Panels or crops may highlight where color appears and how strong it is. Write 1–3 short findings. For each, include: (i) the object or part (if clear), (ii) an approximate location ('upper-left', 'along the mane'), and (iii) the color family you perceive. Avoid aesthetic judgment. Keep each finding under 18 words."

#### Structured Output

Ask the VLM to output strict JSON to simplify post-validation:

```json
{
  "findings": [
    {"noun": "mane", "location": "upper-left", "color": "yellow-green"},
    {"noun": "ear", "location": "upper-left", "color": "yellow-green"}
  ]
}
```

If the model emits text, CN will robustly parse fallback patterns but JSON is preferred.

### 7.4 VLM Adapter

#### Interface (stable)

```python
describe(images: list[PanelOrCrop], model: str, system_prompt: str) -> list[Findings]
```

#### Transport

- OpenAI-style /v1/chat/completions (HTTP).
- **Batching**: Split batch into batch_size groups; within each request, send multiple image inputs if server supports multi-image prompts; otherwise one image per turn with concise context.
- **Timeouts/retries**: Per request timeout (configurable); retry with smaller micro-batches on overload; exponential backoff.

#### Back-ends

- **Phase 1**: vLLM multimodal serving Qwen/Qwen2-VL-7B on the 4080.
- **Upgrade**: Triton/TensorRT-LLM endpoint on the 6000 Pro. Adapter does not change.

### 7.5 Validator (Truth Guard)

CN never publishes a sentence that disagrees with mono-checker hue/chroma.

#### 7.5.1 Hue Word → Numeric Band

Start with a compact 12-color wheel (tune later):

| Word | Degrees (Lab hue) |
|------|-------------------|
| red | [350°, 360°) ∪ [0°, 15°) |
| orange | [15°, 45°) |
| yellow | [45°, 70°) |
| yellow-green | [70°, 100°) |
| green | [100°, 150°) |
| cyan / aqua | [150°, 190°) |
| blue | [190°, 250°) |
| violet | [250°, 275°) |
| purple | [275°, 300°) |
| magenta | [300°, 330°) |
| pink | [330°, 350°) |
| neutral/none | — (reject; not allowed in Phase 1) |

**Region-level check**: Prefer region.mean_hue_deg.

**Global check**: If region hue unavailable (fallback), allow any band overlapping global hue mode(s) in record.

If color is outside the allowed band: reject that finding (do not publish).

#### 7.5.2 Strength Adjective from Chroma

Map mean_cab to the adjective used in the final sentence. Default bins (configurable):

- **very faint**: 0.5 ≤ C*ab < 1.5
- **faint**: 1.5 ≤ C*ab < 3.0
- **noticeable**: 3.0 ≤ C*ab < 6.0
- **strong**: C*ab ≥ 6.0

Values < 0.5 are ignored (below perceptual noise).

#### 7.5.3 Location Normalization

Convert centroid_norm to a simple cardinal / rule-of-thirds grid:

| y\\x | 0..0.33 | 0.33..0.66 | 0.66..1.0 |
|------|---------|------------|-----------|
| 0..0.33 | upper-left | upper-center | upper-right |
| 0.33..0.66 | center-left | center | center-right |
| 0.66..1.0 | lower-left | lower-center | lower-right |

If VLM's location disagrees with the grid, CN replaces it with the grid term (and keeps the noun if present).

#### 7.5.4 Sentence Assembly

Template (per finding):

```
<strength> <color> on the <noun> (<location>).
```

If noun is absent/unsure:

```
<strength> <color> in the <location>.
```

Cap total to sentence_cap (default 3), prioritizing largest regions first.

### 7.6 Writer (JPEG XMP Embed)

#### Fields

- **XMP-xmp:UserComment** → up to sentence_cap lines, each one finding.
- **XMP-dc:Subject** → compact tags:
  - `mono:leak:<color>`
  - `mono:loc:<cardinal>`
  - `mono:strength:<bin>`

#### Method

**Preferred** (consistent with mono-checker practice): ExifTool shell-out to embed an XMP packet into the JPEG itself (APP1). This is the most robust path for Lightroom ingestion.

**Alternative** (Python-only): Use an XMP-capable library; many EXIF-only libs (e.g., piexif) cannot write XMP—keep ExifTool unless you already have a reliable XMP writer in-repo.

**Round-trip verification**: Re-read tags after write and log the normalized values.

## 8. CLI and Python API

### CLI (Phase 1)

```bash
iw-personal color-narrator \
  --images "/…/originals" \
  --overlays "/…/overlays" \
  --mono-jsonl "/…/analysis/mono_results.jsonl" \
  [--out "/…/out"] \
  [--max-regions 3] [--sentence-cap 3] \
  [--backend openai --endpoint-url http://127.0.0.1:8000/v1 --model Qwen/Qwen2-VL-7B-Instruct] \
  [--dry-run]
```

### Python API

```python
from imageworks.apps.color_narrator import run_batch

results = run_batch(
    images_dir=..., overlays_dir=..., mono_jsonl=...,
    model="Qwen/Qwen2-VL-7B-Instruct", endpoint_url="http://127.0.0.1:8000/v1",
    max_regions=3, sentence_cap=3
)
```

Both read defaults from pyproject.toml with explicit flags overriding.

## 9. Environment and Dependencies

### Optional Dependencies Split

In `pyproject.toml`:

```toml
[project.optional-dependencies]
mono = [
  # mono-checker deps (leave exactly as-is today)
]
color-narrator = [
  "httpx>=0.27",
  # VLM client only; servers are separate processes
  # For tests and optional local serving add:
  "torch>=2.3",        # install CUDA wheel via correct index-url per platform
  "vllm>=0.5.0",
  "Pillow>=10.0",
  "opencv-python-headless>=4.9",
]
```

### Two Independent Virtual Environments

```bash
# Mono-checker (unchanged)
uv venv .venv-mono
source .venv-mono/bin/activate
uv pip install -e ".[mono]"
deactivate

# Color-Narrator (new)
uv venv .venv-cn
source .venv-cn/bin/activate
uv pip install -e ".[color-narrator]"
deactivate
```

VS Code Remote-WSL: choose interpreter per task (.venv-mono vs .venv-cn).

uv's multi-threaded resolver/downloader yields faster installs than plain pip, and it keeps lock/solver state stable.

## 10. Logging and Observability

- Per-image log line with: stem, regions used, VLM latency, validator outcome, XMP write status.
- Batch progress bar (images processed/total; ETA).
- Audit artifacts (optional): `…/analysis/cn/<stem>_narration.json` and (for QA) `…/analysis/cn/<stem>_triptych.jpg` or `<stem>_cropK.jpg`, gated by a config flag.

## 11. Performance Plan

### RTX 4080 + Qwen2-VL-7B

- Use batched chat completions (size = batch_size).
- Limit panel resolution to panel_long_edge_px = 1280 (or 1024 if memory tight).
- Keep prompts terse; demand JSON output to minimize tokens.

### RTX 6000 Pro

- Same adapter; increase batch size; or move to Triton/TensorRT for lower latency & larger models.

### I/O

- Threaded prefetch; reuse decoded overlays; avoid re-encoding unless QA artifacts enabled.

## 12. Testing Strategy

### Unit Tests

- Path mapping & stem matching.
- JSONL index building; tolerant parsing.
- Fallback blob extraction from lab_chroma.png (threshold/morph).
- Hue-word ↔ band mapping; chroma → strength bins.
- Centroid → cardinal location mapping.
- XMP round-trip (write → read).

### Integration Tests

- "Zebra"-style case (dominant yellow-green; small % area) → expect noun "mane/ear", location UL; color validated.
- A split-tone example (e.g., aqua in highlights) → ensure validator blocks wrong color words.
- Missing overlays → direct-photo narration still yields conservative, location-agnostic sentence.

### Model-Free CI

- Provide a stub VLM (returns canned JSON findings) so CI doesn't require GPU or server.

## 13. Failure Modes and Recovery

- **Endpoint unreachable/timeout** → retry with backoff; then skip image with logged error; continue batch.
- **Invalid VLM JSON** → one re-prompt ("respond in JSON only"); else drop to no-op for that image.
- **Hue mismatch** → suppress that finding; continue others.
- **JPEG write protected** → emit narration JSON only; warn.
- **Missing overlays/regions** → fallback mode; mark backing.hue_checks = "global-only".

## 14. Security and Safety

- CN writes only to configured JPEGs; never overwrites originals if --dry-run or a --no-write flag is set.
- All external calls are localhost by default; TLS/in-LAN security out of scope for Phase 1 (doc note: secure endpoints when remote).

## 15. Development Milestones

1. Lock regions[] schema and initial hue-word bands.
2. Skeleton implementation: loader, selector, packer (no server), validator, writer.
3. Adapter integration to local OpenAI endpoint (vLLM; Qwen2-VL-7B); smoke test on 2–3 images.
4. Batch evaluation (50 images): measure runtime; refine prompts & bins; confirm metadata round-trip in Lightroom.
5. (Optional) 6000 Pro profile: point adapter to Triton/TensorRT; adjust batch size.

## 16. Repository Structure

```
imageworks/
├─ apps/
│  ├─ mono_checker/                       # existing (unchanged)
│  └─ personal_tagger/
│     └─ color_narrator/                  # CN source (to be created)
├─ docs/
│  ├─ architecture/
│  │  └─ deterministic-model-serving.md    # shared serving rationale
│  ├─ domains/
│  │  └─ color-narrator/
│  │     └─ reference.md                   # THIS FILE (once moved)
│  ├─ guides/
│  │  └─ ide-setup-wsl-vscode.md
│  ├─ proposals/
│  │  └─ color-narrator-*.md               # RFCs / plans
│  ├─ reference/
│  │  └─ prompt-templates.md               # (optional) versioned prompts
│  └─ spec/
│     └─ imageworks-colour-narrator-specification.md
├─ tests/
│  └─ color_narrator/
├─ data/
│  ├─ samples/
│  └─ analysis/
├─ pyproject.toml                          # add [tool.imageworks.color_narrator] + extras
├─ .venv-mono/                             # uv env for mono-checker
└─ .venv-cn/                               # uv env for Color-Narrator
```

## 17. Open Decisions

To finalize before implementation:

1. **Exact regions[] field names** (above are proposed and minimal).
2. **Hue lexicon & bands**: start with 10–12 classes; agree on degree ranges.
3. **Triptych vs crop default**: design sets triptych as default when overlays exist; confirm preference.
4. **ExifTool vs Python XMP writer**: recommend ExifTool for robustness unless you already vend an XMP-capable library.
