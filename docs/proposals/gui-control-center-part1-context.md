# ImageWorks GUI Control Center - Part 1: Context & Requirements

**Document Status**: Specification for Implementation
**Date**: October 26, 2025
**Approach**: Option C - Hybrid (Presets + Advanced Options)

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Conversation History](#conversation-history)
3. [User Requirements](#user-requirements)
4. [Module Workflow Analysis](#module-workflow-analysis)
5. [CLI Complexity Audit](#cli-complexity-audit)
6. [Problem Statement](#problem-statement)
7. [Success Criteria](#success-criteria)

---

## Executive Summary

This document captures the complete context and requirements for building a unified GUI control center for the ImageWorks project. The GUI will provide a streamlined interface for five core modules (mono checker, color narrator, personal tagger, image similarity checker, model downloader) plus infrastructure management (backends, registry, deployment profiles).

**Key Decision**: **Option C - Hybrid Approach**
- Use smart presets by default (Quick/Standard/Thorough)
- Provide "Show Advanced Options" expander for expert users
- Balance simplicity with power user needs

**Primary Goals**:
1. Simplify complex workflows (especially image similarity's 35+ CLI flags)
2. Provide unified interface for common tasks (directory selection, output review, metadata validation)
3. Enable pipeline chaining (mono → color narrator, download → start backend → tag)
4. Centralize model/backend management
5. Reduce context switching between CLI, Lightroom, file browser, and markdown viewers

---

## Conversation History

### Initial Request
User wanted a lightweight GUI to help with mono workflow automation:
- Unzip downloads
- Import to Lightroom
- Run mono analysis
- Review results (markdown summaries)
- Validate before accepting metadata

**Question**: Could Dify/LangFlow help here?
**Answer**: No - Dify/LangFlow are designed for LLM/prompt workflows (Phase 3/4 scope), not filesystem operations.

### Workflow Expansion
User realized personal tagger needs similar workflow patterns:
- Directory selection
- View generated tags without opening Lightroom
- Validate tags before writing to metadata
- Manual editing capability

Additional needs identified:
- Color narrator integration (reads mono results)
- All three modules need similar UI patterns

### Infrastructure Scope Addition
User identified that model management is cross-cutting:
- Model downloader supports all modules (not just chat proxy)
- Registry management needed
- Backend control (vLLM, LMDeploy, Ollama)
- Deployment profile visualization
- Role → model assignment editor

### Image Similarity Addition
User requested image similarity checker be included:
- Most complex CLI (35+ flags)
- Embedding strategies (simple, open_clip, siglip, remote)
- VLM explanation generation
- Cache management
- Performance metrics

**User's Key Concern**: "I don't want to over complicate"

### Design Evolution
1. **Initial**: Simple Streamlit app for mono workflow (2-4 hours)
2. **Expanded**: Multi-page app for 3 workflows (700-850 lines, 2-3 days)
3. **Infrastructure**: Full control center with 8 major sections (2,850 lines, 8-10 days)
4. **Simplified**: Hybrid approach with presets (1,750 lines, 5-6 days)
5. **Final**: **Option C - Presets + Advanced expanders** (realistic, maintainable)

---

## User Requirements

### Functional Requirements

#### FR1: Workflow Automation
- Process downloads (unzip, organize)
- Execute analysis tools (mono, narrator, tagger, similarity)
- Review results before committing
- Write metadata to images

#### FR2: Directory & File Management
- Visual directory browser with recursive scanning
- File type filtering (JPEG, PNG, RAW)
- Batch selection
- Path history/favorites

#### FR3: Output Visualization
- Markdown summary viewer (styled rendering)
- JSONL results browser (paginated, filterable)
- Image viewer with overlays (contamination heatmaps)
- Side-by-side comparison (similarity matches)

#### FR4: Metadata Management
- Preview generated tags/captions/descriptions
- Edit before writing to files
- Approve/reject individual items
- Bulk operations (find/replace keywords)
- XMP/EXIF viewer

#### FR5: Model Infrastructure
- Model registry browser (filter, search, sort)
- Download new models from HuggingFace
- Backend control (start/stop vLLM/LMDeploy/Ollama)
- Deployment profile selection
- Role → model assignment

#### FR6: Configuration Management
- Save/load workflow presets
- Edit pyproject.toml settings
- Environment variable management
- Backend URL configuration

#### FR7: Job Management
- View recent runs across all modules
- Re-run with same configuration
- Compare run results
- Job history with filters

### Non-Functional Requirements

#### NFR1: Simplicity Over Power
- Hide complexity by default
- 3-4 presets per workflow (Quick/Standard/Thorough)
- "Show Advanced Options" for power users
- Smart defaults based on deployment profile

#### NFR2: Low Maintenance
- Reuse existing CLI tools (subprocess wrappers)
- Direct Python imports where possible
- No additional databases or services
- Single file backup (session state)

#### NFR3: Performance
- Responsive UI (< 100ms interactions)
- Background processing for long operations
- Streaming output for subprocess logs
- Cache-aware (don't rebuild unnecessarily)

#### NFR4: Integration
- Work with existing file structures
- Respect pyproject.toml configuration
- Use unified model registry
- Coordinate with chat proxy (Phase 2)

---

## Module Workflow Analysis

### 1. Mono Checker

**Purpose**: Detect non-monochrome images in competition submissions

**Workflow Steps**:
1. Process downloads (optional unzip)
2. Select directory + configure thresholds
3. Run analysis (batch processing)
4. Review results:
   - Filter by verdict (pass/fail/query)
   - View contamination overlays
   - Read markdown summary
5. Export to Lightroom (write metadata)

**Key UI Needs**:
- Directory browser
- Threshold sliders (RGB delta, chroma, hue)
- Image grid with overlay toggle
- Verdict filter (pass/fail/query)
- Markdown viewer
- Export button

**Current CLI**: 10-12 flags ✅ Manageable
- `--input`, `--overlays`, `--output-jsonl`, `--summary`
- `--rgb-delta-threshold`, `--chroma-threshold`, `--hue-consistency-threshold`
- `--min-contaminated-pixels`, `--recursive`, `--dry-run`, `--backup-originals`

**Documentation**: `docs/domains/mono/mono-overview.md`

---

### 2. Color Narrator

**Purpose**: Generate natural language descriptions of color contamination in near-mono images

**Workflow Steps**:
1. Import mono results (load `mono_results.jsonl`)
2. Select images (usually filtered to fail/query verdicts)
3. Configure backend (vLLM/LMDeploy/Triton)
4. Select prompt profile
5. Run narration (VLM inference)
6. Review results:
   - View image + overlay + narration text
   - Edit narration before accepting
   - Approve/reject individual narrations
7. Write to metadata (XMP)

**Key UI Needs**:
- JSONL import (parse mono results)
- Backend selector (with status indicator)
- Prompt profile dropdown (with preview)
- Side-by-side viewer (image, overlay, narration)
- Text editor for narration
- Batch approve/reject
- Pipeline mode (auto-trigger after mono)

**Current CLI**: 15-18 flags ⚠️ Could simplify with presets
- `--images`, `--overlays`, `--mono-jsonl`, `--summary`
- `--backend` (vllm/lmdeploy/triton)
- `--vlm-model`, `--vlm-base-url`, `--vlm-api-key`, `--vlm-timeout`
- `--prompt`, `--list-prompts`, `--regions`
- `--dry-run`, `--backup-original-files`, `--overwrite-existing-metadata`
- `--min-contamination-level`, `--require-overlays`

**Documentation**: `docs/domains/color-narrator/reference.md`

**Integration**: Uses mono output, writes XMP, supports Phase 2 role resolution

---

### 3. Personal Tagger

**Purpose**: Generate captions, keywords, and descriptions for Lightroom libraries

**Workflow Steps**:
1. Select directory (with recursive option)
2. Configure:
   - Enable registry-based model selection
   - Choose roles (caption/keyword/description)
   - Select prompt profile
   - Set batch size
3. Preview tags (dry-run):
   - View image thumbnails with generated tags
   - Edit tags individually
   - Bulk find/replace for keywords
   - Mark for approval
4. Execute tagging (write XMP)
5. Validate existing tags (optional)

**Key UI Needs**:
- Directory browser with preview
- Role selector (3 independent toggles)
- Prompt profile dropdown
- Image grid with tag editor
- Bulk edit tools (find/replace)
- Approve/reject checkboxes
- Dry-run toggle (default ON)

**Current CLI**: 20-25 flags ⚠️ Could benefit from profiles
- `--input`, `--summary`, `--output-jsonl`
- `--use-registry`, `--caption-role`, `--keyword-role`, `--description-role`
- `--backend`, `--base-url`, `--model`, `--api-key`, `--timeout`
- `--prompt-profile`
- `--batch-size`, `--max-workers`
- `--dry-run`, `--no-meta`, `--backup-originals`, `--overwrite-metadata`
- `--recursive`, `--preflight`

**Documentation**: `docs/domains/personal-tagger/overview.md`

**Key Features**:
- Registry integration with role-based selection
- Batch metrics for comparing models/prompts
- XMP metadata writer with graceful fallback
- Preflight checks (backend health)

---

### 4. Image Similarity Checker

**Purpose**: Flag competition entries that match or resemble prior submissions

**Workflow Steps**:
1. Select candidate images (files or directories)
2. Configure strategy:
   - Choose preset (Quick/Standard/Thorough)
   - OR manually select:
     - Strategies (perceptual_hash, embedding)
     - Embedding backend (simple, open_clip, siglip, remote)
     - Embedding model
     - Augmentation options (pooling, grayscale, five-crop)
     - Thresholds (fail, query)
     - VLM explanations (enable/disable)
3. Run analysis
4. Review results:
   - Filter by verdict (pass/fail/query)
   - View side-by-side comparisons
   - Read VLM explanations (if enabled)
   - See similarity scores
5. Export results (JSONL, markdown)
6. Optional: Write metadata

**Key UI Needs**:
- File/directory browser
- **Preset selector** (most important - hides complexity)
- Advanced options expander
- Match viewer (side-by-side with scores)
- Explanation viewer
- Cache management UI
- Performance metrics display

**Current CLI**: **35+ flags** 🔴 MOST COMPLEX - Desperately needs presets
- `--candidates` (repeatable paths)
- `--library-root`
- `--strategy` (repeatable: embedding, perceptual_hash)
- `--embedding-backend` (simple, open_clip, siglip, remote)
- `--embedding-model`
- `--backend`, `--base-url`, `--model`, `--api-key`, `--timeout`
- `--similarity-metric` (cosine, euclidean, manhattan)
- `--fail-threshold`, `--query-threshold`, `--top-matches`
- `--explain/--no-explain`
- `--prompt-profile`
- `--write-metadata/--no-write-metadata`
- `--backup-originals/--no-backup-originals`
- `--overwrite-metadata/--no-overwrite-metadata`
- `--augment-pooling/--no-augment-pooling`
- `--augment-grayscale/--no-augment-grayscale`
- `--augment-five-crop/--no-augment-five-crop`
- `--augment-five-crop-ratio`
- `--use-loader/--no-use-loader`
- `--registry-model`, `--registry-capability` (repeatable)
- `--perf-metrics/--no-perf-metrics`
- `--refresh-library-cache/--no-refresh-library-cache`
- `--manifest-ttl-seconds`
- `--output-jsonl`, `--summary`
- `--dry-run/--no-dry-run`

**Documentation**: `docs/guides/image-similarity-checker.md`

**Complexity Drivers**:
- 4 different embedding backends (each with own config)
- Augmentation strategies (5 flags)
- Dual backend system (embeddings + VLM explanations)
- Cache management (3 flags)
- Registry integration (3 flags)
- Performance tuning (2 flags)

**Preset Benefits**: Can reduce 35 flags to 3 presets + 5-8 common overrides

---

### 5. Model Downloader

**Purpose**: Download models from HuggingFace and sync to unified registry

**Workflow Steps**:
1. Enter HuggingFace URL or owner/repo
2. Select format preference (GGUF, AWQ, GPTQ, safetensors)
3. Choose location (linux_wsl, windows_lmstudio, custom path)
4. Optional: Select specific files (for multi-file repos)
5. Monitor download progress (aria2c)
6. Auto-sync to registry on completion

**Key UI Needs**:
- URL input field with validation
- Format selector (checkboxes for multiple)
- Location dropdown + custom path browser
- File tree viewer (for selective download)
- Progress bar (live aria2c monitoring)
- Download history/queue
- Registry sync status

**Current CLI**: 8-10 flags ✅ Essential
- `model` (positional: URL or owner/repo)
- `--format` (comma-separated preferences)
- `--location` (linux_wsl, windows_lmstudio, custom)
- `--output-dir` (custom path)
- `--resume/--no-resume`
- `--update-registry/--no-update-registry`

**Additional Commands**:
- `list` - Show downloaded models
- `remove` - Delete model from disk + registry
- `sync` - Merge downloader registry into unified registry
- `import-ollama` - Import locally pulled Ollama models

**Documentation**: Inline in `src/imageworks/tools/model_downloader/cli.py`

---

## CLI Complexity Audit

### Complexity Tiers

| Module | Total Flags | Assessment | Simplification Strategy |
|--------|-------------|------------|-------------------------|
| Mono Checker | 10-12 | ✅ Manageable | Direct mapping, minor presets |
| Model Downloader | 8-10 | ✅ Essential | Keep all, add shortcuts |
| Color Narrator | 15-18 | ⚠️ Medium | 3 backend presets + 5 common overrides |
| Personal Tagger | 20-25 | ⚠️ Medium | 3 workflow presets (quick/full/validate) |
| Image Similarity | 35+ | 🔴 High | **Mandatory presets** - 3 levels hiding 25+ flags |

### Flag Category Analysis

#### Image Similarity Breakdown (35 flags)

**Embedding Configuration (8 flags)**:
- `--strategy` (repeatable)
- `--embedding-backend`
- `--embedding-model`
- `--similarity-metric`
- `--augment-pooling`
- `--augment-grayscale`
- `--augment-five-crop`
- `--augment-five-crop-ratio`

**VLM Explanation (5 flags)**:
- `--explain/--no-explain`
- `--backend`
- `--base-url`
- `--model`
- `--prompt-profile`

**Thresholds & Matching (3 flags)**:
- `--fail-threshold`
- `--query-threshold`
- `--top-matches`

**Metadata Management (3 flags)**:
- `--write-metadata`
- `--backup-originals`
- `--overwrite-metadata`

**Registry Integration (3 flags)**:
- `--use-loader`
- `--registry-model`
- `--registry-capability` (repeatable)

**Cache & Performance (5 flags)**:
- `--refresh-library-cache`
- `--manifest-ttl-seconds`
- `--perf-metrics`
- API connection (`--api-key`, `--timeout`)

**I/O & Paths (4 flags)**:
- `--library-root`
- `--output-jsonl`
- `--summary`
- `--dry-run`

**Total**: 31 explicit flags + 4 repeatable/multi-value = 35+ effective flags

### Preset Reduction Strategy

**Quick Preset** (hides 25 flags, exposes 5):
```
Exposed: candidates, library-root, output-jsonl, summary, dry-run
Hidden: Everything else uses defaults
```

**Standard Preset** (hides 20 flags, exposes 10):
```
Exposed: Above + fail-threshold, query-threshold, strategies, explain
Hidden: Embedding details, augmentation, cache, registry integration
```

**Thorough Preset** (hides 15 flags, exposes 15):
```
Exposed: Above + embedding-backend, embedding-model, augmentation toggles, write-metadata
Hidden: Advanced tuning (ratios, timeouts, registry details)
```

**Expert Mode** (show all 35 flags):
```
Organized into collapsible sections:
- Core Settings (paths, thresholds)
- Strategy Configuration (embedding backend, model, metric)
- Augmentation Options (pooling, grayscale, five-crop)
- VLM Explanations (backend, model, prompts)
- Output & Metadata (write, backup, overwrite)
- Registry Integration (use-loader, model, capabilities)
- Performance & Cache (metrics, refresh, TTL)
```

---

## Problem Statement

### Current Pain Points

1. **Tool Fragmentation**:
   - CLI for execution
   - File browser for navigation
   - Text editor for markdown summaries
   - Lightroom for metadata validation
   - Context switching reduces efficiency

2. **Complexity Barrier**:
   - Image similarity: 35 flags is overwhelming for new users
   - Personal tagger: 25 flags with complex registry integration
   - No discoverability (must read docs to know options)

3. **Workflow Interruption**:
   - Mono → color narrator requires manual JSONL passing
   - Download → start backend → select model = 3 separate commands
   - Preview tags requires dry-run + file inspection + re-run

4. **Configuration Management**:
   - pyproject.toml editing requires restart
   - No preset saving/loading
   - Can't compare configurations between runs

5. **Results Review**:
   - JSONL is machine-readable but not human-friendly
   - Markdown summaries lack interactivity
   - No filtering/sorting/searching
   - Image overlays require external viewer

6. **Metadata Anxiety**:
   - No preview before writing XMP
   - Can't easily edit generated tags
   - Bulk operations require scripting
   - Undo requires backups

### Goals

1. **Reduce Cognitive Load**: 3 presets instead of 35 flags
2. **Enable Iteration**: Preview → Edit → Execute loop
3. **Unify Interfaces**: One tool for common workflows
4. **Preserve Power**: Advanced options for experts
5. **Improve Discoverability**: Visible presets = learning path
6. **Accelerate Workflows**: Pipeline chaining, job history, re-run

---

## Success Criteria

### Must Have (MVP)

1. **Workflow Execution**:
   - Run mono checker with results review ✅
   - Run image similarity with preset selection ✅
   - Download models with progress monitoring ✅

2. **Output Visualization**:
   - Markdown viewer (styled) ✅
   - JSONL browser (paginated, filterable) ✅
   - Image viewer with overlays ✅

3. **Configuration**:
   - Preset selector for similarity checker ✅
   - Basic form inputs for other modules ✅
   - Path/directory browser ✅

4. **Infrastructure**:
   - Model registry browser (read-only) ✅
   - Backend status display ✅
   - Basic start/stop controls ✅

### Should Have (Phase 2)

1. **Advanced Workflows**:
   - Personal tagger with tag preview/edit
   - Color narrator with pipeline mode
   - Metadata editor (before write)

2. **Job Management**:
   - History with re-run
   - Configuration comparison
   - Results diff viewer

3. **Enhanced Configuration**:
   - Preset save/load
   - pyproject.toml editor
   - Environment variable manager

4. **Model Management**:
   - Role assignment editor
   - Deployment profile selector
   - Import from Ollama

### Could Have (Future)

1. **Pipeline Builder**: Visual DAG for chaining operations
2. **Metrics Dashboard**: Charts for performance over time
3. **Batch Scheduler**: Queue multiple jobs
4. **Collaborative Features**: Share presets, export configs
5. **Plugin System**: Extend with custom workflows

---

## Next Steps

Proceed to **Part 2: Technical Specification** for:
- Detailed architecture design (Option C implementation)
- Component library specifications
- Preset definitions with exact flag values
- UI wireframes and navigation flow
- Implementation phases and estimates
- Development guidelines

---

**Document Version**: 1.0
**Last Updated**: October 26, 2025
**Author**: GitHub Copilot (from user conversation)
