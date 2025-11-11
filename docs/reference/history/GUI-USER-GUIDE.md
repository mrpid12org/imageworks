# ImageWorks GUI Control Center - User Guide

## Overview

The ImageWorks GUI Control Center provides a user-friendly Streamlit-based interface for all ImageWorks tools. This guide will help you get started and make the most of the GUI.

## Table of Contents

1. [Installation & Launch](#installation--launch)
2. [Navigation](#navigation)
3. [Workflows](#workflows)
4. [Settings & Configuration](#settings--configuration)
5. [Tips & Best Practices](#tips--best-practices)
6. [Troubleshooting](#troubleshooting)

---

## Installation & Launch

### Prerequisites

- Python 3.9+
- uv package manager
- ImageWorks project installed

### Launching the GUI

**Option 1: Using the launch script (recommended)**

```bash
cd /path/to/imageworks
./scripts/launch_gui.sh
```

**Option 2: Manual launch**

```bash
cd /path/to/imageworks
uv run streamlit run src/imageworks/gui/app.py
```

The GUI will open in your default web browser at `http://localhost:8501`.

---

## Navigation

### Sidebar

The left sidebar provides:

- **System Status**: GPU info, VRAM usage
- **Backend Status**: Health of backend services (vLLM, Ollama, Chat Proxy)
- **Quick Actions**: Refresh button, debug mode toggle
- **Help**: Access in-app help documentation

### Main Pages

Navigate using the sidebar menu:

1. **🏠 Dashboard**: System overview and recent jobs
2. **🎯 Models**: Manage models, backends, and profiles
3. **🖼️ Workflows**: Run analysis tools
   - Mono Checker
   - Image Similarity
   - Personal Tagger
   - Judge Vision
   - Color Narrator
4. **📊 Results**: Browse outputs and job history
5. **⚙️ Settings**: Configure preferences

---

## Workflows

### General Workflow Pattern

All ImageWorks modules follow a consistent pattern:

1. **Configure Tab**: Select preset and input parameters
2. **Execute Tab**: Run the tool
3. **Results Tab**: View and analyze outputs

### Preset System

Each module offers three presets:

- **Quick**: Fast processing, lower accuracy
  - Use for: Initial screening, large batches
  - Time: ~10-30s per item

- **Standard** (Recommended): Balanced accuracy and speed
  - Use for: Most use cases
  - Time: ~30-60s per item

- **Thorough**: Maximum accuracy, slower processing
  - Use for: Critical applications, mixed content
  - Time: ~60-120s per item

### Advanced Options

Click "Show Advanced Options" to access all CLI flags:
- Override preset defaults
- Fine-tune thresholds
- Configure backend options
- Enable experimental features

---

## Module Guides

### 🖼️ Mono Checker

**Purpose**: Detect monochrome (grayscale/black-and-white) images

**When to use**:
- Filter color images from B&W scans
- Quality control for digitization projects
- Archive organization

**Quick Start**:
1. Navigate to Mono Checker page
2. Select **Standard** preset
3. Browse and select input directory
4. Click **Run Mono Checker**
5. Review results in Results tab

**Key Settings**:
- `threshold`: Confidence required (0.5-0.9)
- `rgb_delta_threshold`: RGB difference tolerance
- `chroma_threshold`: Color saturation threshold

**Output**: `outputs/mono_results.jsonl` with verdict and certainty per image

---

### 🔍 Image Similarity

**Purpose**: Find duplicate or similar images

**When to use**:
- Deduplicate photo libraries
- Find near-duplicates with minor edits
- Locate similar content across directories

**Quick Start**:
1. Navigate to Image Similarity page
2. Select **Standard** preset
3. Provide:
   - Candidate images (files to check)
   - Library root (directory to search against)
4. Click **Run Similarity Check**
5. View matches in Results tab

**Strategies**:
- `perceptual_hash`: Fast, exact/near-duplicate detection
- `siglip`: Semantic similarity (content-based)
- `resnet`: Deep learning features
- `metadata`: File metadata comparison

**Output**: `outputs/similarity_results.jsonl` with matches per candidate

**Tips**:
- Combine multiple strategies for better results
- Adjust thresholds based on false positive rate
- Use Quick preset for exact duplicate detection

---

### 🏷️ Personal Tagger

**Purpose**: Add AI-generated metadata tags to images

**When to use**:
- Organize large photo libraries
- Add searchable keywords
- Generate descriptions automatically

**Workflow** (4-stage):

1. **Configure Tab**: Select images and options
   - Use **Send This Batch to Judge Vision** when you want competition scoring;
     the dedicated ⚖️ page handles rubric, category, and tournament settings.
2. **Preview Tab**: Run dry-run to see proposed tags
3. **Edit Tab**: Review and modify tags
   - Approve/reject individual tags
   - Bulk find/replace
   - Edit caption/keyword/description fields; competition critiques now live solely
     on the Judge Vision page
4. **Commit Tab**: Write approved tags to files

**IMPORTANT**: Always preview before committing!

**Key Features**:
- Tag categories (content, style, quality, mood)
- Bulk operations (find/replace, approve all)
- Pagination for large batches
- Backup originals option
- Judge Vision hand-off preserves critiques/awards while keeping rubric controls on
  the dedicated scoring page

**Output**: Tags written to image EXIF/XMP metadata

---

### ⚖️ Judge Vision

**Purpose**: Run the multi-stage judging pipeline (compliance, rubric critique, pairwise
tournament) independent of metadata writing.

**When to use**:
- Club competitions that need PAGB-style scoring
- Cross-checking VLM judgements before announcing awards
- Reviewing compliance/technical priors before a human judging night

**Workflow (3 tabs)**:

1. **Brief & Rules** – Point the page at `configs/competitions.toml` (default) and
   select a competition preset generated from the registry. The selection auto-fills
   pairwise rounds, default category, critique template, and judge notes, while you
   choose the input directory and optional overrides. Personal Tagger can still
   pre-stage a batch via the hand-off button.
2. **Run Pipeline** – Executes `imageworks-judge-vision run` in dry-run/no-meta mode,
   surfaces the live progress tracker (per-image filenames plus totals), and keeps
   metadata writes disabled. Command previews help with auditing. The “Execution stage”
   selector now includes **Two-pass (auto IQA → Critique)**, which runs Stage 1 on its
   own process and then automatically launches Stage 2 once the cache is ready. When the
   IQA device is set to GPU, the page requests a temporary GPU lease from the chat proxy:
   vLLM is paused while TensorFlow runs, then transparently restarted before critiques
   begin—no manual GPU juggling required.
3. **Results** – Streams the JSONL + Markdown artifacts to show rubric tables,
   compliance flags, awards, and tournament stability metrics. The default Judge Vision
   outputs also feed the 📊 Results browser automatically.

**Tip**: Leave the JSONL path at its default (`outputs/results/judge_vision.jsonl`) so
the 📊 Results page can pick it up automatically.

---

### 🎨 Color Narrator

**Purpose**: Generate natural language color descriptions

**When to use**:
- Document color palettes
- Generate accessible alt-text
- Analyze color composition

**Pipeline Mode**:
Enable to automatically import Mono Checker results:
- Filter by verdict (monochrome/color/all)
- Process only relevant images
- Seamless workflow integration

**Output**: `outputs/narrator_results.jsonl` with color descriptions

---

### 🎯 Model Manager

**Purpose**: Manage AI models and backend services

**Tabs**:

1. **Registry**: Browse available models
   - Search and filter
   - View model details (size, type, status)
   - Check availability

2. **Download**: Download models from Hugging Face
   - Enter model ID (e.g., `microsoft/Phi-3.5-vision-instruct`)
   - Select GGUF variant
   - Monitor download progress

3. **Backends**: Monitor service health
   - **vLLM**: High-performance inference (:24001)
   - **LMDeploy**: Alternative inference (:23333)
   - **Ollama**: Local serving (:11434)
   - **Chat Proxy**: API gateway (:8100)
   - Test connections
   - View loaded models

4. **Profiles**: Manage model profiles
   - Preset backend configurations
   - Model-specific settings

**Tips**:
- Check backend health before running jobs
- Download models during off-peak hours
- Use Chat Proxy for unified API access

---

### 📊 Results Browser

**Purpose**: Unified view of all module outputs

**Tabs**:

1. **Browse Results**: View output files
   - Select module
   - Browse JSONL/markdown files
   - Filter and search
   - View images side-by-side

2. **Job History**: Track executed jobs
   - View parameters used
   - Re-run previous configurations
   - Filter by module
   - Check status (success/failed)

3. **Statistics**: Aggregate metrics
   - Per-module success rates
   - Processing counts
   - Storage usage
   - Error rates

**Tips**:
- Use history to reproduce successful runs
- Check statistics to optimize workflows
- Export results for external analysis

---

## Settings & Configuration

Access via **⚙️ Settings** page.

### General Settings

- **Auto-save job history**: Enabled by default
- **Cache TTL**: How long to cache data (default: 300s)
- **Default dry-run**: Safety setting (recommended: enabled)
- **Default backup**: Backup files before modification

### Path Configuration

Configure custom paths for:
- Model downloads
- Output directory
- Logs location

### Backend Configuration

Set default backend URLs:
- vLLM: `http://localhost:24001`
- LMDeploy: `http://localhost:23333`
- Ollama: `http://localhost:11434`
- Chat Proxy: `http://localhost:8100`

Test connections directly from Settings.

### Appearance

- Items per page (JSONL viewer)
- Max images in grid
- Debug mode toggle

---

## Tips & Best Practices

### Performance

1. **Use presets**: Start with Standard, adjust if needed
2. **Enable caching**: Significantly speeds up repeated operations
3. **Batch processing**: Process multiple files together
4. **Backend selection**: Use appropriate backend for workload
   - vLLM: Best for high throughput
   - LMDeploy: Good for vision models
   - Ollama: Easy local serving

### Safety

1. **Always dry-run first**: Preview before committing changes
2. **Enable backups**: Protect original files
3. **Check results**: Review outputs before proceeding
4. **Test on small batches**: Validate configuration before large runs

### Workflow Optimization

1. **Use pipelines**: Chain Mono Checker → Color Narrator
2. **Save successful configs**: Re-run from Job History
3. **Monitor backends**: Check health before long jobs
4. **Clear cache**: If seeing stale data

### Debugging

1. **Enable debug mode**: Sidebar → 🐛 Debug checkbox
2. **Check logs**: `logs/chat_proxy.jsonl`
3. **Use help**: Click ❓ Help in sidebar
4. **Test backends**: Settings → Backends → Test Connection

---

## Troubleshooting

### Common Issues

#### Backend Shows "Offline"

**Solutions**:
1. Verify backend is running:
   ```bash
   curl http://localhost:24001/health
   ```
2. Check URL in Settings → Backends
3. Restart backend service
4. Check firewall settings

#### Job Fails or Hangs

**Solutions**:
1. Check logs: `logs/chat_proxy.jsonl`
2. Reduce batch size
3. Increase timeout in Advanced Options
4. Try dry-run to verify configuration
5. Check GPU memory usage

#### GUI is Slow

**Solutions**:
1. Clear cache: Settings → General → Clear All Caches
2. Reduce items per page: Settings → Appearance
3. Restart Streamlit app
4. Check system resources (RAM, GPU)

#### Stale Data

**Solutions**:
1. Click **🔄 Refresh** in sidebar
2. Clear cache in Settings
3. Restart app

#### Cannot Read/Write Files

**Solutions**:
1. Check file permissions
2. Verify paths exist: Settings → Paths
3. Check disk space
4. Use absolute paths

### Getting Help

1. **In-App Help**: Click ❓ Help in sidebar
2. **Documentation**: `docs/index.md`
3. **Debug Mode**: Enable for detailed error messages
4. **Logs**: Check `logs/chat_proxy.jsonl`

---

## Keyboard Shortcuts

(Streamlit default shortcuts)

- `R`: Rerun application
- `C`: Clear cache
- `?`: Show keyboard shortcuts
- `/`: Focus search

---

## Advanced Topics

### Custom Presets

Create custom presets by:
1. Use Advanced Options to configure
2. Export settings: Settings → About → Export Configuration
3. Save as preset file
4. Share with team

### Batch Processing Scripts

For very large batches, consider:
1. Use CLI tools directly (bypass GUI)
2. Enable caching
3. Process in chunks
4. Monitor with Results Browser

### Integration with External Tools

Results are saved in standard formats:
- **JSONL**: One JSON object per line
- **Markdown**: Human-readable summaries
- **Image metadata**: Standard EXIF/XMP

Parse with:
- Python: `json.loads()` per line
- jq: `jq -c . outputs/results.jsonl`
- ExifTool: `exiftool image.jpg`

---

## Appendix

### File Locations

```
imageworks/
├── outputs/           # Module outputs
│   ├── mono_results.jsonl
│   ├── similarity_results.jsonl
│   ├── tagger_results.jsonl
│   ├── judge_vision.jsonl
│   └── narrator_results.jsonl
├── logs/             # Application logs
│   └── chat_proxy.jsonl
├── configs/          # Configuration files
│   └── model_registry.json
└── models/           # Downloaded models
```

### Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- WebP (.webp)
- BMP (.bmp)
- TIFF (.tiff, .tif)

### Backend Requirements

- **vLLM**: CUDA GPU, 8GB+ VRAM
- **LMDeploy**: CUDA GPU, 6GB+ VRAM
- **Ollama**: CPU or GPU
- **Chat Proxy**: Lightweight, any system

---

## Quick Reference Card

### Most Common Tasks

| Task | Steps |
|------|-------|
| Check for monochrome images | Mono Checker → Standard → Select directory → Run |
| Find duplicates | Image Similarity → Quick → Add candidates + library → Run |
| Add tags to images | Personal Tagger → Preview → Edit → Commit |
| Describe colors | Color Narrator → Standard → Select images → Run |
| Download model | Models → Download → Enter ID → Select variant → Download |
| View results | Results → Browse → Select module → View file |
| Re-run job | Results → History → Find job → Re-run |

### Default Shortcuts

| Action | Shortcut |
|--------|----------|
| Refresh app | R |
| Clear cache | C |
| Show help | ? |
| Toggle sidebar | [ |

---

**Version**: 1.0
**Last Updated**: 2024

For more information, see the [main ImageWorks documentation](../index.md).
