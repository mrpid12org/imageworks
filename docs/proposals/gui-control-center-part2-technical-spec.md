# ImageWorks GUI Control Center - Part 2: Technical Specification

**Document Status**: Implementation Blueprint
**Date**: October 26, 2025
**Approach**: Option C - Hybrid (Presets + Advanced Options Expander)

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Technology Stack](#technology-stack)
3. [⚠️ Streamlit Performance & Caching Strategy](#streamlit-performance--caching-strategy)
4. [Application Structure](#application-structure)
5. [Preset System Design](#preset-system-design)
6. [Shared Component Library](#shared-component-library)
7. [Module-Specific Pages](#module-specific-pages)
8. [Implementation Phases](#implementation-phases)
9. [Development Estimates](#development-estimates)
10. [Code Examples](#code-examples)
11. [Testing Strategy](#testing-strategy)

---

## Architecture Overview

### Design Philosophy: Option C Hybrid

**Principle**: "Simple by default, powerful when needed"

```
┌─────────────────────────────────────────────┐
│         User Interaction Layers              │
├─────────────────────────────────────────────┤
│                                             │
│  Layer 1: PRESET SELECTOR (Beginner)       │
│  ┌──────────────────────────────────────┐  │
│  │ [ Quick ] [ Standard ] [ Thorough ]  │  │
│  │  Single click = 90% of use cases     │  │
│  └──────────────────────────────────────┘  │
│                                             │
│  Layer 2: COMMON OVERRIDES (Intermediate)  │
│  ┌──────────────────────────────────────┐  │
│  │ [x] Write Metadata  [x] Dry Run      │  │
│  │ Fail Threshold: [0.92]               │  │
│  │ Output: [outputs/results/...]        │  │
│  └──────────────────────────────────────┘  │
│                                             │
│  Layer 3: ADVANCED OPTIONS (Expert)        │
│  ┌──────────────────────────────────────┐  │
│  │ ▶ Show Advanced Options              │  │
│  │   (Expander collapsed by default)     │  │
│  └──────────────────────────────────────┘  │
│                                             │
│  When expanded:                             │
│  ┌──────────────────────────────────────┐  │
│  │ ▼ Embedding Configuration            │  │
│  │   Backend: [siglip ▼]                │  │
│  │   Model: [google/siglip-...  ▼]      │  │
│  │   Metric: [cosine ▼]                 │  │
│  │                                       │  │
│  │ ▼ Augmentation Options               │  │
│  │   [x] Pooling  [x] Grayscale         │  │
│  │   [x] Five-crop  Ratio: [0.875]      │  │
│  │                                       │  │
│  │ ▼ Cache & Performance                │  │
│  │   [ ] Refresh cache  TTL: [3600]     │  │
│  │   [x] Performance metrics            │  │
│  └──────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

**Key Benefits**:
1. **95% of users** use presets → fast onboarding
2. **Intermediate users** adjust 2-3 common overrides
3. **Expert users** expand for full control
4. **Progressive disclosure** reduces cognitive load

---

## Technology Stack

### Primary Framework: Streamlit

**Rationale**:
- ✅ Python-native (no context switching)
- ✅ Rapid prototyping (100-150 lines per page)
- ✅ Built-in widgets (file picker, sliders, tabs)
- ✅ Session state management
- ✅ Deployment options (local, Docker, cloud)

**Alternatives Rejected**:
- ❌ Dify/LangFlow: Wrong tool (LLM workflows, not filesystem ops)
- ❌ Gradio: Less flexible layout control
- ❌ Tkinter: Desktop-only, requires packaging
- ❌ Rich TUI: No image viewing, limited interactivity
- ❌ VS Code Tasks: Not discoverable for non-developers

**⚠️ CRITICAL RISK: Streamlit Re-execution Model**

Streamlit **reruns the entire script** on every interaction (button click, slider change, text input). This is a **major performance hazard** when working with:
- VLM inference (4-30 seconds per image)
- Large image directories (thousands of files)
- Model loading/unloading
- Expensive embedding computations
- File I/O operations

**Example of Dangerous Code** (DO NOT USE):
```python
# ❌ BAD: This runs VLM inference on EVERY widget interaction!
images = st.file_uploader("Select images", accept_multiple_files=True)
for img in images:
    result = run_vlm_inference(img)  # 🔥 Runs when you adjust ANY slider!
    st.write(result)
```

**Mitigation Required** (see next section for detailed strategies):
1. Use `st.button()` to gate expensive operations
2. Apply `@st.cache_data` for pure functions (embeddings, file lists)
3. Apply `@st.cache_resource` for models/connections
4. Store results in `st.session_state` to survive reruns
5. Use `st.form()` to batch input changes

**This is non-negotiable** - failure to properly cache/gate operations will make the GUI unusable for real workloads.

### Supporting Libraries

| Library | Purpose | Version |
|---------|---------|---------|
| `streamlit` | Core framework | >= 1.28 |
| `streamlit-aggrid` | Advanced tables (registry browser) | >= 0.3 |
| `plotly` | Metrics charts | >= 5.0 |
| `Pillow` | Image display | >= 10.0 |
| `markdown` | Styled markdown rendering | >= 3.4 |
| `psutil` | Process monitoring (backends) | >= 5.9 |
| `watchdog` | File watching (logs) | >= 3.0 |

**No Additional Dependencies**:
- Reuse existing `imageworks` modules
- Subprocess wrappers for CLI tools
- Direct imports where possible

---

## ⚠️ Streamlit Performance & Caching Strategy

### The Rerun Problem

Streamlit's reactive model **reruns the entire script from top to bottom** on every user interaction. This includes:
- Changing a slider
- Typing in a text input
- Selecting a dropdown option
- Opening an expander
- **Navigating between tabs** (within same page)

**Critical Implications for ImageWorks**:

| Operation | Time | Danger if Uncached |
|-----------|------|-------------------|
| VLM inference (single image) | 4-30s | Runs on every slider adjustment! |
| Directory scan (1000+ images) | 1-5s | Rescans when you click anything |
| Model loading (7B VLM) | 10-30s | Reloads on every interaction |
| Embedding computation (batch) | 5-60s | Recomputes on every widget change |
| Registry parsing (large JSON) | 0.1-1s | Rereads file constantly |
| Image overlay generation | 1-3s | Regenerates when adjusting threshold |

### Mitigation Strategies

#### 1. Button Gating (Primary Defense)

**Rule**: All expensive operations MUST be behind `st.button()` or `st.form_submit_button()`

```python
# ✅ GOOD: Gated execution
if st.button("▶️ Run Analysis", type="primary"):
    with st.spinner("Running analysis..."):
        results = run_expensive_analysis(config)
        st.session_state["results"] = results  # Persist across reruns

# Display cached results
if "results" in st.session_state:
    display_results(st.session_state["results"])
```

```python
# ❌ BAD: Runs on every rerun
results = run_expensive_analysis(config)  # 🔥 Disaster!
display_results(results)
```

#### 2. Session State Persistence

**Rule**: Store all computation results in `st.session_state` to survive reruns

```python
# ✅ GOOD: Results persist
if st.button("Tag Images"):
    results = tag_images(directory)
    st.session_state["tagging_results"] = results
    st.session_state["tagging_complete"] = True

# Results available on all subsequent reruns
if st.session_state.get("tagging_complete"):
    st.success("Tagging completed!")
    edit_tags(st.session_state["tagging_results"])
```

#### 3. Data Caching (`@st.cache_data`)

**Use for**: Pure functions returning serializable data (lists, dicts, DataFrames)

```python
# ✅ GOOD: Cache expensive file operations
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_image_list(directory: str, recursive: bool) -> list[str]:
    """Scan directory for images (expensive for large dirs)."""
    images = []
    path = Path(directory)
    pattern = "**/*" if recursive else "*"
    for ext in [".jpg", ".jpeg", ".png"]:
        images.extend(path.glob(f"{pattern}{ext}"))
    return [str(p) for p in images]

# Only runs once per unique (directory, recursive) combination
images = load_image_list(selected_dir, recursive=True)
st.write(f"Found {len(images)} images")  # Instant on reruns
```

```python
# ✅ GOOD: Cache embedding computations
@st.cache_data(show_spinner="Computing embeddings...")
def compute_embeddings(image_paths: list[str], model: str) -> np.ndarray:
    """Compute embeddings (very expensive)."""
    embeddings = []
    for img_path in image_paths:
        emb = embedding_model.encode(img_path)
        embeddings.append(emb)
    return np.array(embeddings)
```

**Cache Invalidation**:
```python
# Clear cache when needed
if st.button("🔄 Refresh File List"):
    load_image_list.clear()  # Clear specific function cache
    st.cache_data.clear()     # Nuclear option: clear all data caches
```

#### 4. Resource Caching (`@st.cache_resource`)

**Use for**: Non-serializable resources (models, database connections, API clients)

```python
# ✅ GOOD: Load model once across all sessions
@st.cache_resource
def load_embedding_model(model_name: str):
    """Load embedding model (expensive, non-serializable)."""
    import torch
    from transformers import AutoModel
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model

# Model loaded once, reused for all users/sessions
model = load_embedding_model("google/siglip-base-patch16-224")
```

```python
# ✅ GOOD: Cache backend clients
@st.cache_resource
def get_vlm_client(base_url: str):
    """Create VLM API client (reusable connection)."""
    from openai import OpenAI
    return OpenAI(base_url=base_url, api_key="EMPTY")

client = get_vlm_client("http://localhost:8100/v1")
```

#### 5. Form Batching

**Use for**: Grouping multiple inputs to prevent premature execution

```python
# ✅ GOOD: Form batches all input changes
with st.form("similarity_config"):
    fail_threshold = st.slider("Fail Threshold", 0.8, 1.0, 0.92)
    query_threshold = st.slider("Query Threshold", 0.7, 0.9, 0.82)
    strategies = st.multiselect("Strategies", ["embedding", "perceptual_hash"])

    # Nothing runs until user clicks submit
    submitted = st.form_submit_button("💾 Apply Configuration")

    if submitted:
        st.session_state["config"] = {
            "fail_threshold": fail_threshold,
            "query_threshold": query_threshold,
            "strategies": strategies,
        }
        st.success("Configuration saved!")
```

```python
# ❌ BAD: Reruns on every slider adjustment
fail_threshold = st.slider("Fail Threshold", 0.8, 1.0, 0.92)  # Rerun!
query_threshold = st.slider("Query Threshold", 0.7, 0.9, 0.82)  # Rerun!
# User can't adjust both before triggering expensive code
```

### Mandatory Patterns for ImageWorks GUI

#### Pattern 1: Configuration → Execute → Review

```python
# Tab 1: Configure (cheap, reruns are fine)
with st.tabs(["⚙️ Configure", "▶️ Execute", "📊 Results"])[0]:
    directory = st.text_input("Directory", value="/path/to/images")
    threshold = st.slider("Threshold", 0.0, 1.0, 0.5)

    # Store in session state immediately
    st.session_state["config"] = {
        "directory": directory,
        "threshold": threshold,
    }

# Tab 2: Execute (expensive, gated by button)
with st.tabs(["⚙️ Configure", "▶️ Execute", "📊 Results"])[1]:
    if st.button("▶️ Run Mono Checker", type="primary"):
        config = st.session_state.get("config", {})

        # Expensive operation only runs when button clicked
        with st.spinner("Running analysis..."):
            results = run_mono_checker(config)
            st.session_state["mono_results"] = results

# Tab 3: Results (display cached, cheap)
with st.tabs(["⚙️ Configure", "▶️ Execute", "📊 Results"])[2]:
    if "mono_results" in st.session_state:
        display_results(st.session_state["mono_results"])
    else:
        st.info("No results yet. Run analysis first.")
```

#### Pattern 2: Preview → Edit → Commit (for Personal Tagger)

```python
# Step 1: Configure + Preview (gated)
if st.button("🔍 Preview Tags (Dry Run)"):
    with st.spinner("Generating tags..."):
        # VLM inference happens here (expensive!)
        preview_tags = generate_tags(directory, dry_run=True)
        st.session_state["preview_tags"] = preview_tags
        st.session_state["preview_complete"] = True

# Step 2: Edit (cheap, uses cached preview)
if st.session_state.get("preview_complete"):
    tags = st.session_state["preview_tags"]

    # User can edit without retriggering VLM
    for i, tag_data in enumerate(tags):
        st.image(tag_data["image"])
        edited_caption = st.text_area(
            f"Caption {i+1}",
            value=tag_data["caption"],
            key=f"caption_{i}"
        )
        # Store edits in session state
        st.session_state["preview_tags"][i]["caption"] = edited_caption

# Step 3: Commit (gated, uses edited tags from session state)
if st.button("✅ Write to Metadata", type="primary"):
    final_tags = st.session_state["preview_tags"]
    write_metadata(final_tags)
    st.success("Metadata written!")
```

#### Pattern 3: Cached File Discovery + Filtered Display

```python
# Cached file discovery (expensive, runs once per directory)
@st.cache_data(ttl=300)
def discover_images(directory: str, recursive: bool) -> list[dict]:
    """Scan directory and extract basic metadata."""
    images = []
    for img_path in Path(directory).glob("**/*.jpg" if recursive else "*.jpg"):
        images.append({
            "path": str(img_path),
            "name": img_path.name,
            "size": img_path.stat().st_size,
        })
    return images

# Discovery runs once, cached
all_images = discover_images(selected_dir, recursive=True)

# Filtering is cheap (reruns are fine)
size_filter = st.slider("Min Size (MB)", 0, 50, 1)
filtered = [img for img in all_images if img["size"] > size_filter * 1024 * 1024]

st.write(f"Showing {len(filtered)} of {len(all_images)} images")
display_image_grid(filtered)  # Cheap to display
```

### Anti-Patterns to AVOID

#### ❌ Anti-Pattern 1: Inference in Main Script Body

```python
# ❌ DISASTER: Runs VLM on every widget change!
directory = st.text_input("Directory")
images = list(Path(directory).glob("*.jpg"))

# 🔥 This runs on EVERY rerun (text input change, slider, anything!)
results = []
for img in images:
    result = vlm_inference(img)  # 5 seconds PER IMAGE!
    results.append(result)

st.write(results)  # GUI is unusable
```

#### ❌ Anti-Pattern 2: Uncached Heavy I/O

```python
# ❌ BAD: Rescans directory on every rerun
directory = st.text_input("Directory")
images = list(Path(directory).glob("**/*.jpg"))  # 🔥 Slow for large dirs

# User changes any widget → rescan entire directory!
st.write(f"Found {len(images)} images")
```

#### ❌ Anti-Pattern 3: Conditional Execution Without State

```python
# ❌ BAD: Loses results on next rerun
run_analysis = st.checkbox("Run Analysis")

if run_analysis:
    results = expensive_analysis()  # 🔥 Runs every rerun while checked!
    st.write(results)  # Results disappear if user unchecks box

# ✅ GOOD: Use session state + button
if st.button("Run Analysis"):
    results = expensive_analysis()
    st.session_state["results"] = results

if "results" in st.session_state:
    st.write(st.session_state["results"])
```

### Performance Testing Checklist

Before declaring any page complete, verify:

- [ ] All VLM/inference calls are behind `st.button()` gates
- [ ] File discovery uses `@st.cache_data` with appropriate TTL
- [ ] Model loading uses `@st.cache_resource`
- [ ] Results persist in `st.session_state`
- [ ] Configuration uses `st.form()` or stores in session state immediately
- [ ] Clicking unrelated widgets doesn't trigger expensive operations
- [ ] Cache clear buttons are provided where needed
- [ ] Spinner/progress indicators show for >1 second operations

### Debug Helpers

```python
# Add to top of page during development
if st.checkbox("🐛 Debug Mode", value=False):
    st.write("**Session State:**", st.session_state)
    st.write("**Cache Stats:**", st.cache_data.cache_info())
    st.write("**Rerun Count:**", st.session_state.get("rerun_count", 0))
    st.session_state["rerun_count"] = st.session_state.get("rerun_count", 0) + 1
```

---

## Application Structure

### Directory Layout

```
src/imageworks/gui/
├── app.py                          # Main Streamlit entry point
├── config.py                       # GUI-specific configuration
├── state.py                        # Session state management
├── presets.py                      # Preset definitions (all modules)
│
├── components/                     # Shared UI components
│   ├── __init__.py
│   ├── file_browser.py            # Directory/file picker
│   ├── process_runner.py          # Subprocess executor with logs
│   ├── preset_selector.py         # Preset dropdown + overrides
│   ├── results_viewer.py          # JSONL/markdown/image viewer
│   ├── markdown_renderer.py       # Styled markdown display
│   ├── image_viewer.py            # Grid + detail view + overlays
│   ├── registry_table.py          # Model registry browser
│   ├── job_history.py             # Recent runs + re-run
│   └── backend_monitor.py         # Backend status cards
│
├── pages/                          # Module-specific pages
│   ├── 1_🏠_Dashboard.py          # System overview
│   ├── 2_🎯_Models.py             # Model management hub
│   ├── 3_🖼️_Mono_Checker.py      # Mono workflow
│   ├── 4_🖼️_Image_Similarity.py   # Similarity checker
│   ├── 5_🖼️_Personal_Tagger.py    # Personal tagger
│   ├── 6_🖼️_Color_Narrator.py     # Color narrator
│   ├── 7_📊_Results.py            # Unified output browser
│   └── 8_⚙️_Settings.py           # Configuration editor
│
└── utils/                          # Helper functions
    ├── __init__.py
    ├── cli_wrapper.py             # Subprocess command builders
    ├── config_loader.py           # Read/write pyproject.toml
    ├── backend_client.py          # Backend health checks
    └── validation.py              # Input validation

scripts/
├── start_gui.py                    # Launch script (uv run imageworks-gui)

pyproject.toml
[project.scripts]
imageworks-gui = "imageworks.gui.app:main"
```

### Navigation Structure

**Sidebar Navigation** (Streamlit multi-page):
```
ImageWorks Control Center
├── 🏠 Dashboard
├── 🎯 Models
│   └── (Tabs: Registry | Download | Backends | Profiles)
├── 🖼️ Workflows
│   ├── Mono Checker
│   ├── Image Similarity
│   ├── Personal Tagger
│   └── Color Narrator
├── 📊 Results
└── ⚙️ Settings
```

---

## Preset System Design

### Core Preset Structure

```python
# src/imageworks/gui/presets.py

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

@dataclass
class PresetConfig:
    """Single preset configuration for a module."""
    name: str                          # "Quick", "Standard", "Thorough"
    description: str                   # User-facing explanation
    flags: Dict[str, Any]             # CLI flags and values
    hidden_flags: List[str] = field(default_factory=list)  # Flags not exposed in UI
    common_overrides: List[str] = field(default_factory=list)  # Flags user can override

@dataclass
class ModulePresets:
    """All presets for a single module."""
    module_name: str
    default_preset: str                # Default selection
    presets: Dict[str, PresetConfig]  # name -> config

    def get_preset(self, name: str) -> PresetConfig:
        return self.presets[name]

    def get_flags_with_overrides(
        self,
        preset_name: str,
        overrides: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge preset flags with user overrides."""
        preset = self.get_preset(preset_name)
        flags = preset.flags.copy()
        flags.update(overrides)
        return flags
```

### Image Similarity Presets (Example)

```python
IMAGE_SIMILARITY_PRESETS = ModulePresets(
    module_name="image_similarity",
    default_preset="standard",
    presets={
        "quick": PresetConfig(
            name="Quick",
            description="Fast duplicate detection using perceptual hashing only. "
                        "No embeddings or VLM explanations. Best for initial screening.",
            flags={
                "strategy": ["perceptual_hash"],
                "fail_threshold": 0.95,
                "query_threshold": 0.90,
                "top_matches": 5,
                "generate_explanations": False,
                "write_metadata": False,
                "enable_perf_metrics": False,
                "enable_augment_pooling": False,
            },
            hidden_flags=[
                "embedding_backend", "embedding_model", "similarity_metric",
                "augment_grayscale", "augment_five_crop", "augment_five_crop_ratio",
                "backend", "base_url", "model", "prompt_profile",
                "use_loader", "registry_model", "registry_capabilities",
                "refresh_library_cache", "manifest_ttl_seconds",
            ],
            common_overrides=[
                "fail_threshold", "query_threshold", "library_root",
                "output_jsonl", "summary_path", "dry_run",
            ],
        ),

        "standard": PresetConfig(
            name="Standard",
            description="Balanced approach using SigLIP embeddings + perceptual hash. "
                        "Recommended for most use cases. No VLM explanations.",
            flags={
                "strategy": ["embedding", "perceptual_hash"],
                "embedding_backend": "siglip",
                "embedding_model": "google/siglip-base-patch16-224",
                "similarity_metric": "cosine",
                "fail_threshold": 0.92,
                "query_threshold": 0.82,
                "top_matches": 10,
                "generate_explanations": False,
                "write_metadata": True,
                "backup_originals": True,
                "enable_perf_metrics": True,
                "enable_augment_pooling": False,
            },
            hidden_flags=[
                "augment_grayscale", "augment_five_crop", "augment_five_crop_ratio",
                "backend", "base_url", "model", "prompt_profile",
                "use_loader", "registry_model", "registry_capabilities",
                "refresh_library_cache", "manifest_ttl_seconds",
            ],
            common_overrides=[
                "fail_threshold", "query_threshold", "library_root",
                "output_jsonl", "summary_path", "dry_run",
                "strategies", "generate_explanations", "write_metadata",
            ],
        ),

        "thorough": PresetConfig(
            name="Thorough",
            description="Comprehensive analysis with large SigLIP model, augmentation pooling, "
                        "and VLM explanations via chat proxy. Slow but most accurate.",
            flags={
                "strategy": ["embedding", "perceptual_hash"],
                "embedding_backend": "siglip",
                "embedding_model": "google/siglip-large-patch16-384",
                "similarity_metric": "cosine",
                "fail_threshold": 0.90,
                "query_threshold": 0.80,
                "top_matches": 15,
                "enable_augment_pooling": True,
                "augment_grayscale": True,
                "augment_five_crop": True,
                "augment_five_crop_ratio": 0.875,
                "generate_explanations": True,
                "backend": "vllm",  # Use chat proxy
                "base_url": "http://localhost:8100/v1",
                "model": "Qwen2.5-VL-7B-AWQ",  # Will be role-resolved
                "prompt_profile": "detailed",
                "write_metadata": True,
                "backup_originals": True,
                "enable_perf_metrics": True,
            },
            hidden_flags=[
                "use_loader", "registry_model", "registry_capabilities",
                "refresh_library_cache", "manifest_ttl_seconds",
            ],
            common_overrides=[
                "fail_threshold", "query_threshold", "library_root",
                "output_jsonl", "summary_path", "dry_run",
                "strategies", "embedding_backend", "embedding_model",
                "enable_augment_pooling", "generate_explanations",
                "write_metadata",
            ],
        ),
    }
)
```

### Personal Tagger Presets

```python
PERSONAL_TAGGER_PRESETS = ModulePresets(
    module_name="personal_tagger",
    default_preset="full_tagging",
    presets={
        "quick_caption": PresetConfig(
            name="Quick Caption",
            description="Generate captions only (no keywords or descriptions). "
                        "Fast preview mode with dry-run enabled.",
            flags={
                "use_registry": True,
                "caption_role": "caption",
                "keyword_role": None,
                "description_role": None,
                "prompt_profile": "concise",
                "batch_size": 8,
                "max_workers": 2,
                "dry_run": True,
                "no_meta": False,
                "backup_originals": True,
                "preflight": True,
                "recursive": True,
            },
            hidden_flags=[
                "backend", "base_url", "model", "api_key", "timeout",
                "overwrite_metadata",
            ],
            common_overrides=[
                "input_paths", "summary_path", "output_jsonl",
                "dry_run", "prompt_profile", "batch_size",
            ],
        ),

        "full_tagging": PresetConfig(
            name="Full Tagging",
            description="Generate captions, keywords, and descriptions. "
                        "Recommended for Lightroom library imports.",
            flags={
                "use_registry": True,
                "caption_role": "caption",
                "keyword_role": "keywords",
                "description_role": "description",
                "prompt_profile": "balanced",
                "batch_size": 4,
                "max_workers": 2,
                "dry_run": True,  # Default to safe mode
                "no_meta": False,
                "backup_originals": True,
                "overwrite_metadata": False,
                "preflight": True,
                "recursive": True,
            },
            hidden_flags=[
                "backend", "base_url", "model", "api_key", "timeout",
            ],
            common_overrides=[
                "input_paths", "summary_path", "output_jsonl",
                "dry_run", "prompt_profile", "batch_size",
                "caption_role", "keyword_role", "description_role",
            ],
        ),

        "validate_existing": PresetConfig(
            name="Validate Existing",
            description="Check already-tagged images and regenerate missing metadata. "
                        "Useful for library maintenance.",
            flags={
                "use_registry": True,
                "caption_role": "caption",
                "keyword_role": "keywords",
                "description_role": "description",
                "prompt_profile": "balanced",
                "batch_size": 4,
                "max_workers": 2,
                "dry_run": True,
                "no_meta": True,  # Don't write, just report
                "backup_originals": True,
                "overwrite_metadata": False,
                "preflight": True,
                "recursive": True,
            },
            hidden_flags=[
                "backend", "base_url", "model", "api_key", "timeout",
            ],
            common_overrides=[
                "input_paths", "summary_path", "output_jsonl",
                "dry_run", "no_meta",
            ],
        ),
    }
)
```

### Mono Checker Presets

```python
MONO_CHECKER_PRESETS = ModulePresets(
    module_name="mono_checker",
    default_preset="balanced",
    presets={
        "strict": PresetConfig(
            name="Strict",
            description="Tight thresholds for competition-grade monochrome validation. "
                        "High sensitivity to color contamination.",
            flags={
                "rgb_delta_threshold": 8.0,
                "chroma_threshold": 3.0,
                "hue_consistency_threshold": 10.0,
                "min_contaminated_pixels": 50,
                "recursive": True,
                "dry_run": False,
                "backup_originals": True,
            },
            hidden_flags=[],
            common_overrides=[
                "input", "overlays", "output_jsonl", "summary",
                "dry_run", "rgb_delta_threshold", "chroma_threshold",
            ],
        ),

        "balanced": PresetConfig(
            name="Balanced",
            description="Recommended defaults for most monochrome checking. "
                        "Balance between false positives and false negatives.",
            flags={
                "rgb_delta_threshold": 10.0,
                "chroma_threshold": 5.0,
                "hue_consistency_threshold": 15.0,
                "min_contaminated_pixels": 100,
                "recursive": True,
                "dry_run": False,
                "backup_originals": True,
            },
            hidden_flags=[],
            common_overrides=[
                "input", "overlays", "output_jsonl", "summary",
                "dry_run", "rgb_delta_threshold", "chroma_threshold",
            ],
        ),

        "lenient": PresetConfig(
            name="Lenient",
            description="Relaxed thresholds for near-monochrome images. "
                        "Allows subtle toning (sepia, blue-tone).",
            flags={
                "rgb_delta_threshold": 15.0,
                "chroma_threshold": 8.0,
                "hue_consistency_threshold": 20.0,
                "min_contaminated_pixels": 200,
                "recursive": True,
                "dry_run": False,
                "backup_originals": True,
            },
            hidden_flags=[],
            common_overrides=[
                "input", "overlays", "output_jsonl", "summary",
                "dry_run", "rgb_delta_threshold", "chroma_threshold",
            ],
        ),
    }
)
```

### Color Narrator Presets

```python
COLOR_NARRATOR_PRESETS = ModulePresets(
    module_name="color_narrator",
    default_preset="lmdeploy_standard",
    presets={
        "vllm_quick": PresetConfig(
            name="vLLM Quick",
            description="Fast narration using vLLM backend with smaller model. "
                        "No region hints.",
            flags={
                "backend": "vllm",
                "vlm_base_url": "http://localhost:8000/v1",
                "vlm_model": "Qwen2-VL-2B-Instruct",
                "prompt": "1",  # Default prompt
                "regions": False,
                "min_contamination_level": 0.1,
                "require_overlays": True,
                "dry_run": False,
                "backup_original_files": True,
                "overwrite_existing_metadata": False,
            },
            hidden_flags=["vlm_api_key", "vlm_timeout"],
            common_overrides=[
                "images", "overlays", "mono_jsonl", "summary",
                "backend", "prompt", "dry_run",
            ],
        ),

        "lmdeploy_standard": PresetConfig(
            name="LMDeploy Standard",
            description="Recommended setup using LMDeploy + Qwen2.5-VL-7B-AWQ. "
                        "Includes region-based prompt hints.",
            flags={
                "backend": "lmdeploy",
                "vlm_base_url": "http://localhost:24001/v1",
                "vlm_model": "Qwen2.5-VL-7B-AWQ",
                "prompt": "2",  # Region-aware prompt
                "regions": True,
                "min_contamination_level": 0.1,
                "require_overlays": True,
                "dry_run": False,
                "backup_original_files": True,
                "overwrite_existing_metadata": False,
            },
            hidden_flags=["vlm_api_key", "vlm_timeout"],
            common_overrides=[
                "images", "overlays", "mono_jsonl", "summary",
                "prompt", "regions", "dry_run",
            ],
        ),

        "proxy_role_based": PresetConfig(
            name="Proxy (Role-Based)",
            description="Use chat proxy with automatic role-based model selection. "
                        "Requires Phase 2 integration.",
            flags={
                "backend": "vllm",  # Proxy appears as vLLM-compatible
                "vlm_base_url": "http://localhost:8100/v1",
                "vlm_model": "auto",  # Will be role-resolved
                "prompt": "2",
                "regions": True,
                "min_contamination_level": 0.1,
                "require_overlays": True,
                "dry_run": False,
                "backup_original_files": True,
                "overwrite_existing_metadata": False,
            },
            hidden_flags=["vlm_api_key", "vlm_timeout"],
            common_overrides=[
                "images", "overlays", "mono_jsonl", "summary",
                "prompt", "regions", "dry_run",
            ],
        ),
    }
)
```

---

## Shared Component Library

### 1. PresetSelector Component

```python
# src/imageworks/gui/components/preset_selector.py

import streamlit as st
from typing import Dict, Any, List, Optional
from ..presets import ModulePresets, PresetConfig

def render_preset_selector(
    module_presets: ModulePresets,
    session_key_prefix: str,
) -> Dict[str, Any]:
    """
    Render preset selector with common overrides and advanced expander.

    Returns:
        Dict of all flags (preset + overrides + advanced)
    """

    # Preset selection
    st.subheader("Configuration")

    preset_name = st.radio(
        "Preset",
        options=list(module_presets.presets.keys()),
        index=list(module_presets.presets.keys()).index(module_presets.default_preset),
        format_func=lambda x: module_presets.presets[x].name,
        key=f"{session_key_prefix}_preset",
        horizontal=True,
    )

    preset = module_presets.get_preset(preset_name)
    st.info(preset.description)

    # Common overrides (always visible)
    st.markdown("### Common Settings")
    overrides = {}

    for flag_name in preset.common_overrides:
        override_value = _render_flag_widget(
            flag_name,
            preset.flags.get(flag_name),
            key=f"{session_key_prefix}_override_{flag_name}"
        )
        if override_value != preset.flags.get(flag_name):
            overrides[flag_name] = override_value

    # Advanced options expander
    with st.expander("🔧 Advanced Options", expanded=False):
        st.markdown("Expert settings - modify with caution")

        advanced_overrides = {}
        for flag_name, flag_value in preset.flags.items():
            if flag_name not in preset.common_overrides and flag_name not in preset.hidden_flags:
                advanced_value = _render_flag_widget(
                    flag_name,
                    flag_value,
                    key=f"{session_key_prefix}_advanced_{flag_name}"
                )
                if advanced_value != flag_value:
                    advanced_overrides[flag_name] = advanced_value

        overrides.update(advanced_overrides)

    # Merge preset + overrides
    final_flags = module_presets.get_flags_with_overrides(preset_name, overrides)

    return final_flags


def _render_flag_widget(flag_name: str, default_value: Any, key: str) -> Any:
    """Render appropriate widget based on flag type."""

    # Boolean flags
    if isinstance(default_value, bool):
        return st.checkbox(
            _humanize_flag_name(flag_name),
            value=default_value,
            key=key,
        )

    # Numeric flags
    elif isinstance(default_value, (int, float)):
        if isinstance(default_value, float):
            return st.number_input(
                _humanize_flag_name(flag_name),
                value=default_value,
                format="%.3f",
                key=key,
            )
        else:
            return st.number_input(
                _humanize_flag_name(flag_name),
                value=default_value,
                key=key,
            )

    # String flags
    elif isinstance(default_value, str):
        return st.text_input(
            _humanize_flag_name(flag_name),
            value=default_value,
            key=key,
        )

    # List flags
    elif isinstance(default_value, list):
        return st.multiselect(
            _humanize_flag_name(flag_name),
            options=default_value,  # For now, use default as options
            default=default_value,
            key=key,
        )

    # None / unhandled
    else:
        return default_value


def _humanize_flag_name(flag_name: str) -> str:
    """Convert flag_name to Human Readable Label."""
    return flag_name.replace("_", " ").title()
```

### 2. ProcessRunner Component

```python
# src/imageworks/gui/components/process_runner.py

import streamlit as st
import subprocess
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class JobResult:
    command: List[str]
    start_time: datetime
    end_time: Optional[datetime]
    stdout: str
    stderr: str
    return_code: Optional[int]
    status: str  # "running", "success", "failed"

def run_cli_command(
    command: List[str],
    description: str,
    session_key: str,
) -> JobResult:
    """
    Execute CLI command with live output streaming.

    Args:
        command: Command and arguments as list
        description: User-facing description
        session_key: Unique key for this job in session state

    Returns:
        JobResult with execution details
    """

    st.markdown(f"### Running: {description}")

    # Command display
    with st.expander("Command Details", expanded=False):
        st.code(" ".join(command), language="bash")

    # Progress indicator
    progress_bar = st.progress(0)
    status_text = st.empty()
    output_container = st.container()

    # Initialize job
    job = JobResult(
        command=command,
        start_time=datetime.now(),
        end_time=None,
        stdout="",
        stderr="",
        return_code=None,
        status="running",
    )

    # Store in session state
    if "job_history" not in st.session_state:
        st.session_state.job_history = []
    st.session_state[session_key] = job

    # Execute
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        # Stream output
        stdout_lines = []
        stderr_lines = []

        while True:
            # Read stdout
            stdout_line = process.stdout.readline()
            if stdout_line:
                stdout_lines.append(stdout_line)
                with output_container:
                    st.text(stdout_line.rstrip())

            # Check if process finished
            if process.poll() is not None:
                break

            # Update progress (indeterminate)
            progress_bar.progress((int(time.time() * 10) % 100) / 100)
            time.sleep(0.1)

        # Capture remaining output
        remaining_stdout, remaining_stderr = process.communicate()
        if remaining_stdout:
            stdout_lines.append(remaining_stdout)
        if remaining_stderr:
            stderr_lines.append(remaining_stderr)

        job.stdout = "".join(stdout_lines)
        job.stderr = "".join(stderr_lines)
        job.return_code = process.returncode
        job.end_time = datetime.now()
        job.status = "success" if process.returncode == 0 else "failed"

        # Update UI
        progress_bar.progress(100)
        if job.status == "success":
            status_text.success(f"✅ Completed in {(job.end_time - job.start_time).total_seconds():.1f}s")
        else:
            status_text.error(f"❌ Failed with exit code {job.return_code}")
            if job.stderr:
                st.error(job.stderr)

    except Exception as e:
        job.status = "failed"
        job.end_time = datetime.now()
        job.stderr = str(e)
        status_text.error(f"❌ Error: {e}")

    # Add to history
    st.session_state.job_history.append(job)

    return job
```

### 3. FileBrowser Component

```python
# src/imageworks/gui/components/file_browser.py

import streamlit as st
from pathlib import Path
from typing import List, Optional

def render_file_browser(
    label: str,
    default_path: Optional[Path] = None,
    file_types: Optional[List[str]] = None,
    allow_multiple: bool = False,
    session_key: str = "file_browser",
) -> Optional[List[Path]]:
    """
    Render directory/file browser with history.

    Args:
        label: Display label
        default_path: Initial directory
        file_types: Filter by extensions (e.g., [".jpg", ".png"])
        allow_multiple: Allow multiple file selection
        session_key: Session state key

    Returns:
        List of selected Paths (or None if no selection)
    """

    st.markdown(f"### {label}")

    # Path history
    if f"{session_key}_history" not in st.session_state:
        st.session_state[f"{session_key}_history"] = []

    # Quick select from history
    if st.session_state[f"{session_key}_history"]:
        quick_select = st.selectbox(
            "Recent Paths",
            options=[""] + st.session_state[f"{session_key}_history"],
            key=f"{session_key}_quick",
        )
        if quick_select:
            default_path = Path(quick_select)

    # Manual path input
    path_input = st.text_input(
        "Path",
        value=str(default_path) if default_path else "",
        key=f"{session_key}_input",
    )

    if not path_input:
        return None

    current_path = Path(path_input)

    # Validate path
    if not current_path.exists():
        st.error(f"Path does not exist: {current_path}")
        return None

    # Directory listing
    if current_path.is_dir():
        st.markdown("**Contents:**")

        # Parent directory button
        if current_path.parent != current_path:
            if st.button("⬆️ Parent Directory", key=f"{session_key}_parent"):
                st.session_state[f"{session_key}_input"] = str(current_path.parent)
                st.rerun()

        # List directory contents
        items = sorted(current_path.iterdir(), key=lambda p: (not p.is_dir(), p.name))

        selected_items = []
        cols = st.columns(4)

        for idx, item in enumerate(items):
            col = cols[idx % 4]

            with col:
                if item.is_dir():
                    if st.button(f"📁 {item.name}", key=f"{session_key}_dir_{idx}"):
                        st.session_state[f"{session_key}_input"] = str(item)
                        st.rerun()
                else:
                    # Filter by file type
                    if file_types and item.suffix.lower() not in file_types:
                        continue

                    if allow_multiple:
                        if st.checkbox(f"📄 {item.name}", key=f"{session_key}_file_{idx}"):
                            selected_items.append(item)
                    else:
                        if st.button(f"📄 {item.name}", key=f"{session_key}_file_{idx}"):
                            selected_items = [item]
                            break

        if selected_items:
            # Add to history
            for item in selected_items:
                path_str = str(item)
                if path_str not in st.session_state[f"{session_key}_history"]:
                    st.session_state[f"{session_key}_history"].insert(0, path_str)
                    # Keep only last 10
                    st.session_state[f"{session_key}_history"] = \
                        st.session_state[f"{session_key}_history"][:10]

            return selected_items

    else:
        # Single file selected
        st.success(f"Selected: {current_path.name}")

        # Add to history
        path_str = str(current_path)
        if path_str not in st.session_state[f"{session_key}_history"]:
            st.session_state[f"{session_key}_history"].insert(0, path_str)
            st.session_state[f"{session_key}_history"] = \
                st.session_state[f"{session_key}_history"][:10]

        return [current_path]

    return None
```

### 4. ResultsViewer Component

```python
# src/imageworks/gui/components/results_viewer.py

import streamlit as st
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

def render_results_viewer(
    jsonl_path: Optional[Path] = None,
    markdown_path: Optional[Path] = None,
):
    """
    Unified results viewer for JSONL + Markdown outputs.
    """

    st.markdown("## Results")

    tabs = st.tabs(["📊 Table View", "📝 Markdown", "🔍 Details"])

    # Tab 1: JSONL as table
    with tabs[0]:
        if jsonl_path and jsonl_path.exists():
            records = _load_jsonl(jsonl_path)
            if records:
                df = pd.DataFrame(records)

                # Add filters
                st.markdown("### Filters")
                cols = st.columns(3)

                filters = {}
                if "verdict" in df.columns:
                    with cols[0]:
                        verdict_filter = st.multiselect(
                            "Verdict",
                            options=df["verdict"].unique(),
                            default=df["verdict"].unique(),
                        )
                        filters["verdict"] = verdict_filter

                # Apply filters
                filtered_df = df.copy()
                for col, values in filters.items():
                    filtered_df = filtered_df[filtered_df[col].isin(values)]

                # Display
                st.dataframe(
                    filtered_df,
                    use_container_width=True,
                    height=600,
                )

                st.markdown(f"**Showing {len(filtered_df)} of {len(df)} records**")
            else:
                st.warning("No records found in JSONL file")
        else:
            st.info("No JSONL file provided")

    # Tab 2: Markdown
    with tabs[1]:
        if markdown_path and markdown_path.exists():
            markdown_content = markdown_path.read_text()
            st.markdown(markdown_content)
        else:
            st.info("No markdown file provided")

    # Tab 3: Raw details
    with tabs[2]:
        if jsonl_path and jsonl_path.exists():
            records = _load_jsonl(jsonl_path)
            if records:
                selected_idx = st.selectbox(
                    "Select Record",
                    options=range(len(records)),
                    format_func=lambda i: f"Record {i+1}",
                )

                st.json(records[selected_idx])
        else:
            st.info("No JSONL file provided")


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file into list of dicts."""
    records = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
```

### 5. ImageViewer Component

```python
# src/imageworks/gui/components/image_viewer.py

import streamlit as st
from pathlib import Path
from typing import List, Optional
from PIL import Image

def render_image_viewer(
    image_paths: List[Path],
    overlay_paths: Optional[List[Path]] = None,
    grid_cols: int = 4,
):
    """
    Display images in grid with optional overlays.

    Args:
        image_paths: List of image file paths
        overlay_paths: Optional list of overlay images (must match length)
        grid_cols: Number of columns in grid
    """

    st.markdown("## Images")

    # View mode
    view_mode = st.radio(
        "View Mode",
        options=["Grid", "Detail"],
        horizontal=True,
        key="image_viewer_mode",
    )

    if view_mode == "Grid":
        _render_grid_view(image_paths, overlay_paths, grid_cols)
    else:
        _render_detail_view(image_paths, overlay_paths)


def _render_grid_view(
    image_paths: List[Path],
    overlay_paths: Optional[List[Path]],
    grid_cols: int,
):
    """Render thumbnail grid."""

    show_overlays = False
    if overlay_paths:
        show_overlays = st.checkbox("Show Overlays", value=False)

    cols = st.columns(grid_cols)

    for idx, img_path in enumerate(image_paths):
        col = cols[idx % grid_cols]

        with col:
            try:
                img = Image.open(img_path)
                img.thumbnail((300, 300))

                # Show overlay or original
                if show_overlays and overlay_paths and idx < len(overlay_paths):
                    overlay_img = Image.open(overlay_paths[idx])
                    overlay_img.thumbnail((300, 300))
                    st.image(overlay_img, caption=img_path.name)
                else:
                    st.image(img, caption=img_path.name)

                # Select button
                if st.button("View", key=f"img_select_{idx}"):
                    st.session_state["selected_image_idx"] = idx
                    st.session_state["image_viewer_mode"] = "Detail"
                    st.rerun()

            except Exception as e:
                st.error(f"Failed to load {img_path.name}: {e}")


def _render_detail_view(
    image_paths: List[Path],
    overlay_paths: Optional[List[Path]],
):
    """Render single image with controls."""

    if "selected_image_idx" not in st.session_state:
        st.session_state["selected_image_idx"] = 0

    idx = st.session_state["selected_image_idx"]

    # Navigation
    cols = st.columns([1, 3, 1])
    with cols[0]:
        if st.button("⬅️ Previous") and idx > 0:
            st.session_state["selected_image_idx"] -= 1
            st.rerun()
    with cols[1]:
        st.markdown(f"**{idx + 1} / {len(image_paths)}**: {image_paths[idx].name}")
    with cols[2]:
        if st.button("Next ➡️") and idx < len(image_paths) - 1:
            st.session_state["selected_image_idx"] += 1
            st.rerun()

    # Image display
    try:
        img = Image.open(image_paths[idx])

        # Toggle overlay
        if overlay_paths and idx < len(overlay_paths):
            show_overlay = st.checkbox("Show Overlay", value=False)
            if show_overlay:
                overlay_img = Image.open(overlay_paths[idx])
                st.image(overlay_img, use_container_width=True)
            else:
                st.image(img, use_container_width=True)
        else:
            st.image(img, use_container_width=True)

        # Image info
        with st.expander("Image Info"):
            st.write(f"**Path**: {image_paths[idx]}")
            st.write(f"**Size**: {img.size}")
            st.write(f"**Mode**: {img.mode}")
            if overlay_paths and idx < len(overlay_paths):
                st.write(f"**Overlay**: {overlay_paths[idx]}")

    except Exception as e:
        st.error(f"Failed to load image: {e}")
```

---

## Module-Specific Pages

### Page: Image Similarity (Full Example with Caching)

```python
# src/imageworks/gui/pages/4_🖼️_Image_Similarity.py

import streamlit as st
from pathlib import Path
from ..components.preset_selector import render_preset_selector
from ..components.file_browser import render_file_browser
from ..components.process_runner import run_cli_command
from ..components.results_viewer import render_results_viewer
from ..presets import IMAGE_SIMILARITY_PRESETS
from ..utils.cli_wrapper import build_similarity_command

st.set_page_config(page_title="Image Similarity", page_icon="🖼️", layout="wide")

st.title("🔍 Image Similarity Checker")
st.markdown("Flag competition entries that match or resemble prior submissions")

# ⚠️ CACHING: File discovery is expensive - cache it!
@st.cache_data(ttl=600)  # Cache for 10 minutes
def discover_candidate_files(directory: str, extensions: list[str]) -> list[str]:
    """Scan directory for candidate images (cached)."""
    files = []
    dir_path = Path(directory)
    if not dir_path.exists():
        return []
    for ext in extensions:
        files.extend([str(p) for p in dir_path.glob(f"**/*{ext}")])
    return sorted(files)

# Tabs for workflow (Configure → Execute → Review pattern)
tabs = st.tabs(["⚙️ Configure", "▶️ Execute", "📊 Results"])

# Tab 1: Configuration (lightweight, reruns are OK)
with tabs[0]:
    st.markdown("## Configuration")

    # ⚠️ IMPORTANT: Store config in session state immediately
    # This prevents losing configuration when navigating tabs

    # Candidate directory selection
    candidate_dir = st.text_input(
        "Candidate Directory",
        value=st.session_state.get("similarity_candidate_dir", ""),
        key="similarity_candidate_dir_input",
    )

    if candidate_dir:
        # Use cached file discovery
        candidates = discover_candidate_files(
            candidate_dir,
            [".jpg", ".jpeg", ".png", ".JPG", ".JPEG"]
        )
        st.success(f"Found {len(candidates)} candidate images")

        # Store in session state
        st.session_state["similarity_candidates"] = candidates
        st.session_state["similarity_candidate_dir"] = candidate_dir
    else:
        st.info("Enter a directory path to scan for candidates")

    # Library root
    library_root = st.text_input(
        "Library Root (Historical Submissions)",
        value=st.session_state.get(
            "similarity_library_root",
            "/mnt/d/Proper Photos/photos/ccc competition images"
        ),
        key="similarity_library_root_input",
    )
    st.session_state["similarity_library_root"] = library_root

    # Preset selector (handles its own session state)
    flags = render_preset_selector(
        module_presets=IMAGE_SIMILARITY_PRESETS,
        session_key_prefix="similarity",
    )

    # Merge flags with paths
    config = {
        "candidates": st.session_state.get("similarity_candidates", []),
        "library_root": library_root,
        **flags,
    }

    # Store complete config in session state
    st.session_state["similarity_config"] = config

    # Preview command (cheap to regenerate)
    if config.get("candidates"):
        with st.expander("Command Preview"):
            command = build_similarity_command(config)
            st.code(" ".join(command), language="bash")

    # Cache clear button
    if st.button("🔄 Refresh File List"):
        discover_candidate_files.clear()
        st.rerun()

# Tab 2: Execution (GATED by button - expensive operations here)
with tabs[1]:
    st.markdown("## Execute Analysis")

    # Retrieve config from session state
    if "similarity_config" not in st.session_state:
        st.warning("⚠️ Please configure settings in the Configure tab first")
    else:
        config = st.session_state["similarity_config"]

        # Show configuration summary
        st.markdown("### Ready to Run")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Candidates", len(config.get("candidates", [])))
        with col2:
            st.metric("Fail Threshold", config.get("fail_threshold", "N/A"))
        with col3:
            strategies = ", ".join(config.get("strategy", []))
            st.metric("Strategies", strategies or "None")

        # ⚠️ CRITICAL: Button gates expensive operation
        # Analysis only runs when user explicitly clicks
        if st.button("▶️ Run Similarity Check", type="primary", use_container_width=True):

            # Validation
            if not config.get("candidates"):
                st.error("No candidates selected!")
            elif not Path(config.get("library_root", "")).exists():
                st.error("Library root does not exist!")
            else:
                # Build command
                command = build_similarity_command(config)

                # Execute (subprocess handles progress display)
                job = run_cli_command(
                    command=command,
                    description="Image Similarity Check",
                    session_key="similarity_current_job",
                )

                # Store results in session state if successful
                if job.status == "success":
                    st.session_state["similarity_last_results"] = {
                        "jsonl": config.get("output_jsonl"),
                        "markdown": config.get("summary_path"),
                        "config": config,
                        "job": job,
                    }
                    st.success("✅ Analysis complete! View results in the Results tab.")
                    st.balloons()

        # Show previous job status (if any)
        if "similarity_current_job" in st.session_state:
            job = st.session_state["similarity_current_job"]
            with st.expander("Last Job Details"):
                st.write(f"**Status**: {job.status}")
                st.write(f"**Duration**: {(job.end_time - job.start_time).total_seconds():.1f}s")
                st.write(f"**Command**: `{' '.join(job.command)}`")

# Tab 3: Results (display from session state - cheap)
with tabs[2]:
    st.markdown("## Results")

    # ⚠️ Results retrieved from session state - no re-computation
    if "similarity_last_results" in st.session_state:
        results_data = st.session_state["similarity_last_results"]

        # Display results viewer (handles own caching internally)
        render_results_viewer(
            jsonl_path=Path(results_data["jsonl"]) if results_data.get("jsonl") else None,
            markdown_path=Path(results_data["markdown"]) if results_data.get("markdown") else None,
        )

        # Provide re-run option
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Re-run Analysis"):
                # Switch to Execute tab
                st.session_state["active_tab"] = "▶️ Execute"
                st.rerun()
        with col2:
            if st.button("🗑️ Clear Results"):
                del st.session_state["similarity_last_results"]
                st.rerun()
    else:
        st.info("📊 No results available yet. Run an analysis first.")
        if st.button("➡️ Go to Execute Tab"):
            st.session_state["active_tab"] = "▶️ Execute"
            st.rerun()
```

**Key Caching Patterns Demonstrated**:
1. ✅ File discovery cached with `@st.cache_data(ttl=600)`
2. ✅ Config stored in `st.session_state` immediately on change
3. ✅ Expensive analysis gated by `st.button()`
5. ✅ Manual cache clear button provided
6. ✅ Tab navigation doesn't trigger re-execution

---

## Implementation Phases

### Phase 1: Foundation (2 days / ~700 lines)

**Goal**: Working prototype with core infrastructure

**Deliverables**:
1. App scaffold with navigation ✅
2. Shared components library (5 core components) ✅
3. Dashboard page (system status) ✅
4. Settings page (read pyproject.toml) ✅
5. One complete workflow (image similarity with presets) ✅

**Files Created**:
- `src/imageworks/gui/app.py`
- `src/imageworks/gui/presets.py`
- `src/imageworks/gui/components/*.py` (5 files)
- `src/imageworks/gui/pages/1_Dashboard.py`
- `src/imageworks/gui/pages/4_Image_Similarity.py`
- `src/imageworks/gui/pages/8_Settings.py`

**Testing Criteria**:
- Launch GUI successfully
- Select preset and generate command
- View system status
- Browse files and directories

---

### Phase 2: Core Workflows (2 days / ~500 lines)

**Goal**: Complete mono checker and model management

**Deliverables**:
1. Mono checker workflow (full cycle) ✅
2. Model registry browser ✅
3. Model downloader integration ✅
4. Backend status monitor ✅

**Files Created**:
- `src/imageworks/gui/pages/3_Mono_Checker.py`
- `src/imageworks/gui/pages/2_Models.py`
- `src/imageworks/gui/components/registry_table.py`
- `src/imageworks/gui/components/backend_monitor.py`

**Testing Criteria**:
- Run mono checker end-to-end
- View results with overlays
- Browse model registry
- Download a model from HuggingFace

---

### Phase 3: Advanced Workflows (2 days / ~550 lines)

**Goal**: Personal tagger and color narrator

**Deliverables**:
1. Personal tagger with preview/edit ✅
2. Color narrator with pipeline mode ✅
3. Results browser (unified) ✅
4. Job history ✅

**Files Created**:
- `src/imageworks/gui/pages/5_Personal_Tagger.py`
- `src/imageworks/gui/pages/6_Color_Narrator.py`
- `src/imageworks/gui/pages/7_Results.py`
- `src/imageworks/gui/components/metadata_editor.py`
- `src/imageworks/gui/components/job_history.py`

**Testing Criteria**:
- Tag images in dry-run mode
- Edit tags before writing
- Chain mono → narrator
- Re-run previous job

---

### Phase 4: Polish & Production (1 day / ~200 lines)

**Goal**: Production-ready release

**Deliverables**:
1. Error handling and validation ✅
2. Help documentation in-app ✅
3. Keyboard shortcuts ✅
4. Performance optimization ✅
5. Deployment packaging ✅

**Testing Criteria**:
- All workflows tested on real data
- Error messages are actionable
- No performance regressions
- Can be deployed with Docker

---

## Development Estimates

| Phase | Duration | Lines of Code | Features |
|-------|----------|---------------|----------|
| Phase 1 | 2 days | 700 | Foundation + Image Similarity |
| Phase 2 | 2 days | 500 | Mono + Models |
| Phase 3 | 2 days | 550 | Tagger + Narrator |
| Phase 4 | 1 day | 200 | Polish |
| **TOTAL** | **7 days** | **~1,950 lines** | **Complete GUI** |

**Assumptions**:
- 6 hours/day focused development
- Existing CLI tools work correctly
- No major refactoring of backend modules
- Testing concurrent with development

**Risk Buffer**: +1-2 days for:
- Integration issues with existing CLI tools
- UI/UX iterations based on user feedback
- Additional preset tuning
- Documentation updates

**Realistic Estimate**: 8-9 days to production-ready v1.0

---

## Code Examples

### Main App Entry Point

```python
# src/imageworks/gui/app.py

import streamlit as st
from pathlib import Path

def main():
    st.set_page_config(
        page_title="ImageWorks Control Center",
        page_icon="🖼️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Sidebar branding
    with st.sidebar:
        st.title("🖼️ ImageWorks")
        st.markdown("Control Center v1.0")
        st.markdown("---")

        # System status quick view
        st.markdown("### System Status")

        # GPU info (from Phase 2)
        try:
            from imageworks.libs.hardware.gpu_detector import GPUDetector
            gpu_info = GPUDetector.detect_gpus()
            if gpu_info:
                gpu = gpu_info[0]
                st.success(f"✅ {gpu.name}")
                st.caption(f"{gpu.vram_total_mb} MB VRAM")
            else:
                st.warning("⚠️ No GPU detected")
        except Exception:
            st.error("❌ GPU detection failed")

        # Backend status
        st.markdown("### Backends")
        # TODO: Add backend health checks
        st.caption("Chat Proxy: :8100")
        st.caption("vLLM: :24001")

        st.markdown("---")

        # Quick links
        st.markdown("### Quick Actions")
        if st.button("🔄 Refresh Status"):
            st.rerun()
        if st.button("📝 View Logs"):
            st.session_state["show_logs"] = True

    # Main page (will be overridden by multi-page navigation)
    st.title("Welcome to ImageWorks Control Center")
    st.markdown("""
    Use the sidebar to navigate between modules:

    - 🏠 **Dashboard**: System overview and recent jobs
    - 🎯 **Models**: Manage models, backends, and profiles
    - 🖼️ **Workflows**: Run analysis tools (mono, similarity, tagger, narrator)
    - 📊 **Results**: Browse outputs and metrics
    - ⚙️ **Settings**: Configure defaults and presets
    """)

if __name__ == "__main__":
    main()
```

### CLI Wrapper Utility

```python
# src/imageworks/gui/utils/cli_wrapper.py

from pathlib import Path
from typing import List, Dict, Any

def build_similarity_command(config: Dict[str, Any]) -> List[str]:
    """Build imageworks-image-similarity command from config dict."""

    cmd = ["uv", "run", "imageworks-image-similarity", "check"]

    # Candidates (positional)
    candidates = config.get("candidates", [])
    for candidate in candidates:
        cmd.append(str(candidate))

    # Library root
    if config.get("library_root"):
        cmd.extend(["--library-root", str(config["library_root"])])

    # Strategies (repeatable)
    strategies = config.get("strategy", [])
    for strategy in strategies:
        cmd.extend(["--strategy", strategy])

    # Thresholds
    if "fail_threshold" in config:
        cmd.extend(["--fail-threshold", str(config["fail_threshold"])])
    if "query_threshold" in config:
        cmd.extend(["--query-threshold", str(config["query_threshold"])])

    # Embedding backend
    if config.get("embedding_backend"):
        cmd.extend(["--embedding-backend", config["embedding_backend"]])
    if config.get("embedding_model"):
        cmd.extend(["--embedding-model", config["embedding_model"]])

    # VLM explanations
    if config.get("generate_explanations"):
        cmd.append("--explain")
        if config.get("backend"):
            cmd.extend(["--backend", config["backend"]])
        if config.get("base_url"):
            cmd.extend(["--base-url", config["base_url"]])
        if config.get("model"):
            cmd.extend(["--model", config["model"]])

    # Augmentation
    if config.get("enable_augment_pooling"):
        cmd.append("--augment-pooling")
    if config.get("augment_grayscale"):
        cmd.append("--augment-grayscale")
    if config.get("augment_five_crop"):
        cmd.append("--augment-five-crop")

    # Metadata
    if config.get("write_metadata"):
        cmd.append("--write-metadata")
    if config.get("backup_originals"):
        cmd.append("--backup-originals")

    # Output paths
    if config.get("output_jsonl"):
        cmd.extend(["--output-jsonl", str(config["output_jsonl"])])
    if config.get("summary_path"):
        cmd.extend(["--summary", str(config["summary_path"])])

    # Dry run
    if config.get("dry_run"):
        cmd.append("--dry-run")

    return cmd


def build_mono_command(config: Dict[str, Any]) -> List[str]:
    """Build imageworks-mono command."""
    # Similar structure to similarity...
    pass


def build_tagger_command(config: Dict[str, Any]) -> List[str]:
    """Build imageworks-personal-tagger command."""
    # Similar structure...
    pass


def build_narrator_command(config: Dict[str, Any]) -> List[str]:
    """Build imageworks-color-narrator command."""
    # Similar structure...
    pass
```

---

## Testing Strategy

### Unit Tests

Test shared components in isolation:

```python
# tests/gui/test_preset_selector.py

def test_preset_flags_merge():
    from imageworks.gui.presets import IMAGE_SIMILARITY_PRESETS

    flags = IMAGE_SIMILARITY_PRESETS.get_flags_with_overrides(
        "standard",
        {"fail_threshold": 0.95}
    )

    assert flags["fail_threshold"] == 0.95  # Override applied
    assert flags["embedding_backend"] == "siglip"  # Preset default
    assert "strategy" in flags  # Preset included
```

### Integration Tests

Test full workflow execution:

```python
# tests/gui/test_similarity_workflow.py

def test_similarity_execution(tmp_path):
    """Test image similarity workflow end-to-end."""

    from imageworks.gui.utils.cli_wrapper import build_similarity_command
    from imageworks.gui.presets import IMAGE_SIMILARITY_PRESETS

    # Build config
    config = {
        "candidates": [tmp_path / "test.jpg"],
        "library_root": tmp_path / "library",
        **IMAGE_SIMILARITY_PRESETS.get_preset("quick").flags,
        "output_jsonl": tmp_path / "results.jsonl",
        "summary_path": tmp_path / "summary.md",
        "dry_run": True,
    }

    # Build command
    command = build_similarity_command(config)

    # Verify command structure
    assert "imageworks-image-similarity" in command
    assert "--strategy" in command
    assert "--dry-run" in command

    # TODO: Execute and verify output files created
```

### Manual Testing Checklist

**Performance/Caching Tests** (CRITICAL):
- [ ] ⚠️ Adjust slider → verify VLM inference does NOT re-run
- [ ] ⚠️ Change text input → verify file discovery uses cache
- [ ] ⚠️ Navigate between tabs → verify expensive ops don't re-trigger
- [ ] ⚠️ Click unrelated widget → verify analysis doesn't restart
- [ ] ⚠️ Run analysis once, adjust threshold, verify results persist
- [ ] ⚠️ Check session_state size doesn't grow unbounded

**Functional Tests**:
- [ ] Launch GUI without errors
- [ ] Navigate all pages
- [ ] Select preset and see description update
- [ ] Expand advanced options
- [ ] Browse files and directories
- [ ] Run mono checker (dry-run)
- [ ] View results in all tabs
- [ ] Download model from HuggingFace
- [ ] Browse model registry
- [ ] Check backend status
- [ ] Re-run job from history
- [ ] Edit pyproject.toml in settings
- [ ] Test with real competition images (1000+ files)

---

## Summary: Critical Performance Requirements

> **⚠️ THIS IS NON-NEGOTIABLE ⚠️**
>
> Streamlit's rerun model makes it **dangerously easy** to accidentally create unusable UIs. Every ImageWorks page MUST follow these patterns:
>
> ### Mandatory Checklist for Every Page
>
> 1. **Button Gating**: All VLM inference, embedding computation, and file processing MUST be behind `st.button()` or `st.form_submit_button()`
>
> 2. **File Discovery Caching**: Use `@st.cache_data(ttl=600)` for directory scanning - a directory with 5000 images takes 3-5 seconds to scan
>
> 3. **Session State Persistence**: Store ALL results in `st.session_state` immediately after generation - users must be able to navigate tabs/adjust widgets without losing results
>
> 4. **Model Loading**: Use `@st.cache_resource` for any model loading - a 7B VLM takes 15-30 seconds to load
>
> 5. **Configuration Isolation**: Store configuration in `st.session_state` as soon as widgets change - don't wait for execution button
>
> 6. **Tab Organization**: Use Configure → Execute → Results pattern with button gates between stages
>
> ### What Happens If You Don't Follow This
>
> - User adjusts a threshold slider → **4-second VLM inference re-runs**
> - User clicks an expander → **Directory with 1000 images re-scans**
> - User navigates to Results tab → **Entire analysis re-executes**
> - User types in text box → **Embedding model reloads (15 seconds)**
>
> This will make the GUI completely unusable and waste hours of debugging time.
>
> ### Code Review Requirement
>
> Before committing any new page:
> 1. Search for function calls that take >1 second
> 2. Verify they are either cached or button-gated
> 3. Test by adjusting an unrelated widget
> 4. Confirm expensive operation doesn't re-run

---

## Next Actions

1. **Create Directory Structure**:
   ```bash
   mkdir -p src/imageworks/gui/{components,pages,utils}
   touch src/imageworks/gui/{__init__,app,config,state,presets}.py
   ```

2. **Install Streamlit**:
   ```bash
   uv add streamlit streamlit-aggrid plotly
   ```

3. **Implement Phase 1** (Foundation):
   - Create `app.py` with sidebar navigation
   - Implement 5 shared components (with caching!)
   - Build dashboard page
   - Create image similarity page with presets (follow caching example)

4. **Test Phase 1**:
   ```bash
   uv run streamlit run src/imageworks/gui/app.py
   ```

   **Critical Performance Test**: Open browser dev tools, watch network tab. Adjust sliders/widgets. Verify no subprocess calls or heavy operations trigger.

5. **Iterate**: Gather feedback, adjust presets, refine UI

---

**Document Version**: 1.1
**Last Updated**: October 26, 2025 (Added critical caching requirements)
**Author**: GitHub Copilot (from user conversation)
**Implementation Status**: Ready for development with performance safeguards
