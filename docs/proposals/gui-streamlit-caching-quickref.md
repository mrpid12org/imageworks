# Streamlit Caching Quick Reference for ImageWorks GUI

**⚠️ CRITICAL: Read this before writing any Streamlit page**

## The Problem

Streamlit **reruns your entire script on every interaction**. This means:
- Slider change → full script rerun
- Text input → full script rerun
- Opening an expander → full script rerun
- **Switching tabs → full script rerun**

For ImageWorks, this is catastrophic:
- VLM inference: 4-30 seconds per image
- Directory scan: 1-5 seconds for 1000+ images
- Model loading: 10-30 seconds for 7B models
- Embedding computation: 5-60 seconds for batches

## The Solution: Three-Layer Defense

### Layer 1: Button Gating (Primary)

**Rule**: ALL expensive operations MUST be behind buttons.

```python
# ✅ GOOD
if st.button("▶️ Run Analysis"):
    with st.spinner("Running..."):
        results = expensive_vlm_inference(images)  # Only runs on click
        st.session_state["results"] = results  # Persist!

# Display cached results
if "results" in st.session_state:
    display_results(st.session_state["results"])
```

```python
# ❌ BAD - Runs on every rerun!
threshold = st.slider("Threshold", 0, 1, 0.5)
results = expensive_vlm_inference(images)  # 🔥 DISASTER!
```

### Layer 2: Data Caching

**Use `@st.cache_data`** for functions returning serializable data (lists, dicts, DataFrames, arrays).

```python
# ✅ GOOD - File discovery cached
@st.cache_data(ttl=600)  # Cache for 10 minutes
def discover_images(directory: str) -> list[str]:
    """Scan directory (expensive for large dirs)."""
    return [str(p) for p in Path(directory).glob("**/*.jpg")]

# Only runs once per unique directory
images = discover_images("/path/to/images")
```

**Cache Invalidation**:
```python
if st.button("🔄 Refresh"):
    discover_images.clear()  # Clear function cache
    # st.cache_data.clear()  # Nuclear option
```

### Layer 3: Resource Caching

**Use `@st.cache_resource`** for non-serializable objects (models, connections, API clients).

```python
# ✅ GOOD - Model loaded once
@st.cache_resource
def load_vlm_model(model_name: str):
    """Load VLM model (very expensive)."""
    from transformers import AutoModel
    model = AutoModel.from_pretrained(model_name)
    return model

# Model persists across all sessions/reruns
model = load_vlm_model("Qwen2.5-VL-7B-AWQ")
```

## ImageWorks-Specific Patterns

### Pattern 1: Configure → Execute → Results

```python
tabs = st.tabs(["⚙️ Configure", "▶️ Execute", "📊 Results"])

# Tab 1: Config (cheap, can rerun)
with tabs[0]:
    directory = st.text_input("Directory")
    threshold = st.slider("Threshold", 0.0, 1.0)

    # Store immediately!
    st.session_state["config"] = {"directory": directory, "threshold": threshold}

# Tab 2: Execute (GATED)
with tabs[1]:
    config = st.session_state.get("config", {})

    if st.button("▶️ Run", type="primary"):  # GATE!
        results = expensive_analysis(config)
        st.session_state["results"] = results  # PERSIST!

# Tab 3: Results (cheap display)
with tabs[2]:
    if "results" in st.session_state:
        display_results(st.session_state["results"])
```

### Pattern 2: Cached File Discovery

```python
@st.cache_data(ttl=300)
def scan_directory(path: str, recursive: bool) -> list[dict]:
    """Expensive file scan - cache it!"""
    files = []
    pattern = "**/*" if recursive else "*"
    for f in Path(path).glob(f"{pattern}.jpg"):
        files.append({"path": str(f), "size": f.stat().st_size})
    return files

# Use cached results
all_files = scan_directory(directory, recursive=True)

# Filtering is cheap (can rerun)
size_filter = st.slider("Min Size (MB)", 0, 50, 1)
filtered = [f for f in all_files if f["size"] > size_filter * 1024**2]
```

### Pattern 3: Preview → Edit → Commit (Personal Tagger)

```python
# Step 1: Preview (GATED)
if st.button("🔍 Preview Tags"):
    tags = generate_tags(images)  # VLM inference
    st.session_state["preview_tags"] = tags

# Step 2: Edit (cheap - uses cached preview)
if "preview_tags" in st.session_state:
    for i, tag in enumerate(st.session_state["preview_tags"]):
        edited = st.text_area(f"Tag {i}", value=tag["text"], key=f"tag_{i}")
        st.session_state["preview_tags"][i]["text"] = edited

# Step 3: Commit (GATED, uses edited tags)
if st.button("✅ Write Metadata"):
    write_metadata(st.session_state["preview_tags"])
```

## Anti-Patterns to AVOID

### ❌ Inference Without Button

```python
# ❌ BAD - Runs on every slider change!
images = st.file_uploader("Images", accept_multiple_files=True)
threshold = st.slider("Threshold", 0, 1, 0.5)

for img in images:
    result = vlm_inference(img)  # 🔥 Runs when you touch ANYTHING!
    st.write(result)
```

### ❌ Uncached Heavy I/O

```python
# ❌ BAD - Rescans on every rerun
directory = st.text_input("Directory")
files = list(Path(directory).glob("**/*.jpg"))  # 🔥 Slow!

# Every widget change → full rescan!
```

### ❌ Conditional Without State

```python
# ❌ BAD - Results disappear on rerun
if st.checkbox("Run Analysis"):
    results = expensive()  # 🔥 Reruns while checked!
    st.write(results)

# ✅ GOOD - Use button + state
if st.button("Run"):
    st.session_state["results"] = expensive()
if "results" in st.session_state:
    st.write(st.session_state["results"])
```

## Pre-Commit Checklist

Before pushing any Streamlit page:

- [ ] All VLM/inference calls behind `st.button()`?
- [ ] File discovery uses `@st.cache_data`?
- [ ] Model loading uses `@st.cache_resource`?
- [ ] Results stored in `st.session_state`?
- [ ] Tested: adjust slider → expensive ops don't rerun?
- [ ] Tested: switch tabs → no re-execution?
- [ ] Cache clear button provided?

## Testing

```python
# Add debug panel to every page during development
if st.checkbox("🐛 Debug", value=False):
    st.write("Session State:", st.session_state)
    st.write("Rerun #:", st.session_state.get("rerun_count", 0))
    st.session_state["rerun_count"] = st.session_state.get("rerun_count", 0) + 1
```

Watch rerun counter - if it increments on unrelated widget changes, you have a caching problem.

## Reference

- Streamlit Caching: https://docs.streamlit.io/develop/concepts/architecture/caching
- Session State: https://docs.streamlit.io/develop/concepts/architecture/session-state
- Forms: https://docs.streamlit.io/develop/api-reference/execution-flow/st.form

---

**Remember**: If it takes >1 second, it MUST be cached or gated. No exceptions.
