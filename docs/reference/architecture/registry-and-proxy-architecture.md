# Model Registry & Chat Proxy Architecture

**Last Updated:** 2025-10-26
**Status:** Living Document

## Overview

The ImageWorks system uses a **layered registry architecture** combined with a **smart chat proxy** to provide flexible model management with automatic backend orchestration. This document describes how these components work together.

---

## Table of Contents

1. [Registry Architecture](#registry-architecture)
2. [Registry Merging & Precedence](#registry-merging--precedence)
3. [Chat Proxy Architecture](#chat-proxy-architecture)
4. [Model Loading & Resolution](#model-loading--resolution)
5. [vLLM Manager & Autostart](#vllm-manager--autostart)
6. [Tips & Best Practices](#tips--best-practices)
7. [Troubleshooting](#troubleshooting)

---

## Registry Architecture

### Three-Layer Registry System

The registry consists of three JSON files:

1. **`model_registry.curated.json`** (Curated Layer)
   - Hand-maintained, stable metadata
   - Contains core model definitions without runtime state
   - Edits should be done manually or through controlled tooling
   - Does NOT contain: download paths, timestamps, performance metrics

2. **`model_registry.discovered.json`** (Discovered Layer)
   - Auto-generated and mutated by download tooling
   - Contains runtime state and dynamic fields
   - Overlays/overrides curated entries when models are downloaded
   - Safe to regenerate from scratch if needed

3. **`model_registry.json`** (Merged Snapshot)
   - **READ-ONLY** materialized union of curated + discovered
   - Auto-regenerated on each registry save
   - Provided for backward compatibility with external tools
   - The proxy and tools load this merged view

### File Locations

Default location: `configs/` directory (configurable via `IMAGEWORKS_REGISTRY_DIR`)

```
configs/
├── model_registry.curated.json      # Hand-maintained definitions
├── model_registry.discovered.json   # Auto-generated overlays
└── model_registry.json               # Merged read-only snapshot
```

### Dynamic vs Static Fields

**Dynamic fields** (discovered layer only):
- `download_path`, `download_location`, `download_format`
- `download_size_bytes`, `download_files`, `download_directory_checksum`
- `downloaded_at`, `last_accessed`
- `artifacts` (file hashes)
- `performance` (rolling metrics)
- `probes` (vision capability checks)
- `version_lock` (verification state)

**Static fields** (curated layer):
- `name`, `display_name`, `backend`
- `backend_config` (port, model_path, extra_args, host, base_url)
- `capabilities` (text, vision, tools, etc.)
- `generation_defaults` (temperature, top_p, etc.)
- `chat_template` (source, path, sha256)
- `roles`, `license`, `source`, `family`, `quantization`

---

## Registry Merging & Precedence

### How Layers Combine

When `load_registry()` is called:

1. **Load Both Layers**
   ```
   curated_raw = load("model_registry.curated.json")
   discovered_raw = load("model_registry.discovered.json")
   ```

2. **Merge Logic** (from `_merge_layered_fragments`):
   ```python
   # Start with discovered layer (base)
   merged = {entry["name"]: entry for entry in discovered_raw}

   # Overlay curated layer (overrides)
   for curated_entry in curated_raw:
       name = curated_entry["name"]
       if name in merged:
           # Deep merge: curated fields override discovered
           merged[name] = deep_merge(merged[name], curated_entry)
       else:
           # New curated-only entry
           merged[name] = curated_entry
   ```

3. **Precedence Rules**:
   - **Curated fields WIN** when both layers define the same field
   - **Discovered provides runtime state** that curated doesn't have
   - **Exception**: If an entry has `metadata.registry_layer = "discovered"`, it's treated as discovered-only

### Key Insight: Curated Overrides Discovered

**Example Scenario:**

```json
// discovered: has download state
{
  "name": "qwen3-vl-8b-instruct_(FP8)",
  "backend": "vllm",
  "backend_config": {
    "port": 24001,
    "extra_args": ["--max-model-len", "4096"]  // Auto-detected
  },
  "download_path": "/home/user/models/qwen3-vl-fp8",
  "downloaded_at": "2025-10-24T08:43:55Z"
}

// curated: manual overrides
{
  "name": "qwen3-vl-8b-instruct_(FP8)",
  "backend": "vllm",
  "backend_config": {
    "extra_args": ["--max-model-len", "6144"]  // Manual override
  },
  "chat_template": {
    "path": "/path/to/qwen_template.jinja"
  }
}

// RESULT (merged):
{
  "name": "qwen3-vl-8b-instruct_(FP8)",
  "backend": "vllm",
  "backend_config": {
    "port": 24001,
    "extra_args": ["--max-model-len", "6144"]  // ← Curated wins!
  },
  "chat_template": {
    "path": "/path/to/qwen_template.jinja"  // ← From curated
  },
  "download_path": "/home/user/models/qwen3-vl-fp8",  // ← From discovered
  "downloaded_at": "2025-10-24T08:43:55Z"  // ← From discovered
}
```

### Migration from Legacy Single-File

If only `model_registry.json` exists (legacy mode):

1. **Automatic Split** on first `load_registry()`:
   - Entries with `metadata.created_from_download = true` → discovered
   - Entries with `backend` in `{"ollama", "unassigned"}` → discovered
   - All others → curated

2. **Backup Created**: Original renamed to `model_registry.backup.pre_split.json`

3. **New Files Generated**:
   - `model_registry.curated.json`
   - `model_registry.discovered.json`
   - `model_registry.json` (regenerated as merged snapshot)

### Environment Variables

- `IMAGEWORKS_REGISTRY_DIR`: Override default `configs/` location
- `IMAGEWORKS_ALLOW_REGISTRY_DUPES`: Tolerate duplicate names (prints warning)
- `IMAGEWORKS_REGISTRY_NO_LAYERING`: Force single-file mode (bypass layering)

---

## Chat Proxy Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────┐
│  FastAPI App (app.py)                               │
│  - /v1/chat/completions                             │
│  - /v1/models                                        │
│  - /v1/metrics                                       │
└────────────┬────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────┐
│  ChatForwarder (forwarder.py)                       │
│  - Resolves model names                             │
│  - Validates capabilities                           │
│  - Routes to backends                               │
│  - Handles streaming                                │
└────────┬───────────────────┬────────────────────────┘
         │                   │
         ▼                   ▼
┌────────────────┐  ┌────────────────────────────────┐
│ AutostartMgr   │  │  Backend Targets               │
│ (autostart.py) │  │  - Ollama (port 11434)         │
└────────┬───────┘  │  - vLLM (port 24001)           │
         │          │  - LMDeploy (port 24001)        │
         ▼          └────────────────────────────────┘
┌────────────────────────────────────────────────────┐
│  VllmManager (vllm_manager.py)                     │
│  - Single-port vLLM orchestration                  │
│  - Process lifecycle management                    │
│  - Health checks & state persistence               │
└────────────────────────────────────────────────────┘
```

### Key Configuration (ProxyConfig)

From `chat_proxy/config.py`:

```python
@dataclass
class ProxyConfig:
    host: str = "127.0.0.1"
    port: int = 8100

    # Registry integration
    require_template: bool = True  # Enforce chat template presence
    include_non_installed: bool = False  # Filter downloaded-only

    # Backend orchestration
    autostart_enabled: bool = True
    autostart_grace_period_s: int = 120

    # vLLM-specific
    vllm_single_port: bool = True
    vllm_port: int = 24_001
    vllm_gpu_memory_utilization: float = 0.75
    vllm_max_model_len: int | None = None  # Global override
    vllm_start_timeout_s: int = 180

    # Image handling
    max_image_bytes: int = 6_000_000

    # Logging & metrics
    log_path: str = "logs/chat_proxy.jsonl"
    enable_metrics: bool = False
```

---

## Model Loading & Resolution

### Request Flow

1. **Client Request** → `/v1/chat/completions`
   ```json
   {
     "model": "qwen3-vl-8b-instruct_(FP8)",
     "messages": [...]
   }
   ```

2. **Model Resolution** (`ChatForwarder.handle_chat`):
   ```python
   # Step 1: Lookup by exact name
   try:
       entry = get_entry(model)
   except KeyError:
       # Step 2: Try display_name matching
       entry = resolve_by_display_name(model)
   ```

3. **Capability Validation**:
   ```python
   has_vision = supports_vision(entry)
   has_images = detect_images(payload["messages"])

   if has_images and not has_vision:
       raise err_capability_mismatch()

   if require_template and not entry.chat_template.path:
       raise err_template_required()
   ```

4. **Backend Resolution** (`_resolve_backend_base`):
   ```python
   # Priority order:
   # 1. backend_config.base_url (explicit override)
   # 2. backend_config.host + backend_config.port
   # 3. Default: 127.0.0.1 + default_port_for_backend

   base_url = entry.backend_config.base_url or \
              f"http://{host}:{port}/v1"
   ```

5. **Model ID Mapping**:
   ```python
   # Determine what ID to send to backend
   backend_id = entry.served_model_id or entry.name

   # Rewrite request for non-Ollama backends
   if entry.backend != "ollama":
       payload["model"] = backend_id
   ```

### Precedence for Backend Parameters

**When vLLM starts, arguments come from** (highest to lowest priority):

1. **`extra_args` in registry** (most specific)
   ```json
   "backend_config": {
     "extra_args": ["--max-model-len", "6144", "--kv-cache-dtype", "fp8"]
   }
   ```

2. **`chat_template.path` from registry** (if not in extra_args)
   ```json
   "chat_template": {
     "path": "/path/to/template.jinja"
   }
   ```

3. **`ProxyConfig` environment defaults** (if not in extra_args)
   ```python
   # From ProxyConfig:
   vllm_gpu_memory_utilization = 0.75
   vllm_max_model_len = None  # Only if set
   ```

4. **vLLM defaults** (fallback)

### Important: Duplicate Argument Handling

The `VllmManager._build_command()` method **checks for duplicates** before adding config defaults:

```python
def _build_command(self, entry, served_model_id, port):
    command = [python, "-m", "vllm.entrypoints.openai.api_server", ...]
    extra = entry.backend_config.extra_args

    # Only add if NOT in extra_args
    if cfg.vllm_gpu_memory_utilization and \
       "--gpu-memory-utilization" not in extra:
        command.extend(["--gpu-memory-utilization", str(cfg.vllm_gpu_memory_utilization)])

    if cfg.vllm_max_model_len and \
       "--max-model-len" not in extra:
        command.extend(["--max-model-len", str(cfg.vllm_max_model_len)])

    if entry.chat_template.path and \
       "--chat-template" not in extra:
        command.extend(["--chat-template", entry.chat_template.path])

    # Finally append extra_args (they win if there were duplicates)
    command.extend(extra)
```

**Key Insight**: Registry `extra_args` ALWAYS win because they're added last and vLLM typically uses the last occurrence of duplicate flags.

---

## vLLM Manager & Autostart

### Single-Port Orchestration

The `VllmManager` ensures **only one vLLM instance** runs at a time on the configured port (default: 24001).

### State Persistence

State file: `_staging/active_vllm.json`

```json
{
  "logical_name": "qwen3-vl-8b-instruct_(FP8)",
  "served_model_id": "qwen3-vl-8b-instruct_(FP8)",
  "port": 24001,
  "pid": 12345,
  "started_at": 1729760000.0
}
```

### Activation Flow

```python
async def activate(entry) -> ActiveVllmState:
    async with exclusive_lock:
        # 1. Check if already running
        state = load_state()
        if state.logical_name == entry.name and process_alive(state.pid):
            return state  # Reuse existing

        # 2. Stop existing if different model
        if state and process_alive(state.pid):
            terminate_process(state.pid)

        # 3. Build command from registry
        command = _build_command(entry, served_model_id, port)

        # 4. Launch process
        pid = subprocess.Popen(command).pid

        # 5. Wait for health check
        if not await wait_for_health(port, timeout=180):
            terminate_process(pid)
            raise VllmActivationError()

        # 6. Persist state
        write_state(ActiveVllmState(...))
        return state
```

### Health Check Logic

```python
async def _wait_for_health(port, started_at):
    deadline = started_at + cfg.vllm_start_timeout_s
    while time.time() < deadline:
        try:
            resp = await http.get(f"http://127.0.0.1:{port}/v1/models")
            if resp.status_code < 500:
                return True  # Success
        except:
            pass
        await asyncio.sleep(2)
    return False  # Timeout
```

### Autostart Behavior

When `autostart_enabled = True` (default):

1. **Proxy receives request** for vLLM model
2. **Checks if backend alive** (HTTP probe)
3. If **NOT alive**:
   - Calls `VllmManager.activate(entry)`
   - Waits for health check (up to `vllm_start_timeout_s`)
   - If successful, proceeds with request
   - If fails, returns 503 error

4. If **alive but wrong model**:
   - Stops current instance
   - Starts requested model
   - Grace period: `autostart_grace_period_s` (120s default)

---

## Tips & Best Practices

### 1. Managing Model Parameters

✅ **DO**: Put model-specific parameters in `extra_args`:
```json
{
  "name": "my-model",
  "backend_config": {
    "extra_args": [
      "--max-model-len", "8192",
      "--kv-cache-dtype", "fp8",
      "--gpu-memory-utilization", "0.95"
    ]
  }
}
```

❌ **DON'T**: Rely on environment variable defaults for model-specific needs

### 2. Chat Template Management

✅ **DO**: Use `chat_template.path` for templates:
```json
{
  "chat_template": {
    "source": "external",
    "path": "/path/to/model/template.jinja",
    "sha256": "abc123..."
  }
}
```

❌ **DON'T**: Duplicate template path in `extra_args` (causes issues)

### 3. Registry Editing

✅ **DO**: Edit `model_registry.curated.json` for stable changes
✅ **DO**: Let download tools manage `model_registry.discovered.json`
❌ **DON'T**: Manually edit `model_registry.json` (it's auto-generated)

### 4. Backend Resolution

✅ **DO**: Use `backend_config.host` for Docker networking:
```json
{
  "backend_config": {
    "host": "imageworks-ollama",  // Default container service (override via IMAGEWORKS_OLLAMA_HOST)
    "port": 11434
  }
}
```

✅ **DO**: Use `backend_config.base_url` for complex scenarios:
```json
{
  "backend_config": {
    "base_url": "https://remote-server.com:8443/api"
  }
}
```

### 5. Capability Filtering

Use `capabilities` to enable/disable features:
```json
{
  "capabilities": {
    "text": true,
    "vision": true,     // ← Enables image content
    "tools": true,      // ← Enables function calling
    "thinking": false,
    "reasoning": false
  }
}
```

The proxy validates requests against capabilities before forwarding.

---

## Troubleshooting

### Issue: Parameters not being passed to vLLM

**Symptoms**: Model starts but doesn't use configured `--max-model-len` or other args

**Diagnosis**:
1. Check proxy logs for vLLM command:
   ```bash
   docker logs imageworks-chat-proxy 2>&1 | grep "vllm-manager.*final.*command"
   ```

2. Verify registry merge:
   ```bash
   jq '.[] | select(.name=="your-model") | .backend_config.extra_args' \
     configs/model_registry.json
   ```

**Solutions**:
- Ensure parameters are in `extra_args` in the **curated** registry
- Check for duplicate arguments (extra_args wins)
- Verify chat template isn't duplicated in extra_args
- Restart proxy after registry changes: `docker restart imageworks-chat-proxy`

### Issue: Model not found

**Symptoms**: `404 Model 'xyz' not found in registry`

**Diagnosis**:
```python
# Check exact name
jq '.[] | .name' configs/model_registry.json | grep -i xyz

# Check display names
jq '.[] | .display_name' configs/model_registry.json | grep -i xyz
```

**Solutions**:
- Use exact `name` field value from registry
- Or use `display_name` (proxy will resolve)
- Check for typos in quantization suffixes: `_(Q4_K_M)` vs `_(q4_k_m)`

### Issue: vLLM won't start / health check fails

**Symptoms**: `503 Service Unavailable` or `VllmActivationError`

**Diagnosis**:
```bash
# Check state file
cat _staging/active_vllm.json

# Check if process running
ps aux | grep vllm

# Check vLLM logs (if started manually)
tail -f /path/to/model/logs/vllm_server.log
```

**Solutions**:
- Increase `vllm_start_timeout_s` (default: 180s)
- Check GPU memory availability: `nvidia-smi`
- Verify model path exists and is readable
- Check for conflicting processes on port 24001
- Review `--max-model-len` (may be too large for available memory)

### Issue: Duplicate entries warning

**Symptoms**: `Warning: duplicates after layering: model-name x2`

**Explanation**: Both curated and discovered contain the same `name`

**Solutions**:
- This is usually **harmless** (curated overrides discovered)
- To eliminate warning: Remove from discovered layer or curate properly
- Set `IMAGEWORKS_ALLOW_REGISTRY_DUPES=1` to silence warning

---

## Code Issues Found During Review

### 1. Inconsistent state file checking

**File**: `chat_proxy/app.py`
**Issue**: Registry reload checks `model_registry.json` mtime, but layered system may update curated/discovered without updating merged snapshot timestamp.

**Recommendation**: Check mtimes of all three registry files or use hash-based change detection.

### 2. Missing documentation for `served_model_id`

**Files**: Multiple
**Issue**: The `served_model_id` field determines what model name is sent to the backend, but this isn't clearly documented.

**Clarification**:
- If `served_model_id` is set: Use it as backend model ID
- If `None` or `"None"` (string): Use logical `name` instead
- This allows registry name to differ from backend-expected name

### 3. Template path resolution ambiguity

**File**: `chat_proxy/vllm_manager.py`
**Issue**: `_resolve_template_path()` may check both host and container paths, but logic isn't clear when path doesn't exist.

**Recommendation**: Add warning logs when template path doesn't exist but continues anyway.

### 4. Ollama streaming with images

**File**: `chat_proxy/forwarder.py`, line ~424
**Code**: Forces non-streaming for Ollama when images present

**Issue**: This is intentional to avoid transfer-encoding issues, but it's not documented why stream is disabled.

**Recommendation**: Add comment explaining why streaming is disabled for Ollama + images.

---

## Summary

The ImageWorks architecture provides:

1. **Flexible registry** with curated/discovered layering
2. **Smart precedence** where curated overrides discovered for stable fields
3. **Automatic orchestration** via VllmManager for vLLM models
4. **Capability-aware routing** with validation before forwarding
5. **Parameter precedence** where registry extra_args beat config defaults

Key to success: **Keep curated layer clean** and **let tooling manage discovered layer**.

---

**Version:** 1.0
**Contributors:** Documentation review 2025-10-26
