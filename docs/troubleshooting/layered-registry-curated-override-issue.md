# Layered Registry: Curated Override Not Applied Issue

**Date:** 2024-10-27
**Status:** Investigation Complete, Fixes Planned
**Severity:** High - Blocks model configuration via curated layer

## Executive Summary

Models downloaded via GUI are added to the discovered layer with empty `extra_args: []`. When we add proper launch parameters to the curated layer, they are correctly merged into the snapshot file, but the running chat-proxy process still uses the empty args. This blocks essential model configuration (context length, enforce-eager, etc.) and causes models to timeout during initialization.

## Problem Description

### Initial Symptom
OpenWebUI hangs when selecting the newly downloaded Qwen 2.5 VL 7B AWQ model, showing error:
```
"Autostart did not make 'qwen2.5-vl-7b-instruct_(AWQ)' healthy in time"
```

The same issue now affects other models (Qwen3 VL FP8) that previously worked.

### Root Cause Investigation

#### Discovery 1: Missing Launch Parameters
The AWQ model was launching with default vLLM settings (128K context length) instead of required `--max-model-len 4096 --enforce-eager --trust-remote-code`, causing initialization timeout.

**Evidence:**
```bash
$ tail logs/vllm/qwen2.5-vl-7b-instruct_(AWQ).log
INFO 10-27 20:38:45 [model.py:1510] Using max model len 128000
TimeoutError: Timed out waiting for engines to send initial message on input socket
```

**Working configuration** (from `configs/working_setups/entries/qwen2.5-vl-7b-awq_vllm.json`):
```json
{
  "server": {
    "flags": {
      "max-model-len": 4096,
      "enforce-eager": true,
      "trust-remote-code": true
    }
  }
}
```

#### Discovery 2: Layered Registry File Structure
The system uses three registry files:
- `model_registry.curated.json` - Hand-maintained configuration (extra_args, context lengths, etc.)
- `model_registry.discovered.json` - Auto-generated from disk scans (paths, sizes, formats)
- `model_registry.json` - Merged snapshot (discovered base + curated overlay)

**Current state:**
```bash
$ grep -c '"name": "qwen2.5-vl-7b-instruct_(AWQ)"' configs/model_registry*.json
configs/model_registry.curated.json:1      # We added entry here
configs/model_registry.discovered.json:1   # GUI download created this
configs/model_registry.json:1              # Merged snapshot
```

#### Discovery 3: Files Contain Correct Values
```bash
# Discovered layer (from GUI download):
$ jq '.[] | select(.name == "qwen2.5-vl-7b-instruct_(AWQ)") | .backend_config' configs/model_registry.discovered.json
{
  "port": 24001,
  "model_path": "/home/stewa/ai-models/weights/Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
  "extra_args": []  # EMPTY
}

# Curated layer (manually added):
$ jq '.[] | select(.name == "qwen2.5-vl-7b-instruct_(AWQ)") | .backend_config' configs/model_registry.curated.json
{
  "port": 24001,
  "model_path": "/home/stewa/ai-models/weights/Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
  "extra_args": [
    "--max-model-len",
    "4096",
    "--enforce-eager",
    "--trust-remote-code"
  ]
}

# Merged snapshot (correctly merged):
$ jq '.[] | select(.name == "qwen2.5-vl-7b-instruct_(AWQ)") | .backend_config' configs/model_registry.json
{
  "port": 24001,
  "model_path": "/home/stewa/ai-models/weights/Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
  "extra_args": [
    "--max-model-len",
    "4096",
    "--enforce-eager",
    "--trust-remote-code"
  ]  # CORRECT - curated values present!
}
```

#### Discovery 4: Docker Container Sees Correct Files
```bash
# Container mounts host configs as read-only
$ grep configs docker-compose.openwebui.yml
- ./configs:/app/configs:ro

# Container CAN read correct values:
$ docker exec imageworks-chat-proxy cat /app/configs/model_registry.json | \
  jq '.[] | select(.name == "qwen2.5-vl-7b-instruct_(AWQ)") | .backend_config.extra_args'
[
  "--max-model-len",
  "4096",
  "--enforce-eager",
  "--trust-remote-code"
]
```

#### Discovery 5: Registry Module Loads Correctly
```bash
# Force reload shows correct values:
$ docker exec imageworks-chat-proxy python3 -c "
from imageworks.model_loader.registry import load_registry, get_entry
load_registry(force=True)
entry = get_entry('qwen2.5-vl-7b-instruct_(AWQ)')
print('extra_args:', entry.backend_config.extra_args)
"
extra_args: ['--max-model-len', '4096', '--enforce-eager', '--trust-remote-code']

# Even in running process:
$ docker exec imageworks-chat-proxy python3 -c "
from imageworks.model_loader.registry import get_entry
entry = get_entry('qwen2.5-vl-7b-instruct_(AWQ)')
print('extra_args:', entry.backend_config.extra_args)
"
extra_args: ['--max-model-len', '4096', '--enforce-eager', '--trust-remote-code']
```

#### Discovery 6: vLLM Manager Still Sees Empty Args
```bash
$ docker logs imageworks-chat-proxy 2>&1 | grep "extra_args.*AWQ"
INFO:imageworks.vllm_manager:[vllm-manager][debug] backend_config.extra_args for 'qwen2.5-vl-7b-instruct_(AWQ)': []
```

**This is the smoking gun!** The registry module has correct values, but vllm_manager receives an entry with empty extra_args.

#### Discovery 7: Merge Logic Works Correctly
```bash
$ docker exec imageworks-chat-proxy python3 -c "
from imageworks.model_loader.registry import _merge_layered_fragments

curated = [{'name': 'test', 'backend_config': {'extra_args': ['--flag', '123']}}]
discovered = [{'name': 'test', 'backend_config': {'extra_args': []}}]

merged, _ = _merge_layered_fragments(curated, discovered)
print('Merged extra_args:', merged[0]['backend_config']['extra_args'])
"
Merged extra_args: ['--flag', '123']  # CURATED WINS ✓
```

#### Discovery 8: Hot-Reload Mechanism Exists But Incomplete
```bash
$ touch configs/model_registry.curated.json && curl -s http://localhost:8100/v1/models > /dev/null
$ docker logs imageworks-chat-proxy 2>&1 | tail -5 | grep reload
INFO:root:[app] Registry file(s) changed, reloading...
```

Hot-reload updates `_REGISTRY_CACHE` but doesn't propagate to all consumers.

#### Discovery 9: RoleSelector Has Separate Cache
From `src/imageworks/chat_proxy/role_selector.py`:
```python
def __init__(self, registry_path: Optional[Path] = None):
    if registry_path is None:
        registry_path = workspace_root / "configs" / "model_registry.curated.json"

    self._models: List[Dict] = []
    self._load_registry()

def _load_registry(self) -> None:
    with open(self.registry_path, "r") as f:  # Direct file read, not using registry module!
        data = json.load(f)
    self._models = data
```

**Issues:**
1. RoleSelector loads ONLY curated.json, not merged snapshot
2. Caches at initialization, doesn't reload on hot-reload
3. Logs show only 11-12 models loaded, but registry has 16

## What We've Tried

### Attempt 1: Edit Merged Snapshot
```bash
$ jq '(.[] | select(.name == "qwen2.5-vl-7b-instruct_(AWQ)") | .backend_config.extra_args) =
  ["--max-model-len", "4096", "--enforce-eager", "--trust-remote-code"]' \
  configs/model_registry.json > /tmp/fixed.json && mv /tmp/fixed.json configs/model_registry.json
```
**Result:** File updated but proxy still read empty args. Merged snapshot is regenerated on save, so edits are overwritten.

### Attempt 2: Add Entry to Curated Layer
Manually created entry in `configs/model_registry.curated.json`:
```json
{
  "name": "qwen2.5-vl-7b-instruct_(AWQ)",
  "display_name": "qwen2.5-vl-7b-instruct (AWQ)",
  "backend": "vllm",
  "backend_config": {
    "port": 24001,
    "model_path": "/home/stewa/ai-models/weights/Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
    "extra_args": [
      "--max-model-len",
      "4096",
      "--enforce-eager",
      "--trust-remote-code"
    ]
  },
  "capabilities": { "vision": true, "multimodal": true, ... },
  "metadata": { "registry_layer": "curated", ... }
}
```
**Result:** Merged snapshot shows correct values, but running proxy still uses empty args.

### Attempt 3: Container Restart
```bash
$ docker-compose -f docker-compose.openwebui.yml restart chat-proxy
```
**Result:** No change. Still logs `extra_args: []`

### Attempt 4: Trigger Hot-Reload
```bash
$ touch configs/model_registry.curated.json configs/model_registry.discovered.json
$ curl http://localhost:8100/v1/models  # Trigger endpoint that calls _refresh_registry_if_changed()
```
**Result:** Logs show "Registry file(s) changed, reloading..." but vllm_manager still sees empty args.

### Attempt 5: Fix Missing Dockerfile Dependency
Discovered container crash due to missing `tomli` dependency:
```bash
$ docker logs imageworks-chat-proxy
ModuleNotFoundError: No module named 'tomli'
```

Fixed in `Dockerfile.chat-proxy`:
```dockerfile
RUN python3 -m pip install --no-cache-dir \
    "fastapi" \
    "uvicorn[standard]" \
    "httpx" \
    "pydantic" \
    "vllm[vision]==0.11.0" \
    "qwen-vl-utils" \
    "tomli"  # ADDED
```
Rebuilt and restarted, but AWQ issue persists.

## User's Design Intent

### Discovered Layer
**Purpose:** Read and store everything discoverable from the model repository on disk (Ollama or HuggingFace).

**What belongs here:**
- Facts you generally **cannot and should not alter**
- Size, format, capabilities, architecture
- Information automatically scanned from model files
- **Exception:** Chat template (can be altered, but is discoverable)

**Key principle:** This layer should be **pure** - never manually tampered with, always rebuildable from disk without risk of losing any curated data.

**What does NOT belong here:**
- `extra_args` - not discoverable from disk at download stage
- Context length overrides - user configuration choice
- Backend selection - user may have multiple backends capable of running same model
- GPU-specific tuning - depends on hardware, not model

### Curated Layer
**Purpose:** Store additional info to augment the discovered data, and potentially regularly changing configuration.

**What belongs here:**
- `extra_args` for model launch (e.g., `--max-model-len`, `--enforce-eager`)
- Context length overrides (may differ per GPU based on VRAM)
- Backend selection (user may choose vLLM vs llama.cpp for same model)
- GPU-specific configurations (different settings for RTX 4080 vs A100)
- Temperature, top_p, and other generation defaults
- **Rare overrides:** Chat template (when you want different template than discovered)

**Key principle:** This layer is for **human configuration** - stuff you intentionally change as your setup evolves.

**Important note:** We may need **multiple curated entries for the same model** (one per GPU configuration), but the discovered layer will remain fixed.

### Merged Registry (Final Output)
**How it works:**
1. Discovered layer provides the **base facts** (immutable, from disk)
2. Curated layer provides the **configuration overlay** (editable, per-GPU/setup)
3. System merges them: discovered first, then curated overlays on top
4. **Curated wins** when both layers have the same field
5. Result written to `model_registry.json` (merged snapshot)
6. All consuming modules (OpenWebUI, mono, narrator, etc.) read **only the merged snapshot**
7. Only the downloader and GUI see the separation into layers

**Transparency principle:** Most of the system should never see complexity - just a single joined-up registry.

### Chat Template Special Case
Chat templates are **discoverable** (can be extracted from `tokenizer_config.json`) but also **configurable** (user may want custom template).

**Proposed approach:**
- Keep in both layers
- Curated takes precedence if non-null
- This allows: discovered scans and sets default, user can override in curated
- Alternative: Move chat template to curated-only, make it pure configuration

**User's preference:** Lean toward making discovered truly pure by moving chat template to curated-only.

## Architectural Analysis

### Current Implementation Gaps

#### Gap 1: save_registry() Writes Everything to Discovered
From `src/imageworks/model_loader/registry.py` lines 856-875:
```python
discovered_out: list[dict] = []
for entry in _REGISTRY_CACHE.values():  # ALL entries, including curated!
    data = _serialize_entry(entry)
    # ... some registry_layer logic ...
    discovered_out.append(discovered_copy)  # EVERY entry goes to discovered!
```

**Problem:** Every time registry is saved (after any download/scan), ALL in-memory entries get written to discovered.json, including those that originated from curated. This causes discovered to have complete entries with empty extra_args, which then override curated on next merge.

#### Gap 2: GUI Download Creates Discovered-Only Entry
When downloading via GUI:
1. `ModelDownloader._register_download()` calls `record_download()`
2. `record_download()` creates `RegistryEntry` with `backend_config.extra_args = []`
3. Entry added to `_REGISTRY_CACHE`
4. `update_entries(save=True)` calls `save_registry()`
5. Entry written to discovered.json with empty extra_args
6. **NO curated entry created**

**Problem:** No placeholder in curated layer for user to add configuration. User must manually create full entry structure.

#### Gap 3: RoleSelector Bypasses Registry Module
RoleSelector reads curated.json directly instead of using `load_registry()`. It:
- Doesn't see discovered entries
- Doesn't benefit from merge logic
- Doesn't reload on hot-reload events
- Maintains stale cache throughout app lifetime

#### Gap 4: Docker Image Bakes in Configs
From `Dockerfile.chat-proxy` line 34:
```dockerfile
COPY configs /app/configs
```

**Problem:** Configs are copied into image at build time. Even though docker-compose mounts the host directory, the built-in files create potential confusion and stale data issues.

#### Gap 5: Chat Template in Both Layers
Chat templates can be in both discovered (scanned from tokenizer_config.json) AND curated (manual override). This violates the "discovered = pure, never tampered" principle.

## Theory: Why Curated Override Fails

### Hypothesis
The vllm_manager receives an entry that comes from a code path that:
1. Either bypasses the registry module's `_REGISTRY_CACHE`, OR
2. Receives the entry BEFORE hot-reload completes, OR
3. Gets a serialized/pickled entry that lost the merged backend_config

### Evidence Supporting Theory

**Evidence A:** Direct registry module calls work correctly
```bash
$ docker exec python3 -c "from imageworks.model_loader.registry import get_entry; ..."
# Returns correct extra_args
```

**Evidence B:** vllm_manager sees stale data
```bash
$ docker logs | grep extra_args.*AWQ
# Shows empty array
```

**Evidence C:** Merge logic proven correct
```bash
$ docker exec python3 -c "from imageworks.model_loader.registry import _merge_layered_fragments; ..."
# Curated correctly overrides discovered
```

**Evidence D:** Files are correct on disk
All three files (curated, discovered, merged) have correct timestamps and content as verified by host and container checks.

### Most Likely Root Cause

The `save_registry()` function writes ALL entries to discovered.json (Gap 1), including complete backend_config objects. When the registry reloads:

1. Load discovered → entry with `backend_config.extra_args: []`
2. Load curated → entry with `backend_config.extra_args: ["--max-model-len", "4096", ...]`
3. Merge: `_merge_entry_dicts(discovered, curated)`
4. **Expected:** Curated's extra_args replaces discovered's
5. **Actual:** Something in the code path prevents this

The merge logic DOES work when tested in isolation, but in the live system, vllm_manager consistently receives empty args. This suggests:
- Timing issue (autostart reads before registry merge completes), OR
- Caching issue (stale entry object held somewhere), OR
- Serialization issue (entry passed through JSON/pickle loses data), OR
- Wrong entry source (reading from discovered directly instead of merged)

## Proposed Fixes

### Immediate Fix: Edit Discovered Layer Directly
**Workaround to get AWQ model working NOW:**
```bash
$ jq '(.[] | select(.name == "qwen2.5-vl-7b-instruct_(AWQ)") | .backend_config.extra_args) =
  ["--max-model-len", "4096", "--enforce-eager", "--trust-remote-code"]' \
  configs/model_registry.discovered.json > /tmp/fixed.json && \
  mv /tmp/fixed.json configs/model_registry.discovered.json
$ touch configs/model_registry.discovered.json
$ docker-compose -f docker-compose.openwebui.yml restart chat-proxy
```

**Caveat:** Next download/scan will overwrite this. Temporary fix only.

### Long-term Architecture Fixes

#### Fix 1: Remove Config Copy from Dockerfile
**File:** `Dockerfile.chat-proxy` line 34

**Change:**
```dockerfile
# BEFORE:
COPY configs /app/configs

# AFTER:
# Configs mounted at runtime via docker-compose volume
# Create empty directory for volume mount point
RUN mkdir -p /app/configs
```

**Rationale:** Configs should ONLY come from runtime volume mount, never baked into image.

#### Fix 2: Fix save_registry() to Respect Layer Separation
**File:** `src/imageworks/model_loader/registry.py` lines 856-880

**Change Logic:**
```python
discovered_out: list[dict] = []
for entry in _REGISTRY_CACHE.values():
    data = _serialize_entry(entry)

    # Only write to discovered if:
    # 1. Entry doesn't exist in curated, OR
    # 2. Entry has dynamic fields that must be persisted
    is_in_curated = data["name"] in curated_map
    has_dynamic_fields = _is_dynamic(entry)

    if not is_in_curated or has_dynamic_fields:
        discovered_out.append(discovered_copy)
    # If in curated WITHOUT dynamic fields, skip writing to discovered
```

**Rationale:** Discovered layer should only contain:
- Entries not yet promoted to curated
- Dynamic runtime fields for curated entries (download_path, last_accessed, etc.)

Entries fully managed in curated should never be written to discovered.

#### Fix 3: Create Curated Stubs on Download
**File:** `src/imageworks/model_loader/download_adapter.py`

**Add function:**
```python
def create_curated_stub(entry: RegistryEntry) -> dict:
    """Create minimal curated entry for user configuration."""
    return {
        "name": entry.name,
        "display_name": entry.display_name or entry.name,
        "backend": entry.backend,
        "backend_config": {
            "port": entry.backend_config.port,
            "model_path": entry.backend_config.model_path,
            "extra_args": []  # Placeholder for user to populate
        },
        "capabilities": entry.capabilities.__dict__ if hasattr(entry.capabilities, '__dict__') else {},
        "generation_defaults": {},
        "chat_template": {},
        "metadata": {
            "registry_layer": "curated",
            "_config_needed": True,  # Flag indicating user should configure
            "notes": "Downloaded via GUI. Configure extra_args, context_length, and other runtime parameters."
        },
        "family": entry.family,
        "source_provider": "huggingface",
        "quantization": entry.quantization
    }
```

**Modify record_download():**
```python
def record_download(...):
    # ... existing code creates entry ...
    update_entries([entry], save=True)  # Writes to discovered

    # Check if curated entry exists
    curated_file = Path("configs/model_registry.curated.json")
    if curated_file.exists():
        curated_data = json.loads(curated_file.read_text())
        existing_names = {e.get("name") for e in curated_data}

        if entry.name not in existing_names:
            # Create stub in curated
            stub = create_curated_stub(entry)
            curated_data.append(stub)
            curated_data.sort(key=lambda e: e.get("name", ""))
            curated_file.write_text(json.dumps(curated_data, indent=2) + "\n")
```

**Rationale:** Every model gets a curated entry for configuration. User doesn't need to manually create structure.

#### Fix 4: Implement GUI Save to Curated Only
**File:** `src/imageworks/gui/pages/2_🎯_Models.py`

**Add function:**
```python
def save_curated_entry(name: str, updated_fields: dict):
    """Save GUI edits directly to curated layer."""
    curated_file = Path("configs/model_registry.curated.json")
    curated_data = json.loads(curated_file.read_text())

    # Find or create entry
    entry = None
    for e in curated_data:
        if e.get("name") == name:
            entry = e
            break

    if not entry:
        # Create minimal entry if not exists
        entry = {"name": name, "metadata": {"registry_layer": "curated"}}
        curated_data.append(entry)

    # Update only the fields user edited
    for key, value in updated_fields.items():
        if "." in key:  # Nested field like "backend_config.extra_args"
            parts = key.split(".")
            target = entry
            for part in parts[:-1]:
                if part not in target:
                    target[part] = {}
                target = target[part]
            target[parts[-1]] = value
        else:
            entry[key] = value

    # Save
    curated_data.sort(key=lambda e: e.get("name", ""))
    curated_file.write_text(json.dumps(curated_data, indent=2) + "\n")

    # Trigger reload
    merged_file = Path("configs/model_registry.json")
    merged_file.touch()  # Trigger hot-reload
```

**Rationale:** GUI writes directly to curated, never touches discovered. Clear separation of concerns.

#### Fix 5: Make RoleSelector Use Registry Module
**File:** `src/imageworks/chat_proxy/role_selector.py`

**Change:**
```python
# BEFORE:
def __init__(self, registry_path: Optional[Path] = None):
    if registry_path is None:
        registry_path = workspace_root / "configs" / "model_registry.curated.json"
    self.registry_path = registry_path
    self._models: List[Dict] = []
    self._load_registry()

def _load_registry(self) -> None:
    with open(self.registry_path, "r") as f:
        data = json.load(f)

# AFTER:
def __init__(self):
    self._models: List[Dict] = []
    self._load_registry()

def _load_registry(self) -> None:
    from imageworks.model_loader.registry import load_registry
    reg = load_registry()
    self._models = [_serialize_entry(entry) for entry in reg.values()]
    logger.info(f"Loaded {len(self._models)} models from registry")

def reload(self) -> None:
    """Reload registry data."""
    self._load_registry()
```

**Update app.py hot-reload:**
```python
def _refresh_registry_if_changed() -> None:
    global _role_selector
    # ... existing hot-reload logic ...
    if changed:
        logging.info("[app] Registry file(s) changed, reloading...")
        load_registry(force=True)
        if _role_selector:
            _role_selector.reload()  # Propagate reload
```

**Rationale:** Single source of truth for registry data. All consumers use merged snapshot.

#### Fix 6: Move Chat Template to Curated Only
**Principle:** Chat template is configuration (user may want to override), so belongs in curated.

**Changes:**
- Remove chat template extraction from discovered layer population
- Document that chat templates should be set in curated
- Keep chat_template.path in discovered only when it's a discovered artifact path
- Merge logic: curated chat_template takes precedence if non-null

## Next Steps

### Phase 1: Immediate Fix (today)
1. ✅ Document issue completely (this doc)
2. ⏳ Apply workaround: Edit discovered.json directly for AWQ model
3. ⏳ Verify AWQ model starts successfully
4. ⏳ Test in OpenWebUI with vision requests

### Phase 2: Docker Fix (today/tomorrow)
1. Remove `COPY configs` from Dockerfile
2. Rebuild image
3. Restart containers
4. Verify hot-reload works properly

### Phase 3: Registry Module Fixes (1-2 days)
1. Implement Fix 2: Update save_registry() logic
2. Implement Fix 3: Create curated stubs on download
3. Add tests for layered merge behavior
4. Verify no regressions

### Phase 4: Consumer Fixes (2-3 days)
1. Implement Fix 5: Update RoleSelector to use registry module
2. Update hot-reload to propagate to all consumers
3. Implement Fix 4: GUI save to curated
4. Add GUI indicator showing which layer values come from

### Phase 5: Validation (1 day)
1. Test full workflow: Download → Configure → Launch
2. Test hot-reload propagation
3. Test GUI edit → save → reload → launch
4. Document new workflow in guides
5. Update architecture docs

## Testing Commands Reference

### Verify File Contents
```bash
# Check all three registry files
for file in configs/model_registry.{curated,discovered,}.json; do
  echo "=== $file ==="
  jq '.[] | select(.name == "qwen2.5-vl-7b-instruct_(AWQ)") | {name, extra_args: .backend_config.extra_args}' "$file"
done

# Check container sees correct files
docker exec imageworks-chat-proxy cat /app/configs/model_registry.json | \
  jq '.[] | select(.name == "qwen2.5-vl-7b-instruct_(AWQ)") | .backend_config.extra_args'
```

### Verify Registry Module State
```bash
# Check what registry module loads
docker exec imageworks-chat-proxy python3 -c "
from imageworks.model_loader.registry import load_registry, get_entry
reg = load_registry(force=True)
entry = get_entry('qwen2.5-vl-7b-instruct_(AWQ)')
print('Loaded entries:', len(reg))
print('Backend:', entry.backend)
print('extra_args:', entry.backend_config.extra_args)
"
```

### Test Merge Logic
```bash
# Verify merge function works correctly
docker exec imageworks-chat-proxy python3 -c "
from imageworks.model_loader.registry import _merge_layered_fragments
curated = [{'name': 'test', 'backend_config': {'extra_args': ['--flag', '123']}}]
discovered = [{'name': 'test', 'backend_config': {'extra_args': []}}]
merged, _ = _merge_layered_fragments(curated, discovered)
print('Result:', merged[0]['backend_config']['extra_args'])
"
```

### Check Runtime State
```bash
# Monitor vllm_manager logs
docker logs imageworks-chat-proxy 2>&1 | grep -E "Starting model qwen2.5-vl.*AWQ|extra_args.*AWQ"

# Check vLLM process logs
tail -50 logs/vllm/qwen2.5-vl-7b-instruct_\(AWQ\).log | grep -E "max.*model.*len|TimeoutError"

# Verify hot-reload triggers
touch configs/model_registry.curated.json
curl -s http://localhost:8100/v1/models > /dev/null
docker logs imageworks-chat-proxy 2>&1 | tail -10 | grep reload
```

### Full Restart Sequence
```bash
# Complete restart with verification
docker-compose -f docker-compose.openwebui.yml restart chat-proxy
sleep 10
docker logs imageworks-chat-proxy 2>&1 | grep -E "Application startup|extra_args.*AWQ" | tail -20
```

## References

- **Architecture Doc:** `docs/architecture/layered-registry.md`
- **Registry Implementation:** `src/imageworks/model_loader/registry.py`
- **Download Adapter:** `src/imageworks/model_loader/download_adapter.py`
- **vLLM Manager:** `src/imageworks/chat_proxy/vllm_manager.py`
- **Role Selector:** `src/imageworks/chat_proxy/role_selector.py`
- **GUI Models Page:** `src/imageworks/gui/pages/2_🎯_Models.py`
- **Working Config:** `configs/working_setups/entries/qwen2.5-vl-7b-awq_vllm.json`

## Related Issues

- GUI download creating models without configuration stubs
- Hot-reload not propagating to all consumers
- Docker image baking in stale configs
- Multiple registry loading mechanisms (file vs module)

---

**Last Updated:** 2024-10-27
**Next Review:** After Phase 1 completion
