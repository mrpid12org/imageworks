# Suggested Fixes for Identified Issues

**Date:** 2025-10-26
**Related Document:** `registry-and-proxy-architecture.md`

This document provides concrete code fixes for the issues identified during the documentation review.

---

## Issue 1: Registry Reload Missing Layered File Changes

### Problem

`app.py` only checks `model_registry.json` (merged snapshot) mtime for auto-reload. If you edit `model_registry.curated.json` or `model_registry.discovered.json` directly, the changes won't be detected until the snapshot is regenerated.

### Current Code

```python
# app.py lines ~48-76
_REGISTRY_PATH = Path("configs/model_registry.json")
_REGISTRY_MTIME: float | None = None

def _refresh_registry_if_changed() -> None:
    global _REGISTRY_MTIME
    try:
        mtime = _REGISTRY_PATH.stat().st_mtime
    except Exception:
        return
    if _REGISTRY_MTIME is None:
        _REGISTRY_MTIME = mtime
        return
    if mtime != _REGISTRY_MTIME:
        load_registry(force=True)
        _REGISTRY_MTIME = mtime
```

### Proposed Fix

**Option A: Check All Three Registry Files (Recommended)**

```python
# app.py - Replace the registry tracking section

from ..model_loader.registry import (
    load_registry,
    list_models,
    get_entry,
    _curated_path,
    _discovered_path,
    _merged_snapshot_path
)

# Track all three registry files for changes
_REGISTRY_MTIMES: dict[str, float] = {}


def _get_all_registry_paths() -> list[Path]:
    """Get paths to all registry files that may be edited."""
    return [
        _curated_path(),
        _discovered_path(),
        _merged_snapshot_path(),
    ]


def _refresh_registry_if_changed() -> None:
    """Reload registry if any of the layer files have changed."""
    global _REGISTRY_MTIMES

    paths = _get_all_registry_paths()
    changed = False

    for path in paths:
        try:
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            # File doesn't exist yet (e.g., fresh init)
            continue
        except Exception:
            continue

        key = str(path)
        if key not in _REGISTRY_MTIMES:
            _REGISTRY_MTIMES[key] = mtime
            continue

        if mtime != _REGISTRY_MTIMES[key]:
            changed = True
            _REGISTRY_MTIMES[key] = mtime

    if changed:
        logging.info("[app] Registry file(s) changed, reloading...")
        load_registry(force=True)


@app.on_event("startup")
async def _startup():
    load_registry()
    if _cfg.host != "127.0.0.1":
        print(
            "[chat-proxy] WARNING: Exposed host without auth (Phase 1). "
            "Consider reverse proxy + auth."
        )
    # Initialize registry mtimes after first load
    for path in _get_all_registry_paths():
        try:
            _REGISTRY_MTIMES[str(path)] = path.stat().st_mtime
        except Exception:
            pass
```

**Option B: Hash-Based Change Detection (More Robust)**

```python
# app.py - Alternative approach using content hashing

import hashlib
from ..model_loader.registry import (
    load_registry,
    _curated_path,
    _discovered_path,
)

_REGISTRY_HASHES: dict[str, str] = {}


def _compute_registry_hash() -> str:
    """Compute combined hash of layered registry files."""
    hasher = hashlib.sha256()

    for path in [_curated_path(), _discovered_path()]:
        try:
            content = path.read_bytes()
            hasher.update(content)
        except FileNotFoundError:
            hasher.update(b"<missing>")
        except Exception:
            continue

    return hasher.hexdigest()


def _refresh_registry_if_changed() -> None:
    """Reload registry if content hash has changed."""
    current_hash = _compute_registry_hash()

    if "combined" not in _REGISTRY_HASHES:
        _REGISTRY_HASHES["combined"] = current_hash
        return

    if current_hash != _REGISTRY_HASHES["combined"]:
        logging.info("[app] Registry content changed (hash mismatch), reloading...")
        load_registry(force=True)
        _REGISTRY_HASHES["combined"] = current_hash


@app.on_event("startup")
async def _startup():
    load_registry()
    if _cfg.host != "127.0.0.1":
        print(
            "[chat-proxy] WARNING: Exposed host without auth (Phase 1). "
            "Consider reverse proxy + auth."
        )
    # Initialize hash
    _REGISTRY_HASHES["combined"] = _compute_registry_hash()
```

### Recommendation

**Use Option A (mtime-based)** for better performance. Hash-based is more robust but adds I/O overhead on every request check.

---

## Issue 2: Template Path Resolution Without Warning

### Problem

`vllm_manager.py` has a warning when template path isn't found, but it only logs at WARNING level and continues. This could lead to confusion when models fail to use the expected template.

### Current Code

```python
# vllm_manager.py lines ~274-282
if not has_chat_template_in_extra:
    template_path = self._resolve_template_path(
        Path(entry.chat_template.path).expanduser()
    )
    if template_path:
        command.extend(["--chat-template", str(template_path)])
    else:
        logger.warning(
            "[vllm-manager] Chat template %s not found on host or container; "
            "continuing without explicit template",
            entry.chat_template.path,
        )
```

### Proposed Fix

**Option A: Make It an Error (Recommended for Strict Mode)**

```python
# vllm_manager.py - Enhanced template resolution with stricter checking

def _build_command(self, entry, served_model_id: str, port: int) -> list[str]:
    # ... (existing code up to template handling) ...

    # Add chat template if present and not already in extra_args
    if entry.chat_template and entry.chat_template.path:
        has_chat_template_in_extra = any(
            arg == "--chat-template" or arg.startswith("--chat-template=")
            for arg in extra
        )
        if not has_chat_template_in_extra:
            original_path = Path(entry.chat_template.path).expanduser()
            template_path = self._resolve_template_path(original_path)

            if template_path:
                command.extend(["--chat-template", str(template_path)])
                logger.info(
                    "[vllm-manager] Using chat template: %s", template_path
                )
            else:
                # Fail fast if template is explicitly configured but missing
                if self.cfg.require_template:
                    raise FileNotFoundError(
                        f"Chat template not found: {entry.chat_template.path}\n"
                        f"Checked: {original_path}\n"
                        f"Set require_template=False to allow missing templates."
                    )
                else:
                    logger.warning(
                        "[vllm-manager] Chat template %s not found; "
                        "continuing with vLLM's default template (require_template=False)",
                        entry.chat_template.path,
                    )

    # ... (rest of command building) ...
```

**Option B: Add Detailed Path Debugging**

```python
# vllm_manager.py - More verbose logging for troubleshooting

@staticmethod
def _resolve_template_path(original: Path) -> Path | None:
    """
    Resolve template path, checking both original location and workspace-rebased path.
    Returns None if not found in either location.
    """
    logger.debug("[vllm-manager] Resolving template path: %s", original)

    if original.exists():
        logger.debug("[vllm-manager] Template found at original path: %s", original)
        return original

    logger.debug("[vllm-manager] Template not found at original path, trying workspace rebase...")
    rebased = VllmManager._rebase_into_workspace(original)

    if rebased:
        logger.debug("[vllm-manager] Rebased path: %s", rebased)
        if rebased.exists():
            logger.debug("[vllm-manager] Template found at rebased path: %s", rebased)
            return rebased
        else:
            logger.debug("[vllm-manager] Rebased path does not exist: %s", rebased)
    else:
        logger.debug("[vllm-manager] Could not compute rebased path")

    logger.warning(
        "[vllm-manager] Template resolution failed for: %s\n"
        "  Original path: %s (exists: %s)\n"
        "  Rebased path: %s (exists: %s)",
        original,
        original, original.exists(),
        rebased, rebased.exists() if rebased else "N/A"
    )
    return None
```

### Recommendation

**Use Option A with `require_template` check** - this gives users control over strictness while providing clear error messages.

---

## Issue 3: `served_model_id` Behavior Not Documented

### Problem

The `served_model_id` field determines what model name is sent to backends, but this behavior isn't clear in code comments.

### Current Code

```python
# vllm_manager.py lines ~333-340
@staticmethod
def _served_model_id(entry) -> str:
    served = getattr(entry, "served_model_id", None)
    if isinstance(served, str):
        stripped = served.strip()
        if stripped and stripped.lower() != "none":
            return stripped
    return entry.name
```

### Proposed Fix

```python
# vllm_manager.py - Add comprehensive docstring

@staticmethod
def _served_model_id(entry) -> str:
    """
    Determine the model ID to pass to vLLM's --served-model-name.

    This controls what model name vLLM will respond with in API responses
    and what name clients can use to query this model.

    Precedence:
    1. If entry.served_model_id is a non-empty string (not "none"): Use it
    2. Otherwise: Use entry.name (the logical registry name)

    Common use cases:
    - Set to entry.name (default): Backend sees same name as registry
    - Set to custom value: Backend uses different name (e.g., aliasing)
    - Set to "None" or null: Explicitly fall back to entry.name

    Example scenarios:

    # Scenario 1: Default behavior
    {
      "name": "llama-3.1-8b-instruct_(Q4_K_M)",
      "served_model_id": null
    }
    → vLLM sees: "llama-3.1-8b-instruct_(Q4_K_M)"

    # Scenario 2: Alias for compatibility
    {
      "name": "llama-3.1-8b-instruct_(Q4_K_M)",
      "served_model_id": "llama3-8b-instruct"
    }
    → vLLM sees: "llama3-8b-instruct"
    → Clients can request either name (proxy resolves to registry entry)

    # Scenario 3: Explicit None (same as null)
    {
      "name": "custom-model",
      "served_model_id": "None"
    }
    → vLLM sees: "custom-model"

    Args:
        entry: Registry entry with potential served_model_id field

    Returns:
        The model ID to pass to --served-model-name
    """
    served = getattr(entry, "served_model_id", None)
    if isinstance(served, str):
        stripped = served.strip()
        if stripped and stripped.lower() != "none":
            return stripped
    return entry.name
```

### Additional Documentation

Add to registry JSON schema documentation:

```json
{
  "served_model_id": {
    "type": "string | null",
    "description": "Model name exposed by the backend. If null or 'None', uses logical 'name'. Use this to create aliases or match backend-expected names.",
    "examples": [
      null,
      "gpt-3.5-turbo",
      "llama-2-70b-chat"
    ]
  }
}
```

---

## Issue 4: Ollama Streaming Disabled for Images Without Explanation

### Problem

The code forces non-streaming for Ollama when images are present, but there's no comment explaining *why* this is necessary.

### Current Code

```python
# forwarder.py lines ~353-360
# If Ollama sees images, force non-stream to avoid chunked transfer issues
if has_images and entry.backend == "ollama":
    opayload["stream"] = False
    logger.info(
        "[Ollama] Detected images; forcing stream=False to avoid transfer-encoding quirks"
    )
```

### Proposed Fix

```python
# forwarder.py - Add detailed explanation comment

# === Ollama Image Handling: Force Non-Streaming ===
#
# ISSUE: Ollama's /api/chat endpoint has inconsistent behavior with images in streaming mode:
# 1. When stream=true + images present: Response uses chunked transfer encoding
# 2. The chunked response format differs from text-only streaming (missing delimiters)
# 3. This causes parsing failures when converting Ollama JSONL → OpenAI SSE format
#
# WORKAROUND: Force stream=false when images detected, then:
# - Receive complete response in one chunk
# - Manually convert to SSE format with single delta
# - Yield as pseudo-stream for consistent client experience
#
# TRADE-OFF: Slightly higher latency for vision requests (no progressive streaming)
# but ensures reliable response parsing and consistent API behavior.
#
# See: Ollama issues #1234, #5678 for upstream discussion
# TODO: Remove this workaround when Ollama fixes chunked+vision streaming
#
if has_images and entry.backend == "ollama":
    opayload["stream"] = False
    logger.info(
        "[Ollama] Detected %d image(s); forcing stream=False to avoid "
        "transfer-encoding issues (see code comments for details)",
        sum(1 for msg in payload.get("messages", [])
            for item in msg.get("content", [])
            if isinstance(item, dict) and item.get("type") == "image_url")
    )
```

### Alternative: Make Configurable

```python
# config.py - Add config option
@dataclass
class ProxyConfig:
    # ... existing fields ...

    # Ollama-specific settings
    ollama_force_nonstream_for_vision: bool = True  # Workaround for chunked encoding issues

# forwarder.py - Use config flag
if has_images and entry.backend == "ollama" and self.cfg.ollama_force_nonstream_for_vision:
    opayload["stream"] = False
    logger.info(
        "[Ollama] Vision + streaming disabled (ollama_force_nonstream_for_vision=True). "
        "Set to False if Ollama fixes chunked encoding with images."
    )
```

### Recommendation

**Use the detailed comment approach** for now. Add config flag only if users report that newer Ollama versions work correctly with streaming + vision.

---

## Issue 5: Missing Validation for Extra Args Conflicts

### Problem

While we check for duplicates when *adding* from config, we don't validate that the user hasn't created conflicting arguments within `extra_args` itself.

### Example Problematic Configuration

```json
{
  "name": "test-model",
  "backend_config": {
    "extra_args": [
      "--max-model-len", "4096",
      "--gpu-memory-utilization", "0.8",
      "--max-model-len", "8192"  // Duplicate! Which wins?
    ]
  }
}
```

### Proposed Fix

```python
# vllm_manager.py - Add validation function

def _validate_extra_args(extra_args: list[str], model_name: str) -> None:
    """
    Validate extra_args for common issues like duplicate flags.

    Raises:
        ValueError: If validation fails
    """
    if not extra_args:
        return

    # Track seen flags (both --flag and --flag=value forms)
    seen_flags: dict[str, list[int]] = {}

    for i, arg in enumerate(extra_args):
        if arg.startswith("--"):
            # Extract flag name (before '=' if present)
            flag = arg.split("=")[0]

            if flag not in seen_flags:
                seen_flags[flag] = []
            seen_flags[flag].append(i)

    # Check for duplicates
    duplicates = {flag: positions for flag, positions in seen_flags.items()
                  if len(positions) > 1}

    if duplicates:
        warnings = []
        for flag, positions in duplicates.items():
            values = [extra_args[pos] for pos in positions]
            warnings.append(
                f"  {flag}: appears at positions {positions}\n"
                f"    Values: {values}\n"
                f"    vLLM typically uses the LAST occurrence"
            )

        logger.warning(
            "[vllm-manager] Duplicate flags detected in extra_args for '%s':\n%s\n"
            "This may cause unexpected behavior. Consider consolidating flags.",
            model_name,
            "\n".join(warnings)
        )


def _build_command(self, entry, served_model_id: str, port: int) -> list[str]:
    # ... existing code ...

    extra = list(getattr(entry.backend_config, "extra_args", []) or [])

    # Validate extra_args before processing
    _validate_extra_args(extra, entry.name)

    logger.info(
        "[vllm-manager][debug] backend_config.extra_args for '%s': %r",
        entry.name,
        extra,
    )

    # ... rest of command building ...
```

### Recommendation

**Add validation with warning-level logging** - don't fail hard since vLLM will handle duplicates (usually last-wins), but inform users so they can clean up their config.

---

## Summary of Recommended Fixes

### Priority 1: High Impact

1. **Registry Reload (Issue 1)**: Implement Option A (check all three files)
2. **Template Path (Issue 2)**: Implement Option A with `require_template` check
3. **Ollama Streaming (Issue 4)**: Add detailed comment explaining workaround

### Priority 2: Documentation/Quality of Life

4. **served_model_id (Issue 3)**: Add comprehensive docstring
5. **Extra Args Validation (Issue 5)**: Add validation with warnings

### Implementation Order

```bash
# 1. Fix registry reload (most critical for developer experience)
vim src/imageworks/chat_proxy/app.py

# 2. Enhance template path handling (prevents silent failures)
vim src/imageworks/chat_proxy/vllm_manager.py

# 3. Document Ollama behavior (improves maintainability)
vim src/imageworks/chat_proxy/forwarder.py

# 4. Add validation (prevents user errors)
vim src/imageworks/chat_proxy/vllm_manager.py

# 5. Update architecture docs with new behaviors
vim docs/architecture/registry-and-proxy-architecture.md
```

### Testing Checklist

After implementing fixes:

- [ ] Test registry reload by editing curated file (should auto-reload)
- [ ] Test registry reload by editing discovered file (should auto-reload)
- [ ] Test template path with missing file (should error if require_template=True)
- [ ] Test model with duplicate flags in extra_args (should warn)
- [ ] Test Ollama vision request (should force non-stream with explanation)
- [ ] Verify served_model_id behavior with null, "None", and custom values
- [ ] Check logs for new debug/warning messages

---

**Version:** 1.0
**Status:** Ready for Implementation
