# Phase 1: Proxy Integration Implementation Plan

**Status**: ✅ **COMPLETED** (October 26, 2025)
**Date**: October 26, 2025
**Priority**: High
**Complexity**: Low (Config-only changes + one code fix)

---

## Implementation Summary

**All 8 tasks completed successfully:**

✅ **Task 1**: Updated personal_tagger config to use proxy (port 8100)
✅ **Task 2**: Updated color_narrator config to use proxy (port 8100)
✅ **Task 3**: Fixed VLMMonoInterpreter to use backend client pattern
✅ **Task 4**: Added role metadata to qwen3-vl-8b-instruct_(FP8)
✅ **Task 5**: Added role metadata to qwen25vl_8.3b_(Q4_K_M)
✅ **Task 6**: Downloaded Florence-2-large model (2.9GB, FP16)
✅ **Task 7**: Added Florence-2 to curated registry with role metadata
✅ **Task 8**: Created comprehensive unit tests for VLMMonoInterpreter

**Test Results**: 131 tests passed (123 existing + 8 new)

**Files Modified**:
- `pyproject.toml` (2 config updates)
- `src/imageworks/apps/color_narrator/core/vlm_mono_interpreter.py` (backend client integration)
- `configs/model_registry.curated.json` (3 model entries with role metadata)

**Backups Created**:
- `pyproject.toml.pre-phase1`
- `vlm_mono_interpreter.py.pre-phase1`
- `model_registry.curated.json.pre-phase1`

---

## Executive Summary

Phase 1 integrates three ImageWorks modules (personal_tagger, color_narrator, and mono_checker) with the centralized chat proxy (port 8100) instead of directly connecting to backend servers (vLLM port 24001, Ollama port 11434). This provides automatic history truncation, model autostart, unified logging, and prepares for role-based model selection in Phase 2.

**Key Benefits**:
- ✅ Automatic vision history truncation (eliminates max_model_len errors on multi-image analysis)
- ✅ Automatic reasoning history truncation (optional, prevents KV cache exhaustion)
- ✅ On-demand model launching with grace periods
- ✅ Unified request logging and monitoring
- ✅ Single configuration point for backend changes
- ✅ Foundation for role-based model selection (Phase 2)

**Changes Required**: 3 config updates + 1 code fix (VLMMonoInterpreter)

---

## Context & Chat History Summary

### Problem Evolution

1. **Initial Issue**: qwen3-vl-fp8 parameters not being passed to vLLM
   - **Root Cause**: Duplicate `--chat-template` arguments
   - **Resolution**: Fixed duplicate detection in `_build_command()`

2. **Vision Context Overflow**: Multi-image analysis hitting max_model_len=6144 errors
   - **Root Cause**: Accumulated image tokens from conversation history
   - **Resolution**: Implemented automatic vision history truncation in chat proxy
   - **Configuration**: `CHAT_PROXY_VISION_TRUNCATE_HISTORY=true` (default enabled)

3. **Reasoning Model Failures**: gpt-oss-20b streaming crashes after long outputs
   - **Root Cause**: KV cache exhaustion + restrictive `--kv-cache-memory` setting
   - **Resolution**: (1) Reasoning history truncation (opt-in), (2) Removed KV cache limit
   - **Configuration**: `CHAT_PROXY_REASONING_TRUNCATE_HISTORY=false` (default disabled)

4. **Module Bypass Discovery**: Modules connect directly to backends, bypass proxy
   - **Analysis**: personal_tagger → port 24001, color_narrator → port 11434/8000
   - **Impact**: Missing automatic truncation, no unified logging, no autostart benefits
   - **Solution**: Phase 1 proxy integration (this document)

5. **Architecture Rethinking**: Multi-model, VRAM-aware selection
   - **User Context**: 16GB RTX 4080 current → 96GB RTX 6000 Pro future
   - **User Clarification**: Prompts stay separate (NOT in registry), need Dify/LangFlow integration later
   - **User Note**: AWQ quantization also suitable for constrained VRAM (in addition to FP8/Q4_K_M)
   - **Strategy**: Model loading/unloading for now, concurrent multi-model after 96GB upgrade
   - **Phase 1 Focus**: Get modules using proxy immediately (config-only changes)
   - **Future Phases**: Role selection (Phase 2), Prompt metrics (Phase 3), Multi-model (Phase 4 with 96GB)

### Available Models (from `uv run imageworks-download list --json`)

| Model | Size (GB) | Quantization | Capabilities | Backend | Port |
|-------|-----------|--------------|--------------|---------|------|
| qwen3-vl-8b-instruct_(FP8) | 10.6 | FP8 | vision+tools+chat | vLLM | 24001 |
| qwen25vl_8.3b_(Q4_K_M) | 6.4 | Q4_K_M | vision+chat | Ollama | 11434 |
| pixtral-local-latest_(Q4_K_M) | 8.9 | Q4_K_M | vision+chat | Ollama | 11434 |
| gpt-oss-20b_(MXFP4) | 13.8 | MXFP4 | reasoning+tools | vLLM | - |
| siglip-large-patch16-384_(FP32) | 2.6 | FP32 | embedding | vLLM | - |

**Constrained VRAM Quantizations** (user confirmed):
- FP8 (10.6GB for qwen3-vl)
- Q4_K_M (6.4GB for qwen2.5vl, 8.9GB for pixtral)
- AWQ (similar to FP8, good quality/size balance)

**16GB VRAM Analysis**:
- Single model: qwen3-vl (10.6GB) with ~5GB headroom ✅
- Dual model (tight): qwen2.5vl (6.4GB) + pixtral (8.9GB) = 15.3GB ⚠️
- Dual model (safe): qwen2.5vl (6.4GB) + siglip (2.6GB) = 9GB ✅
- **Strategy**: Load/unload as needed until 96GB upgrade

---

## Current State Analysis

### Module Configuration (pyproject.toml)

```toml
[tool.imageworks.personal_tagger]
# ... other config ...
default_base_url = "http://localhost:24001/v1"  # ❌ Direct to vLLM
default_use_registry = false  # ❌ Hardcoded model names

[tool.imageworks.color_narrator]
# ... other config ...
# No explicit default_base_url, code defaults vary:
# - vlm.py: uses backend client properly
# - vlm_mono_interpreter.py: hardcoded "http://localhost:8000/v1" in __init__
```

### Code Issues

**VLMMonoInterpreter** (`src/imageworks/apps/color_narrator/core/vlm_mono_interpreter.py`):
```python
class VLMMonoInterpreter:
    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",  # ❌ Hardcoded default
        model: str = "Qwen2-VL-2B-Instruct",
        timeout: int = 120,
        max_tokens: int = 500,
        temperature: float = 0.1,
    ):
        self.base_url = base_url
        self.model = model
        # ...

    def interpret_mono_result(self, ...):
        # ❌ Uses raw requests.post() instead of backend client
        response = requests.post(
            f"{self.base_url}/chat/completions",
            json=request_payload,
            timeout=self.timeout,
        )
```

**Problems**:
1. Direct HTTP requests bypass proxy entirely
2. No automatic history management
3. No error handling from backend client utilities
4. Hardcoded port 8000 (doesn't match any current backend)

---

## Implementation Tasks

### Task 1: Update pyproject.toml - personal_tagger Proxy Config ✅

**File**: `/home/stewa/code/imageworks/pyproject.toml`
**Line**: ~198

**Change**:
```toml
# OLD:
default_base_url = "http://localhost:24001/v1"

# NEW:
default_base_url = "http://localhost:8100/v1"  # Chat proxy with autostart & history management
```

**Impact**:
- personal_tagger now routes through proxy
- Gets automatic vision history truncation (fixes multi-image errors)
- Benefits from model autostart with grace periods
- Future-ready for role-based selection

**Testing**:
```bash
uv run imageworks-personal-tagger --input test_image.jpg
# Should connect to proxy (8100), which forwards to vLLM (24001)
# Check logs/chat_proxy.jsonl for request routing
```

---

### Task 2: Update pyproject.toml - color_narrator Proxy Config ✅

**File**: `/home/stewa/code/imageworks/pyproject.toml`
**Line**: ~241 (image_similarity_checker section)

**Change**:
```toml
# OLD:
default_base_url = "http://localhost:8000/v1"  # Non-existent backend

# NEW:
default_base_url = "http://localhost:8100/v1"  # Chat proxy with autostart & history management
```

**Note**: This affects `image_similarity_checker` which uses VLM embeddings. The `color_narrator` module itself needs code fixes (Task 3).

**Impact**:
- Image similarity checks route through proxy
- Consistent backend URL across all modules

---

### Task 3: Fix VLMMonoInterpreter - Replace requests.post ✅

**File**: `/home/stewa/code/imageworks/src/imageworks/apps/color_narrator/core/vlm_mono_interpreter.py`

**Changes**:

1. **Update imports** (top of file):
```python
# REMOVE:
import requests

# ADD:
from imageworks.libs.vlm import create_backend_client
```

2. **Update __init__ default** (line ~40):
```python
def __init__(
    self,
    base_url: str = "http://localhost:8100/v1",  # Changed from 8000 to 8100 (proxy)
    model: str = "qwen3-vl-8b-instruct_(FP8)",  # Updated to registry name
    timeout: int = 120,
    max_tokens: int = 500,
    temperature: float = 0.1,
):
```

3. **Replace requests.post with backend client** (line ~100-120 in `interpret_mono_result`):
```python
# REMOVE:
response = requests.post(
    f"{self.base_url}/chat/completions",
    json=request_payload,
    timeout=self.timeout,
)
response.raise_for_status()
result_data = response.json()

# REPLACE WITH:
client = create_backend_client(
    base_url=self.base_url,
    api_key="EMPTY",
    timeout=self.timeout,
)

try:
    response = client.chat.completions.create(
        model=self.model,
        messages=request_payload["messages"],
        max_tokens=self.max_tokens,
        temperature=self.temperature,
        stream=False,
    )
    result_data = response.model_dump()
except Exception as e:
    logger.error(f"VLM mono interpretation failed: {e}")
    raise
```

**Impact**:
- VLMMonoInterpreter now uses proxy properly
- Gets automatic history truncation
- Proper error handling from backend client utilities
- Consistent with other module patterns

**Testing**:
```bash
# Run mono checker with VLM interpretation enabled
uv run imageworks-mono --vlm-interpret --input test_image.jpg
# Verify logs show proxy routing, no raw HTTP errors
```

---

### Task 4: Add Role Metadata to qwen3-vl-8b-instruct_(FP8) ✅

**File**: `/home/stewa/code/imageworks/configs/model_registry.curated.json`

**Locate** entry with `"name": "qwen3-vl-8b-instruct_(FP8)"` (around line 900-1100)

**Add/Update Fields**:
```json
{
  "name": "qwen3-vl-8b-instruct_(FP8)",
  "display_name": "qwen3-vl-8b-instruct (FP8)",
  "backend": "vllm",
  // ... existing fields ...

  "roles": [
    "caption",
    "keywords",
    "description",
    "narration",
    "vision_general"
  ],

  "role_priority": {
    "caption": 85,
    "keywords": 80,
    "description": 90,
    "narration": 90,
    "vision_general": 85
  },

  "vram_estimate_mb": 10600,

  // ... rest of existing fields ...
}
```

**Rationale**:
- **caption**: 85 (good, but not specialized)
- **keywords**: 80 (Florence-2 will be better at 95)
- **description**: 90 (excellent at detailed descriptions)
- **narration**: 90 (excellent at storytelling)
- **vision_general**: 85 (solid general-purpose fallback)
- **VRAM**: 10.6GB measured from download list

**Impact**:
- Ready for Phase 2 role-based selection
- Proxy can query `/v1/models/select_by_role?role=description&profile=constrained_16gb`

---

### Task 5: Add Role Metadata to qwen25vl_8.3b_(Q4_K_M) ✅

**File**: `/home/stewa/code/imageworks/configs/model_registry.curated.json`

**Locate** entry with `"name": "qwen25vl_8.3b_(Q4_K_M)"` (around line 800-900)

**Add/Update Fields**:
```json
{
  "name": "qwen25vl_8.3b_(Q4_K_M)",
  "display_name": "qwen25vl 8.3b (Q4 K M)",
  "backend": "ollama",
  // ... existing fields ...

  "roles": [
    "caption",
    "keywords",
    "description",
    "narration",
    "vision_general"
  ],

  "role_priority": {
    "caption": 80,
    "keywords": 75,
    "description": 85,
    "narration": 85,
    "vision_general": 80
  },

  "vram_estimate_mb": 6400,

  // ... rest of existing fields ...
}
```

**Rationale**:
- **Lower priority than qwen3-vl-fp8** (lighter model, slightly lower quality)
- **VRAM advantage**: 6.4GB vs 10.6GB (can fit alongside other models)
- **Fallback role**: When qwen3-vl unavailable or VRAM constrained
- **Q4_K_M quantization**: Good quality/size balance for 16GB card

**Impact**:
- Provides lighter alternative for constrained VRAM scenarios
- Can be loaded alongside pixtral (6.4 + 8.9 = 15.3GB on 16GB card, tight but feasible)

---

### Task 6: Search for Florence-2 Model 🔍

**Command**:
```bash
uv run imageworks-download search "florence" --limit 10
```

**Expected Output**:
- Microsoft Florence-2 models (likely `microsoft/Florence-2-large`, `microsoft/Florence-2-base`)
- Size estimates: ~3-5GB depending on variant
- Specialization: Object detection, captioning, **keyword extraction**

**If Not Found**:
User will provide HuggingFace repo URL (e.g., `microsoft/Florence-2-large`)

**Rationale**:
- Florence-2 specialized for keyword extraction (priority 95 vs qwen's 80)
- Smaller footprint (~3GB) allows concurrent loading on 16GB card
- Future multi-model setup: qwen3-vl (description) + Florence-2 (keywords) + qwen2.5vl (caption)

---

### Task 7: Download Florence-2 Model 📥

**Command** (once repo identified):
```bash
uv run imageworks-download add "microsoft/Florence-2-large" \
  --format safetensors \
  --location linux_wsl \
  --backend vllm
```

**Expected**:
- Model downloads to `~/ai-models/weights/microsoft/Florence-2-large`
- Automatically added to `model_registry.discovered.json`
- ~3-5GB download size

**Verification**:
```bash
uv run imageworks-download list | grep -i florence
```

---

### Task 8: Add Florence-2 to Curated Registry 📝

**File**: `/home/stewa/code/imageworks/configs/model_registry.curated.json`

**Add New Entry** (template, adjust after download discovery):
```json
{
  "name": "florence-2-large_(SAFETENSORS)",
  "display_name": "Florence-2 Large (safetensors)",
  "backend": "vllm",
  "backend_config": {
    "port": 24002,
    "model_path": "/home/stewa/ai-models/weights/microsoft/Florence-2-large",
    "extra_args": [
      "--tensor-parallel-size", "1",
      "--dtype", "auto",
      "--gpu-memory-utilization", "0.85",
      "--max-model-len", "4096",
      "--max-num-seqs", "2"
    ]
  },
  "capabilities": {
    "text": true,
    "vision": true,
    "embedding": false,
    "audio": false,
    "thinking": false,
    "reasoning": false,
    "tools": false,
    "visual": true,
    "multimodal": true,
    "image": true,
    "vl": true
  },
  "generation_defaults": {
    "temperature": 0.3,
    "top_p": 0.9,
    "top_k": 20
  },
  "roles": [
    "keywords",
    "caption",
    "object_detection"
  ],
  "role_priority": {
    "keywords": 95,
    "caption": 75,
    "object_detection": 90
  },
  "vram_estimate_mb": 3500,
  "source": {
    "huggingface_id": "microsoft/Florence-2-large"
  },
  "family": "florence-2-large",
  "source_provider": "hf",
  "quantization": "fp32",
  "download_format": "safetensors",
  "download_location": "linux_wsl"
}
```

**Key Points**:
- **keywords role priority 95**: Highest priority for keyword extraction
- **Dedicated port 24002**: Won't conflict with qwen3-vl (24001)
- **Lower gpu-memory-utilization**: Allows concurrent loading with other models
- **Smaller max-model-len**: Florence-2 doesn't need huge context windows

**Impact**:
- Phase 2 role selection will prefer Florence-2 for keywords tasks
- personal_tagger can use: qwen2.5vl (caption) → Florence-2 (keywords) → qwen3-vl (description)
- 16GB card can load 2 at a time, swap as needed

---

## Validation & Testing

### Post-Implementation Checks

**1. Proxy Routing Verification**:
```bash
# Start chat proxy
uv run uvicorn imageworks.chat_proxy.app:app --host 0.0.0.0 --port 8100

# In another terminal, check proxy health
curl http://localhost:8100/v1/models

# Test personal_tagger routing
uv run imageworks-personal-tagger --input test_image.jpg --verbose

# Check proxy logs
tail -f logs/chat_proxy.jsonl | jq
```

**2. History Truncation Validation**:
```bash
# Multi-image test (should NOT fail with max_model_len error)
uv run imageworks-personal-tagger \
  --input image1.jpg image2.jpg image3.jpg \
  --verbose

# Verify truncation in logs
grep "truncate_history" logs/chat_proxy.jsonl
```

**3. Registry Role Metadata Check**:
```bash
# Verify roles added correctly
uv run imageworks-download list --json | \
  jq '.[] | select(.name | contains("qwen3-vl")) | {name, roles, role_priority, vram_estimate_mb}'

uv run imageworks-download list --json | \
  jq '.[] | select(.name | contains("qwen25vl")) | {name, roles, role_priority, vram_estimate_mb}'
```

**4. VLMMonoInterpreter Fix Validation**:
```bash
# Test mono checker with VLM interpretation
uv run imageworks-mono --vlm-interpret --input test_image.jpg

# Should NOT see raw requests errors, should route through proxy
grep "VLMMonoInterpreter" logs/chat_proxy.jsonl
```

---

## Rollback Plan

If issues arise, revert changes:

**pyproject.toml**:
```toml
# Revert to direct backend connections
[tool.imageworks.personal_tagger]
default_base_url = "http://localhost:24001/v1"

[tool.imageworks.image_similarity_checker]
default_base_url = "http://localhost:8000/v1"
```

**vlm_mono_interpreter.py**:
```python
# Revert to original imports and requests.post() pattern
# (Keep git diff handy)
```

**model_registry.curated.json**:
```bash
# Restore from backup
cp configs/model_registry.curated.json.backup configs/model_registry.curated.json
```

**Backup Command** (run before starting):
```bash
cp pyproject.toml pyproject.toml.pre-phase1
cp configs/model_registry.curated.json configs/model_registry.curated.json.pre-phase1
cp src/imageworks/apps/color_narrator/core/vlm_mono_interpreter.py \
   src/imageworks/apps/color_narrator/core/vlm_mono_interpreter.py.pre-phase1
```

---

## Phase 2 Preview (Role-Based Selection)

After Phase 1 completion, Phase 2 will add:

**New Files**:
- `src/imageworks/chat_proxy/role_selector.py`: Role-based model selection logic
- `src/imageworks/libs/hardware/gpu_detector.py`: GPU auto-detection via nvidia-smi
- `src/imageworks/chat_proxy/profile_manager.py`: Deployment profile management
- `docs/architecture/deployment-profiles.md`: ✅ **COMPLETED** - Comprehensive VRAM profile design

**Profile Auto-Detection** (NEW):
Deployment profiles are **semi-automatic with smart defaults**:
1. System queries `nvidia-smi` for GPU count and VRAM
2. Auto-selects appropriate profile (constrained_16gb, balanced_24gb, multi_gpu_2x16gb, generous_96gb)
3. Users can override via `IMAGEWORKS_DEPLOYMENT_PROFILE=<name>` environment variable
4. Falls back to `development` profile for CPU-only or low-VRAM systems

**Example Auto-Detection Logic**:
```python
# Single RTX 4080 (16GB)    → constrained_16gb profile
# Single RTX 4090 (24GB)    → balanced_24gb profile
# 2x RTX 4080 (32GB total)  → multi_gpu_2x16gb profile
# RTX 6000 Pro (96GB)       → generous_96gb profile
# <12GB or no GPU           → development profile
```

**pyproject.toml Additions**:
```toml
[tool.imageworks.deployment_profiles.constrained_16gb]
description = "RTX 4080 16GB - single model, load/unload strategy"
max_vram_mb = 14000  # 16384 - 2048 headroom
max_concurrent_models = 1
preferred_quantizations = ["q4_k_m", "fp8", "awq", "q5_k_m"]
strategy = "load_unload"
model_selection_bias = "vram_efficient"

[tool.imageworks.deployment_profiles.balanced_24gb]
description = "RTX 4090 24GB - dual model concurrent serving"
max_vram_mb = 22000  # 24576 - 2048 headroom
max_concurrent_models = 2
preferred_quantizations = ["fp8", "awq", "q4_k_m"]
strategy = "concurrent_limited"
model_selection_bias = "balanced"

[tool.imageworks.deployment_profiles.multi_gpu_2x16gb]
description = "2x RTX 4080 16GB - tensor parallel or distributed models"
max_vram_mb = 28000  # (16384 * 2) - 4096 headroom
max_concurrent_models = 3
gpu_count = 2
preferred_quantizations = ["fp8", "awq", "bf16"]
strategy = "tensor_parallel_or_distributed"
model_selection_bias = "quality"

[tool.imageworks.deployment_profiles.generous_96gb]
description = "RTX 6000 Pro 96GB - multi-model concurrent serving"
max_vram_mb = 90000  # 98304 - 8192 headroom
max_concurrent_models = 5
preferred_quantizations = ["fp8", "awq", "bf16", "fp16"]
strategy = "concurrent_full"
model_selection_bias = "quality"
```

**forwarder.py Additions**:
```python
# New endpoint: /v1/models/select_by_role
@router.get("/v1/models/select_by_role")
async def select_by_role(
    role: str,
    profile: Optional[str] = None,
    exclude_models: Optional[List[str]] = None,
):
    """Select best model for given role considering deployment profile."""
    # Reads profile constraints from pyproject.toml
    # Queries registry for models with matching role
    # Returns highest priority model that fits VRAM constraints
```

**personal_tagger Updates**:
```python
# Enable registry-based role selection
use_registry = True  # Now queries proxy's role selection endpoint
caption_model = "role:caption"  # Resolved via proxy
keyword_model = "role:keywords"  # Resolved via proxy
description_model = "role:description"  # Resolved via proxy
```

**Example Flow**:
1. personal_tagger requests `role:keywords`
2. Proxy queries registry: "Give me highest priority model for keywords role on constrained_16gb profile"
3. Registry returns: Florence-2 (priority 95, 3.5GB fits profile)
4. Proxy autostarts Florence-2 on port 24002 if not running
5. Proxy forwards request to Florence-2
6. Result returned to personal_tagger

---

## Success Criteria

Phase 1 considered complete when:

- ✅ pyproject.toml updated with proxy URLs (2 config sections)
- ✅ VLMMonoInterpreter uses `create_backend_client()` instead of raw requests
- ✅ qwen3-vl-8b-instruct_(FP8) has roles/role_priority/vram_estimate_mb
- ✅ qwen25vl_8.3b_(Q4_K_M) has roles/role_priority/vram_estimate_mb
- ✅ Florence-2 searched, downloaded (if found), added to curated registry
- ✅ Multi-image personal-tagger test passes without max_model_len errors
- ✅ Proxy logs show request routing for all three modules
- ✅ No breaking changes to existing functionality

---

## Complete Multi-Phase Roadmap

### Phase 1: Proxy Integration (Current - Config Only) ✅
**Timeline**: 1-2 days
**Complexity**: Low
**Dependencies**: None

**Objectives**:
- Redirect all modules through chat proxy (port 8100)
- Enable automatic history truncation for vision/reasoning models
- Add role metadata to existing models (qwen3-vl, qwen2.5vl)
- Search and download Florence-2 for specialized keyword extraction

**Deliverables**:
- ✅ Updated pyproject.toml (2 base_url changes)
- ✅ Fixed VLMMonoInterpreter (backend client pattern)
- ✅ Role metadata in curated registry (2 models)
- ✅ Florence-2 model added to registry

**Benefits**:
- Eliminates max_model_len errors on multi-image analysis
- Unified logging and monitoring
- Foundation for role-based selection

---

### Phase 2: Role-Based Model Selection (Next - 1-2 Weeks)
**Timeline**: 1-2 weeks
**Complexity**: Medium
**Dependencies**: Phase 1 complete

**Objectives**:
- Implement deployment profile system (constrained_16gb, generous_96gb, development)
- Create RoleSelector class with profile-aware logic
- Add `/v1/models/select_by_role` endpoint to chat proxy
- Update personal_tagger to use role-based model resolution

**Architecture Components**:

**1. Deployment Profiles** (`pyproject.toml`):
```toml
[tool.imageworks.deployment_profiles.constrained_16gb]
description = "RTX 4080 16GB - single model, load/unload strategy"
max_vram_mb = 14000  # Leave 2GB headroom
max_concurrent_models = 1
preferred_quantizations = ["q4_k_m", "fp8", "awq", "q5_k_m"]
strategy = "load_unload"  # Swap models as needed
autostart_timeout_seconds = 120

[tool.imageworks.deployment_profiles.generous_96gb]
description = "RTX 6000 Pro 96GB - multi-model concurrent serving"
max_vram_mb = 90000  # Leave 6GB headroom
max_concurrent_models = 4
preferred_quantizations = ["fp8", "awq", "bf16", "fp16"]
strategy = "concurrent"  # Keep multiple models loaded
autostart_timeout_seconds = 60

[tool.imageworks.deployment_profiles.development]
description = "Development/testing - minimal footprint"
max_vram_mb = 8000
max_concurrent_models = 1
preferred_quantizations = ["q4_k_m", "q5_k_m"]
strategy = "load_unload"
autostart_timeout_seconds = 180
```

**2. RoleSelector Class** (`src/imageworks/chat_proxy/role_selector.py`):
```python
class RoleSelector:
    """Profile-aware model selection by role."""

    def __init__(self, registry: ModelRegistry, config: ProxyConfig):
        self.registry = registry
        self.config = config
        self.profile = self._load_deployment_profile()

    def select_model_for_role(
        self,
        role: str,
        exclude_models: Optional[List[str]] = None,
    ) -> Optional[ModelInfo]:
        """
        Select best model for role considering:
        - Role priority scores
        - VRAM constraints from deployment profile
        - Preferred quantizations
        - Currently loaded models
        """
        candidates = self.registry.get_models_by_role(role)

        # Filter by VRAM constraints
        candidates = [
            m for m in candidates
            if m.vram_estimate_mb <= self.profile.max_vram_mb
        ]

        # Prefer quantizations from profile
        candidates = sorted(
            candidates,
            key=lambda m: (
                m.role_priority.get(role, 0),
                self._quantization_score(m.quantization),
            ),
            reverse=True,
        )

        return candidates[0] if candidates else None
```

**3. Proxy Endpoint** (`src/imageworks/chat_proxy/forwarder.py`):
```python
@router.get("/v1/models/select_by_role")
async def select_by_role(
    role: str,
    profile: Optional[str] = None,
    exclude_models: Optional[List[str]] = None,
):
    """
    Select best model for given role.

    Example:
        GET /v1/models/select_by_role?role=keywords&profile=constrained_16gb

    Returns:
        {
            "model": "florence-2-large_(SAFETENSORS)",
            "role": "keywords",
            "priority": 95,
            "vram_estimate_mb": 3500,
            "profile": "constrained_16gb"
        }
    """
    selector = RoleSelector(registry, config)
    model = selector.select_model_for_role(role, exclude_models)

    if not model:
        raise HTTPException(404, f"No model found for role: {role}")

    return {
        "model": model.name,
        "role": role,
        "priority": model.role_priority.get(role, 0),
        "vram_estimate_mb": model.vram_estimate_mb,
        "profile": selector.profile.name,
    }
```

**4. Module Integration** (`personal_tagger/core/inference.py`):
```python
class TaggerInference:
    def _resolve_role_model(self, role: str) -> str:
        """Resolve model name from role via proxy."""
        if not self.use_registry:
            return self.hardcoded_models.get(role)

        # Query proxy for best model
        response = requests.get(
            f"{self.base_url.replace('/v1', '')}/v1/models/select_by_role",
            params={"role": role},
        )
        response.raise_for_status()
        return response.json()["model"]
```

**Deliverables**:
- `role_selector.py` (~200 lines)
- Deployment profile configuration in pyproject.toml
- `/v1/models/select_by_role` endpoint
- Updated personal_tagger with use_registry=true
- Documentation: `docs/architecture/deployment-profiles.md`

**Testing Strategy**:
```bash
# Test role selection API
curl "http://localhost:8100/v1/models/select_by_role?role=keywords"
# Expected: Florence-2 (priority 95)

# Test with profile constraint
IMAGEWORKS_DEPLOYMENT_PROFILE=constrained_16gb \
  uv run imageworks-personal-tagger --input test.jpg

# Verify model selection in logs
grep "role_selector" logs/chat_proxy.jsonl
```

**Benefits**:
- Automatic model selection based on task requirements
- VRAM-aware decisions (works on 16GB now, scales to 96GB later)
- Easy switching between deployment profiles via environment variable
- No code changes needed when adding new models

---

### Phase 3: Prompt Metrics & Experimentation (Month 2)
**Timeline**: 2-3 weeks
**Complexity**: Medium
**Dependencies**: Phase 2 complete

**Objectives**:
- Track prompt performance across different models and profiles
- Enable A/B testing of prompt variations
- Prepare integration hooks for Dify/LangFlow
- Build analytics for prompt optimization

**Architecture Components**:

**1. PromptMetricsTracker** (`src/imageworks/libs/prompting/metrics.py`):
```python
@dataclass
class PromptExecution:
    """Single prompt execution record."""
    prompt_id: str  # e.g., "personal_tagger.caption.baseline"
    prompt_version: str  # e.g., "v2.1"
    model_name: str
    role: str
    deployment_profile: str

    # Input context
    input_tokens: int
    image_count: int

    # Output metrics
    output_tokens: int
    ttft_ms: float  # Time to first token
    throughput_tps: float  # Tokens per second
    total_time_ms: float

    # Quality indicators
    output_length: int
    contains_refusal: bool
    contains_hallucination_markers: List[str]

    # Metadata
    timestamp: str
    success: bool
    error_message: Optional[str]

class PromptMetricsTracker:
    """JSONL-based prompt execution tracking."""

    def __init__(self, log_path: Path = Path("logs/prompt_metrics.jsonl")):
        self.log_path = log_path

    def log_execution(self, execution: PromptExecution):
        """Append execution record to JSONL log."""
        with open(self.log_path, "a") as f:
            f.write(execution.to_json() + "\n")

    def analyze_prompt_performance(
        self,
        prompt_id: str,
        time_window_hours: int = 24,
    ) -> PromptAnalysis:
        """
        Analyze recent executions of a prompt:
        - Average throughput by model
        - Success rate
        - Output quality indicators
        """
```

**2. Module Integration** (`personal_tagger/core/inference.py`):
```python
class TaggerInference:
    def __init__(self, ...):
        self.metrics_tracker = PromptMetricsTracker()
        self.prompt_library = TaggerPromptLibrary()

    def _run_stage(self, stage: str, image: Image, context: dict):
        """Execute stage with metrics tracking."""
        prompt_profile = self.prompt_library.get(self.prompt_profile_name)
        stage_prompt = getattr(prompt_profile, f"{stage}_prompt")

        start_time = time.time()
        model = self._resolve_role_model(stage)

        # Execute with timing
        result = self.client.chat.completions.create(
            model=model,
            messages=self._build_messages(stage_prompt, image, context),
            max_tokens=stage_prompt.max_new_tokens,
        )

        # Log metrics
        self.metrics_tracker.log_execution(PromptExecution(
            prompt_id=f"personal_tagger.{stage}.{self.prompt_profile_name}",
            prompt_version=prompt_profile.version,
            model_name=model,
            role=stage,
            deployment_profile=os.getenv("IMAGEWORKS_DEPLOYMENT_PROFILE", "default"),
            input_tokens=result.usage.prompt_tokens,
            output_tokens=result.usage.completion_tokens,
            total_time_ms=(time.time() - start_time) * 1000,
            timestamp=datetime.utcnow().isoformat(),
            success=True,
        ))

        return result.choices[0].message.content
```

**3. Analysis Scripts** (`scripts/analyze_prompt_performance.py`):
```python
def analyze_prompts(log_path: Path, output_path: Path):
    """
    Generate prompt performance report:
    - Throughput by model/prompt combination
    - Success rates
    - Output length distributions
    - Recommended model/prompt pairings
    """
    executions = load_jsonl(log_path)

    report = {
        "summary": compute_summary_stats(executions),
        "by_prompt": analyze_by_prompt(executions),
        "by_model": analyze_by_model(executions),
        "recommendations": generate_recommendations(executions),
    }

    write_markdown_report(report, output_path)
```

**4. Dify/LangFlow Integration Hooks** (`src/imageworks/libs/prompting/storage.py`):
```python
class DifyPromptStorage(PromptStorageBackend):
    """
    Load prompts from Dify API.
    Future integration - Phase 3 creates interface, Phase 4 implements.
    """

    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url
        self.api_key = api_key

    def get_prompt(self, prompt_id: str) -> PromptProfileBase:
        """Fetch prompt from Dify workflow."""
        response = requests.get(
            f"{self.api_url}/workflows/{prompt_id}",
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        return self._parse_dify_prompt(response.json())

    def list_prompts(self) -> List[str]:
        """List available prompts from Dify."""
        # Stub for Phase 4 implementation

class LangFlowPromptStorage(PromptStorageBackend):
    """
    Load prompts from LangFlow.
    Future integration - Phase 3 creates interface, Phase 4 implements.
    """
    # Similar structure to DifyPromptStorage
```

**Deliverables**:
- `prompting/metrics.py` (~300 lines)
- `prompting/storage.py` with Dify/LangFlow stubs (~200 lines)
- `scripts/analyze_prompt_performance.py` (~400 lines)
- Metrics logging integrated into personal_tagger and color_narrator
- Documentation: `docs/guides/prompt-metrics-and-experimentation.md`

**Testing Strategy**:
```bash
# Run tagging with metrics enabled
IMAGEWORKS_ENABLE_PROMPT_METRICS=true \
  uv run imageworks-personal-tagger --input test.jpg

# Verify metrics logged
cat logs/prompt_metrics.jsonl | jq

# Generate performance report
uv run python scripts/analyze_prompt_performance.py \
  --input logs/prompt_metrics.jsonl \
  --output outputs/summaries/prompt_performance.md
```

**Benefits**:
- Data-driven prompt optimization
- A/B testing infrastructure
- Foundation for visual prompt management tools
- Performance baselines before 96GB upgrade

---

### Phase 4: Multi-Model Concurrent Serving (After 96GB Upgrade)
**Timeline**: 3-4 weeks
**Complexity**: High
**Dependencies**: Phase 2 complete, 96GB GPU available

**Objectives**:
- Run multiple vLLM instances concurrently (one per model)
- Implement intelligent VRAM allocation
- Enable parallel inference for different roles
- Complete Dify/LangFlow integration

**Architecture Components**:

**1. MultiVllmOrchestrator** (`src/imageworks/chat_proxy/multi_vllm_manager.py`):
```python
@dataclass
class VllmInstance:
    """Single vLLM instance configuration."""
    model_name: str
    port: int
    vram_allocation_mb: int
    gpu_memory_utilization: float
    max_model_len: int
    process: Optional[subprocess.Popen]
    health_status: str  # "starting", "healthy", "unhealthy"

class MultiVllmOrchestrator:
    """
    Manage multiple concurrent vLLM instances.
    Allocates VRAM, assigns ports, monitors health.
    """

    def __init__(self, config: ProxyConfig):
        self.config = config
        self.instances: Dict[str, VllmInstance] = {}
        self.port_allocator = PortAllocator(start=24001, end=24010)
        self.vram_allocator = VramAllocator(total_mb=90000)

    async def start_model_group(
        self,
        models: List[str],
        allocation_strategy: str = "balanced",
    ):
        """
        Start multiple models concurrently.

        allocation_strategy:
        - "balanced": Equal VRAM split
        - "priority": Allocate by role priority
        - "custom": Use vram_estimate_mb from registry
        """
        allocations = self.vram_allocator.compute_allocations(
            models,
            strategy=allocation_strategy,
        )

        for model_name, vram_mb in allocations.items():
            port = self.port_allocator.allocate()
            instance = await self._start_instance(
                model_name=model_name,
                port=port,
                vram_allocation_mb=vram_mb,
            )
            self.instances[model_name] = instance

    async def health_check_loop(self):
        """Monitor all instances, restart if needed."""
        while True:
            for name, instance in self.instances.items():
                if not await self._check_health(instance):
                    logger.warning(f"Instance {name} unhealthy, restarting...")
                    await self._restart_instance(name)
            await asyncio.sleep(30)
```

**2. VRAM Allocation Strategies**:
```python
class VramAllocator:
    """Intelligent VRAM allocation for concurrent models."""

    def compute_allocations(
        self,
        models: List[str],
        strategy: str,
    ) -> Dict[str, int]:
        """
        Example allocation on 96GB GPU (90GB usable):

        Strategy "balanced":
        - qwen3-vl-8b-fp8: 22500 MB (25%)
        - qwen2.5vl-7b-q4: 22500 MB (25%)
        - florence-2-large: 22500 MB (25%)
        - siglip-large: 22500 MB (25%)

        Strategy "priority":
        - qwen3-vl-8b-fp8: 30000 MB (33%, highest priority)
        - florence-2-large: 25000 MB (28%, specialized)
        - qwen2.5vl-7b-q4: 20000 MB (22%, fallback)
        - siglip-large: 15000 MB (17%, embedding)

        Strategy "custom":
        - Use vram_estimate_mb from registry
        - Verify total <= max_vram_mb
        - Adjust gpu_memory_utilization per model
        """
```

**3. Parallel Inference Routing** (`forwarder.py` enhancement):
```python
class MultiModelForwarder:
    """Route requests to appropriate running instance."""

    def __init__(self, orchestrator: MultiVllmOrchestrator):
        self.orchestrator = orchestrator
        self.role_selector = RoleSelector(registry, config)

    async def forward_request(self, request: ChatRequest):
        """
        Route to best available instance:
        1. Determine required role from request
        2. Query role_selector for best model
        3. Check if model instance is running
        4. Route to instance's port
        5. If not running, trigger load/unload or fail
        """
        role = self._infer_role_from_request(request)
        model = self.role_selector.select_model_for_role(role)

        instance = self.orchestrator.instances.get(model.name)
        if not instance or instance.health_status != "healthy":
            raise HTTPException(503, f"Model {model.name} not available")

        # Forward to instance
        return await self._proxy_to_instance(instance, request)
```

**4. Example Multi-Model Configuration**:
```toml
# pyproject.toml deployment profile for 96GB
[tool.imageworks.deployment_profiles.generous_96gb]
strategy = "concurrent"
max_concurrent_models = 4

# Model group definitions
[[tool.imageworks.deployment_profiles.generous_96gb.model_groups]]
name = "vision_processing"
models = [
    "qwen3-vl-8b-instruct_(FP8)",      # Description & narration
    "florence-2-large_(SAFETENSORS)",  # Keywords & object detection
    "qwen25vl_8.3b_(Q4_K_M)",          # Caption fallback
    "siglip-large-patch16-384_(FP32)", # Embeddings
]
allocation_strategy = "custom"
startup_sequence = "parallel"  # Start all at once
health_check_interval_seconds = 30
```

**5. Dify/LangFlow Full Integration**:
```python
# Enable visual prompt editing
class DifyPromptStorage(PromptStorageBackend):
    """Full implementation of Dify integration."""

    def get_prompt(self, prompt_id: str) -> PromptProfileBase:
        """
        Fetch prompt from Dify:
        1. Query workflow by ID
        2. Extract prompt template and parameters
        3. Parse into PromptProfileBase format
        4. Cache locally for performance
        """
        # Full implementation in Phase 4

    def update_prompt(self, prompt: PromptProfileBase):
        """Push prompt changes back to Dify."""
        # Full implementation in Phase 4

# Visual workflow for prompt experimentation
class DifyWorkflowRunner:
    """Execute Dify workflows with ImageWorks models."""

    def run_workflow(
        self,
        workflow_id: str,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Run Dify workflow:
        1. Fetch workflow definition
        2. Replace LLM nodes with ImageWorks proxy calls
        3. Execute workflow steps
        4. Return results with metrics
        """
```

**Deliverables**:
- `multi_vllm_manager.py` (~500 lines)
- `vram_allocator.py` (~300 lines)
- Enhanced `forwarder.py` with multi-instance routing
- Complete Dify/LangFlow integration (~600 lines)
- Model group configuration system
- Documentation: `docs/architecture/multi-model-concurrent-serving.md`

**Testing Strategy**:
```bash
# Start multi-model group
IMAGEWORKS_DEPLOYMENT_PROFILE=generous_96gb \
  uv run uvicorn imageworks.chat_proxy.app:app --port 8100

# Verify all instances healthy
curl http://localhost:8100/v1/models/health

# Test parallel inference
parallel -j4 'curl -X POST http://localhost:8100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d @test_request_{}.json' ::: 1 2 3 4

# Monitor VRAM usage
nvidia-smi dmon -s u
```

**Benefits**:
- 3-4x throughput improvement (parallel processing)
- No model loading delays (all pre-loaded)
- Specialized models for each task (optimal quality)
- Visual prompt management via Dify/LangFlow
- Full utilization of 96GB VRAM

---

### Future Enhancements (Beyond Phase 4)

**Phase 5: Advanced Optimization** (Optional)
- Quantization-aware training for custom models
- Speculative decoding for faster inference
- KV cache sharing between instances
- Dynamic VRAM rebalancing based on load

**Phase 6: Production Hardening** (Optional)
- Kubernetes deployment with auto-scaling
- Model versioning and A/B deployment
- Rate limiting and quota management
- Monitoring and alerting (Prometheus/Grafana)
- Disaster recovery and backup strategies

**Phase 7: Ecosystem Integration** (Optional)
- LangChain/LlamaIndex integration
- OpenAI-compatible API extensions
- Custom tool/function calling registry
- Multi-tenant support
- API gateway integration

---

## Implementation Priority

**Immediate** (Next 2-3 weeks):
1. ✅ Phase 1: Proxy integration (config-only, low risk)
2. → Phase 2: Role-based selection (enables VRAM-aware decisions)

**Short-term** (Month 2):
3. → Phase 3: Prompt metrics (data-driven optimization)
4. → Experiment with 16GB dual-model (qwen2.5vl + pixtral)

**Long-term** (After 96GB upgrade):
5. → Phase 4: Multi-model concurrent (full utilization)
6. → Complete Dify/LangFlow integration

**Future** (As needed):
7. → Phase 5-7 based on production requirements

---

## References

**Related Documentation**:
- `docs/architecture/registry-and-proxy-architecture.md`: System overview
- `docs/guides/vision-history-management.md`: Vision truncation details
- `docs/decisions/`: Architecture decision records (to be created)

**Key Files**:
- `configs/model_registry.curated.json`: Hand-maintained model definitions
- `configs/model_registry.discovered.json`: Auto-generated runtime state
- `configs/model_registry.json`: Merged snapshot (read-only)
- `pyproject.toml`: Module configuration defaults
- `src/imageworks/chat_proxy/forwarder.py`: Request routing with history truncation
- `src/imageworks/chat_proxy/vllm_manager.py`: vLLM orchestration
- `src/imageworks/libs/vlm/`: Backend client utilities

**Commands Reference**:
```bash
# Model management
uv run imageworks-download list [--json]
uv run imageworks-download search <query>
uv run imageworks-download add <hf_repo>

# Module execution
uv run imageworks-personal-tagger --input <image>
uv run imageworks-color-narrator --input <image>
uv run imageworks-mono --input <image> [--vlm-interpret]

# Proxy management
uv run uvicorn imageworks.chat_proxy.app:app --host 0.0.0.0 --port 8100
curl http://localhost:8100/v1/models
tail -f logs/chat_proxy.jsonl | jq
```

---

**Document Version**: 1.0
**Last Updated**: October 26, 2025
**Author**: Copilot + User Collaboration
**Review Status**: Awaiting implementation approval
