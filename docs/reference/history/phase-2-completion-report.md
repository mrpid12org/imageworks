# Phase 2: Deployment Profiles & Role-Based Model Selection
## Implementation Complete Report

**Date**: October 26, 2025
**Status**: ✅ **COMPLETE**
**Test Results**: 54/54 tests passing (41 Phase 2 unit + 13 Phase 1 integration)

---

## Executive Summary

Phase 2 successfully implements intelligent deployment profile management with GPU auto-detection and role-based model selection. The system automatically adapts to available hardware and provides API endpoints for querying configuration and selecting optimal models for specific tasks.

### Key Achievements

1. **GPU Auto-Detection**: Automatic NVIDIA GPU detection via `nvidia-smi` with caching
2. **Profile Management**: 5 deployment profiles covering 16GB to 96GB VRAM scenarios
3. **Role-Based Selection**: Smart model selection based on task roles (keywords, caption, description, etc.)
4. **API Endpoints**: Two new endpoints for profile info and model selection
5. **Comprehensive Testing**: 41 new unit tests covering all scenarios
6. **Zero Regressions**: All Phase 1 tests still passing

---

## Components Implemented

### 1. GPU Detection Module
**File**: `src/imageworks/libs/hardware/gpu_detector.py`

**Features**:
- Queries `nvidia-smi` for GPU information (index, name, VRAM, compute capability, UUID)
- Caches detection results for performance
- Calculates usable VRAM (total - 2GB headroom per GPU)
- Recommends appropriate deployment profile based on hardware

**Auto-Detection Logic**:
```
Single GPU:
  - >= 85GB usable → generous_96gb
  - >= 20GB usable → balanced_24gb
  - >= 14GB usable → constrained_16gb
  - <  14GB usable → development

Multi-GPU (2 similar cards):
  - avg >= 14GB → multi_gpu_2x16gb
  - avg <  14GB → development

3+ GPUs:
  - >= 60GB total → generous_96gb
  - >= 40GB total → balanced_24gb
  - <  40GB total → constrained_16gb
```

**Test Coverage**: 15 unit tests
- GPU detection (single/multiple)
- Caching behavior
- VRAM calculations
- Profile recommendations
- Error handling

---

### 2. Profile Manager
**File**: `src/imageworks/chat_proxy/profile_manager.py`

**Features**:
- Loads deployment profiles from `pyproject.toml`
- Auto-detects GPU hardware and selects appropriate profile
- Supports manual override via `IMAGEWORKS_DEPLOYMENT_PROFILE` environment variable
- Provides comprehensive profile and GPU information via API

**Manual Override Example**:
```bash
export IMAGEWORKS_DEPLOYMENT_PROFILE=development
# System will use development profile regardless of detected hardware
```

**Test Coverage**: 12 unit tests
- Profile loading from config
- Auto-detection integration
- Manual override behavior
- Fallback handling
- Profile information API

---

### 3. Deployment Profiles
**File**: `pyproject.toml` - `[tool.imageworks.deployment_profiles.*]`

#### Profile Definitions

**constrained_16gb** (RTX 4080 16GB)
```toml
max_vram_mb = 14000  # 16384 - 2048 headroom
max_concurrent_models = 1
preferred_quantizations = ["q4_k_m", "fp8", "awq", "q5_k_m", "q8_0"]
strategy = "load_unload"
model_selection_bias = "vram_efficient"
```

**balanced_24gb** (RTX 4090 24GB)
```toml
max_vram_mb = 22000
max_concurrent_models = 2
preferred_quantizations = ["fp8", "awq", "q4_k_m", "q5_k_m"]
strategy = "concurrent_limited"
model_selection_bias = "balanced"
```

**multi_gpu_2x16gb** (2x RTX 4080 16GB)
```toml
max_vram_mb = 28000
max_concurrent_models = 3
gpu_count = 2
preferred_quantizations = ["fp8", "awq", "bf16", "q5_k_m"]
strategy = "tensor_parallel_or_distributed"
model_selection_bias = "quality"
```

**generous_96gb** (RTX 6000 Pro 96GB)
```toml
max_vram_mb = 90000
max_concurrent_models = 5
preferred_quantizations = ["fp8", "awq", "bf16", "fp16"]
strategy = "concurrent_full"
model_selection_bias = "quality"
```

**development** (Testing/Low VRAM)
```toml
max_vram_mb = 10000
max_concurrent_models = 1
preferred_quantizations = ["q4_k_m", "q5_k_m", "q8_0"]
strategy = "load_unload"
model_selection_bias = "vram_efficient"
```

---

### 4. Role Selector
**File**: `src/imageworks/chat_proxy/role_selector.py`

**Features**:
- Loads model registry with role_priority metadata
- Filters models by VRAM constraints from active profile
- Sorts by role_priority scores and quantization preferences
- Supports selection biases (vram_efficient, balanced, quality)

**Supported Roles**:
- `keywords` - Extract keywords from images
- `caption` - Generate short captions
- `description` - Generate detailed descriptions
- `narration` - Create narrative descriptions
- `object_detection` - Detect objects in images
- `vision_general` - General vision tasks

**Selection Algorithm**:
1. Filter models by role support (has role_priority for requested role)
2. Filter by VRAM constraints (vram_estimate_mb <= profile.max_vram_mb)
3. Sort by:
   - Primary: role_priority score (higher = better)
   - Secondary: quantization preference (earlier in list = better)
   - Tertiary: VRAM (lower = better for efficiency bias)
4. Apply profile's model_selection_bias
5. Return top N results

**Test Coverage**: 14 unit tests
- Role-based selection
- VRAM filtering
- Quantization preferences
- Selection biases
- Model lookup and explanations

---

### 5. API Endpoints

#### GET `/v1/config/profile`
Returns active deployment profile and detected GPU information.

**Response Example**:
```json
{
  "active_profile": {
    "name": "constrained_16gb",
    "max_vram_mb": 14000,
    "max_concurrent_models": 1,
    "preferred_quantizations": ["q4_k_m", "fp8", "awq"],
    "strategy": "load_unload",
    "model_selection_bias": "vram_efficient"
  },
  "detected_gpus": [
    {
      "index": 0,
      "name": "NVIDIA GeForce RTX 4080",
      "total_vram_mb": 16384,
      "free_vram_mb": 15234,
      "compute_capability": [8, 9],
      "uuid": "GPU-12345678-abcd-..."
    }
  ],
  "total_usable_vram_mb": 14336,
  "available_profiles": [
    "constrained_16gb",
    "balanced_24gb",
    "multi_gpu_2x16gb",
    "generous_96gb",
    "development"
  ],
  "manual_override": null
}
```

**Use Cases**:
- Check what profile the system selected
- Verify GPU detection
- Troubleshoot VRAM allocation issues
- Monitor hardware utilization

---

#### GET `/v1/models/select_by_role?role={role}&top_n={n}`
Selects best models for a specific task role within profile constraints.

**Parameters**:
- `role` (required): Task role (keywords, caption, description, etc.)
- `top_n` (optional, default=3): Number of models to return

**Response Example**:
```json
{
  "role": "keywords",
  "profile": "constrained_16gb",
  "max_vram_mb": 14000,
  "model_selection_bias": "vram_efficient",
  "top_n": 3,
  "models": [
    {
      "id": "florence-2-large",
      "name": "Florence-2 Large",
      "backend": "ollama",
      "quantization": "Q4_K_M",
      "vram_estimate_mb": 3500,
      "role_priority": 95
    },
    {
      "id": "qwen3-vl",
      "name": "Qwen3-VL",
      "backend": "vllm",
      "quantization": "FP8",
      "vram_estimate_mb": 10600,
      "role_priority": 80
    },
    {
      "id": "qwen25vl",
      "name": "Qwen2.5-VL",
      "backend": "vllm",
      "quantization": "FP8",
      "vram_estimate_mb": 6400,
      "role_priority": 75
    }
  ]
}
```

**Use Cases**:
- Automatically select optimal model for a task
- Compare model options before loading
- Build UI for model selection
- Implement adaptive model routing

**Error Responses**:
- `400 Bad Request`: Invalid role name
- `500 Internal Error`: No active profile
- `503 Service Unavailable`: Profile management not initialized

---

## Integration with Chat Proxy

### Startup Initialization
**File**: `src/imageworks/chat_proxy/app.py`

The chat proxy now initializes profile management on startup:

```python
@app.on_event("startup")
async def _startup():
    global _profile_manager, _role_selector

    load_registry()

    # Phase 2: Initialize profile manager and role selector
    try:
        _profile_manager = ProfileManager()
        _role_selector = RoleSelector()
        profile = _profile_manager.get_active_profile()
        if profile:
            logging.info(f"[app] Active deployment profile: {profile.name}")
    except Exception as e:
        logging.error(f"[app] Failed to initialize profile management: {e}")
```

### Logging Output Example
```
INFO:imageworks.libs.hardware.gpu_detector:Detected 1 GPU(s)
INFO:imageworks.libs.hardware.gpu_detector:  GPU 0: NVIDIA GeForce RTX 4080 (16384MB VRAM)
INFO:imageworks.libs.hardware.gpu_detector:Hardware: 1 GPU(s), 14336MB usable VRAM (after headroom)
INFO:imageworks.libs.hardware.gpu_detector:Recommended profile: constrained_16gb
INFO:imageworks.chat_proxy.profile_manager:Loaded 5 deployment profiles
INFO:imageworks.chat_proxy.profile_manager:Auto-detected profile: constrained_16gb (GPUs: 1, VRAM: 14336MB)
INFO:imageworks.chat_proxy.role_selector:Loaded 18 models from registry
INFO:root:[app] Active deployment profile: constrained_16gb
```

---

## Test Results

### Phase 2 Unit Tests: 41/41 Passing ✅

**test_gpu_detector.py** (15 tests)
- ✅ GPU info dataclass creation
- ✅ Single GPU detection
- ✅ Multiple GPU detection
- ✅ No GPUs handling
- ✅ nvidia-smi failure handling
- ✅ Caching behavior
- ✅ Cache clearing
- ✅ Usable VRAM calculations (single/multiple)
- ✅ Profile recommendations (16GB/24GB/96GB/multi-GPU/no-GPU)
- ✅ Malformed output handling

**test_profile_manager.py** (12 tests)
- ✅ Profile dataclass operations
- ✅ Loading profiles from config
- ✅ Manual profile override
- ✅ Auto-detection integration
- ✅ Fallback behavior
- ✅ Profile retrieval by name
- ✅ Active profile management
- ✅ GPU information API
- ✅ Comprehensive profile info
- ✅ Invalid config handling
- ✅ Override with invalid profile

**test_role_selector.py** (14 tests)
- ✅ Registry loading
- ✅ Available roles enumeration
- ✅ Basic role-based selection
- ✅ VRAM constraint filtering
- ✅ Efficiency bias application
- ✅ Quality bias application
- ✅ No matches handling
- ✅ Top-N limiting
- ✅ Model lookup by ID
- ✅ Profile fitting models
- ✅ Selection explanations (eligible/rejected/no-support/not-found)

### Phase 1 Integration Tests: 13/13 Passing ✅

All existing Phase 1 tests continue to pass with no regressions:
- ✅ Proxy integration tests
- ✅ Backend client usage
- ✅ Registry validation
- ✅ Role metadata presence
- ✅ VRAM estimates

**Total Test Suite**: 54/54 passing (100%)

---

## Usage Examples

### 1. Check Active Profile
```bash
curl http://localhost:8100/v1/config/profile | jq
```

### 2. Select Models for a Task
```bash
# Get best models for keyword extraction
curl "http://localhost:8100/v1/models/select_by_role?role=keywords&top_n=3" | jq

# Get best models for description generation
curl "http://localhost:8100/v1/models/select_by_role?role=description&top_n=2" | jq
```

### 3. Override Profile Manually
```bash
# Use development profile regardless of hardware
export IMAGEWORKS_DEPLOYMENT_PROFILE=development
./scripts/start_chat_proxy.sh

# Or for generous profile
export IMAGEWORKS_DEPLOYMENT_PROFILE=generous_96gb
./scripts/start_chat_proxy.sh
```

### 4. Python Integration
```python
from imageworks.chat_proxy.profile_manager import ProfileManager
from imageworks.chat_proxy.role_selector import RoleSelector

# Initialize
pm = ProfileManager()
rs = RoleSelector()

# Get active profile
profile = pm.get_active_profile()
print(f"Using profile: {profile.name} ({profile.max_vram_mb}MB limit)")

# Select models for a role
models = rs.select_for_role("keywords", profile, top_n=3)
for model in models:
    print(f"  - {model['name']}: {model['vram_estimate_mb']}MB, "
          f"priority={model['role_priority']['keywords']}")

# Get explanation for specific model
explanation = rs.explain_selection("keywords", profile, "florence-2-large")
print(f"Status: {explanation['status']}")
print(f"Reason: {explanation['reason']}")
```

---

## Architecture Decisions

### 1. Semi-Automatic Profile Selection
**Decision**: Auto-detect with manual override capability
**Rationale**:
- Provides smart defaults for common scenarios
- Allows developers to test different profiles
- Supports edge cases (e.g., shared GPU systems)
- Environment variable override is simple and standard

### 2. Profile Definitions in pyproject.toml
**Decision**: Configuration in project config file, not separate file
**Rationale**:
- Single source of truth for project configuration
- Version-controlled alongside code
- Standard TOML parsing (Python 3.11+ native)
- Clear project structure

### 3. VRAM Headroom Strategy
**Decision**: Reserve 2GB per GPU as headroom
**Rationale**:
- Accounts for OS/driver overhead
- Prevents out-of-memory errors
- Conservative but reliable
- Adjustable via function parameter

### 4. Role-Based vs. Model-Based Selection
**Decision**: Role-first, then model selection
**Rationale**:
- Task-oriented API (what to do, not which model)
- Enables automatic optimization as models improve
- Supports A/B testing different models
- Simplifies client code

### 5. Caching Strategy
**Decision**: Cache GPU detection results, not profile selection
**Rationale**:
- GPU hardware doesn't change at runtime
- Profile selection may consider runtime factors (future feature)
- Clear cache invalidation path (`clear_cache()`)
- Performance-sensitive operation (nvidia-smi query)

---

## Future Enhancements (Out of Scope for Phase 2)

1. **Runtime VRAM Monitoring**: Track actual VRAM usage and adjust profiles dynamically
2. **Model Load/Unload Orchestration**: Implement the strategies defined in profiles
3. **Tensor Parallel Support**: Leverage multi_gpu_2x16gb profile for large models
4. **Profile Metrics**: Track which models run in which profiles and performance
5. **Custom Profiles**: Allow users to define profiles via config files
6. **Cloud GPU Detection**: Support cloud providers (AWS, GCP, Azure) GPU detection
7. **Profile Migration**: Automatically switch profiles when hardware changes
8. **Quantization Conversion**: Auto-convert models to preferred quantizations

---

## Known Limitations

1. **NVIDIA Only**: Currently only supports NVIDIA GPUs via `nvidia-smi`
2. **Static Detection**: GPU detection happens at startup only
3. **No AMD/Intel**: No support for AMD ROCm or Intel Arc GPUs
4. **Manual Registry**: Role metadata must be manually added to model registry
5. **No Load Balancing**: Multi-GPU profiles don't implement actual distribution yet
6. **Simple VRAM Estimates**: Uses fixed estimates, not runtime measurements

---

## Migration Notes

### From Phase 1 to Phase 2

**No Breaking Changes**: Phase 2 is fully backward compatible with Phase 1.

**New Features Available**:
- Two new API endpoints (`/v1/config/profile`, `/v1/models/select_by_role`)
- Automatic profile selection on startup
- Manual profile override via environment variable

**No Action Required**: Existing deployments continue to work without changes.

**Optional Enhancements**:
1. Add role_priority metadata to model registry for better model selection
2. Set `IMAGEWORKS_DEPLOYMENT_PROFILE` env var if manual control desired
3. Update applications to use role-based model selection API

---

## Conclusion

Phase 2 successfully implements intelligent deployment profile management with GPU auto-detection and role-based model selection. The implementation is:

- ✅ **Complete**: All 10 planned tasks finished
- ✅ **Tested**: 41 new unit tests, all passing
- ✅ **Documented**: Comprehensive docs and examples
- ✅ **Backward Compatible**: No breaking changes to Phase 1
- ✅ **Production Ready**: Robust error handling and logging

The system is ready for deployment and will automatically adapt to available hardware, making it easy to run on different GPU configurations from 16GB to 96GB.

---

**Next Steps**: Phase 3 will focus on runtime model orchestration, implementing the load/unload strategies defined in profiles, and adding actual multi-model concurrent serving capabilities.
