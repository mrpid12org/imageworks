# Deployment Profiles: Auto-Detection & Configuration

**Status**: Design Specification (Phase 2)
**Date**: October 26, 2025
**Priority**: High

---

## Overview

Deployment profiles automatically configure ImageWorks behavior based on available GPU hardware. The system detects VRAM capacity and adjusts model loading strategies, quantization preferences, and concurrent serving capabilities.

---

## Profile Selection Strategy

### **Automatic Detection (Recommended)**

The system auto-detects GPU configuration at startup:

```python
# Startup detection flow
1. Query nvidia-smi for GPU info (count, VRAM per GPU, model names)
2. Calculate usable VRAM (total VRAM - 2GB system headroom per GPU)
3. Match to predefined profiles OR create dynamic profile
4. Cache detection result for session duration
```

### **Manual Override (Optional)**

Users can explicitly set a profile via environment variable:

```bash
# Force specific profile
export IMAGEWORKS_DEPLOYMENT_PROFILE=constrained_16gb

# Or via CLI flag
uv run uvicorn imageworks.chat_proxy.app:app --env-profile constrained_16gb
```

---

## Predefined Profiles

### **1. `constrained_16gb` (RTX 4080, RTX 4060 Ti 16GB)**

**Hardware**: Single 16GB GPU
**Strategy**: Load/unload models on demand, single model active at a time

```toml
[tool.imageworks.deployment_profiles.constrained_16gb]
description = "RTX 4080 16GB - single model, load/unload strategy"
max_vram_mb = 14000  # 16384 - 2048 headroom
max_concurrent_models = 1
preferred_quantizations = ["q4_k_m", "fp8", "awq", "q5_k_m", "q8_0"]
strategy = "load_unload"
autostart_timeout_seconds = 120
grace_period_seconds = 300  # Keep model loaded 5min after last request

# Model priorities optimized for constrained VRAM
model_selection_bias = "vram_efficient"
```

**Auto-Detection**:
```python
if total_vram_mb >= 14000 and total_vram_mb < 20000:
    if gpu_count == 1:
        profile = "constrained_16gb"
```

**Example Model Fits**:
- ✅ qwen3-vl-8b-instruct_(FP8): 10.6GB
- ✅ qwen25vl_8.3b_(Q4_K_M): 6.4GB
- ✅ florence-2-large_(FP16): 3.5GB
- ✅ Dual-model (swap): qwen2.5vl (6.4GB) + florence-2 (3.5GB) = 9.9GB ✅

---

### **2. `balanced_24gb` (RTX 4090, RTX A5000)**

**Hardware**: Single 24GB GPU
**Strategy**: Dual-model concurrent serving, smart VRAM allocation

```toml
[tool.imageworks.deployment_profiles.balanced_24gb]
description = "RTX 4090 24GB - dual model concurrent serving"
max_vram_mb = 22000  # 24576 - 2048 headroom
max_concurrent_models = 2
preferred_quantizations = ["fp8", "awq", "q4_k_m", "q5_k_m"]
strategy = "concurrent_limited"
autostart_timeout_seconds = 90
grace_period_seconds = 600  # 10min

model_selection_bias = "balanced"
```

**Auto-Detection**:
```python
if total_vram_mb >= 20000 and total_vram_mb < 35000:
    if gpu_count == 1:
        profile = "balanced_24gb"
```

**Example Model Combinations**:
- ✅ qwen3-vl-fp8 (10.6GB) + florence-2 (3.5GB) = 14.1GB
- ✅ qwen3-vl-fp8 (10.6GB) + qwen2.5vl (6.4GB) = 17GB
- ✅ Triple-model (swap): florence-2 + qwen2.5vl concurrently, swap qwen3-vl as needed

---

### **3. `multi_gpu_2x16gb` (2x RTX 4080)**

**Hardware**: 2x 16GB GPUs (32GB total)
**Strategy**: Model sharding via tensor-parallel or separate models per GPU

```toml
[tool.imageworks.deployment_profiles.multi_gpu_2x16gb]
description = "2x RTX 4080 16GB - tensor parallel or distributed models"
max_vram_mb = 28000  # (16384 * 2) - 4096 headroom
max_concurrent_models = 3
gpu_count = 2
preferred_quantizations = ["fp8", "awq", "bf16", "q5_k_m"]
strategy = "tensor_parallel_or_distributed"
autostart_timeout_seconds = 150
grace_period_seconds = 600

# Prefer tensor-parallel for large models, distributed for small
tensor_parallel_threshold_mb = 12000
model_selection_bias = "quality"
```

**Auto-Detection**:
```python
if gpu_count == 2:
    if all(gpu.vram_mb >= 14000 and gpu.vram_mb < 20000 for gpu in gpus):
        profile = "multi_gpu_2x16gb"
```

**Example Model Deployments**:
- ✅ qwen3-vl-fp8 (tensor-parallel 2-way) across both GPUs
- ✅ GPU0: qwen3-vl-fp8 (10.6GB), GPU1: florence-2 (3.5GB) + qwen2.5vl (6.4GB)
- ✅ Larger FP16 models with tensor-parallel sharding

---

### **4. `generous_96gb` (RTX 6000 Pro 96GB, Future Upgrade)**

**Hardware**: Single 96GB GPU
**Strategy**: Multi-model concurrent serving, keep 4+ models loaded

```toml
[tool.imageworks.deployment_profiles.generous_96gb]
description = "RTX 6000 Pro 96GB - multi-model concurrent serving"
max_vram_mb = 90000  # 98304 - 8192 headroom
max_concurrent_models = 5
preferred_quantizations = ["fp8", "awq", "bf16", "fp16"]
strategy = "concurrent_full"
autostart_timeout_seconds = 60
grace_period_seconds = 3600  # Keep models hot for 1 hour

model_selection_bias = "quality"
```

**Auto-Detection**:
```python
if total_vram_mb >= 85000:
    if gpu_count == 1:
        profile = "generous_96gb"
```

**Example Model Groups**:
```python
# Vision processing group (auto-loaded at startup)
models = [
    "qwen3-vl-8b-instruct_(FP8)",      # 10.6GB - description/narration
    "florence-2-large_(FP16)",          # 3.5GB  - keywords/object detection
    "qwen25vl_8.3b_(Q4_K_M)",          # 6.4GB  - caption fallback
    "siglip-large-patch16-384_(FP32)", # 2.6GB  - embeddings
    "gpt-oss-20b_(MXFP4)",             # 13.8GB - reasoning (optional)
]
# Total: ~37GB, leaves 53GB for KV cache and additional models
```

---

### **5. `development` (Low VRAM / Testing)**

**Hardware**: <12GB GPU (RTX 3060, GTX 1080 Ti) or CPU-only
**Strategy**: Minimal footprint, Q4 quantization only

```toml
[tool.imageworks.deployment_profiles.development]
description = "Development/testing - minimal footprint"
max_vram_mb = 10000
max_concurrent_models = 1
preferred_quantizations = ["q4_k_m", "q5_k_m", "q8_0"]
strategy = "load_unload"
autostart_timeout_seconds = 180
grace_period_seconds = 120

model_selection_bias = "vram_efficient"
```

**Auto-Detection**:
```python
if total_vram_mb < 12000:
    profile = "development"
```

---

## Auto-Detection Implementation

### **GPU Detection Module** (`src/imageworks/libs/hardware/gpu_detector.py`)

```python
import subprocess
import json
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class GPUInfo:
    """Single GPU information."""
    index: int
    name: str
    vram_total_mb: int
    vram_free_mb: int
    compute_capability: tuple[int, int]
    uuid: str

class GPUDetector:
    """Detect and analyze available GPU hardware."""

    def __init__(self):
        self._cached_info: Optional[List[GPUInfo]] = None

    def detect_gpus(self) -> List[GPUInfo]:
        """Query nvidia-smi for GPU information."""
        if self._cached_info is not None:
            return self._cached_info

        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total,memory.free,compute_cap,uuid",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            gpus = []
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                index, name, vram_total, vram_free, compute_cap, uuid = parts

                major, minor = map(int, compute_cap.split("."))

                gpus.append(GPUInfo(
                    index=int(index),
                    name=name,
                    vram_total_mb=int(vram_total),
                    vram_free_mb=int(vram_free),
                    compute_capability=(major, minor),
                    uuid=uuid,
                ))

            self._cached_info = gpus
            return gpus

        except (subprocess.CalledProcessError, FileNotFoundError):
            # No NVIDIA GPU or nvidia-smi not available
            return []

    def get_usable_vram_mb(self, headroom_per_gpu_mb: int = 2048) -> int:
        """Calculate total usable VRAM across all GPUs."""
        gpus = self.detect_gpus()
        if not gpus:
            return 0

        total = sum(gpu.vram_total_mb for gpu in gpus)
        return total - (headroom_per_gpu_mb * len(gpus))

    def recommend_profile(self) -> str:
        """Auto-recommend deployment profile based on hardware."""
        gpus = self.detect_gpus()

        if not gpus:
            return "development"  # CPU-only or no GPU

        gpu_count = len(gpus)
        total_vram = sum(gpu.vram_total_mb for gpu in gpus)
        usable_vram = self.get_usable_vram_mb()

        # Single GPU profiles
        if gpu_count == 1:
            if usable_vram >= 85000:
                return "generous_96gb"
            elif usable_vram >= 20000:
                return "balanced_24gb"
            elif usable_vram >= 14000:
                return "constrained_16gb"
            else:
                return "development"

        # Multi-GPU profiles
        elif gpu_count == 2:
            # Check if GPUs are similar (within 20% VRAM)
            vram_variance = max(gpu.vram_total_mb for gpu in gpus) / min(gpu.vram_total_mb for gpu in gpus)

            if vram_variance < 1.2:  # Similar GPUs
                avg_vram = total_vram / gpu_count
                if avg_vram >= 14000:
                    return "multi_gpu_2x16gb"
                else:
                    return "development"
            else:
                # Mismatched GPUs, use primary GPU rules
                primary_vram = gpus[0].vram_total_mb - 2048
                if primary_vram >= 20000:
                    return "balanced_24gb"
                elif primary_vram >= 14000:
                    return "constrained_16gb"
                else:
                    return "development"

        else:
            # 3+ GPUs: use total VRAM approach
            if usable_vram >= 60000:
                return "generous_96gb"  # Treat as high-capacity
            elif usable_vram >= 40000:
                return "balanced_24gb"  # Treat as balanced
            else:
                return "constrained_16gb"
```

### **Profile Manager** (`src/imageworks/chat_proxy/profile_manager.py`)

```python
import os
from dataclasses import dataclass
from typing import List, Optional
import toml
from pathlib import Path

from imageworks.libs.hardware.gpu_detector import GPUDetector

@dataclass
class DeploymentProfile:
    """Deployment profile configuration."""
    name: str
    description: str
    max_vram_mb: int
    max_concurrent_models: int
    preferred_quantizations: List[str]
    strategy: str
    autostart_timeout_seconds: int
    grace_period_seconds: int
    model_selection_bias: str

class ProfileManager:
    """Manage deployment profile selection and loading."""

    def __init__(self, config_path: Path = Path("pyproject.toml")):
        self.config_path = config_path
        self.gpu_detector = GPUDetector()
        self._profiles: dict[str, DeploymentProfile] = {}
        self._active_profile: Optional[DeploymentProfile] = None

        self._load_profiles()
        self._select_active_profile()

    def _load_profiles(self):
        """Load profile definitions from config."""
        with open(self.config_path, "rb") as f:
            config = toml.load(f)

        profiles_config = config.get("tool", {}).get("imageworks", {}).get("deployment_profiles", {})

        for profile_name, profile_data in profiles_config.items():
            self._profiles[profile_name] = DeploymentProfile(
                name=profile_name,
                description=profile_data.get("description", ""),
                max_vram_mb=profile_data.get("max_vram_mb", 10000),
                max_concurrent_models=profile_data.get("max_concurrent_models", 1),
                preferred_quantizations=profile_data.get("preferred_quantizations", ["q4_k_m"]),
                strategy=profile_data.get("strategy", "load_unload"),
                autostart_timeout_seconds=profile_data.get("autostart_timeout_seconds", 120),
                grace_period_seconds=profile_data.get("grace_period_seconds", 300),
                model_selection_bias=profile_data.get("model_selection_bias", "balanced"),
            )

    def _select_active_profile(self):
        """Select active profile: manual override > auto-detect > default."""
        # 1. Check environment variable override
        manual_profile = os.getenv("IMAGEWORKS_DEPLOYMENT_PROFILE")
        if manual_profile and manual_profile in self._profiles:
            self._active_profile = self._profiles[manual_profile]
            return

        # 2. Auto-detect based on hardware
        recommended = self.gpu_detector.recommend_profile()
        if recommended in self._profiles:
            self._active_profile = self._profiles[recommended]
            return

        # 3. Fallback to development profile
        self._active_profile = self._profiles.get("development")

    @property
    def active_profile(self) -> DeploymentProfile:
        """Get currently active deployment profile."""
        if self._active_profile is None:
            raise RuntimeError("No active profile selected")
        return self._active_profile

    def get_profile(self, name: str) -> Optional[DeploymentProfile]:
        """Get profile by name."""
        return self._profiles.get(name)

    def list_profiles(self) -> List[str]:
        """List all available profile names."""
        return list(self._profiles.keys())
```

---

## Usage Examples

### **Automatic (Recommended)**

```bash
# System auto-detects GPU and selects profile
uv run uvicorn imageworks.chat_proxy.app:app --port 8100

# Logs show:
# [INFO] Detected: 1x RTX 4080 (16GB VRAM)
# [INFO] Auto-selected profile: constrained_16gb
# [INFO] Strategy: load_unload, max_concurrent_models=1
```

### **Manual Override**

```bash
# Force specific profile (testing, development, or preference)
export IMAGEWORKS_DEPLOYMENT_PROFILE=balanced_24gb
uv run uvicorn imageworks.chat_proxy.app:app --port 8100

# Logs show:
# [INFO] Detected: 1x RTX 4080 (16GB VRAM)
# [INFO] Manual override: balanced_24gb (note: may exceed available VRAM)
```

### **Query Active Profile via API**

```bash
curl http://localhost:8100/v1/config/profile

# Response:
{
  "profile": "constrained_16gb",
  "description": "RTX 4080 16GB - single model, load/unload strategy",
  "max_vram_mb": 14000,
  "max_concurrent_models": 1,
  "strategy": "load_unload",
  "detected_gpus": [
    {
      "index": 0,
      "name": "NVIDIA GeForce RTX 4080",
      "vram_total_mb": 16384,
      "vram_free_mb": 15200
    }
  ]
}
```

---

## Profile-Aware Model Selection

Role selector uses active profile to filter model candidates:

```python
class RoleSelector:
    def __init__(self, registry: ModelRegistry, profile_manager: ProfileManager):
        self.registry = registry
        self.profile = profile_manager.active_profile

    def select_model_for_role(self, role: str) -> Optional[ModelInfo]:
        """Select model considering profile constraints."""
        candidates = self.registry.get_models_by_role(role)

        # Filter by VRAM constraints
        candidates = [
            m for m in candidates
            if m.vram_estimate_mb <= self.profile.max_vram_mb
        ]

        # Sort by role priority, then quantization preference
        candidates = sorted(
            candidates,
            key=lambda m: (
                m.role_priority.get(role, 0),  # Higher priority first
                self._quantization_score(m.quantization),  # Preferred quant
            ),
            reverse=True,
        )

        return candidates[0] if candidates else None

    def _quantization_score(self, quant: str) -> int:
        """Score quantization based on profile preference."""
        try:
            return self.profile.preferred_quantizations.index(quant.lower())
        except ValueError:
            return 999  # Unknown quantization = low priority
```

---

## Benefits of Auto-Detection

1. **Zero Configuration**: Works out-of-the-box on any hardware
2. **Optimal Performance**: Automatically adjusts to available resources
3. **Prevents OOM**: Respects VRAM limits, avoids crashes
4. **Portable**: Same codebase works on 16GB laptop and 96GB workstation
5. **Manual Override**: Power users can force specific profiles for testing
6. **Future-Proof**: Adding new GPUs only requires profile definitions

---

## Phase 2 Implementation Checklist

- [ ] Implement `GPUDetector` class with nvidia-smi integration
- [ ] Implement `ProfileManager` with auto-detection logic
- [ ] Add profile definitions to `pyproject.toml`
- [ ] Integrate profile manager into chat proxy startup
- [ ] Add `/v1/config/profile` API endpoint
- [ ] Update `RoleSelector` to use active profile constraints
- [ ] Add profile override environment variable support
- [ ] Create profile detection tests
- [ ] Document profile selection behavior
- [ ] Add logging for profile detection and selection

---

## Testing Strategy

### **Unit Tests**

```python
def test_gpu_detector_single_gpu():
    """Test detection of single GPU."""
    detector = GPUDetector()
    # Mock nvidia-smi output
    assert detector.recommend_profile() == "constrained_16gb"

def test_profile_vram_filtering():
    """Test that profile max_vram_mb filters models."""
    profile = DeploymentProfile(
        name="test",
        max_vram_mb=8000,
        # ... other fields
    )
    selector = RoleSelector(registry, profile)

    # Should exclude qwen3-vl (10.6GB > 8GB limit)
    model = selector.select_model_for_role("keywords")
    assert model.vram_estimate_mb <= 8000
```

### **Integration Tests**

```python
def test_auto_profile_selection():
    """Test automatic profile selection on startup."""
    # Start proxy
    app = create_app()

    # Verify profile was auto-selected
    response = client.get("/v1/config/profile")
    assert response.json()["profile"] in VALID_PROFILES
```

---

**Next**: Phase 2 implementation begins after Phase 1 completion.
