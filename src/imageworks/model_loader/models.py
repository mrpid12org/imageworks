"""Dataclasses defining the deterministic model registry schema (Phase 1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class BackendConfig:
    port: int
    model_path: str
    extra_args: List[str] = field(default_factory=list)


@dataclass
class ArtifactFile:
    path: str
    sha256: str


@dataclass
class Artifacts:
    aggregate_sha256: str
    files: List[ArtifactFile] = field(default_factory=list)


@dataclass
class ChatTemplate:
    source: str  # embedded | external
    path: Optional[str]
    sha256: Optional[str]


@dataclass
class VersionLock:
    locked: bool
    expected_aggregate_sha256: Optional[str]
    last_verified: Optional[str]


@dataclass
class PerformanceLastSample:
    ttft_ms: Optional[float]
    tokens_generated: Optional[int]
    duration_ms: Optional[float]


@dataclass
class PerformanceSummary:
    rolling_samples: int
    ttft_ms_avg: Optional[float]
    throughput_toks_per_s_avg: Optional[float]
    last_sample: Optional[PerformanceLastSample]


@dataclass
class VisionProbe:
    vision_ok: bool
    timestamp: str
    probe_version: str
    latency_ms: Optional[float]
    notes: Optional[str]


@dataclass
class Probes:
    vision: Optional[VisionProbe]


@dataclass
class RegistryEntry:
    name: str
    display_name: Optional[str]
    backend: str  # vllm | ollama | gguf | lmdeploy
    backend_config: BackendConfig
    capabilities: Dict[str, bool]
    artifacts: Artifacts
    chat_template: ChatTemplate
    version_lock: VersionLock
    performance: PerformanceSummary
    probes: Probes
    profiles_placeholder: Optional[Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    served_model_id: Optional[str] = (
        None  # actual identifier required by backend (e.g. Ollama tag with colon)
    )
    model_aliases: List[str] = field(
        default_factory=list
    )  # alternative names resolvable for preflight or selection
    roles: List[str] = field(
        default_factory=list
    )  # functional roles (caption, keywords, description, narration, embedding)
    license: Optional[str] = None
    source: Optional[Dict[str, Any]] = (
        None  # raw downloader-like source info (legacy compatibility)
    )
    deprecated: bool = False
    # --- New variant classification fields ---
    family: Optional[str] = None  # canonical base family e.g. qwen2.5-vl-7b
    source_provider: Optional[str] = None  # hf | ollama | other
    quantization: Optional[str] = None  # AWQ, Q4_K_M, FP16, BF16, etc.
    backend_alternatives: List[str] = field(
        default_factory=list
    )  # other viable backends
    role_priority: Dict[str, int] = field(
        default_factory=dict
    )  # per-role ordering (lower=preferred)
    # New unified download tracking (merged from model_downloader.registry.ModelEntry)
    download_format: Optional[str] = None  # e.g. safetensors | gguf | awq
    download_location: Optional[str] = (
        None  # logical location label (linux_wsl, windows_lmstudio, etc.)
    )
    download_path: Optional[str] = None  # absolute or ~ path to directory
    download_size_bytes: Optional[int] = None  # cumulative size
    download_files: List[str] = field(default_factory=list)  # relative file paths
    download_directory_checksum: Optional[str] = (
        None  # directory content hash (first 16 chars of sha256)
    )
    downloaded_at: Optional[str] = None  # ISO timestamp when first registered
    last_accessed: Optional[str] = (
        None  # ISO timestamp when last used (updated by downloader)
    )

    def require_capabilities(self, required: List[str]) -> None:
        missing = [c for c in required if not self.capabilities.get(c)]
        if missing:
            raise ValueError(
                f"Model '{self.name}' missing required capabilities: {', '.join(missing)}"
            )


@dataclass
class SelectedModel:
    logical_name: str
    endpoint_url: str
    internal_model_id: str
    backend: str
    capabilities: Dict[str, bool]


__all__ = [
    "BackendConfig",
    "ArtifactFile",
    "Artifacts",
    "ChatTemplate",
    "VersionLock",
    "PerformanceLastSample",
    "PerformanceSummary",
    "VisionProbe",
    "Probes",
    "RegistryEntry",
    "SelectedModel",
]
