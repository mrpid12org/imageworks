"""
Configuration management for model downloader.

Handles paths, formats, user preferences, and integration with imageworks settings.
"""

import os
from pathlib import Path
from typing import List, Optional
import tomllib
from dataclasses import dataclass, field


@dataclass
class DirectoryConfig:
    """Configuration for a model storage directory."""

    root: Path
    formats: List[str]
    use_cases: List[str]
    publisher_structure: bool = False


@dataclass
class DownloaderConfig:
    """Main configuration for the model downloader."""

    # Directory configurations
    linux_wsl: DirectoryConfig = field(
        default_factory=lambda: DirectoryConfig(
            root=Path("~/ai-models").expanduser(),
            formats=["safetensors", "pytorch", "awq", "gptq", "onnx"],
            use_cases=["vllm", "transformers", "training"],
        )
    )

    windows_lmstudio: DirectoryConfig = field(
        default_factory=lambda: DirectoryConfig(
            root=Path("/mnt/d/ai stuff/models/llm models"),
            formats=["gguf"],
            use_cases=["lmstudio", "llama.cpp"],
            publisher_structure=True,
        )
    )

    # Registry and cache paths
    registry_path: Path = field(
        default_factory=lambda: Path("~/ai-models/registry").expanduser()
    )
    cache_path: Path = field(
        default_factory=lambda: Path("~/ai-models/cache").expanduser()
    )

    # Download settings
    max_connections_per_server: int = 16
    max_concurrent_downloads: int = 8
    enable_resume: bool = True
    include_optional_files: bool = False

    # Format preferences
    preferred_formats: List[str] = field(
        default_factory=lambda: ["awq", "safetensors", "gguf"]
    )

    @classmethod
    def from_pyproject(
        cls, pyproject_path: Optional[Path] = None
    ) -> "DownloaderConfig":
        """Load configuration from pyproject.toml file."""
        if pyproject_path is None:
            # Default to imageworks pyproject.toml
            pyproject_path = Path(__file__).parents[4] / "pyproject.toml"

        config = cls()

        try:
            with open(pyproject_path, "rb") as f:
                data = tomllib.load(f)

            # Load model downloader specific settings
            downloader_config = (
                data.get("tool", {}).get("imageworks", {}).get("model-downloader", {})
            )

            if "linux_wsl_root" in downloader_config:
                config.linux_wsl.root = Path(
                    downloader_config["linux_wsl_root"]
                ).expanduser()

            if "windows_lmstudio_root" in downloader_config:
                config.windows_lmstudio.root = Path(
                    downloader_config["windows_lmstudio_root"]
                )

            if "max_connections" in downloader_config:
                config.max_connections_per_server = downloader_config["max_connections"]

            if "preferred_formats" in downloader_config:
                config.preferred_formats = downloader_config["preferred_formats"]

        except Exception as e:
            print(f"Warning: Could not load config from {pyproject_path}: {e}")

        return config

    @classmethod
    def from_env(cls) -> "DownloaderConfig":
        """Load configuration from environment variables."""
        config = cls()

        if "IMAGEWORKS_MODEL_ROOT" in os.environ:
            # IMAGEWORKS_MODEL_ROOT should point to the weights directory
            # So the root should be its parent (e.g., if IMAGEWORKS_MODEL_ROOT=/path/to/weights, root=/path/to)
            model_root = Path(os.environ["IMAGEWORKS_MODEL_ROOT"])
            if model_root.name == "weights":
                config.linux_wsl.root = model_root.parent
            else:
                # If IMAGEWORKS_MODEL_ROOT doesn't end in 'weights', assume it's the root
                config.linux_wsl.root = model_root

        if "IMAGEWORKS_LMSTUDIO_ROOT" in os.environ:
            config.windows_lmstudio.root = Path(os.environ["IMAGEWORKS_LMSTUDIO_ROOT"])

        return config

    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            self.linux_wsl.root,
            self.linux_wsl.root / "weights",
            self.registry_path,
            self.cache_path,
        ]

        # Only create LM Studio directory if it's accessible
        if (
            self.windows_lmstudio.root.exists()
            or self.windows_lmstudio.root.parent.exists()
        ):
            directories.append(self.windows_lmstudio.root)

        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except (OSError, PermissionError) as e:
                print(f"Warning: Could not create directory {directory}: {e}")

    def get_target_directory(
        self, format_type: str, model_name: str, publisher: Optional[str] = None
    ) -> Path:
        """Determine target directory for a model based on format."""
        if format_type == "gguf":
            if self.windows_lmstudio.publisher_structure and publisher:
                return self.windows_lmstudio.root / publisher / model_name
            else:
                return self.windows_lmstudio.root / model_name
        else:
            return self.linux_wsl.root / "weights" / model_name

    def get_compatible_backends(self, format_type: str) -> List[str]:
        """Get compatible backends for a given format."""
        format_backends = {
            "gguf": ["lmstudio", "llama.cpp", "ollama"],
            "awq": ["vllm", "autoawq", "transformers"],
            "gptq": ["vllm", "autogptq", "transformers"],
            "safetensors": ["vllm", "transformers", "diffusers"],
            "pytorch": ["transformers", "diffusers"],
            "onnx": ["onnxruntime", "transformers"],
        }
        return format_backends.get(format_type, ["unknown"])


# Global config instance
_config_instance: Optional[DownloaderConfig] = None


def get_config() -> DownloaderConfig:
    """Get the global configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = DownloaderConfig.from_pyproject()
        _config_instance.ensure_directories()
    return _config_instance


def set_config(config: DownloaderConfig) -> None:
    """Set the global configuration instance."""
    global _config_instance
    _config_instance = config
