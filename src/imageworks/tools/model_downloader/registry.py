"""
Model registry for tracking downloaded models across directories.

Maintains a unified registry of all downloaded models, their locations, formats,
and metadata to avoid duplicates and enable smart management.
"""

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict, field
from .config import get_config


@dataclass
class ModelEntry:
    """Registry entry for a downloaded model."""

    model_name: str  # e.g., "microsoft/DialoGPT-medium"
    format_type: str  # e.g., "safetensors", "gguf", "awq"
    location: str  # "linux_wsl" or "windows_lmstudio"
    path: str  # Full path to model directory
    size_bytes: int  # Total size on disk
    files: List[str]  # List of model files
    downloaded_at: str  # ISO timestamp
    last_accessed: Optional[str] = None  # Last access timestamp
    metadata: Dict[str, Any] = field(default_factory=dict)  # Extra metadata
    checksum: Optional[str] = None  # Directory content hash

    @property
    def path_obj(self) -> Path:
        """Get path as Path object."""
        return Path(self.path).expanduser()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelEntry":
        """Create from dictionary."""
        return cls(**data)


class ModelRegistry:
    """Registry for tracking downloaded models."""

    def __init__(self, registry_path: Optional[Path] = None):
        if registry_path is None:
            config = get_config()
            self.registry_path = config.registry_path / "models.json"
        else:
            self.registry_path = registry_path

        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: Dict[str, ModelEntry] = {}
        self._load_registry()

    def _load_registry(self) -> None:
        """Load registry from disk."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, "r") as f:
                    data = json.load(f)

                for key, entry_data in data.items():
                    self._entries[key] = ModelEntry.from_dict(entry_data)

            except (json.JSONDecodeError, KeyError, TypeError) as e:
                print(f"Warning: Could not load registry {self.registry_path}: {e}")
                self._entries = {}
        else:
            self._entries = {}

    def _save_registry(self) -> None:
        """Save registry to disk."""
        try:
            data = {key: entry.to_dict() for key, entry in self._entries.items()}

            with open(self.registry_path, "w") as f:
                json.dump(data, f, indent=2, sort_keys=True)

        except (OSError, json.JSONEncodeError) as e:
            print(f"Warning: Could not save registry {self.registry_path}: {e}")

    def _generate_key(self, model_name: str, format_type: str, location: str) -> str:
        """Generate unique key for registry entry."""
        # Normalize model name
        normalized_name = model_name.lower().replace("/", "_").replace("-", "_")
        return f"{normalized_name}_{format_type}_{location}"

    def _calculate_directory_checksum(self, directory: Path) -> str:
        """Calculate checksum of directory contents."""
        if not directory.exists():
            return ""

        hasher = hashlib.sha256()

        # Sort files for consistent hash
        files = sorted(directory.rglob("*"))

        for file_path in files:
            if file_path.is_file():
                # Include relative path and size in hash
                rel_path = file_path.relative_to(directory)
                hasher.update(str(rel_path).encode())
                hasher.update(str(file_path.stat().st_size).encode())

        return hasher.hexdigest()[:16]  # First 16 chars

    def add_model(
        self,
        model_name: str,
        format_type: str,
        location: str,
        path: Union[str, Path],
        size_bytes: Optional[int] = None,
        files: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add a model to the registry."""

        path_obj = Path(path).expanduser()

        # Auto-discover files if not provided
        if files is None:
            files = []
            if path_obj.exists():
                files = [
                    str(f.relative_to(path_obj))
                    for f in path_obj.rglob("*")
                    if f.is_file()
                ]

        # Auto-calculate size if not provided
        if size_bytes is None:
            size_bytes = 0
            if path_obj.exists():
                size_bytes = sum(
                    f.stat().st_size for f in path_obj.rglob("*") if f.is_file()
                )

        # Generate checksum
        checksum = self._calculate_directory_checksum(path_obj)

        # Create entry
        entry = ModelEntry(
            model_name=model_name,
            format_type=format_type,
            location=location,
            path=str(path),
            size_bytes=size_bytes,
            files=files or [],
            downloaded_at=datetime.now(timezone.utc).isoformat(),
            metadata=metadata or {},
            checksum=checksum,
        )

        key = self._generate_key(model_name, format_type, location)
        self._entries[key] = entry
        self._save_registry()

        return key

    def find_model(
        self,
        model_name: str,
        format_type: Optional[str] = None,
        location: Optional[str] = None,
    ) -> List[ModelEntry]:
        """Find models matching criteria."""
        matches = []

        for entry in self._entries.values():
            if entry.model_name.lower() != model_name.lower():
                continue

            if format_type and entry.format_type != format_type:
                continue

            if location and entry.location != location:
                continue

            # Lightweight defensive check: does the model directory still exist?
            if not entry.path_obj.exists():
                print(
                    f"⚠️  Warning: Model {entry.model_name} registry entry found but directory missing: {entry.path}"
                )
                continue

            matches.append(entry)

        return matches

    def get_all_models(self) -> List[ModelEntry]:
        """Get all registered models."""
        # Lightweight defensive check: filter out models with missing directories
        valid_models = []
        for entry in self._entries.values():
            if entry.path_obj.exists():
                valid_models.append(entry)
            else:
                # Note: We don't automatically remove from registry to avoid data loss
                # User can manually clean up or use a future 'verify' command
                entry.metadata = entry.metadata or {}
                entry.metadata["directory_missing"] = True
                valid_models.append(entry)  # Include but mark as problematic

        return valid_models

    def remove_model(self, model_name: str, format_type: str, location: str) -> bool:
        """Remove a model from registry."""
        key = self._generate_key(model_name, format_type, location)

        if key in self._entries:
            del self._entries[key]
            self._save_registry()
            return True

        return False

    def update_access_time(
        self, model_name: str, format_type: str, location: str
    ) -> None:
        """Update last access time for a model."""
        key = self._generate_key(model_name, format_type, location)

        if key in self._entries:
            self._entries[key].last_accessed = datetime.now(timezone.utc).isoformat()
            self._save_registry()

    def verify_model_integrity(
        self, model_name: str, format_type: str, location: str
    ) -> bool:
        """Verify that a registered model still exists and hasn't changed."""
        entry = self.find_model(model_name, format_type, location)

        if not entry:
            return False

        entry = entry[0]  # Take first match
        path_obj = Path(entry.path).expanduser()

        if not path_obj.exists():
            return False

        # Check if checksum matches
        current_checksum = self._calculate_directory_checksum(path_obj)
        return current_checksum == entry.checksum

    def cleanup_missing_models(self) -> List[str]:
        """Remove registry entries for models that no longer exist."""
        removed_keys = []

        for key, entry in list(self._entries.items()):
            if not Path(entry.path).expanduser().exists():
                del self._entries[key]
                removed_keys.append(key)

        if removed_keys:
            self._save_registry()

        return removed_keys

    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics."""
        total_models = len(self._entries)
        total_size = sum(entry.size_bytes for entry in self._entries.values())

        # Group by format
        by_format = {}
        for entry in self._entries.values():
            format_type = entry.format_type
            if format_type not in by_format:
                by_format[format_type] = {"count": 0, "size": 0}
            by_format[format_type]["count"] += 1
            by_format[format_type]["size"] += entry.size_bytes

        # Group by location
        by_location = {}
        for entry in self._entries.values():
            location = entry.location
            if location not in by_location:
                by_location[location] = {"count": 0, "size": 0}
            by_location[location]["count"] += 1
            by_location[location]["size"] += entry.size_bytes

        return {
            "total_models": total_models,
            "total_size_bytes": total_size,
            "by_format": by_format,
            "by_location": by_location,
        }

    def export_model_list(self, format: str = "json") -> str:
        """Export model list in various formats."""
        models_data = [entry.to_dict() for entry in self._entries.values()]

        if format.lower() == "json":
            return json.dumps(models_data, indent=2, sort_keys=True)
        elif format.lower() == "csv":
            import csv
            import io

            output = io.StringIO()
            if models_data:
                writer = csv.DictWriter(output, fieldnames=models_data[0].keys())
                writer.writeheader()
                writer.writerows(models_data)

            return output.getvalue()
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def search_models(
        self, query: str, search_fields: List[str] = None
    ) -> List[ModelEntry]:
        """Search models by query string."""
        if search_fields is None:
            search_fields = ["model_name", "format_type", "path"]

        query_lower = query.lower()
        matches = []

        for entry in self._entries.values():
            for search_field in search_fields:
                field_value = str(getattr(entry, search_field, "")).lower()
                if query_lower in field_value:
                    matches.append(entry)
                    break  # Don't add duplicate matches

        return matches


# Global registry instance
_registry_instance: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    """Get the global registry instance."""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ModelRegistry()
    return _registry_instance


def find_existing_model(model_name: str) -> List[ModelEntry]:
    """Convenience function to find existing models."""
    registry = get_registry()
    return registry.find_model(model_name)
