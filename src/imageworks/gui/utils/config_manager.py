"""Configuration manager for reading/writing pyproject.toml defaults."""

import sys
from pathlib import Path
from typing import Dict, Any

# Use tomllib (Python 3.11+) or tomli (older versions)
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

# For writing TOML
try:
    import tomli_w
except ImportError:
    tomli_w = None


def load_pyproject_config(project_root: Path) -> Dict[str, Any]:
    """
    Load configuration from pyproject.toml.

    Args:
        project_root: Path to project root directory

    Returns:
        Dict with parsed TOML configuration
    """
    pyproject_path = project_root / "pyproject.toml"

    if not pyproject_path.exists():
        return {}

    with open(pyproject_path, "rb") as f:
        return tomllib.load(f)


def save_pyproject_config(project_root: Path, config: Dict[str, Any]) -> None:
    """
    Save configuration to pyproject.toml.

    Args:
        project_root: Path to project root directory
        config: Dict with configuration to save
    """
    if tomli_w is None:
        raise ImportError(
            "tomli_w is required for writing TOML files. Install with: pip install tomli-w"
        )

    pyproject_path = project_root / "pyproject.toml"

    with open(pyproject_path, "wb") as f:
        tomli_w.dump(config, f)


def get_tool_config(config: Dict[str, Any], tool_name: str) -> Dict[str, Any]:
    """
    Get configuration for a specific tool.

    Args:
        config: Full pyproject.toml config
        tool_name: Tool name (e.g., 'imageworks.mono')

    Returns:
        Tool-specific configuration dict
    """
    # Handle nested structure: tool.imageworks.mono
    parts = tool_name.split(".")
    result = config.get("tool", {})

    for part in parts:
        if isinstance(result, dict):
            result = result.get(part, {})
        else:
            return {}

    return result if isinstance(result, dict) else {}


def update_tool_config(
    config: Dict[str, Any], tool_name: str, updates: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Update configuration for a specific tool.

    Args:
        config: Full pyproject.toml config
        tool_name: Tool name (e.g., 'imageworks.mono')
        updates: Dict with values to update

    Returns:
        Updated configuration dict
    """
    # Ensure tool section exists
    if "tool" not in config:
        config["tool"] = {}

    # Handle nested structure: tool.imageworks.mono
    parts = tool_name.split(".")
    current = config["tool"]

    # Navigate to the parent of the target
    for part in parts[:-1]:
        if part not in current:
            current[part] = {}
        current = current[part]

    # Update the target section
    if parts[-1] not in current:
        current[parts[-1]] = {}

    current[parts[-1]].update(updates)

    return config


# Default path mappings for each module
MODULE_PATH_CONFIGS = {
    "mono": {
        "tool_name": "imageworks.mono",
        "fields": {
            "default_folder": "Input Directory",
            "default_jsonl": "Output JSONL",
            "default_summary": "Summary Markdown",
        },
    },
    "personal_tagger": {
        "tool_name": "imageworks.personal_tagger",
        "fields": {
            "default_output_jsonl": "Output JSONL",
            "default_summary_path": "Summary Markdown",
        },
    },
    "image_similarity_checker": {
        "tool_name": "imageworks.image_similarity_checker",
        "fields": {
            "default_library_root": "Library Root",
            "default_output_jsonl": "Output JSONL",
            "default_summary_path": "Summary Markdown",
            "default_cache_dir": "Cache Directory",
        },
    },
    "color_narrator": {
        "tool_name": "imageworks.color_narrator",
        "fields": {
            "default_images_dir": "Images Directory",
            "default_overlays_dir": "Overlays Directory",
            "default_mono_jsonl": "Mono JSONL Input",
        },
    },
}
