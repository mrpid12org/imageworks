"""Error handling and validation utilities."""

import streamlit as st
from pathlib import Path
from typing import Optional, Union, List, Dict, Any
import subprocess
import json


class ValidationError(Exception):
    """Custom validation error."""

    pass


def handle_error(error: Exception, context: str = "") -> None:
    """
    Display error to user with context.

    Args:
        error: The exception that occurred
        context: Additional context about where/why error occurred
    """
    error_msg = f"**Error:** {str(error)}"
    if context:
        error_msg = f"{context}\n\n{error_msg}"

    st.error(error_msg)

    # Show debug info if enabled
    if st.session_state.get("debug_mode", False):
        with st.expander("🐛 Debug Information"):
            st.code(f"Type: {type(error).__name__}\n{str(error)}")


def validate_path(
    path: Union[str, Path],
    must_exist: bool = True,
    must_be_file: bool = False,
    must_be_dir: bool = False,
    extensions: Optional[List[str]] = None,
) -> Path:
    """
    Validate a path with various checks.

    Args:
        path: Path to validate
        must_exist: If True, path must exist
        must_be_file: If True, path must be a file
        must_be_dir: If True, path must be a directory
        extensions: If provided, file must have one of these extensions

    Returns:
        Validated Path object

    Raises:
        ValidationError: If validation fails
    """
    if not path:
        raise ValidationError("Path cannot be empty")

    path_obj = Path(path)

    if must_exist and not path_obj.exists():
        raise ValidationError(f"Path does not exist: {path}")

    if must_be_file and path_obj.exists() and not path_obj.is_file():
        raise ValidationError(f"Path is not a file: {path}")

    if must_be_dir and path_obj.exists() and not path_obj.is_dir():
        raise ValidationError(f"Path is not a directory: {path}")

    if extensions and path_obj.exists():
        if path_obj.suffix.lower() not in [
            f".{ext.lower().lstrip('.')}" for ext in extensions
        ]:
            raise ValidationError(
                f"File must have one of these extensions: {', '.join(extensions)}"
            )

    return path_obj


def validate_url(url: str) -> str:
    """
    Validate URL format.

    Args:
        url: URL to validate

    Returns:
        Validated URL

    Raises:
        ValidationError: If URL is invalid
    """
    if not url:
        raise ValidationError("URL cannot be empty")

    if not url.startswith(("http://", "https://")):
        raise ValidationError(f"URL must start with http:// or https://: {url}")

    return url


def validate_model_name(model_name: str) -> str:
    """
    Validate model name format.

    Args:
        model_name: Model name to validate

    Returns:
        Validated model name

    Raises:
        ValidationError: If model name is invalid
    """
    if not model_name:
        raise ValidationError("Model name cannot be empty")

    if (
        not model_name.replace("-", "")
        .replace("_", "")
        .replace("/", "")
        .replace(".", "")
        .isalnum()
    ):
        raise ValidationError(f"Model name contains invalid characters: {model_name}")

    return model_name


def validate_threshold(
    value: float, min_val: float = 0.0, max_val: float = 1.0
) -> float:
    """
    Validate threshold value is in range.

    Args:
        value: Threshold to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        Validated threshold

    Raises:
        ValidationError: If threshold is out of range
    """
    if not min_val <= value <= max_val:
        raise ValidationError(
            f"Threshold must be between {min_val} and {max_val}, got {value}"
        )

    return value


def safe_subprocess_run(
    command: List[str], timeout: int = 300, check: bool = False
) -> subprocess.CompletedProcess:
    """
    Safely run subprocess command with error handling.

    Args:
        command: Command to run
        timeout: Timeout in seconds
        check: If True, raise exception on non-zero exit

    Returns:
        CompletedProcess object

    Raises:
        ValidationError: If command fails
    """
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=check
        )
        return result

    except subprocess.TimeoutExpired:
        raise ValidationError(f"Command timed out after {timeout} seconds")

    except subprocess.CalledProcessError as e:
        raise ValidationError(
            f"Command failed with exit code {e.returncode}\n{e.stderr}"
        )

    except FileNotFoundError:
        raise ValidationError(f"Command not found: {command[0]}")


def safe_json_load(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Safely load JSON file with error handling.

    Args:
        path: Path to JSON file

    Returns:
        Loaded JSON data

    Raises:
        ValidationError: If loading fails
    """
    path_obj = validate_path(
        path, must_exist=True, must_be_file=True, extensions=["json"]
    )

    try:
        with open(path_obj, "r") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValidationError(f"Invalid JSON in {path}: {e}")
    except Exception as e:
        raise ValidationError(f"Failed to load JSON from {path}: {e}")


def safe_json_save(data: Dict[str, Any], path: Union[str, Path]) -> None:
    """
    Safely save JSON file with error handling.

    Args:
        data: Data to save
        path: Path to save to

    Raises:
        ValidationError: If saving fails
    """
    path_obj = Path(path)

    # Create parent directory if needed
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(path_obj, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        raise ValidationError(f"Failed to save JSON to {path}: {e}")


def require_session_state(*keys: str) -> None:
    """
    Require session state keys to exist.

    Args:
        *keys: Session state keys that must exist

    Raises:
        ValidationError: If any key is missing
    """
    missing = [k for k in keys if k not in st.session_state]
    if missing:
        raise ValidationError(f"Missing required session state: {', '.join(missing)}")


def validate_mono_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Validate mono checker results format.

    Args:
        results: Results to validate

    Returns:
        Validated results

    Raises:
        ValidationError: If results are invalid
    """
    if not isinstance(results, list):
        raise ValidationError("Results must be a list")

    required_keys = {"image_path", "verdict", "certainty"}

    for i, result in enumerate(results):
        if not isinstance(result, dict):
            raise ValidationError(f"Result {i} must be a dictionary")

        missing = required_keys - set(result.keys())
        if missing:
            raise ValidationError(f"Result {i} missing keys: {', '.join(missing)}")

    return results


def validate_similarity_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Validate similarity checker results format.

    Args:
        results: Results to validate

    Returns:
        Validated results

    Raises:
        ValidationError: If results are invalid
    """
    if not isinstance(results, list):
        raise ValidationError("Results must be a list")

    required_keys = {"candidate_path", "matches"}

    for i, result in enumerate(results):
        if not isinstance(result, dict):
            raise ValidationError(f"Result {i} must be a dictionary")

        missing = required_keys - set(result.keys())
        if missing:
            raise ValidationError(f"Result {i} missing keys: {', '.join(missing)}")

        if not isinstance(result.get("matches"), list):
            raise ValidationError(f"Result {i} 'matches' must be a list")

    return results


def validate_backend_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate backend configuration.

    Args:
        config: Backend config to validate

    Returns:
        Validated config

    Raises:
        ValidationError: If config is invalid
    """
    required_keys = {"backend", "base_url"}

    missing = required_keys - set(config.keys())
    if missing:
        raise ValidationError(f"Backend config missing keys: {', '.join(missing)}")

    # Validate backend type
    valid_backends = {"vllm", "lmdeploy", "ollama"}
    if config["backend"] not in valid_backends:
        raise ValidationError(
            f"Invalid backend: {config['backend']}. Must be one of {valid_backends}"
        )

    # Validate URL
    validate_url(config["base_url"])

    return config


def check_disk_space(path: Union[str, Path], required_mb: int = 100) -> bool:
    """
    Check if sufficient disk space is available.

    Args:
        path: Path to check disk space for
        required_mb: Required space in MB

    Returns:
        True if sufficient space available

    Raises:
        ValidationError: If insufficient space
    """
    import shutil

    path_obj = Path(path)

    # Get parent directory if path doesn't exist
    if not path_obj.exists():
        path_obj = path_obj.parent

    stat = shutil.disk_usage(path_obj)
    available_mb = stat.free / (1024 * 1024)

    if available_mb < required_mb:
        raise ValidationError(
            f"Insufficient disk space: {available_mb:.0f} MB available, {required_mb} MB required"
        )

    return True


def validate_image_extensions(images: List[Path]) -> List[Path]:
    """
    Validate image file extensions.

    Args:
        images: List of image paths

    Returns:
        Validated image paths

    Raises:
        ValidationError: If invalid extensions found
    """
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}

    invalid = [img for img in images if img.suffix.lower() not in valid_extensions]

    if invalid:
        raise ValidationError(
            f"Invalid image extensions found: {', '.join(str(i) for i in invalid[:5])}"
            + (f" and {len(invalid)-5} more" if len(invalid) > 5 else "")
        )

    return images
