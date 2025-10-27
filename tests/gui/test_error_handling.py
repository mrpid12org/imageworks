"""Unit tests for error handling utilities."""

import pytest
from pathlib import Path
import tempfile

from imageworks.gui.utils.error_handling import (
    ValidationError,
    validate_path,
    validate_url,
    validate_model_name,
    validate_threshold,
    safe_json_load,
    safe_json_save,
    validate_mono_results,
    validate_similarity_results,
    check_disk_space,
    validate_image_extensions,
)


# ===== Path Validation Tests =====


def test_validate_path_exists():
    """Test path validation with existing path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        result = validate_path(path, must_exist=True)
        assert result == path


def test_validate_path_not_exists():
    """Test path validation with non-existing path."""
    with pytest.raises(ValidationError, match="does not exist"):
        validate_path("/nonexistent/path", must_exist=True)


def test_validate_path_must_be_file():
    """Test path validation for file requirement."""
    with tempfile.NamedTemporaryFile() as tmp:
        path = Path(tmp.name)
        result = validate_path(path, must_exist=True, must_be_file=True)
        assert result == path


def test_validate_path_must_be_dir():
    """Test path validation for directory requirement."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        result = validate_path(path, must_exist=True, must_be_dir=True)
        assert result == path


def test_validate_path_wrong_type():
    """Test path validation fails when type is wrong."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        with pytest.raises(ValidationError, match="not a file"):
            validate_path(path, must_exist=True, must_be_file=True)


def test_validate_path_extensions():
    """Test path validation with file extensions."""
    with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
        path = Path(tmp.name)
        result = validate_path(path, must_exist=True, extensions=["json"])
        assert result == path


def test_validate_path_wrong_extension():
    """Test path validation fails with wrong extension."""
    with tempfile.NamedTemporaryFile(suffix=".txt") as tmp:
        path = Path(tmp.name)
        with pytest.raises(ValidationError, match="must have one of these extensions"):
            validate_path(path, must_exist=True, extensions=["json"])


def test_validate_path_empty():
    """Test path validation with empty path."""
    with pytest.raises(ValidationError, match="cannot be empty"):
        validate_path("")


# ===== URL Validation Tests =====


def test_validate_url_http():
    """Test URL validation with http."""
    url = "http://example.com"
    result = validate_url(url)
    assert result == url


def test_validate_url_https():
    """Test URL validation with https."""
    url = "https://example.com"
    result = validate_url(url)
    assert result == url


def test_validate_url_invalid():
    """Test URL validation fails without protocol."""
    with pytest.raises(ValidationError, match="must start with"):
        validate_url("example.com")


def test_validate_url_empty():
    """Test URL validation with empty URL."""
    with pytest.raises(ValidationError, match="cannot be empty"):
        validate_url("")


# ===== Model Name Validation Tests =====


def test_validate_model_name_valid():
    """Test model name validation with valid names."""
    names = [
        "microsoft/Phi-3.5-vision-instruct",
        "llava-v1.5-7b",
        "model_name_123",
    ]
    for name in names:
        result = validate_model_name(name)
        assert result == name


def test_validate_model_name_empty():
    """Test model name validation with empty name."""
    with pytest.raises(ValidationError, match="cannot be empty"):
        validate_model_name("")


def test_validate_model_name_invalid_chars():
    """Test model name validation with invalid characters."""
    with pytest.raises(ValidationError, match="invalid characters"):
        validate_model_name("model name with spaces!")


# ===== Threshold Validation Tests =====


def test_validate_threshold_valid():
    """Test threshold validation with valid values."""
    assert validate_threshold(0.5) == 0.5
    assert validate_threshold(0.0) == 0.0
    assert validate_threshold(1.0) == 1.0


def test_validate_threshold_too_low():
    """Test threshold validation fails when too low."""
    with pytest.raises(ValidationError, match="must be between"):
        validate_threshold(-0.1)


def test_validate_threshold_too_high():
    """Test threshold validation fails when too high."""
    with pytest.raises(ValidationError, match="must be between"):
        validate_threshold(1.5)


def test_validate_threshold_custom_range():
    """Test threshold validation with custom range."""
    assert validate_threshold(50, min_val=0, max_val=100) == 50


# ===== JSON File Tests =====


def test_safe_json_save_and_load():
    """Test JSON save and load cycle."""
    data = {"key": "value", "number": 42}

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        safe_json_save(data, tmp_path)
        loaded = safe_json_load(tmp_path)
        assert loaded == data
    finally:
        tmp_path.unlink()


def test_safe_json_load_invalid():
    """Test JSON load with invalid JSON."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        tmp.write("{invalid json")
        tmp_path = Path(tmp.name)

    try:
        with pytest.raises(ValidationError, match="Invalid JSON"):
            safe_json_load(tmp_path)
    finally:
        tmp_path.unlink()


def test_safe_json_save_creates_dir():
    """Test JSON save creates parent directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "subdir" / "file.json"
        data = {"test": "data"}

        safe_json_save(data, path)

        assert path.exists()
        assert path.parent.exists()


# ===== Results Validation Tests =====


def test_validate_mono_results_valid():
    """Test mono results validation with valid data."""
    results = [
        {"image_path": "a.jpg", "verdict": "monochrome", "certainty": 0.9},
        {"image_path": "b.jpg", "verdict": "color", "certainty": 0.8},
    ]

    validated = validate_mono_results(results)
    assert validated == results


def test_validate_mono_results_missing_key():
    """Test mono results validation fails with missing key."""
    results = [
        {"image_path": "a.jpg", "verdict": "monochrome"},  # Missing certainty
    ]

    with pytest.raises(ValidationError, match="missing keys"):
        validate_mono_results(results)


def test_validate_mono_results_not_list():
    """Test mono results validation fails if not list."""
    with pytest.raises(ValidationError, match="must be a list"):
        validate_mono_results({"not": "a list"})


def test_validate_similarity_results_valid():
    """Test similarity results validation with valid data."""
    results = [
        {"candidate_path": "a.jpg", "matches": []},
        {"candidate_path": "b.jpg", "matches": [{"path": "c.jpg", "score": 0.9}]},
    ]

    validated = validate_similarity_results(results)
    assert validated == results


def test_validate_similarity_results_missing_key():
    """Test similarity results validation fails with missing key."""
    results = [
        {"candidate_path": "a.jpg"},  # Missing matches
    ]

    with pytest.raises(ValidationError, match="missing keys"):
        validate_similarity_results(results)


# ===== Disk Space Tests =====


def test_check_disk_space_sufficient():
    """Test disk space check passes with sufficient space."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Should pass with small requirement
        result = check_disk_space(tmpdir, required_mb=1)
        assert result is True


def test_check_disk_space_insufficient():
    """Test disk space check fails with insufficient space."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Should fail with huge requirement
        with pytest.raises(ValidationError, match="Insufficient disk space"):
            check_disk_space(tmpdir, required_mb=999999999)


# ===== Image Extension Tests =====


def test_validate_image_extensions_valid():
    """Test image extension validation with valid extensions."""
    images = [
        Path("a.jpg"),
        Path("b.png"),
        Path("c.webp"),
        Path("d.JPEG"),  # Case insensitive
    ]

    result = validate_image_extensions(images)
    assert result == images


def test_validate_image_extensions_invalid():
    """Test image extension validation fails with invalid extension."""
    images = [
        Path("a.jpg"),
        Path("b.txt"),  # Invalid
    ]

    with pytest.raises(ValidationError, match="Invalid image extensions"):
        validate_image_extensions(images)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
