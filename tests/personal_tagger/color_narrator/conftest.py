"""Pytest configuration and fixtures for color-narrator tests.

Provides shared fixtures, test configuration, and utilities for
color-narrator test suite.
"""

import pytest
from pathlib import Path
import tempfile
import shutil
import json
from datetime import datetime

from imageworks.apps.personal_tagger.color_narrator.core.data_loader import (
    ColorNarratorItem,
)
from imageworks.apps.personal_tagger.color_narrator.core.vlm import VLMResponse
from imageworks.apps.personal_tagger.color_narrator.core.metadata import (
    ColorNarrationMetadata,
)


@pytest.fixture
def sample_mono_data():
    """Provide sample mono-checker analysis data."""
    return {
        "filename": "test_image.jpg",
        "contamination_level": 0.45,
        "hue_analysis": "Warm color cast detected in highlights",
        "chroma_analysis": "Low to moderate chroma levels throughout",
        "processing_timestamp": "2024-01-15T14:30:00",
        "total_pixels": 1920000,
        "contaminated_pixels": 864000,
    }


@pytest.fixture
def sample_vlm_response():
    """Provide sample VLM inference response."""
    return VLMResponse(
        description="The image shows subtle warm color contamination primarily in the background highlights and mid-tones, with some cooler tones visible in the shadow areas.",
        confidence=0.82,
        color_regions=["background", "highlights"],
        processing_time=1.4,
        error=None,
    )


@pytest.fixture
def sample_color_narration_metadata():
    """Provide sample color narration metadata."""
    return ColorNarrationMetadata(
        description="Subtle warm color cast visible in background and clothing areas",
        confidence_score=0.78,
        color_regions=["background", "clothing"],
        processing_timestamp=datetime.now().isoformat(),
        mono_contamination_level=0.35,
        vlm_model="Qwen/Qwen2-VL-7B-Instruct",
        vlm_processing_time=1.2,
        hue_analysis="Warm cast detected",
        chroma_analysis="Low chroma levels",
    )


@pytest.fixture
def sample_narrator_item():
    """Provide sample ColorNarratorItem for testing."""
    return ColorNarratorItem(
        image_path=Path("/test/images/sample.jpg"),
        overlay_path=Path("/test/overlays/sample_lab_chroma.png"),
        mono_data={
            "contamination_level": 0.3,
            "hue_analysis": "Warm tones",
            "chroma_analysis": "Moderate chroma",
        },
        has_existing_xmp=False,
    )


@pytest.fixture
def temp_test_workspace():
    """Create a complete temporary workspace for integration testing."""
    temp_dir = Path(tempfile.mkdtemp())

    # Create directory structure
    images_dir = temp_dir / "images"
    overlays_dir = temp_dir / "overlays"
    images_dir.mkdir()
    overlays_dir.mkdir()

    # Create sample image files
    sample_images = ["test1.jpg", "test2.jpg", "test3.jpg"]
    for img in sample_images:
        (images_dir / img).write_text("fake_image_data")

    # Create sample overlay files
    for img in sample_images:
        overlay_name = img.replace(".jpg", "_lab_chroma.png")
        (overlays_dir / overlay_name).write_text("fake_overlay_data")

    # Create sample mono JSONL data
    mono_jsonl = temp_dir / "mono_results.jsonl"
    sample_mono_data = [
        {"filename": "test1.jpg", "contamination_level": 0.3, "hue_analysis": "warm"},
        {"filename": "test2.jpg", "contamination_level": 0.15, "hue_analysis": "cool"},
        {"filename": "test3.jpg", "contamination_level": 0.6, "hue_analysis": "mixed"},
    ]

    with open(mono_jsonl, "w") as f:
        for data in sample_mono_data:
            f.write(json.dumps(data) + "\n")

    workspace = {
        "temp_dir": temp_dir,
        "images_dir": images_dir,
        "overlays_dir": overlays_dir,
        "mono_jsonl": mono_jsonl,
        "sample_images": sample_images,
    }

    yield workspace

    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_vlm_client():
    """Provide a mock VLM client for testing."""
    from unittest.mock import Mock

    mock_client = Mock()
    mock_client.health_check.return_value = True
    mock_client.infer_single.return_value = VLMResponse(
        description="Mock color description",
        confidence=0.8,
        color_regions=["background"],
        processing_time=1.0,
        error=None,
    )
    mock_client.infer_batch.return_value = [
        VLMResponse(
            description=f"Mock description {i}",
            confidence=0.8 + (i * 0.05),
            color_regions=[f"region_{i}"],
            processing_time=1.0 + i,
            error=None,
        )
        for i in range(3)
    ]

    return mock_client


@pytest.fixture
def mock_metadata_writer():
    """Provide a mock XMP metadata writer for testing."""
    from unittest.mock import Mock

    mock_writer = Mock()
    mock_writer.write_metadata.return_value = True
    mock_writer.read_metadata.return_value = None
    mock_writer.has_color_narration.return_value = False
    mock_writer.remove_metadata.return_value = True

    return mock_writer


# Test configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "vlm: mark test as requiring VLM server")


# Custom test utilities
class TestUtils:
    """Utility functions for color-narrator tests."""

    @staticmethod
    def create_test_image_file(path: Path, content: str = "fake_image_data"):
        """Create a test image file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    @staticmethod
    def create_test_jsonl(path: Path, data_list: list):
        """Create a test JSONL file from list of dictionaries."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for data in data_list:
                f.write(json.dumps(data) + "\n")
        return path

    @staticmethod
    def assert_valid_metadata(metadata: ColorNarrationMetadata):
        """Assert that metadata meets validation criteria."""
        assert metadata.description and len(metadata.description) >= 10
        assert 0.0 <= metadata.confidence_score <= 1.0
        assert isinstance(metadata.color_regions, list)
        assert metadata.processing_timestamp
        assert metadata.mono_contamination_level >= 0.0
        assert metadata.vlm_model
        assert metadata.vlm_processing_time >= 0.0

    @staticmethod
    def create_mock_processing_results(count: int = 3):
        """Create mock processing results for testing."""
        from imageworks.apps.personal_tagger.color_narrator.core.narrator import (
            ProcessingResult,
        )

        results = []
        for i in range(count):
            item = ColorNarratorItem(
                image_path=Path(f"/test/image_{i}.jpg"),
                overlay_path=Path(f"/test/overlay_{i}.png"),
                mono_data={"contamination_level": 0.3 + (i * 0.1)},
            )

            vlm_response = VLMResponse(
                description=f"Color description {i}",
                confidence=0.8 + (i * 0.05),
                color_regions=[f"region_{i}"],
                processing_time=1.0 + i,
            )

            result = ProcessingResult(
                item=item,
                vlm_response=vlm_response,
                metadata_written=True,
                error=None,
                processing_time=2.0 + i,
            )
            results.append(result)

        return results


@pytest.fixture
def test_utils():
    """Provide TestUtils instance for tests."""
    return TestUtils
