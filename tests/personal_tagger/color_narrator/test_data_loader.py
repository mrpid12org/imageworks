"""Tests for data loading and validation functionality.

Tests data loader initialization, JSONL parsing, image/overlay matching,
filtering, and batch generation for color narrator processing.
"""

import pytest
from pathlib import Path
import json
import tempfile
import shutil

from imageworks.apps.personal_tagger.color_narrator.core.data_loader import (
    ColorNarratorDataLoader,
    DataLoaderConfig,
    ColorNarratorItem,
)


class TestDataLoaderConfig:
    """Test cases for DataLoaderConfig."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = DataLoaderConfig(
            images_dir=Path("/test/images"),
            overlays_dir=Path("/test/overlays"),
            mono_jsonl=Path("/test/mono.jsonl"),
        )

        assert config.image_extensions == [".jpg", ".jpeg", ".JPG", ".JPEG"]
        assert config.overlay_extensions == [".png", ".PNG"]
        assert config.min_contamination_level == 0.1
        assert config.require_overlays is True

    def test_config_custom_values(self):
        """Test custom configuration values."""
        config = DataLoaderConfig(
            images_dir=Path("/test/images"),
            overlays_dir=Path("/test/overlays"),
            mono_jsonl=Path("/test/mono.jsonl"),
            image_extensions=[".jpg"],
            overlay_extensions=[".png"],
            min_contamination_level=0.5,
            require_overlays=False,
        )

        assert config.image_extensions == [".jpg"]
        assert config.overlay_extensions == [".png"]
        assert config.min_contamination_level == 0.5
        assert config.require_overlays is False


class TestColorNarratorDataLoader:
    """Test cases for ColorNarratorDataLoader."""

    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing."""
        temp_dir = Path(tempfile.mkdtemp())

        images_dir = temp_dir / "images"
        overlays_dir = temp_dir / "overlays"
        images_dir.mkdir()
        overlays_dir.mkdir()

        # Create test JSONL file
        mono_jsonl = temp_dir / "mono.jsonl"
        test_data = [
            {
                "filename": "test1.jpg",
                "contamination_level": 0.3,
                "hue_analysis": "warm",
            },
            {
                "filename": "test2.jpg",
                "contamination_level": 0.05,
                "hue_analysis": "cool",
            },
            {
                "filename": "test3.jpg",
                "contamination_level": 0.8,
                "hue_analysis": "mixed",
            },
        ]

        with open(mono_jsonl, "w") as f:
            for item in test_data:
                f.write(json.dumps(item) + "\n")

        yield {
            "temp_dir": temp_dir,
            "images_dir": images_dir,
            "overlays_dir": overlays_dir,
            "mono_jsonl": mono_jsonl,
        }

        # Cleanup
        shutil.rmtree(temp_dir)

    def test_loader_initialization(self, temp_dirs):
        """Test data loader initialization."""
        config = DataLoaderConfig(
            images_dir=temp_dirs["images_dir"],
            overlays_dir=temp_dirs["overlays_dir"],
            mono_jsonl=temp_dirs["mono_jsonl"],
        )

        loader = ColorNarratorDataLoader(config)
        assert loader.config == config
        assert not loader._loaded
        assert len(loader._mono_data) == 0

    def test_load_success(self, temp_dirs):
        """Test successful data loading."""
        config = DataLoaderConfig(
            images_dir=temp_dirs["images_dir"],
            overlays_dir=temp_dirs["overlays_dir"],
            mono_jsonl=temp_dirs["mono_jsonl"],
        )

        loader = ColorNarratorDataLoader(config)
        loader.load()

        assert loader._loaded
        assert len(loader._mono_data) == 3
        assert "test1.jpg" in loader._mono_data
        assert loader._mono_data["test1.jpg"]["contamination_level"] == 0.3

    def test_load_missing_directories(self, temp_dirs):
        """Test loading with missing directories."""
        config = DataLoaderConfig(
            images_dir=Path("/nonexistent/images"),
            overlays_dir=temp_dirs["overlays_dir"],
            mono_jsonl=temp_dirs["mono_jsonl"],
        )

        loader = ColorNarratorDataLoader(config)

        with pytest.raises(FileNotFoundError, match="Images directory not found"):
            loader.load()

    def test_load_missing_jsonl(self, temp_dirs):
        """Test loading with missing JSONL file."""
        config = DataLoaderConfig(
            images_dir=temp_dirs["images_dir"],
            overlays_dir=temp_dirs["overlays_dir"],
            mono_jsonl=Path("/nonexistent/mono.jsonl"),
        )

        loader = ColorNarratorDataLoader(config)

        with pytest.raises(FileNotFoundError, match="Mono JSONL file not found"):
            loader.load()

    def test_load_malformed_jsonl(self, temp_dirs):
        """Test loading with malformed JSONL data."""
        # Create malformed JSONL
        malformed_jsonl = temp_dirs["temp_dir"] / "malformed.jsonl"
        with open(malformed_jsonl, "w") as f:
            f.write('{"filename": "test1.jpg"}\n')  # Valid
            f.write("invalid json line\n")  # Invalid
            f.write('{"filename": "test2.jpg"}\n')  # Valid

        config = DataLoaderConfig(
            images_dir=temp_dirs["images_dir"],
            overlays_dir=temp_dirs["overlays_dir"],
            mono_jsonl=malformed_jsonl,
        )

        loader = ColorNarratorDataLoader(config)
        loader.load()

        # Should load valid lines, skip malformed ones
        assert len(loader._mono_data) == 2
        assert "test1.jpg" in loader._mono_data
        assert "test2.jpg" in loader._mono_data

    def test_find_overlay_for_image(self, temp_dirs):
        """Test overlay finding logic."""
        # Create test image and overlays
        test_image = temp_dirs["images_dir"] / "test.jpg"
        test_image.write_text("fake image")

        overlay1 = temp_dirs["overlays_dir"] / "test.png"
        overlay2 = temp_dirs["overlays_dir"] / "test_lab_chroma.png"
        overlay3 = temp_dirs["overlays_dir"] / "test_residual.png"

        config = DataLoaderConfig(
            images_dir=temp_dirs["images_dir"],
            overlays_dir=temp_dirs["overlays_dir"],
            mono_jsonl=temp_dirs["mono_jsonl"],
        )
        loader = ColorNarratorDataLoader(config)

        # Test exact match
        overlay1.write_text("fake overlay")
        result = loader._find_overlay_for_image(test_image)
        assert result == overlay1
        overlay1.unlink()

        # Test _lab_chroma suffix
        overlay2.write_text("fake overlay")
        result = loader._find_overlay_for_image(test_image)
        assert result == overlay2
        overlay2.unlink()

        # Test _residual suffix
        overlay3.write_text("fake overlay")
        result = loader._find_overlay_for_image(test_image)
        assert result == overlay3
        overlay3.unlink()

        # Test no overlay found
        result = loader._find_overlay_for_image(test_image)
        assert result is None

    def test_get_items_filtering(self, temp_dirs):
        """Test item filtering by contamination level."""
        # Create test images
        (temp_dirs["images_dir"] / "test1.jpg").write_text("image1")
        (temp_dirs["images_dir"] / "test2.jpg").write_text("image2")  # Below threshold
        (temp_dirs["images_dir"] / "test3.jpg").write_text("image3")

        # Create corresponding overlays
        (temp_dirs["overlays_dir"] / "test1.png").write_text("overlay1")
        (temp_dirs["overlays_dir"] / "test2.png").write_text("overlay2")
        (temp_dirs["overlays_dir"] / "test3.png").write_text("overlay3")

        config = DataLoaderConfig(
            images_dir=temp_dirs["images_dir"],
            overlays_dir=temp_dirs["overlays_dir"],
            mono_jsonl=temp_dirs["mono_jsonl"],
            min_contamination_level=0.1,
        )

        loader = ColorNarratorDataLoader(config)

        # Get all items in single batch
        batch_gen = loader.get_items(batch_size=None)
        items = next(batch_gen)

        # Should filter out test2.jpg (contamination 0.05 < 0.1)
        assert len(items) == 2
        filenames = [item.image_path.name for item in items]
        assert "test1.jpg" in filenames
        assert "test2.jpg" not in filenames  # Filtered out
        assert "test3.jpg" in filenames

    def test_get_items_batching(self, temp_dirs):
        """Test item batching functionality."""
        # Create test images with overlays
        for i in range(5):
            (temp_dirs["images_dir"] / f"test{i}.jpg").write_text(f"image{i}")
            (temp_dirs["overlays_dir"] / f"test{i}.png").write_text(f"overlay{i}")

        # Update JSONL with valid contamination levels
        test_data = [
            {"filename": f"test{i}.jpg", "contamination_level": 0.3} for i in range(5)
        ]

        with open(temp_dirs["mono_jsonl"], "w") as f:
            for item in test_data:
                f.write(json.dumps(item) + "\n")

        config = DataLoaderConfig(
            images_dir=temp_dirs["images_dir"],
            overlays_dir=temp_dirs["overlays_dir"],
            mono_jsonl=temp_dirs["mono_jsonl"],
        )

        loader = ColorNarratorDataLoader(config)

        # Test batch size of 2
        batches = list(loader.get_items(batch_size=2))
        assert len(batches) == 3  # 5 items in batches of 2 = 3 batches
        assert len(batches[0]) == 2
        assert len(batches[1]) == 2
        assert len(batches[2]) == 1

    def test_get_statistics(self, temp_dirs):
        """Test statistics calculation."""
        # Create test images and overlays
        (temp_dirs["images_dir"] / "test1.jpg").write_text("image1")
        (temp_dirs["images_dir"] / "test3.jpg").write_text("image3")
        (temp_dirs["overlays_dir"] / "test1.png").write_text("overlay1")
        (temp_dirs["overlays_dir"] / "test3.png").write_text("overlay3")

        config = DataLoaderConfig(
            images_dir=temp_dirs["images_dir"],
            overlays_dir=temp_dirs["overlays_dir"],
            mono_jsonl=temp_dirs["mono_jsonl"],
        )

        loader = ColorNarratorDataLoader(config)
        stats = loader.get_statistics()

        assert stats["mono_results"] == 3
        assert stats["valid_items"] == 2  # test1.jpg and test3.jpg (test2.jpg filtered)
        assert stats["avg_contamination"] > 0.0
        assert stats["max_contamination"] == 0.8

    def test_validate_data_consistency(self, temp_dirs):
        """Test data consistency validation."""
        # Create image that doesn't have mono data
        (temp_dirs["images_dir"] / "orphan.jpg").write_text("orphan image")

        # Create mono data for missing image
        extra_data = {"filename": "missing.jpg", "contamination_level": 0.5}
        with open(temp_dirs["mono_jsonl"], "a") as f:
            f.write(json.dumps(extra_data) + "\n")

        config = DataLoaderConfig(
            images_dir=temp_dirs["images_dir"],
            overlays_dir=temp_dirs["overlays_dir"],
            mono_jsonl=temp_dirs["mono_jsonl"],
        )

        loader = ColorNarratorDataLoader(config)
        issues = loader.validate_data_consistency()

        # Should find orphaned mono result
        assert len(issues) > 0
        orphan_issues = [issue for issue in issues if "missing.jpg" in issue]
        assert len(orphan_issues) > 0


class TestColorNarratorItem:
    """Test cases for ColorNarratorItem dataclass."""

    def test_item_creation(self):
        """Test ColorNarratorItem creation."""
        item = ColorNarratorItem(
            image_path=Path("/test/image.jpg"),
            overlay_path=Path("/test/overlay.png"),
            mono_data={"contamination_level": 0.5},
            has_existing_xmp=True,
        )

        assert item.image_path == Path("/test/image.jpg")
        assert item.overlay_path == Path("/test/overlay.png")
        assert item.mono_data == {"contamination_level": 0.5}
        assert item.has_existing_xmp is True

    def test_item_defaults(self):
        """Test ColorNarratorItem with default values."""
        item = ColorNarratorItem(
            image_path=Path("/test/image.jpg"),
            overlay_path=Path("/test/overlay.png"),
            mono_data={},
        )

        assert item.has_existing_xmp is False
