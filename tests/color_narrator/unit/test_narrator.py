"""Tests for the main color narrator orchestration module.

Tests narrator configuration, processing pipeline integration, VLM coordination,
metadata writing, batch processing, and validation functionality.
"""

import pytest
from unittest.mock import patch
from pathlib import Path
from datetime import datetime
import tempfile
import shutil

from imageworks.apps.color_narrator.core import prompts
from imageworks.apps.color_narrator.core.narrator import (
    ColorNarrator,
    NarrationConfig,
    ProcessingResult,
)
from imageworks.apps.color_narrator.core.data_loader import (
    ColorNarratorItem,
)
from imageworks.apps.color_narrator.core.vlm import VLMBackend, VLMResponse
from imageworks.apps.color_narrator.core.metadata import (
    ColorNarrationMetadata,
)


class TestNarrationConfig:
    """Test cases for NarrationConfig."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = NarrationConfig(
            images_dir=Path("/test/images"),
            overlays_dir=Path("/test/overlays"),
            mono_jsonl=Path("/test/mono.jsonl"),
        )

        assert config.vlm_base_url == "http://localhost:24001/v1"
        assert config.vlm_model == "Qwen2.5-VL-7B-AWQ"
        assert config.vlm_timeout == 120
        assert config.vlm_backend == VLMBackend.LMDEPLOY.value
        assert config.vlm_api_key == "EMPTY"
        assert config.vlm_backend_options is None
        assert config.batch_size == 4
        assert config.min_contamination_level == 0.1
        assert config.require_overlays is True
        assert config.prompt_id == prompts.CURRENT_PROMPT_ID
        assert config.use_regions is False
        assert config.allowed_verdicts is None
        assert config.dry_run is False
        assert config.debug is False
        assert config.overwrite_existing is False

    def test_config_custom_values(self):
        """Test custom configuration values."""
        config = NarrationConfig(
            images_dir=Path("/custom/images"),
            overlays_dir=Path("/custom/overlays"),
            mono_jsonl=Path("/custom/mono.jsonl"),
            vlm_base_url="http://custom:9000/v1",
            vlm_model="custom/model",
            vlm_backend="lmdeploy",
            vlm_api_key="secret",
            vlm_backend_options={"tp": 2},
            batch_size=8,
            dry_run=True,
            debug=True,
            prompt_id=3,
            use_regions=True,
            allowed_verdicts={"fail"},
        )

        assert config.vlm_base_url == "http://custom:9000/v1"
        assert config.vlm_model == "custom/model"
        assert config.vlm_backend == "lmdeploy"
        assert config.vlm_api_key == "secret"
        assert config.vlm_backend_options == {"tp": 2}
        assert config.batch_size == 8
        assert config.dry_run is True
        assert config.debug is True
        assert config.prompt_id == 3
        assert config.use_regions is True
        assert config.allowed_verdicts == {"fail"}


class TestProcessingResult:
    """Test cases for ProcessingResult dataclass."""

    def test_result_creation(self):
        """Test ProcessingResult creation."""
        item = ColorNarratorItem(
            image_path=Path("/test.jpg"), overlay_path=Path("/test.png"), mono_data={}
        )

        vlm_response = VLMResponse(
            description="Test description",
            confidence=0.8,
            color_regions=["background"],
            processing_time=1.5,
        )

        result = ProcessingResult(
            item=item,
            vlm_response=vlm_response,
            metadata_written=True,
            error=None,
            processing_time=2.0,
        )

        assert result.item == item
        assert result.vlm_response == vlm_response
        assert result.metadata_written is True
        assert result.error is None
        assert result.processing_time == 2.0

    def test_result_defaults(self):
        """Test ProcessingResult with default values."""
        item = ColorNarratorItem(
            image_path=Path("/test.jpg"), overlay_path=Path("/test.png"), mono_data={}
        )

        result = ProcessingResult(item=item)

        assert result.vlm_response is None
        assert result.metadata_written is False
        assert result.error is None
        assert result.processing_time == 0.0


class TestColorNarrator:
    """Test cases for ColorNarrator main orchestration."""

    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for testing."""
        temp_dir = Path(tempfile.mkdtemp())

        images_dir = temp_dir / "images"
        overlays_dir = temp_dir / "overlays"
        images_dir.mkdir()
        overlays_dir.mkdir()

        mono_jsonl = temp_dir / "mono.jsonl"
        mono_jsonl.write_text('{"filename": "test.jpg", "contamination_level": 0.5}\n')

        yield {
            "temp_dir": temp_dir,
            "images_dir": images_dir,
            "overlays_dir": overlays_dir,
            "mono_jsonl": mono_jsonl,
        }

        # Cleanup
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_config(self, temp_workspace):
        """Create sample configuration for testing."""
        return NarrationConfig(
            images_dir=temp_workspace["images_dir"],
            overlays_dir=temp_workspace["overlays_dir"],
            mono_jsonl=temp_workspace["mono_jsonl"],
            batch_size=2,
            dry_run=True,  # Safe for testing
        )

    def test_narrator_initialization(self, sample_config):
        """Test ColorNarrator initialization."""
        narrator = ColorNarrator(sample_config)

        assert narrator.config == sample_config
        assert narrator.data_loader is not None
        assert narrator.vlm_client is not None
        assert narrator.metadata_writer is not None

    @patch("imageworks.apps.color_narrator.core.vlm.VLMClient.infer_single")
    @patch("imageworks.apps.color_narrator.core.vlm.VLMClient.health_check")
    @patch(
        "imageworks.apps.color_narrator.core.data_loader.ColorNarratorDataLoader.load"
    )
    @patch(
        "imageworks.apps.color_narrator.core.data_loader.ColorNarratorDataLoader.get_statistics"
    )
    @patch(
        "imageworks.apps.color_narrator.core.data_loader.ColorNarratorDataLoader.get_items"
    )
    def test_process_all_dry_run(
        self,
        mock_get_items,
        mock_get_stats,
        mock_load,
        mock_health,
        mock_infer_single,
        sample_config,
    ):
        """Test process_all in dry run mode."""
        # Setup mocks
        mock_health.return_value = True
        mock_get_stats.return_value = {"valid_items": 1}

        # Create mock item
        mock_item = ColorNarratorItem(
            image_path=Path("/test/image.jpg"),
            overlay_path=Path("/test/overlay.png"),
            mono_data={"contamination_level": 0.5},
        )
        mock_get_items.return_value = [[mock_item]]  # Single batch with one item

        mock_infer_single.return_value = VLMResponse(
            description="Mock dry-run description",
            confidence=0.9,
            color_regions=["region"],
            processing_time=1.2,
            error=None,
        )

        narrator = ColorNarrator(sample_config)
        results = narrator.process_all()

        assert len(results) == 1
        assert results[0].vlm_response.description == "Mock dry-run description"
        assert results[0].metadata_written is False
        assert results[0].error is None
        mock_infer_single.assert_called_once()

    @patch("imageworks.apps.color_narrator.core.vlm.VLMClient.health_check")
    def test_process_all_vlm_server_unavailable(self, mock_health, sample_config):
        """Test process_all with unavailable VLM server."""
        mock_health.return_value = False

        narrator = ColorNarrator(sample_config)

        with pytest.raises(RuntimeError, match="VLM backend"):
            narrator.process_all()

    def test_get_prompt_template(self, sample_config):
        """Test prompt template retrieval."""
        narrator = ColorNarrator(sample_config)
        template = narrator._get_prompt_template()

        assert isinstance(template, str)
        assert len(template) > 0
        assert "You are a precise technical image analyst" in template

    def test_create_metadata(self, sample_config):
        """Test metadata creation from processing results."""
        narrator = ColorNarrator(sample_config)

        item = ColorNarratorItem(
            image_path=Path("/test/image.jpg"),
            overlay_path=Path("/test/overlay.png"),
            mono_data={"contamination_level": 0.4},
        )

        vlm_response = VLMResponse(
            description="Color found in background",
            confidence=0.85,
            color_regions=["background"],
            processing_time=1.2,
        )

        metadata = narrator._create_metadata(item, vlm_response)

        assert isinstance(metadata, ColorNarrationMetadata)
        assert metadata.description == "Color found in background"
        assert metadata.confidence_score == 0.85
        assert metadata.color_regions == ["background"]
        assert metadata.mono_contamination_level == 0.4
        assert metadata.vlm_model == sample_config.vlm_model
        assert metadata.vlm_processing_time == 1.2

    @patch(
        "imageworks.apps.color_narrator.core.metadata.XMPMetadataWriter.read_metadata"
    )
    def test_validate_existing(self, mock_read_metadata, temp_workspace):
        """Test validation of existing color narrations."""
        # Create test image
        test_image = temp_workspace["images_dir"] / "test.jpg"
        test_image.write_text("fake image")

        # Mock metadata reading
        sample_metadata = ColorNarrationMetadata(
            description="Valid description",
            confidence_score=0.8,
            color_regions=["background"],
            processing_timestamp=datetime.now().isoformat(),
            mono_contamination_level=0.3,
            vlm_model="test-model",
            vlm_processing_time=1.0,
        )
        mock_read_metadata.return_value = sample_metadata

        config = NarrationConfig(
            images_dir=temp_workspace["images_dir"],
            overlays_dir=temp_workspace["overlays_dir"],
            mono_jsonl=temp_workspace["mono_jsonl"],
        )

        narrator = ColorNarrator(config)
        results = narrator.validate_existing(temp_workspace["images_dir"])

        assert results["total_images"] == 1
        assert results["with_metadata"] == 1
        assert results["valid_metadata"] == 1
        assert len(results["validation_errors"]) == 0

    def test_validate_metadata_content_valid(self, sample_config):
        """Test metadata content validation - valid case."""
        narrator = ColorNarrator(sample_config)

        valid_metadata = ColorNarrationMetadata(
            description="This is a valid description with good length",
            confidence_score=0.75,
            color_regions=["background"],
            processing_timestamp=datetime.now().isoformat(),
            mono_contamination_level=0.3,
            vlm_model="test-model",
            vlm_processing_time=1.0,
        )

        assert narrator._validate_metadata_content(valid_metadata) is True

    def test_validate_metadata_content_invalid(self, sample_config):
        """Test metadata content validation - invalid cases."""
        narrator = ColorNarrator(sample_config)

        # Empty description
        invalid1 = ColorNarrationMetadata(
            description="",
            confidence_score=0.75,
            color_regions=[],
            processing_timestamp=datetime.now().isoformat(),
            mono_contamination_level=0.3,
            vlm_model="test",
            vlm_processing_time=1.0,
        )
        assert narrator._validate_metadata_content(invalid1) is False

        # Invalid confidence score
        invalid2 = ColorNarrationMetadata(
            description="Valid description",
            confidence_score=1.5,  # > 1.0
            color_regions=[],
            processing_timestamp=datetime.now().isoformat(),
            mono_contamination_level=0.3,
            vlm_model="test",
            vlm_processing_time=1.0,
        )
        assert narrator._validate_metadata_content(invalid2) is False

        # Description too short
        invalid3 = ColorNarrationMetadata(
            description="Short",  # < 10 chars
            confidence_score=0.75,
            color_regions=[],
            processing_timestamp=datetime.now().isoformat(),
            mono_contamination_level=0.3,
            vlm_model="test",
            vlm_processing_time=1.0,
        )
        assert narrator._validate_metadata_content(invalid3) is False

    @patch("imageworks.apps.color_narrator.core.vlm.VLMClient.infer_single")
    @patch(
        "imageworks.apps.color_narrator.core.metadata.XMPMetadataWriter.write_metadata"
    )
    def test_process_batch_success(
        self, mock_write_metadata, mock_infer, sample_config
    ):
        """Test successful batch processing."""
        # Setup mocks
        mock_vlm_response = VLMResponse(
            description="Color found in image",
            confidence=0.8,
            color_regions=["background"],
            processing_time=1.0,
        )
        mock_infer.return_value = mock_vlm_response
        mock_write_metadata.return_value = True

        # Create test item
        test_item = ColorNarratorItem(
            image_path=Path("/test/image.jpg"),
            overlay_path=Path("/test/overlay.png"),
            mono_data={"contamination_level": 0.5},
        )

        # Set non-dry-run mode for this test
        sample_config.dry_run = False
        narrator = ColorNarrator(sample_config)

        results = narrator._process_batch([test_item])

        assert len(results) == 1
        assert results[0].vlm_response == mock_vlm_response
        assert results[0].metadata_written is True
        assert results[0].error is None

    @patch("imageworks.apps.color_narrator.core.vlm.VLMClient.infer_single")
    @patch(
        "imageworks.apps.color_narrator.core.metadata.XMPMetadataWriter.write_metadata"
    )
    def test_process_batch_metadata_write_failure(
        self, mock_write_metadata, mock_infer, sample_config
    ):
        """Test batch processing with metadata write failure."""
        # Setup mocks
        mock_vlm_response = VLMResponse(
            description="Color found",
            confidence=0.8,
            color_regions=[],
            processing_time=1.0,
        )
        mock_infer.return_value = mock_vlm_response
        mock_write_metadata.side_effect = Exception("Write failed")

        # Create test item
        test_item = ColorNarratorItem(
            image_path=Path("/test/image.jpg"),
            overlay_path=Path("/test/overlay.png"),
            mono_data={},
        )

        sample_config.dry_run = False
        narrator = ColorNarrator(sample_config)

        results = narrator._process_batch([test_item])

        assert len(results) == 1
        assert results[0].metadata_written is False
        assert results[0].error is not None
        assert "Metadata write error" in results[0].error

    def test_process_batch_skip_existing_xmp(self, sample_config):
        """Test batch processing skips items with existing XMP when not overwriting."""
        test_item = ColorNarratorItem(
            image_path=Path("/test/image.jpg"),
            overlay_path=Path("/test/overlay.png"),
            mono_data={},
            has_existing_xmp=True,
        )

        # Ensure overwrite_existing is False
        sample_config.overwrite_existing = False
        narrator = ColorNarrator(sample_config)

        results = narrator._process_batch([test_item])

        # Should return empty results since item was skipped
        assert len(results) == 0
