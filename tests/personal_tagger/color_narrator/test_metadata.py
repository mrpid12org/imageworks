"""Tests for XMP metadata reading and writing functionality.

Tests metadata dataclass creation, XMP embedding simulation, sidecar JSON handling,
batch operations, and validation logic for color narrator metadata.
"""

import pytest
from unittest.mock import patch
from pathlib import Path
import json
import tempfile
from datetime import datetime

from imageworks.apps.personal_tagger.color_narrator.core.metadata import (
    XMPMetadataWriter,
    ColorNarrationMetadata,
    XMPMetadataBatch,
)


class TestColorNarrationMetadata:
    """Test cases for ColorNarrationMetadata dataclass."""

    def test_metadata_creation(self):
        """Test ColorNarrationMetadata creation with required fields."""
        metadata = ColorNarrationMetadata(
            description="Test color description",
            confidence_score=0.85,
            color_regions=["background", "skin"],
            processing_timestamp="2024-01-15T10:30:00",
            mono_contamination_level=0.3,
            vlm_model="Qwen/Qwen2-VL-7B-Instruct",
            vlm_processing_time=1.5,
        )

        assert metadata.description == "Test color description"
        assert metadata.confidence_score == 0.85
        assert metadata.color_regions == ["background", "skin"]
        assert metadata.processing_timestamp == "2024-01-15T10:30:00"
        assert metadata.mono_contamination_level == 0.3
        assert metadata.vlm_model == "Qwen/Qwen2-VL-7B-Instruct"
        assert metadata.vlm_processing_time == 1.5

    def test_metadata_optional_fields(self):
        """Test ColorNarrationMetadata with optional fields."""
        metadata = ColorNarrationMetadata(
            description="Test description",
            confidence_score=0.8,
            color_regions=[],
            processing_timestamp="2024-01-15T10:30:00",
            mono_contamination_level=0.2,
            vlm_model="test-model",
            vlm_processing_time=1.0,
            hue_analysis="Warm cast",
            chroma_analysis="Low chroma",
            validation_status="Validated",
        )

        assert metadata.hue_analysis == "Warm cast"
        assert metadata.chroma_analysis == "Low chroma"
        assert metadata.validation_status == "Validated"

    def test_metadata_defaults(self):
        """Test ColorNarrationMetadata default values for optional fields."""
        metadata = ColorNarrationMetadata(
            description="Test",
            confidence_score=0.5,
            color_regions=[],
            processing_timestamp="2024-01-15T10:30:00",
            mono_contamination_level=0.1,
            vlm_model="test",
            vlm_processing_time=0.5,
        )

        assert metadata.hue_analysis is None
        assert metadata.chroma_analysis is None
        assert metadata.validation_status is None


class TestXMPMetadataWriter:
    """Test cases for XMP metadata writer."""

    @pytest.fixture
    def temp_image_file(self):
        """Create a temporary image file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"fake_image_data")
            temp_path = Path(f.name)

        yield temp_path

        # Cleanup
        if temp_path.exists():
            temp_path.unlink()

        # Also cleanup potential sidecar files
        sidecar_path = temp_path.with_suffix(f"{temp_path.suffix}.cn_metadata.json")
        if sidecar_path.exists():
            sidecar_path.unlink()

    @pytest.fixture
    def sample_metadata(self):
        """Create sample metadata for testing."""
        return ColorNarrationMetadata(
            description="Sample color description",
            confidence_score=0.9,
            color_regions=["background", "foreground"],
            processing_timestamp=datetime.now().isoformat(),
            mono_contamination_level=0.4,
            vlm_model="Qwen/Qwen2-VL-7B-Instruct",
            vlm_processing_time=2.1,
        )

    def test_writer_initialization(self):
        """Test XMP metadata writer initialization."""
        writer = XMPMetadataWriter()
        assert writer.backup_original is True

        writer = XMPMetadataWriter(backup_original=False)
        assert writer.backup_original is False

    def test_write_metadata_success(self, temp_image_file, sample_metadata):
        """Test successful metadata writing (sidecar implementation)."""
        writer = XMPMetadataWriter(backup_original=False)

        result = writer.write_metadata(temp_image_file, sample_metadata)

        assert result is True

        # Check sidecar file was created
        sidecar_path = writer._get_sidecar_path(temp_image_file)
        assert sidecar_path.exists()

        # Verify sidecar content
        with open(sidecar_path) as f:
            data = json.load(f)

        assert data["description"] == "Sample color description"
        assert data["confidence_score"] == 0.9
        assert data["color_regions"] == ["background", "foreground"]
        assert data["source_file"] == temp_image_file.name
        assert data["metadata_version"] == "1.0"

    def test_read_metadata_success(self, temp_image_file, sample_metadata):
        """Test successful metadata reading (sidecar implementation)."""
        writer = XMPMetadataWriter(backup_original=False)

        # Write metadata first
        writer.write_metadata(temp_image_file, sample_metadata)

        # Read it back
        read_metadata = writer.read_metadata(temp_image_file)

        assert read_metadata is not None
        assert read_metadata.description == sample_metadata.description
        assert read_metadata.confidence_score == sample_metadata.confidence_score
        assert read_metadata.color_regions == sample_metadata.color_regions

    def test_read_metadata_not_found(self, temp_image_file):
        """Test reading metadata when none exists."""
        writer = XMPMetadataWriter()

        metadata = writer.read_metadata(temp_image_file)

        assert metadata is None

    def test_has_color_narration(self, temp_image_file, sample_metadata):
        """Test checking for existing color narration metadata."""
        writer = XMPMetadataWriter(backup_original=False)

        # Initially should not have metadata
        assert writer.has_color_narration(temp_image_file) is False

        # After writing, should have metadata
        writer.write_metadata(temp_image_file, sample_metadata)
        assert writer.has_color_narration(temp_image_file) is True

    def test_remove_metadata(self, temp_image_file, sample_metadata):
        """Test metadata removal."""
        writer = XMPMetadataWriter(backup_original=False)

        # Write metadata first
        writer.write_metadata(temp_image_file, sample_metadata)
        assert writer.has_color_narration(temp_image_file) is True

        # Remove metadata
        result = writer.remove_metadata(temp_image_file)

        assert result is True
        assert writer.has_color_narration(temp_image_file) is False

    def test_backup_file_creation(self, temp_image_file):
        """Test backup file creation."""
        writer = XMPMetadataWriter(backup_original=True)

        backup_path = writer._backup_file(temp_image_file)

        assert backup_path.exists()
        assert backup_path.name.startswith(temp_image_file.stem)
        assert "backup_" in backup_path.name

        # Verify backup content matches original
        assert backup_path.read_bytes() == temp_image_file.read_bytes()

        # Cleanup
        backup_path.unlink()

    def test_get_sidecar_path(self, temp_image_file):
        """Test sidecar path generation."""
        writer = XMPMetadataWriter()

        sidecar_path = writer._get_sidecar_path(temp_image_file)

        expected = temp_image_file.with_suffix(
            f"{temp_image_file.suffix}.cn_metadata.json"
        )
        assert sidecar_path == expected

    def test_read_malformed_sidecar(self, temp_image_file):
        """Test reading malformed sidecar JSON."""
        writer = XMPMetadataWriter()
        sidecar_path = writer._get_sidecar_path(temp_image_file)

        # Create malformed JSON
        sidecar_path.write_text("invalid json content")

        metadata = writer.read_metadata(temp_image_file)

        assert metadata is None

    @patch.object(
        XMPMetadataWriter, "_write_sidecar_json", side_effect=Exception("Write failed")
    )
    def test_write_metadata_failure(self, mock_write, temp_image_file, sample_metadata):
        """Test metadata write failure handling."""
        writer = XMPMetadataWriter()

        result = writer.write_metadata(temp_image_file, sample_metadata)

        assert result is False


class TestXMPMetadataBatch:
    """Test cases for batch XMP metadata operations."""

    @pytest.fixture
    def temp_files_and_metadata(self):
        """Create multiple temporary files with sample metadata."""
        files = []
        metadata_list = []

        for i in range(3):
            # Create temp file
            with tempfile.NamedTemporaryFile(suffix=f"_test{i}.jpg", delete=False) as f:
                f.write(b"fake_image_data")
                temp_path = Path(f.name)
                files.append(temp_path)

            # Create metadata
            metadata = ColorNarrationMetadata(
                description=f"Description {i}",
                confidence_score=0.8 + (i * 0.05),
                color_regions=[f"region{i}"],
                processing_timestamp=datetime.now().isoformat(),
                mono_contamination_level=0.2 + (i * 0.1),
                vlm_model="test-model",
                vlm_processing_time=1.0 + i,
            )
            metadata_list.append(metadata)

        yield list(zip(files, metadata_list))

        # Cleanup
        for file_path, _ in zip(files, metadata_list):
            if file_path.exists():
                file_path.unlink()

            # Cleanup sidecar files
            writer = XMPMetadataWriter()
            sidecar_path = writer._get_sidecar_path(file_path)
            if sidecar_path.exists():
                sidecar_path.unlink()

    def test_batch_write_success(self, temp_files_and_metadata):
        """Test successful batch metadata writing."""
        writer = XMPMetadataWriter(backup_original=False)
        batch = XMPMetadataBatch(writer)

        results = batch.batch_write(temp_files_and_metadata)

        assert results["total"] == 3
        assert results["successful"] == 3
        assert results["failed"] == 0
        assert len(results["errors"]) == 0

        # Verify all files have metadata
        for file_path, _ in temp_files_and_metadata:
            assert writer.has_color_narration(file_path)

    def test_batch_write_partial_failure(self, temp_files_and_metadata):
        """Test batch write with some failures."""
        writer = XMPMetadataWriter(backup_original=False)
        batch = XMPMetadataBatch(writer)

        # Make one file path invalid to cause failure
        temp_files_and_metadata[1] = (
            Path("/invalid/path.jpg"),
            temp_files_and_metadata[1][1],
        )

        results = batch.batch_write(temp_files_and_metadata)

        assert results["total"] == 3
        assert results["successful"] == 2
        assert results["failed"] == 1
        assert len(results["errors"]) == 1

    def test_batch_read_success(self, temp_files_and_metadata):
        """Test successful batch metadata reading."""
        writer = XMPMetadataWriter(backup_original=False)
        batch = XMPMetadataBatch(writer)

        # Write metadata first
        batch.batch_write(temp_files_and_metadata)

        # Read back
        file_paths = [fp for fp, _ in temp_files_and_metadata]
        results = batch.batch_read(file_paths)

        assert len(results) == 3
        for file_path in file_paths:
            assert file_path in results
            assert results[file_path] is not None
            assert isinstance(results[file_path], ColorNarrationMetadata)

    def test_batch_read_missing_files(self):
        """Test batch read with missing files."""
        writer = XMPMetadataWriter()
        batch = XMPMetadataBatch(writer)

        file_paths = [Path("/nonexistent1.jpg"), Path("/nonexistent2.jpg")]
        results = batch.batch_read(file_paths)

        assert len(results) == 2
        for file_path in file_paths:
            assert results[file_path] is None
