"""Integration tests for mono checker using shared test assets.

These tests use real competition images from tests/shared/ directory
to validate end-to-end mono checker functionality.
"""

import pytest
from pathlib import Path

# Shared test asset locations
SHARED_IMAGES = Path("tests/shared/images")
SHARED_OVERLAYS = Path("tests/shared/overlays")
SHARED_DATA = Path("tests/shared/sample_data")
TEST_OUTPUT = Path("tests/test_output")


class TestMonoIntegration:
    """Integration tests for mono checker with real images."""

    @pytest.fixture(autouse=True)
    def setup_test_output(self):
        """Ensure test output directory exists."""
        TEST_OUTPUT.mkdir(exist_ok=True)

    @pytest.mark.skipif(
        not SHARED_IMAGES.exists(), reason="Shared test images not available"
    )
    def test_mono_analysis_production_images(self):
        """Test mono analysis with real competition images."""
        # Skip if shared assets not available (CI environments)
        if not (SHARED_IMAGES / "production_images").exists():
            pytest.skip("Production test images not available")

        # TODO: Implement integration test using shared production images
        # This would test the full mono analysis pipeline
        pass

    @pytest.mark.skipif(
        not SHARED_DATA.exists(), reason="Shared test data not available"
    )
    def test_mono_with_sample_data(self):
        """Test mono checker with shared sample data."""
        # TODO: Implement test using production_sample.jsonl from shared data
        pass

    def test_mono_output_to_test_directory(self):
        """Test that mono output can be directed to test directory."""
        # TODO: Run mono analysis with output to test directory
        # output_file = TEST_OUTPUT / "test_mono_results.jsonl"
        # Verify output doesn't interfere with production outputs/
        pass
