"""Integration tests for color narrator using shared test assets.

These tests use real competition images and mono analysis results
to validate end-to-end color narrator functionality.
"""

import pytest
from pathlib import Path

# Shared test asset locations
SHARED_IMAGES = Path("tests/shared/sample_production_images")
SHARED_OVERLAYS = SHARED_IMAGES
SHARED_DATA = Path("tests/shared/sample_production_mono_json_output")
TEST_OUTPUT = Path("tests/test_output")


class TestColorNarratorIntegration:
    """Integration tests for color narrator with real images."""

    @pytest.fixture(autouse=True)
    def setup_test_output(self):
        """Ensure test output directory exists."""
        TEST_OUTPUT.mkdir(exist_ok=True)

    @pytest.mark.skipif(
        not SHARED_IMAGES.exists(), reason="Shared test images not available"
    )
    def test_enhance_mono_production_images(self):
        """Test enhance-mono command with real competition images."""
        # Skip if shared assets not available (CI environments)
        if not any(SHARED_IMAGES.glob("*.jpg")):
            pytest.skip("Sample production images not available")

        # TODO: Test enhance-mono command with production images
        # Use TEST_OUTPUT for results to avoid overwriting production summaries
        pass

    @pytest.mark.skipif(
        not SHARED_DATA.exists(), reason="Shared test data not available"
    )
    def test_vlm_analysis_with_regions(self):
        """Test VLM analysis with grid regions using sample data."""
        sample_data = SHARED_DATA / "production_sample.jsonl"
        if not sample_data.exists():
            pytest.skip("Sample data not available")

        # TODO: Test VLM analysis with regions enabled
        # Test both --regions and --no-regions modes
        pass

    def test_prompt_template_switching(self):
        """Test different prompt template versions."""
        # TODO: Test --prompt v6/v5/v4/legacy options
        # Verify outputs differ appropriately
        pass

    def test_output_to_test_directory(self):
        """Test that color narrator output can be directed to test directory."""
        # TODO: Run enhance-mono with output to test directory
        # test_summary = TEST_OUTPUT / "test_enhancement_summary.md"
        # Verify test output doesn't overwrite production summaries
        pass
