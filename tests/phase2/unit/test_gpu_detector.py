"""
Unit tests for GPU detection module
"""

from unittest.mock import patch, MagicMock
from imageworks.libs.hardware.gpu_detector import GPUDetector, GPUInfo


class TestGPUInfo:
    """Test GPUInfo dataclass."""

    def test_gpu_info_creation(self):
        """Test creating GPUInfo instance."""
        gpu = GPUInfo(
            index=0,
            name="NVIDIA RTX 4080",
            vram_total_mb=16384,
            vram_free_mb=15000,
            compute_capability=(8, 9),
            uuid="GPU-12345678",
        )

        assert gpu.index == 0
        assert gpu.name == "NVIDIA RTX 4080"
        assert gpu.vram_total_mb == 16384
        assert gpu.vram_free_mb == 15000
        assert gpu.compute_capability == (8, 9)
        assert gpu.uuid == "GPU-12345678"


class TestGPUDetector:
    """Test GPUDetector class."""

    @patch("subprocess.run")
    def test_detect_single_gpu(self, mock_run):
        """Test detecting a single GPU."""
        # Mock nvidia-smi output with correct format
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="0, NVIDIA GeForce RTX 4080, 16384, 15000, 8.9, GPU-12345678\n",
        )

        detector = GPUDetector()
        gpus = detector.detect_gpus()

        assert len(gpus) == 1
        assert gpus[0].index == 0
        assert gpus[0].name == "NVIDIA GeForce RTX 4080"
        assert gpus[0].vram_total_mb == 16384
        assert gpus[0].vram_free_mb == 15000
        assert gpus[0].compute_capability == (8, 9)
        assert gpus[0].uuid == "GPU-12345678"

    @patch("subprocess.run")
    def test_detect_multiple_gpus(self, mock_run):
        """Test detecting multiple GPUs."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=(
                "0, NVIDIA GeForce RTX 4080, 16384, 15000, 8.9, GPU-12345678\n"
                "1, NVIDIA GeForce RTX 4080, 16384, 14500, 8.9, GPU-87654321\n"
            ),
        )

        detector = GPUDetector()
        gpus = detector.detect_gpus()

        assert len(gpus) == 2
        assert gpus[0].index == 0
        assert gpus[1].index == 1
        assert gpus[1].vram_free_mb == 14500

    @patch("subprocess.run")
    def test_detect_no_gpus(self, mock_run):
        """Test when no GPUs are detected."""
        mock_run.return_value = MagicMock(returncode=0, stdout="")

        detector = GPUDetector()
        gpus = detector.detect_gpus()

        assert len(gpus) == 0

    @patch("subprocess.run")
    def test_detect_nvidia_smi_failure(self, mock_run):
        """Test handling nvidia-smi failure."""
        mock_run.side_effect = FileNotFoundError("nvidia-smi not found")

        detector = GPUDetector()
        gpus = detector.detect_gpus()

        assert len(gpus) == 0

    @patch("subprocess.run")
    def test_caching_behavior(self, mock_run):
        """Test that GPU detection results are cached."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="0, NVIDIA GeForce RTX 4080, 16384, 15000, 8.9, GPU-12345678\n",
        )

        detector = GPUDetector()

        # First call should invoke nvidia-smi
        gpus1 = detector.detect_gpus()
        assert mock_run.call_count == 1

        # Second call should use cached result
        gpus2 = detector.detect_gpus()
        assert mock_run.call_count == 1  # Still 1, not 2

        # Results should be identical
        assert len(gpus1) == len(gpus2)
        assert gpus1[0].name == gpus2[0].name

    @patch("subprocess.run")
    def test_clear_cache(self, mock_run):
        """Test that clear_cache() clears the cache."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="0, NVIDIA GeForce RTX 4080, 16384, 15000, 8.9, GPU-12345678\n",
        )

        detector = GPUDetector()

        # First call
        detector.detect_gpus()
        assert mock_run.call_count == 1

        # Clear cache
        detector.clear_cache()

        # Next call should re-query
        detector.detect_gpus()
        assert mock_run.call_count == 2

    @patch("subprocess.run")
    def test_get_usable_vram_single_gpu(self, mock_run):
        """Test usable VRAM calculation for single GPU."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="0, NVIDIA GeForce RTX 4080, 16384, 15000, 8.9, GPU-12345678\n",
        )

        detector = GPUDetector()
        usable_vram = detector.get_usable_vram_mb()

        # Should be total - headroom (16384 - 2048)
        assert usable_vram == 14336

    @patch("subprocess.run")
    def test_get_usable_vram_multiple_gpus(self, mock_run):
        """Test usable VRAM calculation for multiple GPUs."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=(
                "0, NVIDIA GeForce RTX 4080, 16384, 15000, 8.9, GPU-12345678\n"
                "1, NVIDIA GeForce RTX 4080, 16384, 14000, 8.9, GPU-87654321\n"
            ),
        )

        detector = GPUDetector()
        usable_vram = detector.get_usable_vram_mb()

        # Should be total - (headroom * gpu_count) = 32768 - 4096
        assert usable_vram == 28672

    @patch("subprocess.run")
    def test_recommend_profile_16gb_single(self, mock_run):
        """Test profile recommendation for 16GB single GPU."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="0, NVIDIA GeForce RTX 4080, 16384, 15000, 8.9, GPU-12345678\n",
        )

        detector = GPUDetector()
        profile = detector.recommend_profile()

        assert profile == "constrained_16gb"

    @patch("subprocess.run")
    def test_recommend_profile_24gb_single(self, mock_run):
        """Test profile recommendation for 24GB single GPU."""
        mock_run.return_value = MagicMock(
            returncode=0, stdout="0, NVIDIA RTX 4090, 24576, 22000, 8.9, GPU-12345678\n"
        )

        detector = GPUDetector()
        profile = detector.recommend_profile()

        assert profile == "balanced_24gb"

    @patch("subprocess.run")
    def test_recommend_profile_multi_gpu(self, mock_run):
        """Test profile recommendation for multiple 16GB GPUs."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=(
                "0, NVIDIA GeForce RTX 4080, 16384, 15000, 8.9, GPU-12345678\n"
                "1, NVIDIA GeForce RTX 4080, 16384, 14000, 8.9, GPU-87654321\n"
            ),
        )

        detector = GPUDetector()
        profile = detector.recommend_profile()

        assert profile == "multi_gpu_2x16gb"

    @patch("subprocess.run")
    def test_recommend_profile_96gb_single(self, mock_run):
        """Test profile recommendation for 96GB single GPU."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="0, NVIDIA RTX 6000 Ada, 98304, 95000, 8.9, GPU-12345678\n",
        )

        detector = GPUDetector()
        profile = detector.recommend_profile()

        assert profile == "generous_96gb"

    @patch("subprocess.run")
    def test_recommend_profile_no_gpus(self, mock_run):
        """Test profile recommendation when no GPUs detected."""
        mock_run.return_value = MagicMock(returncode=0, stdout="")

        detector = GPUDetector()
        profile = detector.recommend_profile()

        assert profile == "development"

    @patch("subprocess.run")
    def test_malformed_nvidia_smi_output(self, mock_run):
        """Test handling of malformed nvidia-smi output."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="0, NVIDIA RTX 4080, not_a_number, 15000, 8.9\n",  # Missing UUID field
        )

        detector = GPUDetector()
        gpus = detector.detect_gpus()

        # Should skip malformed line
        assert len(gpus) == 0
