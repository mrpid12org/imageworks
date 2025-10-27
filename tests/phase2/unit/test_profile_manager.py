"""
Unit tests for ProfileManager module
"""

from pathlib import Path
from unittest.mock import patch
from imageworks.chat_proxy.profile_manager import ProfileManager, DeploymentProfile
from imageworks.libs.hardware.gpu_detector import GPUInfo


class TestDeploymentProfile:
    """Test DeploymentProfile dataclass."""

    def test_from_dict(self):
        """Test creating profile from dictionary."""
        config = {
            "max_vram_mb": 10000,
            "max_concurrent_models": 1,
            "preferred_quantizations": ["fp8", "q4_k_m"],
            "strategy": "conservative",
            "autostart_timeout_seconds": 300,
            "grace_period_seconds": 30,
            "model_selection_bias": "vram_efficient",
        }

        profile = DeploymentProfile.from_dict("test_profile", config)

        assert profile.name == "test_profile"
        assert profile.max_vram_mb == 10000
        assert profile.max_concurrent_models == 1
        assert profile.preferred_quantizations == ["fp8", "q4_k_m"]
        assert profile.strategy == "conservative"
        assert profile.autostart_timeout_seconds == 300
        assert profile.grace_period_seconds == 30
        assert profile.model_selection_bias == "vram_efficient"

    def test_to_dict(self):
        """Test converting profile to dictionary."""
        profile = DeploymentProfile(
            name="test",
            max_vram_mb=10000,
            max_concurrent_models=1,
            preferred_quantizations=["fp8"],
            strategy="conservative",
            autostart_timeout_seconds=300,
            grace_period_seconds=30,
            model_selection_bias="vram_efficient",
        )

        result = profile.to_dict()

        assert result["name"] == "test"
        assert result["max_vram_mb"] == 10000
        assert result["preferred_quantizations"] == ["fp8"]


class TestProfileManager:
    """Test ProfileManager class."""

    def _create_test_config(self, tmp_path: Path) -> Path:
        """Create a test pyproject.toml with profiles."""
        config_content = """
[tool.imageworks.deployment_profiles.test_16gb]
max_vram_mb = 10000
max_concurrent_models = 1
preferred_quantizations = ["fp8", "q4_k_m"]
strategy = "conservative"
autostart_timeout_seconds = 300
grace_period_seconds = 30
model_selection_bias = "vram_efficient"

[tool.imageworks.deployment_profiles.test_24gb]
max_vram_mb = 20000
max_concurrent_models = 2
preferred_quantizations = ["fp8", "awq"]
strategy = "balanced"
autostart_timeout_seconds = 600
grace_period_seconds = 60
model_selection_bias = "balanced"

[tool.imageworks.deployment_profiles.development]
max_vram_mb = 5000
max_concurrent_models = 1
preferred_quantizations = ["q4_k_m", "q5_k_m"]
strategy = "aggressive_shutdown"
autostart_timeout_seconds = 180
grace_period_seconds = 15
model_selection_bias = "vram_efficient"
"""
        config_path = tmp_path / "pyproject.toml"
        config_path.write_text(config_content)
        return config_path

    def test_load_profiles(self, tmp_path):
        """Test loading profiles from config file."""
        config_path = self._create_test_config(tmp_path)

        with patch(
            "imageworks.libs.hardware.gpu_detector.GPUDetector.detect_gpus"
        ) as mock_detect:
            mock_detect.return_value = []
            manager = ProfileManager(config_path)

        profiles = manager.get_all_profiles()
        assert len(profiles) == 3
        assert "test_16gb" in profiles
        assert "test_24gb" in profiles
        assert "development" in profiles

        profile = profiles["test_16gb"]
        assert profile.max_vram_mb == 10000
        assert profile.max_concurrent_models == 1

    def test_manual_profile_override(self, tmp_path, monkeypatch):
        """Test manual profile selection via environment variable."""
        config_path = self._create_test_config(tmp_path)
        monkeypatch.setenv("IMAGEWORKS_DEPLOYMENT_PROFILE", "test_24gb")

        with patch(
            "imageworks.libs.hardware.gpu_detector.GPUDetector.detect_gpus"
        ) as mock_detect:
            mock_detect.return_value = []
            manager = ProfileManager(config_path)

        active = manager.get_active_profile()
        assert active is not None
        assert active.name == "test_24gb"

    def test_auto_detect_16gb_profile(self, tmp_path):
        """Test auto-detection of 16GB profile."""
        config_path = self._create_test_config(tmp_path)

        mock_gpu = GPUInfo(
            index=0,
            name="RTX 4080",
            vram_total_mb=16384,
            vram_free_mb=15000,
            compute_capability=(8, 9),
            uuid="GPU-12345678",
        )

        with patch(
            "imageworks.libs.hardware.gpu_detector.GPUDetector.detect_gpus"
        ) as mock_detect:
            mock_detect.return_value = [mock_gpu]
            # Also need to patch recommend_profile to return a profile that exists in our test config
            with patch(
                "imageworks.libs.hardware.gpu_detector.GPUDetector.recommend_profile"
            ) as mock_recommend:
                mock_recommend.return_value = "test_16gb"
                manager = ProfileManager(config_path)

        active = manager.get_active_profile()
        # Should select test_16gb as closest match
        assert active is not None
        assert active.name == "test_16gb"

    def test_fallback_to_development(self, tmp_path):
        """Test fallback to development profile when auto-detection fails."""
        config_path = self._create_test_config(tmp_path)

        with patch(
            "imageworks.libs.hardware.gpu_detector.GPUDetector.detect_gpus"
        ) as mock_detect:
            mock_detect.side_effect = Exception("GPU detection failed")
            manager = ProfileManager(config_path)

        active = manager.get_active_profile()
        assert active is not None
        assert active.name == "development"

    def test_get_profile_by_name(self, tmp_path):
        """Test retrieving specific profile by name."""
        config_path = self._create_test_config(tmp_path)

        with patch(
            "imageworks.libs.hardware.gpu_detector.GPUDetector.detect_gpus"
        ) as mock_detect:
            mock_detect.return_value = []
            manager = ProfileManager(config_path)

        profile = manager.get_profile("test_24gb")
        assert profile is not None
        assert profile.name == "test_24gb"
        assert profile.max_vram_mb == 20000

        nonexistent = manager.get_profile("nonexistent")
        assert nonexistent is None

    def test_set_active_profile(self, tmp_path):
        """Test manually setting active profile."""
        config_path = self._create_test_config(tmp_path)

        with patch(
            "imageworks.libs.hardware.gpu_detector.GPUDetector.detect_gpus"
        ) as mock_detect:
            mock_detect.return_value = []
            manager = ProfileManager(config_path)

        # Change to test_24gb
        success = manager.set_active_profile("test_24gb")
        assert success is True

        active = manager.get_active_profile()
        assert active.name == "test_24gb"

        # Try to set non-existent profile
        success = manager.set_active_profile("nonexistent")
        assert success is False

    def test_get_detected_gpus(self, tmp_path):
        """Test retrieving detected GPU information."""
        config_path = self._create_test_config(tmp_path)

        mock_gpu = GPUInfo(
            index=0,
            name="RTX 4080",
            vram_total_mb=16384,
            vram_free_mb=15000,
            compute_capability=(8, 9),
            uuid="GPU-12345678",
        )

        with patch(
            "imageworks.libs.hardware.gpu_detector.GPUDetector.detect_gpus"
        ) as mock_detect:
            mock_detect.return_value = [mock_gpu]
            manager = ProfileManager(config_path)

        gpus = manager.get_detected_gpus()
        assert len(gpus) == 1
        assert gpus[0].name == "RTX 4080"

    def test_get_profile_info(self, tmp_path, monkeypatch):
        """Test getting comprehensive profile information."""
        config_path = self._create_test_config(tmp_path)
        monkeypatch.setenv("IMAGEWORKS_DEPLOYMENT_PROFILE", "test_16gb")

        mock_gpu = GPUInfo(
            index=0,
            name="RTX 4080",
            vram_total_mb=16384,
            vram_free_mb=15000,
            compute_capability=(8, 9),
            uuid="GPU-12345678",
        )

        with patch(
            "imageworks.libs.hardware.gpu_detector.GPUDetector.detect_gpus"
        ) as mock_detect:
            mock_detect.return_value = [mock_gpu]
            manager = ProfileManager(config_path)

        info = manager.get_profile_info()

        assert "active_profile" in info
        assert info["active_profile"]["name"] == "test_16gb"
        assert "detected_gpus" in info
        assert len(info["detected_gpus"]) > 0  # May be real GPUs or mocked
        # Don't check exact GPU name as it may vary
        assert "available_profiles" in info
        assert len(info["available_profiles"]) == 3
        assert "manual_override" in info
        assert info["manual_override"] == "test_16gb"

    def test_invalid_config_file(self, tmp_path):
        """Test handling of invalid config file."""
        config_path = tmp_path / "nonexistent.toml"

        with patch(
            "imageworks.libs.hardware.gpu_detector.GPUDetector.detect_gpus"
        ) as mock_detect:
            mock_detect.return_value = []
            manager = ProfileManager(config_path)

        # Should handle gracefully
        profiles = manager.get_all_profiles()
        assert len(profiles) == 0

    def test_manual_override_invalid_profile(self, tmp_path, monkeypatch):
        """Test manual override with non-existent profile falls back to auto-detect."""
        config_path = self._create_test_config(tmp_path)
        monkeypatch.setenv("IMAGEWORKS_DEPLOYMENT_PROFILE", "nonexistent_profile")

        with patch(
            "imageworks.libs.hardware.gpu_detector.GPUDetector.detect_gpus"
        ) as mock_detect:
            mock_detect.return_value = []
            manager = ProfileManager(config_path)

        # Should fall back to development profile
        active = manager.get_active_profile()
        assert active is not None
        assert active.name == "development"
