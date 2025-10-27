"""
Integration tests for Phase 2: Deployment Profiles and Role-Based Model Selection

Tests end-to-end functionality of:
- GPU detection
- Profile auto-detection and manual override
- Role-based model selection
- API endpoints for profile and role selection
"""

import pytest
import json
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient

# Import after ensuring imageworks package is available
from imageworks.libs.hardware.gpu_detector import GPUInfo
from imageworks.chat_proxy.profile_manager import ProfileManager
from imageworks.chat_proxy.role_selector import RoleSelector


class TestPhase2Integration:
    """Integration tests for Phase 2 functionality."""

    @pytest.fixture
    def workspace_root(self):
        """Get workspace root directory."""
        return Path(__file__).parent.parent.parent

    @pytest.fixture
    def pyproject_path(self, workspace_root):
        """Get path to pyproject.toml."""
        return workspace_root / "pyproject.toml"

    @pytest.fixture
    def registry_path(self, workspace_root):
        """Get path to curated model registry."""
        return workspace_root / "configs" / "model_registry.curated.json"

    def test_pyproject_has_deployment_profiles(self, pyproject_path):
        """Test that pyproject.toml contains deployment profile definitions."""
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib

        with open(pyproject_path, "rb") as f:
            config = tomllib.load(f)

        profiles = (
            config.get("tool", {}).get("imageworks", {}).get("deployment_profiles", {})
        )

        assert len(profiles) > 0, "No deployment profiles found in pyproject.toml"

        # Check for expected profiles
        expected_profiles = [
            "constrained_16gb",
            "balanced_24gb",
            "multi_gpu_2x16gb",
            "generous_96gb",
            "development",
        ]
        for profile_name in expected_profiles:
            assert profile_name in profiles, f"Profile '{profile_name}' not found"

            # Check required fields
            profile = profiles[profile_name]
            assert "max_vram_mb" in profile
            assert "max_concurrent_models" in profile
            assert "preferred_quantizations" in profile
            assert "strategy" in profile
            assert "timeout_seconds" in profile
            assert "grace_period_seconds" in profile
            assert "model_selection_bias" in profile

    def test_profile_manager_initialization(self, pyproject_path):
        """Test that ProfileManager initializes successfully."""
        with patch(
            "imageworks.libs.hardware.gpu_detector.GPUDetector.detect_gpus"
        ) as mock_detect:
            mock_detect.return_value = []

            manager = ProfileManager(pyproject_path)

            assert manager is not None
            profiles = manager.get_all_profiles()
            assert len(profiles) >= 5

    def test_profile_auto_detection_16gb(self, pyproject_path):
        """Test auto-detection for 16GB GPU."""
        mock_gpu = GPUInfo(
            index=0,
            name="NVIDIA RTX 4080",
            total_vram_mb=16384,
            free_vram_mb=15000,
            driver_version="535.54",
        )

        with patch(
            "imageworks.libs.hardware.gpu_detector.GPUDetector.detect_gpus"
        ) as mock_detect:
            mock_detect.return_value = [mock_gpu]

            manager = ProfileManager(pyproject_path)
            active = manager.get_active_profile()

            assert active is not None
            assert active.name == "constrained_16gb"
            assert active.max_vram_mb == 10000

    def test_profile_auto_detection_24gb(self, pyproject_path):
        """Test auto-detection for 24GB GPU."""
        mock_gpu = GPUInfo(
            index=0,
            name="NVIDIA RTX 4090",
            total_vram_mb=24576,
            free_vram_mb=23000,
            driver_version="535.54",
        )

        with patch(
            "imageworks.libs.hardware.gpu_detector.GPUDetector.detect_gpus"
        ) as mock_detect:
            mock_detect.return_value = [mock_gpu]

            manager = ProfileManager(pyproject_path)
            active = manager.get_active_profile()

            assert active is not None
            assert active.name == "balanced_24gb"

    def test_profile_auto_detection_multi_gpu(self, pyproject_path):
        """Test auto-detection for multiple 16GB GPUs."""
        mock_gpus = [
            GPUInfo(0, "NVIDIA RTX 4080", 16384, 15000, "535.54"),
            GPUInfo(1, "NVIDIA RTX 4080", 16384, 14500, "535.54"),
        ]

        with patch(
            "imageworks.libs.hardware.gpu_detector.GPUDetector.detect_gpus"
        ) as mock_detect:
            mock_detect.return_value = mock_gpus

            manager = ProfileManager(pyproject_path)
            active = manager.get_active_profile()

            assert active is not None
            assert active.name == "multi_gpu_2x16gb"

    def test_profile_auto_detection_96gb(self, pyproject_path):
        """Test auto-detection for 96GB GPU."""
        mock_gpu = GPUInfo(
            index=0,
            name="NVIDIA RTX 6000 Ada",
            total_vram_mb=98304,
            free_vram_mb=95000,
            driver_version="535.54",
        )

        with patch(
            "imageworks.libs.hardware.gpu_detector.GPUDetector.detect_gpus"
        ) as mock_detect:
            mock_detect.return_value = [mock_gpu]

            manager = ProfileManager(pyproject_path)
            active = manager.get_active_profile()

            assert active is not None
            assert active.name == "generous_96gb"

    def test_profile_manual_override(self, pyproject_path, monkeypatch):
        """Test manual profile override via environment variable."""
        monkeypatch.setenv("IMAGEWORKS_DEPLOYMENT_PROFILE", "development")

        # Even with 16GB GPU, should use development profile
        mock_gpu = GPUInfo(0, "NVIDIA RTX 4080", 16384, 15000, "535.54")

        with patch(
            "imageworks.libs.hardware.gpu_detector.GPUDetector.detect_gpus"
        ) as mock_detect:
            mock_detect.return_value = [mock_gpu]

            manager = ProfileManager(pyproject_path)
            active = manager.get_active_profile()

            assert active is not None
            assert active.name == "development"

    def test_role_selector_with_registry(self, registry_path, pyproject_path):
        """Test RoleSelector with real model registry."""
        selector = RoleSelector(registry_path)

        # Load a profile
        with patch(
            "imageworks.libs.hardware.gpu_detector.GPUDetector.detect_gpus"
        ) as mock_detect:
            mock_detect.return_value = []
            manager = ProfileManager(pyproject_path)

        profile = manager.get_profile("constrained_16gb")
        assert profile is not None

        # Get available roles
        roles = selector.get_available_roles()
        assert len(roles) > 0

        # Try selecting for a role if available
        if "keywords" in roles:
            results = selector.select_for_role("keywords", profile, top_n=3)
            # Should return at least some models (depending on registry content)
            assert isinstance(results, list)

    def test_registry_has_role_priority_metadata(self, registry_path):
        """Test that curated registry has role_priority metadata."""
        with open(registry_path, "r") as f:
            registry = json.load(f)

        models = registry.get("models", [])
        assert len(models) > 0, "Registry is empty"

        # At least some models should have role_priority
        models_with_roles = [m for m in models if "role_priority" in m]
        assert len(models_with_roles) > 0, "No models have role_priority metadata"

        # Check structure of role_priority
        for model in models_with_roles:
            role_priority = model["role_priority"]
            assert isinstance(role_priority, dict)

            # Each role should have a numeric score
            for role, score in role_priority.items():
                assert isinstance(role, str)
                assert isinstance(score, (int, float))
                assert 0 <= score <= 100

    def test_registry_has_vram_estimates(self, registry_path):
        """Test that models have VRAM estimates for profile filtering."""
        with open(registry_path, "r") as f:
            registry = json.load(f)

        models = registry.get("models", [])

        # Models with role_priority should also have vram_estimate_mb
        models_with_roles = [m for m in models if "role_priority" in m]

        for model in models_with_roles:
            assert (
                "vram_estimate_mb" in model
            ), f"Model {model.get('id')} missing vram_estimate_mb"
            vram = model["vram_estimate_mb"]
            assert isinstance(vram, (int, float))
            assert vram > 0

    def test_vram_filtering_logic(self, registry_path, pyproject_path):
        """Test that models are properly filtered by VRAM constraints."""
        selector = RoleSelector(registry_path)

        with patch(
            "imageworks.libs.hardware.gpu_detector.GPUDetector.detect_gpus"
        ) as mock_detect:
            mock_detect.return_value = []
            manager = ProfileManager(pyproject_path)

        # Use constrained profile (10000 MB limit)
        constrained_profile = manager.get_profile("constrained_16gb")
        assert constrained_profile is not None

        # Get models that fit
        fitting_models = selector.get_models_for_profile(constrained_profile)

        # All fitting models should be under the VRAM limit
        for model in fitting_models:
            vram = model.get("vram_estimate_mb", float("inf"))
            assert (
                vram <= constrained_profile.max_vram_mb
            ), f"Model {model.get('id')} ({vram} MB) exceeds profile limit ({constrained_profile.max_vram_mb} MB)"

    def test_role_selection_respects_profile_bias(self, registry_path, pyproject_path):
        """Test that role selection respects model_selection_bias."""
        selector = RoleSelector(registry_path)

        with patch(
            "imageworks.libs.hardware.gpu_detector.GPUDetector.detect_gpus"
        ) as mock_detect:
            mock_detect.return_value = []
            manager = ProfileManager(pyproject_path)

        # Get profiles with different biases
        efficiency_profile = manager.get_profile("constrained_16gb")
        assert efficiency_profile.model_selection_bias == "efficiency"

        balanced_profile = manager.get_profile("balanced_24gb")
        assert balanced_profile.model_selection_bias == "balanced"

        # Selection should work with both profiles
        roles = selector.get_available_roles()
        if len(roles) > 0:
            test_role = roles[0]

            efficiency_results = selector.select_for_role(
                test_role, efficiency_profile, top_n=3
            )
            balanced_results = selector.select_for_role(
                test_role, balanced_profile, top_n=3
            )

            # Both should return valid results
            assert isinstance(efficiency_results, list)
            assert isinstance(balanced_results, list)


class TestPhase2APIIntegration:
    """Test Phase 2 API endpoints integration."""

    @pytest.fixture
    def app_client(self):
        """Create FastAPI test client."""
        # Import app after mocking
        with patch(
            "imageworks.libs.hardware.gpu_detector.GPUDetector.detect_gpus"
        ) as mock_detect:
            mock_detect.return_value = []

            from imageworks.chat_proxy.app import app

            return TestClient(app)

    def test_profile_endpoint_available(self, app_client):
        """Test that /v1/config/profile endpoint exists."""
        response = app_client.get("/v1/config/profile")

        # Should return 200 or 503 (if not initialized), not 404
        assert response.status_code in [200, 503]

    def test_profile_endpoint_returns_valid_data(self):
        """Test that profile endpoint returns expected structure."""
        mock_gpu = GPUInfo(0, "NVIDIA RTX 4080", 16384, 15000, "535.54")

        with patch(
            "imageworks.libs.hardware.gpu_detector.GPUDetector.detect_gpus"
        ) as mock_detect:
            mock_detect.return_value = [mock_gpu]

            from imageworks.chat_proxy.app import app

            client = TestClient(app)

            response = client.get("/v1/config/profile")

            if response.status_code == 200:
                data = response.json()

                assert "active_profile" in data
                assert "detected_gpus" in data
                assert "available_profiles" in data

                # Check GPU data
                if len(data["detected_gpus"]) > 0:
                    gpu = data["detected_gpus"][0]
                    assert "name" in gpu
                    assert "total_vram_mb" in gpu

    def test_select_by_role_endpoint_available(self, app_client):
        """Test that /v1/models/select_by_role endpoint exists."""
        response = app_client.get("/v1/models/select_by_role?role=keywords&top_n=3")

        # Should return 200, 400, or 503, not 404
        assert response.status_code in [200, 400, 503]

    def test_select_by_role_endpoint_with_valid_role(self):
        """Test role selection endpoint with valid role."""
        mock_gpu = GPUInfo(0, "NVIDIA RTX 4080", 16384, 15000, "535.54")

        with patch(
            "imageworks.libs.hardware.gpu_detector.GPUDetector.detect_gpus"
        ) as mock_detect:
            mock_detect.return_value = [mock_gpu]

            from imageworks.chat_proxy.app import app

            client = TestClient(app)

            # First, get available roles
            from imageworks.chat_proxy.role_selector import RoleSelector

            selector = RoleSelector()
            roles = selector.get_available_roles()

            if len(roles) > 0:
                test_role = roles[0]
                response = client.get(
                    f"/v1/models/select_by_role?role={test_role}&top_n=3"
                )

                if response.status_code == 200:
                    data = response.json()

                    assert "role" in data
                    assert "profile" in data
                    assert "models" in data
                    assert data["role"] == test_role

    def test_select_by_role_endpoint_invalid_role(self):
        """Test role selection endpoint with invalid role."""
        with patch(
            "imageworks.libs.hardware.gpu_detector.GPUDetector.detect_gpus"
        ) as mock_detect:
            mock_detect.return_value = []

            from imageworks.chat_proxy.app import app

            client = TestClient(app)

            response = client.get(
                "/v1/models/select_by_role?role=nonexistent_role&top_n=3"
            )

            # Should return 400 for invalid role
            assert response.status_code == 400
            data = response.json()
            assert "error" in data
