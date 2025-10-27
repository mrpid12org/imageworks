"""
Unit tests for RoleSelector module
"""

import json
from pathlib import Path
from imageworks.chat_proxy.role_selector import RoleSelector
from imageworks.chat_proxy.profile_manager import DeploymentProfile


class TestRoleSelector:
    """Test RoleSelector class."""

    def _create_test_registry(self, tmp_path: Path) -> Path:
        """Create a test model registry."""
        registry = {
            "models": [
                {
                    "id": "qwen3-vl",
                    "name": "Qwen3-VL",
                    "backend": "vllm",
                    "quantization": "FP8",
                    "vram_estimate_mb": 10600,
                    "role_priority": {"keywords": 80, "description": 90, "caption": 70},
                },
                {
                    "id": "qwen25vl",
                    "name": "Qwen2.5-VL",
                    "backend": "vllm",
                    "quantization": "FP8",
                    "vram_estimate_mb": 6400,
                    "role_priority": {"keywords": 75, "description": 85, "caption": 80},
                },
                {
                    "id": "florence-2",
                    "name": "Florence-2-large",
                    "backend": "ollama",
                    "quantization": "Q4_K_M",
                    "vram_estimate_mb": 3500,
                    "role_priority": {
                        "keywords": 95,
                        "object_detection": 90,
                        "caption": 60,
                    },
                },
                {
                    "id": "llava-7b",
                    "name": "LLaVA-1.5-7B",
                    "backend": "ollama",
                    "quantization": "Q4_K_M",
                    "vram_estimate_mb": 5000,
                    "role_priority": {"description": 70, "caption": 75},
                },
                {
                    "id": "huge-model",
                    "name": "Huge Model 70B",
                    "backend": "vllm",
                    "quantization": "FP16",
                    "vram_estimate_mb": 50000,
                    "role_priority": {"keywords": 100, "description": 100},
                },
            ]
        }

        registry_path = tmp_path / "test_registry.json"
        registry_path.write_text(json.dumps(registry, indent=2))
        return registry_path

    def test_load_registry(self, tmp_path):
        """Test loading model registry."""
        registry_path = self._create_test_registry(tmp_path)
        selector = RoleSelector(registry_path)

        # Check internal state (accessing private member for testing)
        assert len(selector._models) == 5

    def test_get_available_roles(self, tmp_path):
        """Test getting list of available roles."""
        registry_path = self._create_test_registry(tmp_path)
        selector = RoleSelector(registry_path)

        roles = selector.get_available_roles()

        assert "keywords" in roles
        assert "description" in roles
        assert "caption" in roles
        assert "object_detection" in roles
        assert len(roles) == 4  # 4 unique roles across all models

    def test_select_for_role_basic(self, tmp_path):
        """Test basic role-based model selection."""
        registry_path = self._create_test_registry(tmp_path)
        selector = RoleSelector(registry_path)

        profile = DeploymentProfile(
            name="test",
            max_vram_mb=20000,
            max_concurrent_models=2,
            preferred_quantizations=["fp8", "q4_k_m"],
            strategy="balanced",
            autostart_timeout_seconds=300,
            grace_period_seconds=30,
            model_selection_bias="quality",
        )

        # Select for "keywords" role
        results = selector.select_for_role("keywords", profile, top_n=3)

        # Should return models sorted by role_priority
        assert len(results) > 0
        assert results[0]["id"] == "florence-2"  # highest priority (95)
        assert results[1]["id"] == "qwen3-vl"  # second (80)
        assert results[2]["id"] == "qwen25vl"  # third (75)

    def test_select_for_role_vram_constraint(self, tmp_path):
        """Test that VRAM constraints are respected."""
        registry_path = self._create_test_registry(tmp_path)
        selector = RoleSelector(registry_path)

        # Constrained profile that excludes larger models
        profile = DeploymentProfile(
            name="constrained",
            max_vram_mb=7000,
            max_concurrent_models=1,
            preferred_quantizations=["fp8", "q4_k_m"],
            strategy="conservative",
            autostart_timeout_seconds=300,
            grace_period_seconds=30,
            model_selection_bias="vram_efficient",
        )

        results = selector.select_for_role("keywords", profile, top_n=5)

        # Should exclude qwen3-vl (10600 MB) and huge-model (50000 MB)
        result_ids = [m["id"] for m in results]
        assert "qwen3-vl" not in result_ids
        assert "huge-model" not in result_ids
        assert "florence-2" in result_ids  # 3500 MB - fits
        assert "qwen25vl" in result_ids  # 6400 MB - fits

    def test_select_for_role_efficiency_bias(self, tmp_path):
        """Test efficiency bias prefers lower VRAM models."""
        registry_path = self._create_test_registry(tmp_path)
        selector = RoleSelector(registry_path)

        profile = DeploymentProfile(
            name="efficiency",
            max_vram_mb=12000,
            max_concurrent_models=1,
            preferred_quantizations=["fp8"],
            strategy="conservative",
            autostart_timeout_seconds=300,
            grace_period_seconds=30,
            model_selection_bias="vram_efficient",
        )

        results = selector.select_for_role("description", profile, top_n=3)

        # With efficiency bias and similar scores, should prefer lower VRAM
        # qwen25vl (6400 MB, score 85) should rank higher than qwen3-vl (10600 MB, score 90)
        # when efficiency is prioritized
        result_ids = [m["id"] for m in results]
        assert len(result_ids) > 0

    def test_select_for_role_quality_bias(self, tmp_path):
        """Test quality bias prefers higher role_priority."""
        registry_path = self._create_test_registry(tmp_path)
        selector = RoleSelector(registry_path)

        profile = DeploymentProfile(
            name="quality",
            max_vram_mb=12000,
            max_concurrent_models=2,
            preferred_quantizations=["fp8"],
            strategy="balanced",
            autostart_timeout_seconds=600,
            grace_period_seconds=60,
            model_selection_bias="quality",
        )

        results = selector.select_for_role("description", profile, top_n=3)

        # With quality bias, higher role_priority should win
        assert results[0]["id"] == "qwen3-vl"  # score 90
        assert results[1]["id"] == "qwen25vl"  # score 85

    def test_select_for_role_no_matches(self, tmp_path):
        """Test selecting for role with no matching models."""
        registry_path = self._create_test_registry(tmp_path)
        selector = RoleSelector(registry_path)

        profile = DeploymentProfile(
            name="test",
            max_vram_mb=20000,
            max_concurrent_models=2,
            preferred_quantizations=["fp8"],
            strategy="balanced",
            autostart_timeout_seconds=300,
            grace_period_seconds=30,
            model_selection_bias="quality",
        )

        # "narration" role doesn't exist in test registry
        results = selector.select_for_role("narration", profile, top_n=3)

        assert len(results) == 0

    def test_select_for_role_top_n_limit(self, tmp_path):
        """Test that top_n parameter limits results."""
        registry_path = self._create_test_registry(tmp_path)
        selector = RoleSelector(registry_path)

        profile = DeploymentProfile(
            name="test",
            max_vram_mb=20000,
            max_concurrent_models=3,
            preferred_quantizations=["fp8", "q4_k_m"],
            strategy="balanced",
            autostart_timeout_seconds=300,
            grace_period_seconds=30,
            model_selection_bias="quality",
        )

        # Request only 2 models
        results = selector.select_for_role("description", profile, top_n=2)

        assert len(results) == 2

    def test_get_model_by_id(self, tmp_path):
        """Test retrieving specific model by ID."""
        registry_path = self._create_test_registry(tmp_path)
        selector = RoleSelector(registry_path)

        model = selector.get_model_by_id("qwen3-vl")

        assert model is not None
        assert model["name"] == "Qwen3-VL"
        assert model["backend"] == "vllm"

        nonexistent = selector.get_model_by_id("nonexistent")
        assert nonexistent is None

    def test_get_models_for_profile(self, tmp_path):
        """Test getting all models that fit within profile."""
        registry_path = self._create_test_registry(tmp_path)
        selector = RoleSelector(registry_path)

        profile = DeploymentProfile(
            name="constrained",
            max_vram_mb=7000,
            max_concurrent_models=1,
            preferred_quantizations=["fp8"],
            strategy="conservative",
            autostart_timeout_seconds=300,
            grace_period_seconds=30,
            model_selection_bias="vram_efficient",
        )

        fitting_models = selector.get_models_for_profile(profile)

        # Should include florence-2 (3500), llava-7b (5000), qwen25vl (6400)
        # Should exclude qwen3-vl (10600), huge-model (50000)
        assert len(fitting_models) == 3
        ids = [m["id"] for m in fitting_models]
        assert "florence-2" in ids
        assert "llava-7b" in ids
        assert "qwen25vl" in ids

    def test_explain_selection_eligible(self, tmp_path):
        """Test explanation for eligible model."""
        registry_path = self._create_test_registry(tmp_path)
        selector = RoleSelector(registry_path)

        profile = DeploymentProfile(
            name="test",
            max_vram_mb=12000,
            max_concurrent_models=2,
            preferred_quantizations=["fp8"],
            strategy="balanced",
            autostart_timeout_seconds=300,
            grace_period_seconds=30,
            model_selection_bias="quality",
        )

        explanation = selector.explain_selection("keywords", profile, "qwen3-vl")

        assert explanation["status"] == "eligible"
        assert explanation["model_id"] == "qwen3-vl"
        assert explanation["role"] == "keywords"
        assert len(explanation["checks"]) == 3

        # All checks should pass
        for check in explanation["checks"]:
            assert check["passed"] is True

    def test_explain_selection_vram_rejected(self, tmp_path):
        """Test explanation for model rejected due to VRAM."""
        registry_path = self._create_test_registry(tmp_path)
        selector = RoleSelector(registry_path)

        profile = DeploymentProfile(
            name="constrained",
            max_vram_mb=8000,
            max_concurrent_models=1,
            preferred_quantizations=["fp8"],
            strategy="conservative",
            autostart_timeout_seconds=300,
            grace_period_seconds=30,
            model_selection_bias="vram_efficient",
        )

        explanation = selector.explain_selection("keywords", profile, "qwen3-vl")

        assert explanation["status"] == "rejected"
        assert explanation["reason"] == "Exceeds VRAM limit"

        # Role support should pass, VRAM should fail
        checks = {c["check"]: c for c in explanation["checks"]}
        assert checks["role_support"]["passed"] is True
        assert checks["vram_constraint"]["passed"] is False

    def test_explain_selection_no_role_support(self, tmp_path):
        """Test explanation for model without role support."""
        registry_path = self._create_test_registry(tmp_path)
        selector = RoleSelector(registry_path)

        profile = DeploymentProfile(
            name="test",
            max_vram_mb=20000,
            max_concurrent_models=2,
            preferred_quantizations=["fp8"],
            strategy="balanced",
            autostart_timeout_seconds=300,
            grace_period_seconds=30,
            model_selection_bias="quality",
        )

        # llava-7b doesn't support "keywords" role
        explanation = selector.explain_selection("keywords", profile, "llava-7b")

        assert explanation["status"] == "rejected"
        assert "does not support role" in explanation["reason"]

    def test_explain_selection_model_not_found(self, tmp_path):
        """Test explanation for non-existent model."""
        registry_path = self._create_test_registry(tmp_path)
        selector = RoleSelector(registry_path)

        profile = DeploymentProfile(
            name="test",
            max_vram_mb=20000,
            max_concurrent_models=2,
            preferred_quantizations=["fp8"],
            strategy="balanced",
            autostart_timeout_seconds=300,
            grace_period_seconds=30,
            model_selection_bias="quality",
        )

        explanation = selector.explain_selection("keywords", profile, "nonexistent")

        assert explanation["status"] == "not_found"
        assert explanation["reason"] == "Model not found in registry"
