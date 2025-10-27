"""
Integration test for Phase 1: Proxy Integration.
Validates that modules can route through chat proxy correctly.
"""

from pathlib import Path


class TestPhase1ProxyIntegration:
    """Integration tests for Phase 1 proxy routing."""

    def test_personal_tagger_config_uses_proxy(self):
        """Verify personal_tagger default config points to proxy."""
        try:
            import tomllib  # Python 3.11+
        except ImportError:
            import tomli as tomllib  # Fallback for Python 3.10

        config_path = Path("pyproject.toml")
        with open(config_path, "rb") as f:
            config = tomllib.load(f)

        personal_tagger_config = config["tool"]["imageworks"]["personal_tagger"]
        assert personal_tagger_config["default_base_url"] == "http://localhost:8100/v1"

    def test_image_similarity_checker_config_uses_proxy(self):
        """Verify image_similarity_checker default config points to proxy."""
        try:
            import tomllib  # Python 3.11+
        except ImportError:
            import tomli as tomllib  # Fallback for Python 3.10

        config_path = Path("pyproject.toml")
        with open(config_path, "rb") as f:
            config = tomllib.load(f)

        similarity_config = config["tool"]["imageworks"]["image_similarity_checker"]
        assert similarity_config["default_base_url"] == "http://localhost:8100/v1"

    def test_registry_has_role_metadata(self):
        """Verify model registry contains role metadata for key models."""
        import json

        registry_path = Path("configs/model_registry.curated.json")
        with open(registry_path) as f:
            registry = json.load(f)

        # Find qwen3-vl
        qwen3_vl = next(
            (m for m in registry if m["name"] == "qwen3-vl-8b-instruct_(FP8)"), None
        )
        assert qwen3_vl is not None, "qwen3-vl model not found"
        assert "roles" in qwen3_vl
        assert "role_priority" in qwen3_vl
        assert "vram_estimate_mb" in qwen3_vl
        assert qwen3_vl["vram_estimate_mb"] == 10600
        assert "keywords" in qwen3_vl["roles"]
        assert qwen3_vl["role_priority"]["keywords"] == 80

        # Find qwen2.5vl
        qwen25_vl = next(
            (m for m in registry if m["name"] == "qwen25vl_8.3b_(Q4_K_M)"), None
        )
        assert qwen25_vl is not None, "qwen2.5vl model not found"
        assert "roles" in qwen25_vl
        assert "role_priority" in qwen25_vl
        assert "vram_estimate_mb" in qwen25_vl
        assert qwen25_vl["vram_estimate_mb"] == 6400

        # Find Florence-2
        florence = next(
            (m for m in registry if m["name"] == "florence-2-large_(FP16)"), None
        )
        assert florence is not None, "Florence-2 model not found"
        assert "roles" in florence
        assert "role_priority" in florence
        assert "vram_estimate_mb" in florence
        assert florence["vram_estimate_mb"] == 3500
        assert "keywords" in florence["roles"]
        assert florence["role_priority"]["keywords"] == 95  # Highest priority

    def test_florence2_config_correct(self):
        """Verify Florence-2 model configuration is correct."""
        import json

        registry_path = Path("configs/model_registry.curated.json")
        with open(registry_path) as f:
            registry = json.load(f)

        florence = next(
            (m for m in registry if m["name"] == "florence-2-large_(FP16)"), None
        )

        assert florence["backend"] == "vllm"
        assert florence["backend_config"]["port"] == 24002  # Separate port
        assert florence["family"] == "florence-2-large"
        assert florence["quantization"] == "fp16"
        assert "--trust-remote-code" in florence["backend_config"]["extra_args"]

        # Check roles
        assert set(florence["roles"]) == {"keywords", "caption", "object_detection"}

        # Check role priorities
        assert florence["role_priority"]["keywords"] == 95  # Highest
        assert florence["role_priority"]["caption"] == 75
        assert florence["role_priority"]["object_detection"] == 90

    def test_vlm_mono_interpreter_imports_backend_client(self):
        """Verify VLMMonoInterpreter imports backend client, not requests."""
        from imageworks.apps.color_narrator.core import vlm_mono_interpreter

        # Check that create_backend_client is imported
        assert hasattr(vlm_mono_interpreter, "create_backend_client")

        # Check that requests is NOT imported at module level

        module_code = open(vlm_mono_interpreter.__file__).read()
        assert "import requests" not in module_code
        assert "create_backend_client" in module_code

    def test_vlm_mono_interpreter_default_config(self):
        """Verify VLMMonoInterpreter defaults are correct."""
        from imageworks.apps.color_narrator.core.vlm_mono_interpreter import (
            VLMMonoInterpreter,
        )

        interpreter = VLMMonoInterpreter()

        # Check default base_url is proxy
        assert interpreter.base_url == "http://localhost:8100/v1"

        # Check default model is registry name
        assert interpreter.model == "qwen3-vl-8b-instruct_(FP8)"

    def test_role_priority_ordering(self):
        """Verify role priorities are ordered correctly across models."""
        import json

        registry_path = Path("configs/model_registry.curated.json")
        with open(registry_path) as f:
            registry = json.load(f)

        # Get all models with keywords role
        keyword_models = [
            (m["name"], m["role_priority"].get("keywords", 0))
            for m in registry
            if "keywords" in m.get("roles", [])
        ]

        # Sort by priority (descending)
        keyword_models.sort(key=lambda x: x[1], reverse=True)

        # Florence-2 should be highest priority for keywords
        assert keyword_models[0][0] == "florence-2-large_(FP16)"
        assert keyword_models[0][1] == 95

        # qwen3-vl should be second
        assert keyword_models[1][0] == "qwen3-vl-8b-instruct_(FP8)"
        assert keyword_models[1][1] == 80

        # qwen2.5vl should be third
        assert keyword_models[2][0] == "qwen25vl_8.3b_(Q4_K_M)"
        assert keyword_models[2][1] == 75

    def test_vram_estimates_for_16gb_planning(self):
        """Verify VRAM estimates support 16GB deployment planning."""
        import json

        registry_path = Path("configs/model_registry.curated.json")
        with open(registry_path) as f:
            registry = json.load(f)

        # Get vision models with VRAM estimates
        vision_models = [
            (m["name"], m.get("vram_estimate_mb", 0))
            for m in registry
            if m.get("vram_estimate_mb") and m["capabilities"].get("vision")
        ]

        # Verify we have the three key models
        model_dict = dict(vision_models)

        assert model_dict.get("qwen3-vl-8b-instruct_(FP8)") == 10600  # 10.6GB
        assert model_dict.get("qwen25vl_8.3b_(Q4_K_M)") == 6400  # 6.4GB
        assert model_dict.get("florence-2-large_(FP16)") == 3500  # 3.5GB

        # Verify 16GB constraints
        # Single model: qwen3-vl (10.6GB) fits with headroom
        assert model_dict["qwen3-vl-8b-instruct_(FP8)"] < 14000  # Leave 2GB headroom

        # Dual model: qwen2.5vl + florence-2
        dual_vram = (
            model_dict["qwen25vl_8.3b_(Q4_K_M)"] + model_dict["florence-2-large_(FP16)"]
        )
        assert dual_vram == 9900  # 9.9GB, well within 16GB

    def test_backups_created(self):
        """Verify backup files were created."""
        backups = [
            Path("pyproject.toml.pre-phase1"),
            Path("configs/model_registry.curated.json.pre-phase1"),
            Path(
                "src/imageworks/apps/color_narrator/core/vlm_mono_interpreter.py.pre-phase1"
            ),
        ]

        for backup_path in backups:
            assert backup_path.exists(), f"Backup file not found: {backup_path}"

    def test_json_validity(self):
        """Verify model registry JSON is valid."""
        import json

        registry_path = Path("configs/model_registry.curated.json")

        # Should not raise exception
        with open(registry_path) as f:
            registry = json.load(f)

        # Basic structure validation
        assert isinstance(registry, list)
        assert len(registry) > 0

        # Each entry should have required fields
        for model in registry:
            assert "name" in model
            assert "backend" in model
            assert "capabilities" in model


class TestPhase1Documentation:
    """Test that documentation was properly updated."""

    def test_phase1_doc_marked_complete(self):
        """Verify Phase 1 implementation doc is marked as completed."""
        doc_path = Path("docs/implementation/phase-1-proxy-integration.md")
        assert doc_path.exists()

        with open(doc_path) as f:
            content = f.read()

        assert "COMPLETED" in content or "✅" in content

    def test_completion_report_exists(self):
        """Verify Phase 1 completion report was created."""
        report_path = Path("docs/implementation/phase-1-completion-report.md")
        assert report_path.exists()

        with open(report_path) as f:
            content = f.read()

        # Should contain key sections
        assert "Overview" in content
        assert "Changes Implemented" in content
        assert "Test Results" in content
        assert "Success Criteria" in content

    def test_vlm_mono_interpreter_tests_exist(self):
        """Verify new VLMMonoInterpreter tests were created."""
        test_path = Path("tests/color_narrator/unit/test_vlm_mono_interpreter.py")
        assert test_path.exists()

        with open(test_path) as f:
            content = f.read()

        # Should test key functionality
        assert "test_default_base_url_is_proxy" in content
        assert "test_uses_backend_client" in content
        assert "create_backend_client" in content
