"""Unit tests for preset selector component."""

import pytest

from imageworks.gui.components.preset_selector import (
    PresetConfig,
    ModulePresets,
)


def test_preset_config_structure():
    """Test PresetConfig dataclass."""
    config = PresetConfig(
        name="test",
        description="Test preset",
        flags={"key": "value"},
        hidden_flags=[],
        common_overrides=[],
    )

    assert config.name == "test"
    assert config.description == "Test preset"
    assert config.flags == {"key": "value"}


def test_module_presets_structure():
    """Test ModulePresets dataclass."""
    quick = PresetConfig(
        "quick",
        "Quick preset",
        flags={"speed": "fast"},
        hidden_flags=[],
        common_overrides=[],
    )
    standard = PresetConfig(
        "standard",
        "Standard preset",
        flags={"speed": "medium"},
        hidden_flags=[],
        common_overrides=[],
    )
    thorough = PresetConfig(
        "thorough",
        "Thorough preset",
        flags={"speed": "slow"},
        hidden_flags=[],
        common_overrides=[],
    )

    presets = ModulePresets(
        module_name="test",
        default_preset="standard",
        presets={"quick": quick, "standard": standard, "thorough": thorough},
    )

    assert presets.module_name == "test"
    assert presets.default_preset == "standard"
    assert "quick" in presets.presets
    assert "standard" in presets.presets
    assert "thorough" in presets.presets


def test_preset_config_merge():
    """Test merging preset config with custom overrides."""
    base_config = {"threshold": 0.7, "timeout": 30}
    overrides = {"threshold": 0.9, "retries": 3}

    # Merge (override behavior)
    merged = {**base_config, **overrides}

    assert merged["threshold"] == 0.9  # Overridden
    assert merged["timeout"] == 30  # Preserved
    assert merged["retries"] == 3  # Added


def test_all_preset_levels_available():
    """Test that similarity presets have the expected structure."""
    from imageworks.gui.presets import IMAGE_SIMILARITY_PRESETS

    # Check image similarity has all preset levels
    assert "quick" in IMAGE_SIMILARITY_PRESETS.presets
    assert "standard" in IMAGE_SIMILARITY_PRESETS.presets
    assert "thorough" in IMAGE_SIMILARITY_PRESETS.presets

    assert IMAGE_SIMILARITY_PRESETS.presets["quick"].name == "Quick"
    assert IMAGE_SIMILARITY_PRESETS.presets["standard"].name == "Standard"
    assert IMAGE_SIMILARITY_PRESETS.presets["thorough"].name == "Thorough"


def test_preset_configs_not_empty():
    """Test that preset configs contain actual configuration."""
    from imageworks.gui.presets import IMAGE_SIMILARITY_PRESETS

    assert len(IMAGE_SIMILARITY_PRESETS.presets["quick"].flags) > 0
    assert len(IMAGE_SIMILARITY_PRESETS.presets["standard"].flags) > 0
    assert len(IMAGE_SIMILARITY_PRESETS.presets["thorough"].flags) > 0


def test_preset_descriptions_exist():
    """Test that all presets have descriptions."""
    from imageworks.gui.presets import IMAGE_SIMILARITY_PRESETS

    assert len(IMAGE_SIMILARITY_PRESETS.presets["quick"].description) > 0
    assert len(IMAGE_SIMILARITY_PRESETS.presets["standard"].description) > 0
    assert len(IMAGE_SIMILARITY_PRESETS.presets["thorough"].description) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
