"""Tests unified model registry role integrity for Personal Tagger usage.

Validates that:
- Registry loads without error.
- Expected roles are present on at least one model.
- Each model's roles list has no duplicates and only contains lowercase tokens.
"""

from __future__ import annotations

from pathlib import Path

from imageworks.model_loader.registry import load_registry


def test_registry_roles_integrity():
    reg = load_registry(Path("configs/model_registry.json"), force=True)
    assert reg, "Registry should not be empty"

    # Collect role counts
    role_map = {}
    for entry in reg.values():
        # roles must be unique & lowercase
        assert len(entry.roles) == len(
            set(entry.roles)
        ), f"Duplicate roles in {entry.name}"
        for r in entry.roles:
            assert r == r.lower(), f"Role '{r}' in {entry.name} not lowercase"
            role_map.setdefault(r, []).append(entry.name)

    for required in ["caption", "description", "keywords"]:
        assert required in role_map, f"Missing required role: {required}"
        assert role_map[required], f"No models advertise role: {required}"

    # At least one multimodal role set should include caption+description
    multi = [n for n in role_map["caption"] if n in role_map["description"]]
    assert (
        multi
    ), "Expected at least one model providing both caption and description roles"
