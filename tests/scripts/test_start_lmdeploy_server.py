"""Tests for the LMDeploy startup helper script."""

from __future__ import annotations

from importlib import util
from pathlib import Path

import types


def load_module() -> types.ModuleType:
    module_path = Path(__file__).resolve().parents[2] / "scripts" / "start_lmdeploy_server.py"
    spec = util.spec_from_file_location("start_lmdeploy_server", module_path)
    module = util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def test_resolve_default_model_path_uses_weights_suffix(tmp_path):
    module = load_module()
    env = {"IMAGEWORKS_MODEL_ROOT": str(tmp_path / "models")}
    resolved = module.resolve_default_model_path(env=env, home=tmp_path)
    expected = tmp_path / "models" / "weights" / module.DEFAULT_MODEL_REPO
    assert resolved == expected


def test_resolve_default_model_path_accepts_weights_root(tmp_path):
    module = load_module()
    env = {"IMAGEWORKS_MODEL_ROOT": str(tmp_path / "weights")}
    resolved = module.resolve_default_model_path(env=env, home=tmp_path)
    expected = tmp_path / "weights" / module.DEFAULT_MODEL_REPO
    assert resolved == expected


def test_resolve_default_model_path_home_fallback(tmp_path):
    module = load_module()
    resolved = module.resolve_default_model_path(env={}, home=tmp_path)
    expected = tmp_path / "ai-models" / "weights" / module.DEFAULT_MODEL_REPO
    assert resolved == expected
