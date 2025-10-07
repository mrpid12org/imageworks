"""Tests for the LMDeploy startup helper script."""

from __future__ import annotations

from importlib import util
from pathlib import Path

import types


import pytest



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



def test_validate_model_directory_requires_tokenizer_assets(tmp_path):
    module = load_module()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    (model_dir / "tokenizer_config.json").write_text("{}")

    with pytest.raises(RuntimeError):
        module.validate_model_directory(model_dir)


def test_validate_model_directory_warns_on_optional_assets(tmp_path):
    module = load_module()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    (model_dir / "tokenizer_config.json").write_text("{}")
    (model_dir / "tokenizer.json").write_text("{}")
    (model_dir / "weights.safetensors").write_text("stub")

    warnings = module.validate_model_directory(model_dir)
    assert any("chat_template.json" in warning for warning in warnings)


def test_build_command_strips_remainder_sentinel():
    module = load_module()
    ns = types.SimpleNamespace(
        model_path="/models/demo",
        host="0.0.0.0",
        port=9000,
        model_name="demo",
        backend="pytorch",
        device="cuda",
        vision_max_batch_size=1,
        max_batch_size=None,
        eager=True,
        disable_fastapi_docs=False,
        api_keys=None,
        extra=["--", "--enable-auto-tool-choice", "--tool-call-parser", "openai"],
    )

    command = module.build_command(ns)
    assert "--" not in command
    assert "--enable-auto-tool-choice" in command
    assert "--tool-call-parser" in command
