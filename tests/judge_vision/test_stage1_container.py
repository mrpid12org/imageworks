from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from imageworks.apps.judge_vision.config import JudgeVisionConfig
from imageworks.apps.judge_vision import stage1_container as container


def _config(tmp_path: Path) -> JudgeVisionConfig:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    return JudgeVisionConfig(
        input_paths=[input_dir],
        recursive=False,
        image_extensions=(".jpg",),
        backend="vllm",
        base_url="http://localhost:8100/v1",
        api_key="",
        timeout=30,
        max_new_tokens=256,
        temperature=0.2,
        top_p=0.9,
        model=None,
        use_registry=False,
        critique_role="judge",
        skip_preflight=True,
        dry_run=False,
        competition_id=None,
        competition_config=None,
        pairwise_rounds=None,
        pairwise_enabled=False,
        pairwise_threshold=17,
        critique_title_template="{stem}",
        critique_category=None,
        critique_notes="",
        output_jsonl=tmp_path / "outputs" / "judge.jsonl",
        summary_path=tmp_path / "outputs" / "judge.md",
        progress_path=tmp_path / "outputs" / "progress.json",
        enable_musiq=True,
        enable_nima=True,
        iqa_cache_path=tmp_path / "outputs" / "cache.jsonl",
        stage="iqa",
        iqa_device="gpu",
    )


def test_serialize_config_forces_stage_iqa(tmp_path, monkeypatch):
    cfg = replace(_config(tmp_path), stage="full")
    monkeypatch.setattr(container, "PROJECT_ROOT", tmp_path)
    payload = container.serialize_config(cfg)
    assert payload.parent == tmp_path / "tmp" / container.CONFIG_SUBDIR
    data = json.loads(payload.read_text(encoding="utf-8"))
    assert data["stage"] == "iqa"


def test_build_docker_command_includes_mounts(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    payload = tmp_path / "tmp" / "judge_stage1" / "config.json"
    payload.parent.mkdir(parents=True, exist_ok=True)
    payload.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(container, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(container, "TF_RUNTIME_ROOT", tmp_path / ".tf-backend")
    monkeypatch.setattr(container, "TF_ENV_DIR", ".venv-test")
    monkeypatch.setattr(container, "TF_CACHE_DIR", str(tmp_path / ".tf-cache"))
    monkeypatch.setattr(container, "TF_UV_BIN", str(tmp_path / ".tf-backend/bin/uv"))
    monkeypatch.setattr(container, "TF_EXTRA_MOUNTS", "")
    monkeypatch.setattr(container, "TF_DOCKER_ARGS", "")

    image, cmd = container.build_docker_command(cfg, payload, image="test:image")

    assert image == "test:image"
    assert cmd[0:4] == ["docker", "run", "--rm", "--gpus"]
    mounts = [
        cmd[idx + 1]
        for idx, token in enumerate(cmd)
        if token == "-v" and idx + 1 < len(cmd)
    ]
    assert any(str(tmp_path) in entry for entry in mounts)
    assert cmd[-3] == "bash"
    assert cmd[-2] == "-lc"
    assert str(payload) in cmd[-1]
