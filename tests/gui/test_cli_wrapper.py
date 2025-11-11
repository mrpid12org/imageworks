from types import SimpleNamespace

from imageworks.gui.utils import cli_wrapper


def _fake_settings():
    return SimpleNamespace(
        default_backend="ollama",
        default_base_url="http://localhost:11434/v1",
        default_api_key="EMPTY",
        default_timeout=60,
        default_max_new_tokens=512,
        default_temperature=0.2,
        default_top_p=0.9,
        default_prompt_profile="club_judge_json",
        default_batch_size=4,
        default_max_workers=2,
        default_use_registry=True,
    )


def test_build_tagger_command_ignores_judge_flags(tmp_path, monkeypatch):
    monkeypatch.setattr(cli_wrapper, "load_config", _fake_settings)

    registry = tmp_path / "competitions.toml"
    registry.write_text("[competition]", encoding="utf-8")

    config = {
        "input": ["/tmp/images"],
        "competition_config": str(registry),
        "competition": "club_open_2025",
        "pairwise_rounds": 3,
        "no_meta": True,
        "dry_run": True,
    }

    command = cli_wrapper.build_tagger_command(config)

    assert "--competition-config" not in command
    assert "--competition" not in command
    assert "--pairwise-rounds" not in command


def test_build_judge_command_invokes_module(monkeypatch, tmp_path):
    """Judge Vision command should run module path so no script install is required."""

    monkeypatch.setattr(cli_wrapper, "load_config", _fake_settings)

    image = tmp_path / "image.jpg"
    image.write_bytes(b"\x00")

    config = {
        "input": [str(image)],
        "use_registry": False,
        "skip_preflight": True,
        "progress_file": tmp_path / "progress.json",
        "enable_musiq": False,
        "enable_nima": False,
        "iqa_cache": tmp_path / "iqa.jsonl",
        "stage": "critique",
        "iqa_device": "gpu",
        "pairwise_rounds": 4,
        "pairwise_enabled": True,
        "pairwise_threshold": 17,
    }

    command = cli_wrapper.build_judge_command(config)

    assert command[:5] == [
        "uv",
        "run",
        "python",
        "-m",
        "imageworks.apps.judge_vision.cli.main",
    ]
    assert "--no-registry" in command
    assert "--skip-preflight" in command
    assert "--disable-musiq" in command
    assert "--disable-nima" in command
    assert "--iqa-cache" in command
    stage_index = command.index("--stage")
    assert command[stage_index + 1] == "critique"
    device_index = command.index("--iqa-device")
    assert command[device_index + 1] == "gpu"
    assert "--pairwise" in command
    thresh_index = command.index("--pairwise-threshold")
    assert command[thresh_index + 1] == "17"
    pairwise_index = command.index("--pairwise-rounds")
    assert command[pairwise_index + 1] == "4"
