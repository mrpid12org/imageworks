from imageworks.gui.pages.judge_vision_presets import build_competition_presets


def test_build_competition_presets_from_registry(tmp_path):
    registry = tmp_path / "competitions.toml"
    registry.write_text(
        """
[competition.club_open]
categories = ["Open", "Mono"]
pairwise_rounds = 4
notes = "Club DPI season round"

[competition.club_open.rules]
max_width = 1600
max_height = 1200
colour_space = "sRGB"

[competition.nature]
categories = ["Nature"]
pairwise_rounds = 5

[competition.nature.rules]
max_width = 1920
max_height = 1080
""",
        encoding="utf-8",
    )

    presets, registry_obj, errors = build_competition_presets(str(registry))

    assert not errors
    assert registry_obj is not None
    assert "club_open" in presets.presets
    flags = presets.presets["club_open"].flags
    assert flags["competition"] == "club_open"
    assert flags["pairwise_rounds"] == 4
    assert flags["available_categories"] == ["Open", "Mono"]


def test_build_competition_presets_missing_file(tmp_path):
    missing = tmp_path / "missing.toml"
    presets, registry_obj, errors = build_competition_presets(str(missing))

    assert registry_obj is None
    assert errors
    assert "setup_required" in presets.presets
