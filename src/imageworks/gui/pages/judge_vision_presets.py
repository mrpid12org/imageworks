"""Dynamic presets for Judge Vision competitions."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from imageworks.apps.judge_vision import (
    CompetitionConfig,
    CompetitionRegistry,
    load_competition_registry,
)
from imageworks.gui.components.preset_selector import ModulePresets, PresetConfig
from imageworks.gui.config import (
    DEFAULT_INPUT_DIR,
    JUDGE_VISION_DEFAULT_COMPETITION_CONFIG,
    JUDGE_VISION_DEFAULT_OUTPUT_JSONL,
    JUDGE_VISION_DEFAULT_SUMMARY_PATH,
)

BASE_JUDGE_FLAGS: Dict[str, object] = {
    "input": [str(DEFAULT_INPUT_DIR)],
    "use_registry": True,
    "caption_role": "caption",
    "keyword_role": "keywords",
    "description_role": "description",
    "prompt_profile": "club_judge_json",
    "dry_run": True,
    "no_meta": True,
    "backup_originals": False,
    "overwrite_metadata": False,
    "critique_title_template": "{stem}",
    "critique_category": "Open",
    "critique_notes": "",
    "pairwise_rounds": 0,
    "output_jsonl": str(JUDGE_VISION_DEFAULT_OUTPUT_JSONL),
    "summary": str(JUDGE_VISION_DEFAULT_SUMMARY_PATH),
    "progress_file": "outputs/metrics/judge_vision_progress.json",
}

HIDDEN_FLAGS = ["api_key", "timeout", "batch_size", "max_workers"]
COMMON_OVERRIDE_KEYS = [
    "input",
    "output_jsonl",
    "summary",
    "pairwise_rounds",
    "critique_title_template",
    "critique_category",
    "critique_notes",
]


def _preset_name(config: CompetitionConfig) -> str:
    display_name = getattr(config, "display_name", None)
    if display_name:
        return display_name
    return config.identifier.replace("_", " ").title()


def _build_flags(
    base: Dict[str, object],
    competition: CompetitionConfig,
    registry_path: Path,
) -> Dict[str, object]:
    flags = dict(base)
    flags["competition_config"] = str(registry_path)
    flags["competition"] = competition.identifier
    if competition.pairwise_rounds:
        flags["pairwise_rounds"] = competition.pairwise_rounds
    if competition.categories:
        flags["available_categories"] = competition.categories
        flags["critique_category"] = competition.categories[0]
    if competition.notes:
        flags["critique_notes"] = competition.notes
        flags["competition_notes"] = competition.notes
    flags["rules_text"] = competition.rules.describe()
    flags["competition_name"] = _preset_name(competition)
    return flags


def _placeholder_presets(message: str, registry_path: Path) -> ModulePresets:
    presets = {
        "setup_required": PresetConfig(
            name="Setup Required",
            description=message,
            flags={
                **BASE_JUDGE_FLAGS,
                "competition_config": str(registry_path),
                "competition": "",
            },
            hidden_flags=HIDDEN_FLAGS,
            common_overrides=COMMON_OVERRIDE_KEYS,
        )
    }
    return ModulePresets(
        module_name="judge_vision",
        default_preset="setup_required",
        presets=presets,
    )


def build_competition_presets(
    registry_path: str | Path,
) -> Tuple[ModulePresets, Optional[CompetitionRegistry], List[str]]:
    """Build ModulePresets from the TOML competition registry."""

    errors: List[str] = []
    path = (
        Path(registry_path).expanduser()
        if registry_path
        else JUDGE_VISION_DEFAULT_COMPETITION_CONFIG
    )
    try:
        registry = load_competition_registry(path)
    except FileNotFoundError:
        message = f"Competition registry not found at {path}."
        errors.append(message)
        return _placeholder_presets(message, Path(path)), None, errors
    except Exception as exc:  # noqa: BLE001
        message = f"Failed to load registry: {exc}"
        errors.append(message)
        return _placeholder_presets(message, Path(path)), None, errors

    if not registry.competitions:
        message = "Competition registry is empty."
        errors.append(message)
        return _placeholder_presets(message, Path(path)), registry, errors

    presets: Dict[str, PresetConfig] = {}
    for identifier, competition in registry.competitions.items():
        name = _preset_name(competition)
        description = competition.rules.describe() or "Competition preset"
        flags = _build_flags(BASE_JUDGE_FLAGS, competition, Path(path))
        presets[identifier] = PresetConfig(
            name=name,
            description=description,
            flags=flags,
            hidden_flags=HIDDEN_FLAGS,
            common_overrides=COMMON_OVERRIDE_KEYS,
        )

    default_name = next(iter(presets.keys()))
    return (
        ModulePresets(
            module_name="judge_vision", default_preset=default_name, presets=presets
        ),
        registry,
        errors,
    )


__all__ = ["build_competition_presets", "BASE_JUDGE_FLAGS"]
