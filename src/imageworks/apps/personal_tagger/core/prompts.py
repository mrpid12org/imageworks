"""Prompt profiles for the personal tagger."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from imageworks.libs.prompting import PromptLibrary, PromptProfileBase

_LOGGER = logging.getLogger(__name__)

_OVERRIDE_PATH = Path("configs/personal_tagger_prompts.user.json")


def _stage_to_dict(stage: StagePrompt) -> Dict[str, object]:
    return {
        "system": stage.system,
        "user_template": stage.user_template,
        "max_new_tokens": stage.max_new_tokens,
        "expects_json": stage.expects_json,
    }


def _stage_from_dict(data: Dict[str, object]) -> StagePrompt:
    return StagePrompt(
        system=str(data.get("system", "")),
        user_template=str(data.get("user_template", "")),
        max_new_tokens=data.get("max_new_tokens"),
        expects_json=bool(data.get("expects_json", False)),
    )


def _profile_to_dict(profile: TaggerPromptProfile) -> Dict[str, object]:
    return {
        "id": profile.id,
        "name": profile.name,
        "description": profile.description,
        "caption_stage": _stage_to_dict(profile.caption_stage),
        "keyword_stage": _stage_to_dict(profile.keyword_stage),
        "description_stage": _stage_to_dict(profile.description_stage),
    }


def _profile_from_dict(data: Dict[str, object]) -> TaggerPromptProfile:
    return TaggerPromptProfile(
        id=int(data.get("id")),
        name=str(data.get("name", "")),
        description=str(data.get("description", "")),
        caption_stage=_stage_from_dict(data.get("caption_stage", {})),
        keyword_stage=_stage_from_dict(data.get("keyword_stage", {})),
        description_stage=_stage_from_dict(data.get("description_stage", {})),
    )


__all__ = [
    "StagePrompt",
    "TaggerPromptProfile",
    "PROMPT_LIBRARY",
    "DEFAULT_PROMPT_ID",
    "get_prompt_profile",
    "list_prompt_profiles",
    "serialize_prompt_profile",
    "save_prompt_profile_data",
    "reset_prompt_profile",
    "is_user_profile",
]


@dataclass(frozen=True)
class StagePrompt:
    """Encapsulates the system and user prompt for a single stage."""

    system: str
    user_template: str
    max_new_tokens: Optional[int] = None
    expects_json: bool = False

    def render(self, **context: str) -> str:
        class _MissingDefault(dict):
            def __missing__(self, key: str) -> str:
                return ""

        safe_context: Dict[str, Any] = {
            key: "" if value is None else str(value) for key, value in context.items()
        }
        return self.user_template.format_map(_MissingDefault(safe_context))


@dataclass(frozen=True)
class TaggerPromptProfile(PromptProfileBase):
    """Prompt variants used by the tagging stages."""

    caption_stage: StagePrompt
    keyword_stage: StagePrompt
    description_stage: StagePrompt


_BASE_PROFILES: Dict[int, TaggerPromptProfile] = {
    1: TaggerPromptProfile(
        id=1,
        name="default",
        description="Baseline prompts tuned for concise caption, keyword JSON, and rich description outputs.",
        caption_stage=StagePrompt(
            system="You write concise, photographic captions for personal photo libraries.",
            user_template=(
                "Provide an active-voice caption describing this image. "
                "Limit the caption to at most two sentences and fewer than 200 characters. "
                "Do not include quotation marks or extra commentary."
            ),
            max_new_tokens=128,
        ),
        keyword_stage=StagePrompt(
            system=(
                "Return carefully curated keyword lists for personal photography archives. "
                "Avoid generic photographic terminology or subjective adjectives."
            ),
            user_template=(
                "Generate a ranked JSON array named keywords containing 25 distinct, specific keywords that describe this photograph. "
                "Use lowercase text, avoid duplicates, and prefer noun phrases that a photographer would use for search. "
                "Return ONLY valid JSON."
            ),
            max_new_tokens=256,
        ),
        description_stage=StagePrompt(
            system=(
                "Write rich, accessibility-friendly descriptions of photographs. "
                "Compose warm but factual prose suitable for metadata fields."
            ),
            user_template=(
                "Using the provided caption and keywords as context, craft a vivid 3-4 sentence description of the image. "
                "Do not repeat the caption verbatim; expand on important subjects, setting, light, and mood. "
                "Caption: {caption}. Keywords: {keyword_preview}."
            ),
            max_new_tokens=512,
        ),
    ),
    2: TaggerPromptProfile(
        id=2,
        name="narrative_v2",
        description="Alternative prompt focusing on storyteller tone and broader keyword diversity.",
        caption_stage=StagePrompt(
            system="You provide evocative photo captions suitable for personal albums while staying factual.",
            user_template=(
                "Write a short caption (≤2 sentences) that highlights the main subject and setting. "
                "Keep it grounded in what is visible, avoiding speculation or punctuation embellishments."
            ),
            max_new_tokens=160,
        ),
        keyword_stage=StagePrompt(
            system=(
                "Produce keyword sets that improve searchability in photo library software."
            ),
            user_template=(
                "Return a JSON object with a keywords array of 20 distinct entries ranked by relevance. "
                "Mix specific subject terms, contextual cues, and compositional hints. "
                "Only output JSON with the key 'keywords'."
            ),
            max_new_tokens=256,
        ),
        description_stage=StagePrompt(
            system=(
                "Compose accessible descriptions that read naturally while covering subject, environment, and mood."
            ),
            user_template=(
                "Write three sentences describing the image. First sentence: subject and action. "
                "Second: surrounding environment or supporting details. Third: light, colour tone, or atmosphere. "
                "Caption context: {caption}. Keyword highlights: {keyword_preview}."
            ),
            max_new_tokens=512,
        ),
    ),
    3: TaggerPromptProfile(
        id=3,
        name="phototools_prompt",
        description="PhotoTools-inspired prompts focused on expert photo editing perspective with ranked keyword generation.",
        caption_stage=StagePrompt(
            system="You write concise, photographic captions for personal photo libraries.",
            user_template=(
                "Provide an active-voice caption describing this image. "
                "Limit the caption to at most two sentences and fewer than 200 characters. "
                "Do not include quotation marks or extra commentary."
            ),
            max_new_tokens=128,
        ),
        keyword_stage=StagePrompt(
            system="Act as an expert photo editor. Your task is to provide a ranked list of keywords for the image.",
            user_template=(
                "Generate a ranked list of the 25 most relevant Keywords, with the most important first. "
                "Keywords must be specific and distinct. "
                "Return ONLY valid JSON."
            ),
            max_new_tokens=256,
        ),
        description_stage=StagePrompt(
            system=(
                "Write rich, accessibility-friendly descriptions of photographs. "
                "Compose warm but factual prose suitable for metadata fields."
            ),
            user_template=(
                "Using the provided caption and keywords as context, craft a vivid 3-4 sentence description of the image. "
                "Do not repeat the caption verbatim; expand on important subjects, setting, light, and mood. "
                "Caption: {caption}. Keywords: {keyword_preview}."
            ),
            max_new_tokens=512,
        ),
    ),
}

DEFAULT_PROMPT_ID = 1


def _load_user_overrides() -> (
    Tuple[Dict[int, TaggerPromptProfile], Dict[int, Dict[str, object]]]
):
    if not _OVERRIDE_PATH.exists():
        return {}, {}

    try:
        data = json.loads(_OVERRIDE_PATH.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        _LOGGER.warning("Failed to load prompt overrides: %s", exc)
        return {}, {}

    overrides: Dict[int, TaggerPromptProfile] = {}
    raw_data: Dict[int, Dict[str, object]] = {}
    for entry in data.get("profiles", []):
        try:
            profile = _profile_from_dict(entry)
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning("Invalid prompt override skipped: %s", exc)
            continue
        overrides[profile.id] = profile
        raw_data[profile.id] = _profile_to_dict(profile)
    return overrides, raw_data


def _write_overrides(raw: Dict[int, Dict[str, object]]) -> None:
    if not raw:
        if _OVERRIDE_PATH.exists():
            _OVERRIDE_PATH.unlink()
        return

    payload = {"profiles": list(raw.values())}
    _OVERRIDE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _OVERRIDE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _combine_profiles(
    overrides: Dict[int, TaggerPromptProfile],
) -> Dict[int, TaggerPromptProfile]:
    merged = dict(_BASE_PROFILES)
    merged.update(overrides)
    return merged


_USER_OVERRIDE_PROFILES, _USER_OVERRIDE_RAW = _load_user_overrides()


def _build_prompt_library() -> PromptLibrary:
    return PromptLibrary(
        _combine_profiles(_USER_OVERRIDE_PROFILES), default_id=DEFAULT_PROMPT_ID
    )


PROMPT_LIBRARY = _build_prompt_library()


def _next_profile_id() -> int:
    all_ids = set(_BASE_PROFILES.keys()) | set(_USER_OVERRIDE_PROFILES.keys())
    return max(all_ids) + 1 if all_ids else 1


def get_prompt_profile(identifier=None) -> TaggerPromptProfile:
    """Return the selected prompt profile (defaults to the configured profile)."""

    return PROMPT_LIBRARY.get(identifier)


def list_prompt_profiles():
    """List available prompt profiles."""

    return PROMPT_LIBRARY.list()


def serialize_prompt_profile(identifier) -> Dict[str, object]:
    profile = get_prompt_profile(identifier)
    return _profile_to_dict(profile)


def _refresh_library() -> None:
    global PROMPT_LIBRARY
    PROMPT_LIBRARY = _build_prompt_library()


def save_prompt_profile_data(
    data: Dict[str, object], *, as_new: bool = False
) -> TaggerPromptProfile:
    """Persist a prompt profile override and refresh the library.

    Args:
        data: Serialized prompt profile data (matching serialize_prompt_profile).
        as_new: When True, allocate a new profile id; otherwise overwrite provided id.
    """

    global _USER_OVERRIDE_PROFILES, _USER_OVERRIDE_RAW

    payload = dict(data)
    target_id = payload.get("id")
    if as_new or target_id is None:
        payload["id"] = _next_profile_id()
    else:
        payload["id"] = int(target_id)

    profile = _profile_from_dict(payload)
    _USER_OVERRIDE_PROFILES[profile.id] = profile
    _USER_OVERRIDE_RAW[profile.id] = _profile_to_dict(profile)
    _write_overrides(_USER_OVERRIDE_RAW)
    _refresh_library()
    return profile


def reset_prompt_profile(profile_id: int) -> None:
    """Remove an override so the base definition is used."""

    global _USER_OVERRIDE_PROFILES, _USER_OVERRIDE_RAW

    if profile_id in _USER_OVERRIDE_PROFILES:
        _USER_OVERRIDE_PROFILES.pop(profile_id, None)
        _USER_OVERRIDE_RAW.pop(profile_id, None)
        _write_overrides(_USER_OVERRIDE_RAW)
        _refresh_library()


def is_user_profile(profile_id: int) -> bool:
    return profile_id in _USER_OVERRIDE_RAW
