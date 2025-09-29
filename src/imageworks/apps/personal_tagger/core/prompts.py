"""Prompt profiles for the personal tagger."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from imageworks.libs.prompting import PromptLibrary, PromptProfileBase

__all__ = [
    "StagePrompt",
    "TaggerPromptProfile",
    "PROMPT_LIBRARY",
    "DEFAULT_PROMPT_ID",
    "get_prompt_profile",
    "list_prompt_profiles",
]


@dataclass(frozen=True)
class StagePrompt:
    """Encapsulates the system and user prompt for a single stage."""

    system: str
    user_template: str
    max_new_tokens: Optional[int] = None

    def render(self, **context: str) -> str:
        return self.user_template.format(**context)


@dataclass(frozen=True)
class TaggerPromptProfile(PromptProfileBase):
    """Prompt variants used by the three tagging stages."""

    caption_stage: StagePrompt
    keyword_stage: StagePrompt
    description_stage: StagePrompt


PROFILES: Dict[int, TaggerPromptProfile] = {
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
}

DEFAULT_PROMPT_ID = 1

PROMPT_LIBRARY = PromptLibrary(PROFILES, default_id=DEFAULT_PROMPT_ID)


def get_prompt_profile(identifier=None) -> TaggerPromptProfile:
    """Return the selected prompt profile (defaults to the configured profile)."""

    return PROMPT_LIBRARY.get(identifier)


def list_prompt_profiles():
    """List available prompt profiles."""

    return PROMPT_LIBRARY.list()
