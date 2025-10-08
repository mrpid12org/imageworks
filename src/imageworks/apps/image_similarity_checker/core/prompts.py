"""Prompt profiles for similarity explanations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from imageworks.libs.prompting import PromptLibrary, PromptProfileBase


@dataclass(frozen=True)
class SimilarityPromptProfile(PromptProfileBase):
    """Prompt template for describing similarity verdicts."""

    system_prompt: str
    user_template: str

    def render_user(self, **context: object) -> str:
        return self.user_template.format(**context)


PROFILES: Dict[int, SimilarityPromptProfile] = {
    1: SimilarityPromptProfile(
        id=1,
        name="baseline",
        description="Explain similarity verdicts using computed metrics and thresholds.",
        system_prompt=(
            "You are an expert photo competition judge."
            " Explain to organisers whether two images appear to be duplicates,"
            " considering quantitative similarity metrics and any reviewer notes."
            " Keep the tone factual and concise."
        ),
        user_template=(
            "Candidate image: {candidate}\n"
            "Best library match: {best_match}\n"
            "Similarity score: {score:.3f}\n"
            "Fail threshold: {fail:.2f}\n"
            "Query threshold: {query:.2f}\n"
            "Strategy scores: {strategies}\n"
            "Current verdict: {verdict}\n"
            "Existing notes: {notes}\n"
            "Provide a short justification (2-3 sentences) clarifying whether this should be PASS, QUERY, or FAIL."
        ),
    ),
    2: SimilarityPromptProfile(
        id=2,
        name="conservative",
        description="Emphasise caution by highlighting uncertainty and recommending manual review when appropriate.",
        system_prompt=(
            "You assist competition moderators by summarising similarity evidence."
            " Focus on highlighting risks and suggesting manual review when data is borderline."
        ),
        user_template=(
            "Candidate: {candidate}\n"
            "Closest match: {best_match}\n"
            "Similarity score: {score:.3f}\n"
            "Fail threshold: {fail:.2f}, Query threshold: {query:.2f}\n"
            "Strategy breakdown: {strategies}\n"
            "Observed notes: {notes}\n"
            "Explain the recommendation in 3 sentences, flagging any uncertainty."
        ),
    ),
}

PROMPT_LIBRARY = PromptLibrary(PROFILES, default_id=1)


def get_prompt_profile(identifier=None) -> SimilarityPromptProfile:
    return PROMPT_LIBRARY.get(identifier)
