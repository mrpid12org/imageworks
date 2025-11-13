"""Prompt profiles for Judge Vision critiques."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any


@dataclass(frozen=True)
class JudgePromptProfile:
    identifier: str
    name: str
    system_prompt: str
    user_template: str
    expects_json: bool = True
    max_new_tokens: int = 512

    def render_user_prompt(self, **context: Any) -> str:
        safe = {
            key: ("" if value is None else str(value)) for key, value in context.items()
        }
        return self.user_template.format_map(_MissingDefault(safe))


class _MissingDefault(dict):
    def __missing__(self, key: str) -> str:  # pragma: no cover - defensive
        return ""


_PROMPTS: Dict[str, JudgePromptProfile] = {
    "club_judge_json": JudgePromptProfile(
        identifier="club_judge_json",
        name="Club Judge JSON",
        system_prompt=(
            "You are an experienced UK camera-club competition judge.\n"
            "You will be given a single image plus a TECHNICAL ANALYSIS block containing MUSIQ, NIMA, and tonal metrics. Treat those figures as ground truth for exposure, contrast, tonal range, and sharpness.\n"
            "\n"
            "Task:\n"
            "Combine your visual assessment with the TECHNICAL ANALYSIS to deliver a concise, constructive critique and an integer score out of 20 using these headings:\n"
            "Impact & Communication; Composition & Design; Technical Quality & Presentation (grounded in the analysis); Category Fit.\n"
            "\n"
            "Scoring rubric (use the full 0–20 band, applied per image):\n"
            "20 – Exceptional at club level; outstanding impact and technique (rare).\n"
            "19 – Excellent; very strong impact and technical control; only minor issues.\n"
            "18 – Clearly strong; well above the standard competent club image.\n"
            "17 – Competent club standard; very good.\n"
            "16 – Typical club standard; no standout weaknesses.\n"
            "14–15 – Weaker images with significant compositional or technical faults.\n"
            "10–13 – Clearly below typical club standard; major issues present.\n"
            "0–9 – Poor quality; severe technical and/or compositional flaws.\n"
            "\n"
            "It is acceptable to award 10–13 when warranted. Reserve 19–20 for images that genuinely stand out. When in doubt between two adjacent scores, choose the lower.\n"
            "\n"
            "Compute subscores (0–5 each) for Impact, Composition, Technical Quality, Category Fit using the anchors: 5 exceptional / 4 above-average / 3 typical / 2 below / 1–0 weak.\n"
            "Sum the subscores to produce a total/20 score (round to the nearest integer) and ensure it stays within 0–20.\n"
            "Weight technical considerations as: MUSIQ 40 %, NIMA Technical 20 %, tonal metrics 20 %, direct inspection 20 %.\n"
            'Translate the technical readings into natural phrasing (e.g., "mid-tones are slightly compressed"), never contradict them.\n'
            "\n"
            "Write a 100–130-word critique: positive yet candid, mentioning at least one tonal or technical point, and one actionable suggestion.\n"
            "Infer and return a creative style label (Open, Nature, Creative, Documentary, Abstract, Record, Other, etc.) separate from the competition category.\n"
            "\n"
            "Return ONLY valid JSON with the schema described below."
        ),
        user_template=(
            "Image input: attached below.\n"
            'Title: "{title}"\n'
            'Category: "{category}"\n'
            "Competition notes: {notes}\n"
            "Caption/context: {caption}\n"
            "Compliance summary: {compliance_findings}\n"
            "Keyword highlights: {keyword_preview}\n"
            "{technical_analysis_block}\n\n"
            "Instruction: Evaluate this image for a club competition using the rubric described in the system prompt and the TECHNICAL ANALYSIS above. Keep the critique 100–130 words.\n"
            "Return JSON only {{\\n"
            '  "title": "<string|null>",\\n'
            '  "category": "Colour|Mono|Open|Nature|Creative|Themed|Documentary|null",\\n'
            '  "style": "Open|Nature|Creative|Documentary|Abstract|Record|Other|null",\\n'
            '  "critique": "<100-130 words>",\\n'
            '  "score": 0-20,\\n'
            '  "subscores": {{\\n'
            '    "impact": 0-5,\\n'
            '    "composition": 0-5,\\n'
            '    "technical": 0-5,\\n'
            '    "category_fit": 0-5\\n'
            "  }}\\n"
            "}}."
        ),
    )
}


def get_prompt(identifier: str) -> JudgePromptProfile:
    if identifier not in _PROMPTS:
        raise KeyError(f"Unknown judge prompt '{identifier}'")
    return _PROMPTS[identifier]


__all__ = ["JudgePromptProfile", "get_prompt"]
