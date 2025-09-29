"""Keyword and text post-processing helpers for the personal tagger."""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence, Set

# Keywords that should never appear in the final list.
BANNED_WORDS: Set[str] = {
    "no",
    "unspecified",
    "unknown",
    "standard",
    "unidentified",
    "time",
    "category",
    "actions",
    "setting",
    "objects",
    "visual",
    "elements",
    "activities",
    "appearance",
    "professions",
    "relationships",
    "identify",
    "photography",
    "photographic",
    "topiary",
    "composition",
    "beauty",
    "mood",
    "various",
    "item",
    "image",
    "photo",
    "picture",
}

GENERIC_SINGLE_WORD_BANS: Set[str] = {
    "angle",
    "color",
    "contrast",
    "focus",
    "light",
    "lighting",
    "sharpness",
    "shot",
}

AND_EXCEPTIONS: Set[str] = {
    "rock and roll",
    "black and white",
    "bread and butter",
    "salt and pepper",
    "give and take",
    "ups and downs",
    "pros and cons",
    "spick and span",
}

SINGULAR_S_WORDS: Set[str] = {
    "alias",
    "analysis",
    "apparatus",
    "asbestos",
    "atlas",
    "axis",
    "basis",
    "bias",
    "billiards",
    "bonus",
    "bus",
    "business",
    "bypass",
    "canvas",
    "chaos",
    "chassis",
    "circus",
    "class",
    "corps",
    "cosmos",
    "crisis",
    "customs",
    "debris",
    "diabetes",
    "diagnosis",
    "dominoes",
    "duress",
    "ethos",
    "eyeglass",
    "focus",
    "fungus",
    "gas",
    "glass",
    "headquarters",
    "hypnosis",
    "jeans",
    "kudos",
    "lens",
    "mathematics",
    "mumps",
    "news",
    "octopus",
    "pathos",
    "pelvis",
    "physics",
    "pliers",
    "plus",
    "process",
    "progress",
    "prognosis",
    "rabies",
    "series",
    "scissors",
    "species",
    "status",
    "surplus",
    "synthesis",
    "thanks",
    "thesis",
    "tongs",
    "trousers",
    "virus",
    "wanderlust",
}


def de_pluralize(word: str) -> str:
    """Apply lightweight de-pluralisation heuristics."""

    if not word or word in SINGULAR_S_WORDS:
        return word
    if word.endswith("ies") and len(word) > 3:
        return word[:-3] + "y"
    if word.endswith("ses") and len(word) > 3:
        return word[:-2]
    if word.endswith("es") and len(word) > 2:
        return word[:-2]
    if word.endswith("s") and len(word) > 1:
        return word[:-1]
    return word


def normalise_keyword(keyword: str) -> Iterable[str]:
    """Normalise a keyword candidate, yielding zero or more tokens."""

    keyword = keyword.strip().lower()
    keyword = re.sub(r"[^\w\s-]", "", keyword)
    keyword = re.sub(r"\s+", " ", keyword)

    if not keyword:
        return []

    words = keyword.split()
    if words[0] in BANNED_WORDS:
        return []
    if len(words) > 3:
        return []

    if any(banned in words for banned in BANNED_WORDS):
        return []

    if len(words) == 1 and words[0] in GENERIC_SINGLE_WORD_BANS:
        return []

    if (
        len(words) == 3
        and words[1] in {"and", "or"}
        and " ".join(words) not in AND_EXCEPTIONS
    ):
        first = de_pluralize(words[0])
        second = de_pluralize(words[2])
        return [first, second]

    if len(words) > 1:
        words[-1] = de_pluralize(words[-1])
    else:
        words[0] = de_pluralize(words[0])

    return [" ".join(words)]


def filter_keywords(keywords: Sequence[str]) -> List[str]:
    """Remove substrings and near-duplicates while preserving order."""

    # Remove duplicates while preserving order.
    deduped: List[str] = []
    seen: Set[str] = set()
    for keyword in keywords:
        lowered = keyword.lower()
        if lowered not in seen:
            seen.add(lowered)
            deduped.append(keyword)

    # Remove keywords that are strict substrings of others.
    filtered: List[str] = []
    for keyword in deduped:
        if any(keyword != other and keyword in other for other in deduped):
            continue
        filtered.append(keyword)

    # Basic stemming heuristic on the final word.
    final: List[str] = []
    seen_roots: Set[str] = set()
    for keyword in filtered:
        last_word = keyword.split()[-1]
        root = de_pluralize(last_word)
        if root not in seen_roots:
            seen_roots.add(root)
            final.append(keyword)

    return final


def clean_keywords(raw_keywords: Iterable[str]) -> List[str]:
    """Return a cleaned, deduplicated keyword list."""

    normalised: List[str] = []
    for keyword in raw_keywords:
        if not isinstance(keyword, str):
            continue
        normalised.extend(normalise_keyword(keyword))

    return filter_keywords(normalised)


def tidy_caption(text: str) -> str:
    """Normalise caption spacing and ensure trailing punctuation."""

    caption = " ".join(text.strip().split())
    if not caption:
        return ""
    if caption[-1] not in {".", "!", "?"}:
        caption += "."
    return caption


def tidy_description(text: str) -> str:
    """Collapse whitespace in descriptions."""

    return " ".join(text.strip().split())
