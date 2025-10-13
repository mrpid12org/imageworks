"""Shared quantization token utilities.

Centralises detection of quant tokens across importers and scanners so that
future additions (e.g. q3_k_m, nf4, gptq families) only require editing here.
"""

from __future__ import annotations

import re

_QUANT_PATTERN = re.compile(
    r"^(q\d(?:_[01]|_k(?:_[sml])?)?|int4|int8|fp16|f16|bf16|fp32|f32|awq|gptq|nf4|fp8|iq4_xs)$",
    re.IGNORECASE,
)


def is_quant_token(token: str) -> bool:
    """Return True if the provided token string represents a recognized quant token.

    Normalizes underscores & case; keeps underscores significant for matching patterns like q4_k_m.
    """
    if not token:
        return False
    return bool(_QUANT_PATTERN.match(token.strip()))


__all__ = ["is_quant_token"]
