"""Optional LLM explainer for similarity results."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Optional

from imageworks.libs.vlm import VLMBackend, VLMBackendError, create_backend_client

from .config import SimilarityConfig
from .models import CandidateSimilarity
from .prompts import SimilarityPromptProfile, get_prompt_profile

logger = logging.getLogger(__name__)


@dataclass
class SimilarityExplainer:
    """Generate natural-language rationales using an OpenAI-compatible backend."""

    config: SimilarityConfig
    prompt_profile: SimilarityPromptProfile

    def __post_init__(self) -> None:
        backend_name = self.config.backend.lower()
        try:
            self._backend = VLMBackend(backend_name)
        except ValueError as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Unknown backend '{self.config.backend}' for explanations"
            ) from exc
        self._client = create_backend_client(
            self._backend,
            base_url=self.config.base_url,
            model_name=self.config.model,
            api_key=self.config.api_key,
            timeout=self.config.timeout,
        )

    def explain(self, result: CandidateSimilarity) -> Optional[str]:
        if not result.matches:
            return None
        payload = self._build_payload(result)
        try:
            response = self._client.chat_completions(payload)
        except VLMBackendError as exc:  # pragma: no cover - runtime only
            logger.error("Explanation backend error: %s", exc)
            return None
        if response.status_code != 200:
            logger.error("Explanation request failed: HTTP %s", response.status_code)
            return None
        data = response.json()
        content = (
            data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        )
        return content or None

    def close(self) -> None:  # pragma: no cover - exercised in runtime
        self._client.close()

    def _build_payload(self, result: CandidateSimilarity) -> dict:
        best = result.matches[0]
        strategies = best.extra.get("strategies", {})
        notes = "; ".join(result.notes) if result.notes else "(none)"
        user_message = self.prompt_profile.render_user(
            candidate=str(result.candidate),
            best_match=str(best.reference),
            score=result.top_score,
            fail=self.config.fail_threshold,
            query=self.config.query_threshold,
            strategies=json.dumps(strategies, ensure_ascii=False, sort_keys=True),
            verdict=result.verdict.value.upper(),
            notes=notes,
        )
        messages = [
            {"role": "system", "content": self.prompt_profile.system_prompt},
            {"role": "user", "content": user_message},
        ]
        return {
            "model": self.config.model,
            "messages": messages,
            "temperature": 0.2,
            "top_p": 0.9,
            "max_tokens": 256,
        }


def create_explainer(config: SimilarityConfig) -> SimilarityExplainer:
    profile = get_prompt_profile(config.prompt_profile)
    return SimilarityExplainer(config=config, prompt_profile=profile)
