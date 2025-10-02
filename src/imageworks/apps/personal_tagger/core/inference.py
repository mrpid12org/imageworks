"""Inference backends for the personal tagger using OpenAI-compatible APIs."""

from __future__ import annotations

import base64
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from imageworks.libs.vlm import VLMBackend, VLMBackendError, create_backend_client
from imageworks.model_loader.role_selection import select_by_role
from imageworks.model_loader.service import select_model, CapabilityError

from . import prompts as prompt_registry
from .config import PersonalTaggerConfig
from .models import GenerationModels, KeywordPrediction, PersonalTaggerRecord
from .post_processing import clean_keywords, tidy_caption, tidy_description

logger = logging.getLogger(__name__)


@dataclass
class ChatResult:
    """Simple container for chat completion results."""

    content: str
    raw_response: Dict[str, object]


class OpenAIChatClient:
    """Thin wrapper around the OpenAI-compatible backend adapters."""

    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        api_key: str,
        backend: str,
        timeout: int,
    ) -> None:
        enum_backend = self._resolve_backend(backend)
        self.model_name = model_name
        self._client = create_backend_client(
            enum_backend,
            base_url=base_url,
            model_name=model_name,
            api_key=api_key,
            timeout=timeout,
        )
        self.timeout = timeout

    @staticmethod
    def _resolve_backend(value: str) -> VLMBackend:
        try:
            return VLMBackend(value.lower())
        except ValueError as exc:  # noqa: BLE001
            raise ValueError(f"Unknown backend '{value}'") from exc

    def chat(
        self,
        *,
        messages: List[Dict[str, object]],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> ChatResult:
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max(1, max_tokens),
            "temperature": max(0.0, temperature),
            "top_p": max(0.0, min(top_p, 1.0)),
            "stream": False,
        }

        response = self._client.chat_completions(payload)
        if response.status_code != 200:
            raise VLMBackendError(f"HTTP {response.status_code}: {response.text[:200]}")

        data = response.json()
        content = (
            data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        )
        return ChatResult(content=content, raw_response=data)

    def close(self) -> None:
        self._client.close()


class BaseInferenceEngine:
    """Common interface for tagger inference backends."""

    def __init__(self, config: PersonalTaggerConfig) -> None:
        self.config = config

    def process(self, image_path: Path) -> PersonalTaggerRecord:
        raise NotImplementedError

    def close(self) -> None:
        """Hook for releasing backend resources."""


class OpenAIInferenceEngine(BaseInferenceEngine):
    """Sequential inference pipeline using OpenAI-compatible servers."""

    def __init__(self, config: PersonalTaggerConfig) -> None:
        super().__init__(config)
        self._clients: Dict[str, OpenAIChatClient] = {}
        self._prompt_profile = prompt_registry.get_prompt_profile(config.prompt_profile)

    def close(self) -> None:  # pragma: no cover - exercised in runtime only
        for client in self._clients.values():
            client.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process(self, image_path: Path) -> PersonalTaggerRecord:
        start = time.perf_counter()
        image_b64 = self._encode_image(image_path)

        # Resolve role → logical model names once per run if registry mode enabled
        if self.config.use_registry:
            # Cache resolved names to avoid re-resolution on each stage
            self._resolved_caption_model = getattr(
                self, "_resolved_caption_model", None
            ) or self._resolve_role_model(
                self.config.caption_role, self.config.caption_model
            )
            self._resolved_keyword_model = getattr(
                self, "_resolved_keyword_model", None
            ) or self._resolve_role_model(
                self.config.keyword_role, self.config.keyword_model
            )
            self._resolved_description_model = getattr(
                self, "_resolved_description_model", None
            ) or self._resolve_role_model(
                self.config.description_role, self.config.description_model
            )
            logger.info(
                "role_resolution",
                extra={
                    "event_type": "role_resolution",
                    "caption_role": self.config.caption_role,
                    "caption_model": self._resolved_caption_model,
                    "keyword_role": self.config.keyword_role,
                    "keyword_model": self._resolved_keyword_model,
                    "description_role": self.config.description_role,
                    "description_model": self._resolved_description_model,
                },
            )
        else:
            self._resolved_caption_model = self.config.caption_model
            self._resolved_keyword_model = self.config.keyword_model
            self._resolved_description_model = self.config.description_model

        caption_text, caption_error = self._run_caption_stage(image_b64)
        keyword_strings, keyword_error = self._run_keyword_stage(image_b64)
        description_text, description_error = self._run_description_stage(
            image_b64, caption_text, keyword_strings
        )

        keywords = self._convert_keywords(keyword_strings)
        duration = time.perf_counter() - start

        notes: List[str] = []
        for marker in (caption_error, keyword_error, description_error):
            if marker:
                notes.append(marker)

        record = PersonalTaggerRecord(
            image=image_path,
            keywords=keywords,
            caption=caption_text,
            description=description_text,
            duration_seconds=duration,
            backend=self.config.backend,
            models=GenerationModels(
                caption=self._resolved_caption_model,
                keywords=self._resolved_keyword_model,
                description=self._resolved_description_model,
            ),
            metadata_written=False,
            notes="; ".join(notes),
        )
        return record

    # ------------------------------------------------------------------
    # Stage executors
    # ------------------------------------------------------------------
    def _run_caption_stage(self, image_b64: str) -> tuple[str, Optional[str]]:
        stage = self._prompt_profile.caption_stage
        try:
            model_name = getattr(
                self, "_resolved_caption_model", self.config.caption_model
            )
            result = self._chat(
                model=model_name,
                messages=[
                    {"role": "system", "content": stage.system},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": stage.render()},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}"
                                },
                            },
                        ],
                    },
                ],
                max_tokens=stage.max_new_tokens or self.config.max_new_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Caption generation failed")
            return "", f"caption_error: {exc}"

        caption = tidy_caption(self._extract_text(result.content))
        return caption, None if caption else "caption_empty"

    def _run_keyword_stage(self, image_b64: str) -> tuple[List[str], Optional[str]]:
        stage = self._prompt_profile.keyword_stage
        try:
            model_name = getattr(
                self, "_resolved_keyword_model", self.config.keyword_model
            )
            result = self._chat(
                model=model_name,
                messages=[
                    {"role": "system", "content": stage.system},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": stage.render()},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}"
                                },
                            },
                        ],
                    },
                ],
                max_tokens=stage.max_new_tokens or max(self.config.max_new_tokens, 300),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Keyword generation failed")
            return [], f"keyword_error: {exc}"

        keywords = self._parse_keyword_response(result.content)
        cleaned = clean_keywords(keywords)
        return cleaned, None if cleaned else "keyword_empty"

    def _run_description_stage(
        self,
        image_b64: str,
        caption: str,
        keywords: List[str],
    ) -> tuple[str, Optional[str]]:
        stage = self._prompt_profile.description_stage
        try:
            keyword_preview = ", ".join(keywords[:10]) if keywords else "none"
            model_name = getattr(
                self, "_resolved_description_model", self.config.description_model
            )
            result = self._chat(
                model=model_name,
                messages=[
                    {"role": "system", "content": stage.system},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": stage.render(
                                    caption=caption or "N/A",
                                    keyword_preview=keyword_preview,
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}"
                                },
                            },
                        ],
                    },
                ],
                max_tokens=stage.max_new_tokens or max(self.config.max_new_tokens, 600),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Description generation failed")
            return "", f"description_error: {exc}"

        description = tidy_description(self._extract_text(result.content))
        return description, None if description else "description_empty"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _chat(
        self,
        *,
        model: str,
        messages: List[Dict[str, object]],
        max_tokens: Optional[int] = None,
    ) -> ChatResult:
        client = self._get_client(model)
        result = client.chat(
            messages=messages,
            max_tokens=max_tokens or self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )
        return result

    def _get_client(self, model_name: str) -> OpenAIChatClient:
        # If registry mode enabled, use selection service to derive endpoint + internal id.
        if self.config.use_registry:
            try:
                selected = select_model(model_name, require_capabilities=["vision"])
            except CapabilityError as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"Registry model '{model_name}' missing capabilities: {exc}"
                ) from exc
            endpoint = selected.endpoint_url
            internal_name = selected.internal_model_id
            cache_key = f"{model_name}@{endpoint}"
            client = self._clients.get(cache_key)
            if client is None:
                client = OpenAIChatClient(
                    base_url=endpoint,
                    model_name=internal_name,
                    api_key=self.config.api_key,
                    backend=selected.backend,
                    timeout=self.config.timeout,
                )
                self._clients[cache_key] = client
            return client

        # Legacy direct backend path
        client = self._clients.get(model_name)
        if client is None:
            client = OpenAIChatClient(
                base_url=self.config.base_url,
                model_name=model_name,
                api_key=self.config.api_key,
                backend=self.config.backend,
                timeout=self.config.timeout,
            )
            self._clients[model_name] = client
        return client

    # Registry-role resolution helpers (used before stages when enabled)
    def _resolve_role_model(self, role: str, fallback: str) -> str:
        if not self.config.use_registry:
            return fallback
        try:
            logical_name = select_by_role(role, preferred=[fallback])
            return logical_name
        except CapabilityError as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to resolve role '{role}': {exc}") from exc

    @staticmethod
    def _encode_image(path: Path) -> str:
        try:
            with path.open("rb") as handle:
                return base64.b64encode(handle.read()).decode("utf-8")
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to read image {path}: {exc}") from exc

    @staticmethod
    def _extract_text(content: str) -> str:
        text = content.strip()
        if text.startswith("```"):
            # Strip markdown code fences if provided.
            text = text.removeprefix("```json").removeprefix("```")
            if text.endswith("```"):
                text = text[:-3]
        return text.strip().strip('"')

    @staticmethod
    def _parse_keyword_response(content: str) -> List[str]:
        text = OpenAIInferenceEngine._extract_text(content)
        payload = OpenAIInferenceEngine._safe_json_loads(text)
        if isinstance(payload, dict):
            payload = payload.get("keywords")
        if not isinstance(payload, list):
            return []
        result: List[str] = []
        for item in payload:
            if isinstance(item, str):
                result.append(item)
        return result

    @staticmethod
    def _safe_json_loads(text: str) -> Optional[object]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("[")
            end = text.rfind("]")
            if start != -1 and end != -1 and start < end:
                snippet = text[start : end + 1]
                try:
                    return json.loads(snippet)
                except json.JSONDecodeError:
                    pass
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and start < end:
                snippet = text[start : end + 1]
                try:
                    return json.loads(snippet)
                except json.JSONDecodeError:
                    pass
        return None

    @staticmethod
    def _convert_keywords(keywords: Iterable[str]) -> List[KeywordPrediction]:
        keywords_list = [kw.strip() for kw in keywords if kw.strip()]
        total = len(keywords_list)
        predictions: List[KeywordPrediction] = []
        for index, keyword in enumerate(keywords_list):
            score = 1.0 - (index / max(total, 1))
            predictions.append(KeywordPrediction(keyword=keyword, score=score))
        return predictions


class FakeInferenceEngine(BaseInferenceEngine):
    """Test inference engine that generates fake data without AI calls."""

    def __init__(self, config: PersonalTaggerConfig) -> None:
        super().__init__(config)
        self._models = GenerationModels(
            caption=f"fake/{config.caption_model}",
            keywords=f"fake/{config.keyword_model}",
            description=f"fake/{config.description_model}",
        )

    def process(self, image_path: Path) -> PersonalTaggerRecord:
        # Generate fake but realistic-looking data
        import time

        time.sleep(0.01)  # Small delay to simulate processing

        return PersonalTaggerRecord(
            image=image_path,
            keywords=[
                KeywordPrediction(keyword="motorcycle", score=1.0),
                KeywordPrediction(keyword="racing", score=0.95),
                KeywordPrediction(keyword="speed", score=0.90),
                KeywordPrediction(keyword="helmet", score=0.85),
                KeywordPrediction(keyword="number", score=0.80),
                KeywordPrediction(keyword="11", score=0.75),
                KeywordPrediction(keyword="sponsor", score=0.70),
                KeywordPrediction(keyword="advertisement", score=0.65),
            ],
            caption="A rider in a red and white suit leans into a turn on a track, with a blurred background of another motorcycle.",
            description="A rider clad in a striking red and white suit, marked with the number 11, leans into a sharp turn on a racetrack, showcasing the precision and speed of motorcycle racing. The rider's helmet, adorned with a shark logo, adds to the intensity of the scene. The blurred background of another motorcycle emphasizes the high-speed motion, while the track's edge, lined with red and white, frames the action.",
            duration_seconds=0.01,
            backend=f"fake-{self.config.backend}",
            models=self._models,
            metadata_written=False,
            notes="dry-run: fake test data",
        )


def create_inference_engine(config: PersonalTaggerConfig) -> BaseInferenceEngine:
    """Factory to create the appropriate inference engine."""

    if config.dry_run:
        return FakeInferenceEngine(config)
    else:
        return OpenAIInferenceEngine(config)
