"""Inference utilities for Judge Vision."""

from __future__ import annotations

import base64
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

import httpx

from imageworks.apps.judge_vision import (
    CompetitionConfig,
    evaluate_compliance,
    TechnicalSignalExtractor,
)
from imageworks.apps.judge_vision.judge_types import TechnicalSignals
from imageworks.apps.judge_vision.models import JudgeVisionRecord
from imageworks.apps.judge_vision.prompts import get_prompt

logger = logging.getLogger(__name__)


def _clamp_subscore(value: Optional[float]) -> float:
    try:
        numeric = float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        numeric = 0.0
    return max(0.0, min(5.0, numeric))


def _normalise_score(total: float) -> int:
    """Clamp summed subscores into an integer 0–20 band."""
    if total is None or isinstance(total, bool):
        total = 0.0
    try:
        numeric = float(total)
    except (TypeError, ValueError):
        numeric = 0.0
    numeric = max(0.0, min(20.0, numeric))
    return int(round(numeric))


class OpenAIChatClient:
    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        api_key: str,
        timeout: int,
    ) -> None:
        self.model_name = model_name
        self._client = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else None,
            timeout=timeout,
        )

    def chat(
        self,
        messages: List[Dict[str, object]],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> Dict[str, object]:
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max(1, max_tokens),
            "temperature": max(0.0, temperature),
            "top_p": max(0.0, min(1.0, top_p)),
            "stream": False,
        }
        response = self._client.post("/chat/completions", json=payload)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:  # noqa: BLE001
            snippet = ""
            try:
                snippet = exc.response.text[:600]
            except Exception:  # noqa: BLE001
                snippet = "<unavailable>"
            logger.error(
                "Judge Vision backend returned HTTP %s for model '%s': %s",
                exc.response.status_code if exc.response else "???",
                self.model_name,
                snippet,
            )
            raise
        return response.json()

    def close(self) -> None:
        self._client.close()


class JudgeVisionInferenceEngine:
    def __init__(
        self,
        config,
        *,
        competition: Optional[CompetitionConfig] = None,
        precomputed_signals: Optional[Dict[str, TechnicalSignals]] = None,
    ) -> None:
        self.config = config
        self.competition = competition
        self._precomputed_signals = precomputed_signals or {}
        self._technical = TechnicalSignalExtractor(
            enable_nima=config.enable_nima,
            enable_musiq=config.enable_musiq,
            use_gpu=(config.iqa_device.lower() == "gpu"),
        )
        self._prompt = get_prompt("club_judge_json")
        self._client_cache: Dict[str, OpenAIChatClient] = {}

    def close(self) -> None:
        for client in self._client_cache.values():
            client.close()

    def process(self, image_path: Path) -> JudgeVisionRecord:
        start = time.perf_counter()
        competition_category = self._infer_competition_category(image_path)
        image_title = self._read_image_title(image_path)
        compliance = evaluate_compliance(image_path, self.competition)
        technical = self._lookup_or_compute_signals(image_path)

        critique_payload = self._run_critique_stage(
            image_path=image_path,
            compliance=compliance,
            technical=technical,
            competition_category=competition_category,
            image_title=image_title,
        )

        subscores_payload = critique_payload.get("subscores") or {}
        impact = _clamp_subscore(_safe_float(subscores_payload.get("impact")))
        composition = _clamp_subscore(_safe_float(subscores_payload.get("composition")))
        technical_score = _clamp_subscore(
            _safe_float(subscores_payload.get("technical"))
        )
        category_fit = _clamp_subscore(
            _safe_float(subscores_payload.get("category_fit"))
        )
        subscores_total = impact + composition + technical_score + category_fit
        mapped_score = _normalise_score(subscores_total)

        inferred_style = self._normalise_style_label(critique_payload.get("style"))
        if inferred_style is None:
            inferred_style = self._normalise_style_label(
                critique_payload.get("category")
            )

        record = JudgeVisionRecord(
            image=image_path,
            critique=critique_payload.get("critique", ""),
            critique_title=critique_payload.get("title"),
            critique_category=competition_category,
            competition_category=competition_category,
            style_inference=inferred_style,
            image_title=image_title,
            critique_total_initial=subscores_total,
            critique_total=float(mapped_score),
            critique_award=critique_payload.get("award_suggestion"),
            critique_compliance_flag=critique_payload.get("compliance_flag"),
            compliance=compliance,
            technical_signals=technical,
            backend=self.config.backend,
            model=self._resolved_model_name(),
            duration_seconds=time.perf_counter() - start,
        )

        record.critique_subscores.impact = impact
        record.critique_subscores.composition = composition
        record.critique_subscores.technical = technical_score
        record.critique_subscores.category_fit = category_fit
        return record

    def _lookup_or_compute_signals(self, image_path: Path) -> TechnicalSignals:
        key = str(image_path.resolve())
        cached = self._precomputed_signals.get(key)
        if cached:
            return cached
        cached = self._technical.run(image_path)
        self._precomputed_signals[key] = cached
        return cached

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolved_model_name(self) -> str:
        if self.config.model:
            return self.config.model
        return self.config.critique_role or "judge"

    def _get_client(self) -> OpenAIChatClient:
        key = self._resolved_model_name()
        if key not in self._client_cache:
            self._client_cache[key] = OpenAIChatClient(
                base_url=self.config.base_url,
                model_name=key,
                api_key=self.config.api_key,
                timeout=self.config.timeout,
            )
        return self._client_cache[key]

    def _run_critique_stage(
        self,
        *,
        image_path: Path,
        compliance,
        technical,
        competition_category: str,
        image_title: Optional[str],
    ) -> Dict[str, object]:
        encoded = self._encode_image(image_path)
        messages = [
            {"role": "system", "content": self._prompt.system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": self._prompt.render_user_prompt(
                            title=image_path.stem,
                            category=competition_category,
                            notes=self.config.critique_notes
                            or (
                                self.competition.rules.describe()
                                if self.competition
                                else ""
                            ),
                            caption=image_title or "",
                            keyword_preview="",
                            compliance_findings=(
                                compliance.to_prompt() if compliance else ""
                            ),
                            technical_analysis_block=technical.technical_analysis_block(),
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                    },
                ],
            },
        ]

        response = self._get_client().chat(
            messages,
            max_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )
        try:
            content = (
                response.get("choices", [{}])[0].get("message", {}).get("content", "")
            )
            payload = json.loads(content)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to parse critique JSON: %s", exc)
            payload = {"critique": content}
        return payload

    @staticmethod
    def _encode_image(image_path: Path) -> str:
        data = image_path.read_bytes()
        return base64.b64encode(data).decode("utf-8")

    @staticmethod
    def _infer_competition_category(image_path: Path) -> str:
        stem = image_path.name.upper()
        if stem.startswith("01_"):
            return "Mono"
        return "Colour"

    @staticmethod
    def _normalise_style_label(value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _read_image_title(image_path: Path) -> Optional[str]:
        try:
            result = subprocess.run(
                ["exiftool", "-j", str(image_path)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except FileNotFoundError:
            return None
        except subprocess.CalledProcessError:
            return None

        try:
            data = json.loads(result.stdout)[0]
        except Exception:  # noqa: BLE001
            return None

        title_fields = [
            "Title",
            "ObjectName",
            "Description",
            "ImageDescription",
            "XMP-dc:Title",
            "IPTC:ObjectName",
        ]
        for field in title_fields:
            value = data.get(field)
            if isinstance(value, list):
                value = value[0] if value else None
            if value:
                text = str(value).strip()
                if text:
                    return text
        return None


def _safe_float(value: object) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):  # noqa: BLE001
        return None


__all__ = ["JudgeVisionInferenceEngine"]
