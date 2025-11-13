"""Configuration helpers for Judge Vision."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

from imageworks.apps.personal_tagger.core.config import (
    PersonalTaggerSettings,
    load_config as load_personal_settings,
)


@dataclass(frozen=True)
class JudgeVisionSettings(PersonalTaggerSettings):
    """Inherit defaults from personal tagger settings for reuse."""


@dataclass
class JudgeVisionConfig:
    input_paths: List[Path]
    recursive: bool
    image_extensions: Tuple[str, ...]
    backend: str
    base_url: str
    api_key: str
    timeout: int
    max_new_tokens: int
    temperature: float
    top_p: float
    model: Optional[str]
    use_registry: bool
    critique_role: Optional[str]
    skip_preflight: bool
    dry_run: bool
    competition_id: Optional[str]
    competition_config: Optional[Path]
    pairwise_rounds: Optional[int]
    pairwise_enabled: bool
    pairwise_threshold: int
    critique_title_template: str
    critique_category: Optional[str]
    critique_notes: str
    output_jsonl: Path
    summary_path: Path
    progress_path: Path
    enable_musiq: bool
    enable_nima: bool
    iqa_cache_path: Path
    stage: str
    iqa_device: str
    gpu_lease_token: Optional[str] = None

    def resolved_input_paths(self) -> List[Path]:
        return [path.expanduser().resolve() for path in self.input_paths]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["input_paths"] = [str(path) for path in self.input_paths]
        data["competition_config"] = (
            str(self.competition_config) if self.competition_config else None
        )
        data["output_jsonl"] = str(self.output_jsonl)
        data["summary_path"] = str(self.summary_path)
        data["progress_path"] = str(self.progress_path)
        data["iqa_cache_path"] = str(self.iqa_cache_path)
        data["pairwise_threshold"] = self.pairwise_threshold
        data["pairwise_enabled"] = self.pairwise_enabled
        data["gpu_lease_token"] = self.gpu_lease_token
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "JudgeVisionConfig":
        def _to_path(value):
            return Path(value) if value is not None else None

        return cls(
            input_paths=[Path(path) for path in data["input_paths"]],
            recursive=bool(data["recursive"]),
            image_extensions=tuple(data["image_extensions"]),
            backend=data["backend"],
            base_url=data["base_url"],
            api_key=data["api_key"],
            timeout=int(data["timeout"]),
            max_new_tokens=int(data["max_new_tokens"]),
            temperature=float(data["temperature"]),
            top_p=float(data["top_p"]),
            model=data.get("model"),
            use_registry=bool(data["use_registry"]),
            critique_role=data.get("critique_role"),
            skip_preflight=bool(data["skip_preflight"]),
            dry_run=bool(data["dry_run"]),
            competition_id=data.get("competition_id"),
            competition_config=_to_path(data.get("competition_config")),
            pairwise_rounds=(
                int(data["pairwise_rounds"])
                if data.get("pairwise_rounds") is not None
                else None
            ),
            pairwise_enabled=bool(data.get("pairwise_enabled")),
            pairwise_threshold=int(data.get("pairwise_threshold", 17)),
            critique_title_template=data["critique_title_template"],
            critique_category=data.get("critique_category"),
            critique_notes=data.get("critique_notes") or "",
            output_jsonl=Path(data["output_jsonl"]),
            summary_path=Path(data["summary_path"]),
            progress_path=Path(data["progress_path"]),
            enable_musiq=bool(data["enable_musiq"]),
            enable_nima=bool(data["enable_nima"]),
            iqa_cache_path=Path(data["iqa_cache_path"]),
            stage=data.get("stage", "full"),
            iqa_device=data.get("iqa_device", "cpu"),
            gpu_lease_token=data.get("gpu_lease_token"),
        )


def build_runtime_config(
    *,
    settings: Optional[JudgeVisionSettings] = None,
    input_dirs: Optional[List[Path]] = None,
    recursive: Optional[bool] = None,
    backend: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: Optional[int] = None,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    model: Optional[str] = None,
    use_registry: Optional[bool] = None,
    critique_role: Optional[str] = None,
    skip_preflight: Optional[bool] = None,
    dry_run: Optional[bool] = None,
    competition_config: Optional[Path] = None,
    competition_id: Optional[str] = None,
    pairwise_rounds: Optional[int] = None,
    pairwise_enabled: Optional[bool] = None,
    pairwise_threshold: Optional[int] = None,
    critique_title_template: Optional[str] = None,
    critique_category: Optional[str] = None,
    critique_notes: Optional[str] = None,
    output_jsonl: Optional[Path] = None,
    summary_path: Optional[Path] = None,
    progress_path: Optional[Path] = None,
    enable_musiq: Optional[bool] = None,
    enable_nima: Optional[bool] = None,
    iqa_cache_path: Optional[Path] = None,
    stage: Optional[str] = None,
    iqa_device: Optional[str] = None,
) -> JudgeVisionConfig:
    settings = settings or load_personal_settings()
    input_paths = input_dirs or settings.default_input_dirs
    if not input_paths:
        raise ValueError("At least one --input-dir is required")

    progress_default = Path("outputs/metrics/judge_vision_progress.json")

    return JudgeVisionConfig(
        input_paths=input_paths,
        recursive=bool(
            recursive if recursive is not None else settings.default_recursive
        ),
        image_extensions=tuple(settings.image_extensions),
        backend=(backend or settings.default_backend).strip(),
        base_url=(base_url or settings.default_base_url).strip(),
        api_key=(api_key or settings.default_api_key or "").strip(),
        timeout=timeout if timeout is not None else settings.default_timeout,
        max_new_tokens=(
            max_new_tokens
            if max_new_tokens is not None
            else settings.default_max_new_tokens
        ),
        temperature=(
            temperature if temperature is not None else settings.default_temperature
        ),
        top_p=top_p if top_p is not None else settings.default_top_p,
        model=model
        or getattr(settings, "description_model", None)
        or getattr(settings, "default_model", ""),
        use_registry=bool(
            use_registry if use_registry is not None else settings.default_use_registry
        ),
        critique_role=critique_role or "description",
        skip_preflight=bool(skip_preflight),
        dry_run=True if dry_run is None else dry_run,
        competition_config=competition_config,
        competition_id=competition_id,
        pairwise_rounds=pairwise_rounds,
        pairwise_enabled=bool(pairwise_enabled) or bool(pairwise_rounds),
        pairwise_threshold=pairwise_threshold if pairwise_threshold is not None else 17,
        critique_title_template=critique_title_template or "{stem}",
        critique_category=critique_category,
        critique_notes=critique_notes or "",
        output_jsonl=output_jsonl or Path("outputs/results/judge_vision.jsonl"),
        summary_path=summary_path or Path("outputs/summaries/judge_vision_summary.md"),
        progress_path=progress_path or progress_default,
        enable_musiq=True if enable_musiq is None else enable_musiq,
        enable_nima=True if enable_nima is None else enable_nima,
        iqa_cache_path=iqa_cache_path or Path("outputs/cache/judge_vision_iqa.jsonl"),
        stage=(stage or "full"),
        iqa_device=(iqa_device or "cpu"),
    )


__all__ = ["JudgeVisionConfig", "JudgeVisionSettings", "build_runtime_config"]
