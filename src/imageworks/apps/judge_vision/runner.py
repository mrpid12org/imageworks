"""Execution harness for Judge Vision pipeline."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime

try:  # Python 3.11+
    from datetime import UTC  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - Py3.10 compatibility
    from datetime import timezone

    UTC = timezone.utc
from pathlib import Path
from typing import Dict, List, Optional

from imageworks.apps.judge_vision import (
    CompetitionConfig,
    load_competition_registry,
)
from imageworks.apps.judge_vision.config import JudgeVisionConfig
from imageworks.apps.judge_vision.inference import JudgeVisionInferenceEngine
from imageworks.apps.judge_vision.models import JudgeVisionRecord
from imageworks.apps.judge_vision.progress import ProgressTracker
from imageworks.apps.judge_vision.technical_signals import TechnicalSignalExtractor
from imageworks.apps.judge_vision.judge_types import (
    ComplianceReport,
    RubricScores,
    TechnicalSignals,
)
from imageworks.apps.judge_vision.pairwise_playoff import (
    PairwisePlayoffRunner,
    recommended_pairwise_rounds,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _PairwiseCategoryPlan:
    category: str
    records: List["JudgeVisionRecord"]
    eligible_count: int
    comparisons: int


def _phase_label(stage: str) -> str:
    stage = stage.lower()
    if stage == "iqa":
        return "Stage 1 – IQA"
    if stage == "critique":
        return "Stage 2 – Critique"
    if stage == "pairwise":
        return "Stage 3 – Pairwise"
    if stage == "full":
        return "Judge Vision"
    return stage.title() if stage else "Judge Vision"


class JudgeVisionRunner:
    def __init__(self, config: JudgeVisionConfig) -> None:
        self.config = config
        self.progress = ProgressTracker(config.progress_path)

    def run(self) -> List[JudgeVisionRecord]:
        images = self._discover_images()
        if not images:
            logger.warning("Judge Vision discovered no images")
            return []

        stage = (self.config.stage or "full").lower()
        self.progress.reset(total=len(images), phase=_phase_label(stage))
        competition = self._load_competition()

        if stage == "iqa":
            self._run_stage_iqa_only(images)
            self.progress.complete()
            logger.info("Judge Vision IQA stage completed for %d images", len(images))
            return []

        precomputed = None
        if stage == "critique":
            precomputed = self._load_iqa_cache()

        engine = JudgeVisionInferenceEngine(
            self.config,
            competition=competition,
            precomputed_signals=precomputed,
        )
        records: List[JudgeVisionRecord] = []
        current_path: Optional[Path] = None
        self.config.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        jsonl_handle = self.config.output_jsonl.open("w", encoding="utf-8")
        write_iqa = stage == "full"
        try:
            for idx, image_path in enumerate(images, 1):
                current_path = image_path
                record = engine.process(image_path)
                records.append(record)
                self._append_jsonl_record(jsonl_handle, record)
                if write_iqa:
                    self._write_iqa_cache(record)
                self.progress.update(processed=idx, current_image=str(image_path))

            should_run_pairwise = self.config.pairwise_enabled and stage in {
                "full",
                "critique",
            }
            if should_run_pairwise:
                # Close out the critique phase so the GUI records completion before Stage 3 starts.
                self.progress.complete()
                plans = self._build_pairwise_plans(records)
                if not plans:
                    self.progress.reset(total=1, phase=_phase_label("pairwise"))
                    msg = f"No eligible ≥{self.config.pairwise_threshold} images"
                    self.progress.update(processed=1, current_image=msg)
                    self.progress.complete()
                else:
                    model_name = engine._resolved_model_name()
                    for plan in plans:
                        self._run_pairwise_plan(plan, model_name=model_name)
            else:
                self.progress.complete()

            jsonl_handle.close()
            self._rewrite_jsonl(records)
            self._write_summary(records)
            logger.info("Judge Vision completed for %d images", len(records))
            return records
        except Exception as exc:
            jsonl_handle.close()
            self.progress.fail(
                processed=len(records),
                current_image=str(current_path) if current_path else None,
                message=str(exc),
            )
            logger.exception("Judge Vision failed after %d image(s)", len(records))
            raise
        finally:
            engine.close()

    # ------------------------------------------------------------------
    def _discover_images(self) -> List[Path]:
        discovered: List[Path] = []
        for base in self.config.resolved_input_paths():
            if not base.exists():
                logger.warning("Input path does not exist: %s", base)
                continue
            if base.is_file():
                if self._matches_extension(base) and not self._should_ignore_file(base):
                    discovered.append(base)
                continue
            iterator = base.rglob("*") if self.config.recursive else base.glob("*")
            for candidate in iterator:
                if (
                    candidate.is_file()
                    and self._matches_extension(candidate)
                    and not self._should_ignore_file(candidate)
                ):
                    discovered.append(candidate)
        return sorted({path.resolve() for path in discovered})

    def _matches_extension(self, path: Path) -> bool:
        suffix = path.suffix.lower()
        return suffix in {ext.lower() for ext in self.config.image_extensions}

    @staticmethod
    def _should_ignore_file(path: Path) -> bool:
        stem = path.stem.lower()
        return "_lab_" in stem or "backup" in stem

    def _load_competition(self) -> Optional[CompetitionConfig]:
        if not self.config.competition_config or not self.config.competition_id:
            return None
        try:
            registry = load_competition_registry(self.config.competition_config)
            return registry.get(self.config.competition_id)
        except FileNotFoundError:
            logger.warning(
                "Competition config not found: %s", self.config.competition_config
            )
        except Exception:  # noqa: BLE001
            logger.exception("Failed to load competition registry")
        return None

    def _write_jsonl(self, records: List[JudgeVisionRecord]) -> None:
        self.config.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with self.config.output_jsonl.open("w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record.to_json("judge-1.0")) + "\n")

    def _append_jsonl_record(self, handle, record: JudgeVisionRecord) -> None:
        handle.write(json.dumps(record.to_json("judge-1.0")) + "\n")
        handle.flush()

    def _rewrite_jsonl(self, records: List[JudgeVisionRecord]) -> None:
        self._write_jsonl(records)

    def _write_summary(self, records: List[JudgeVisionRecord]) -> None:
        self.config.summary_path.parent.mkdir(parents=True, exist_ok=True)
        lines: List[str] = []
        lines.append(f"# Judge Vision Summary ({datetime.now(UTC).isoformat()})")
        lines.append("")

        grouped: dict[str, List[JudgeVisionRecord]] = {}
        for record in records:
            grouped.setdefault(record.competition_category, []).append(record)

        for category in sorted(grouped):
            lines.append(f"## {category} Entries")
            lines.append("")
            for record in grouped[category]:
                lines.append(f"### {record.image.name}")
                lines.append(
                    f"- Image title: {record.image_title or record.critique_title or 'n/a'}"
                )
                lines.append(f"- Style: {record.style_inference or 'n/a'}")
                final_total = record.critique_total
                raw_total = record.critique_total_initial
                pairwise_initial = record.pairwise_score_initial
                if pairwise_initial is not None and final_total is not None:
                    lines.append(
                        f"- Score: {final_total} (pairwise baseline {pairwise_initial})"
                    )
                elif raw_total is not None and final_total is not None:
                    lines.append(
                        f"- Score: {final_total} (subscore sum {raw_total:.2f})"
                    )
                else:
                    lines.append(f"- Score: {final_total or 'n/a'}")
                lines.append(
                    f"- Deterministic IQA: {record.technical_signals.deterministic_summary()}"
                )
                if record.critique_award:
                    lines.append(f"- Award: {record.critique_award}")
                if record.pairwise_win_ratio is not None:
                    lines.append(
                        f"- Pairwise win ratio: {record.pairwise_win_ratio:.2f} "
                        f"({record.pairwise_wins}/{record.pairwise_comparisons})"
                    )
                lines.append(f"- Critique: {record.critique or '<none>'}")
                lines.append("")

        playoff_records = [
            record
            for record in records
            if record.pairwise_wins is not None and record.pairwise_comparisons
        ]
        if playoff_records:
            lines.append("## Pairwise Adjustments")
            for record in playoff_records:
                lines.append(
                    f"- {record.image.name}: ratio {record.pairwise_win_ratio:.2f} "
                    f"({record.pairwise_wins}/{record.pairwise_comparisons}) → {record.critique_total}"
                )

        self.config.summary_path.write_text("\n".join(lines), encoding="utf-8")

    def _write_iqa_cache(self, record: JudgeVisionRecord) -> None:
        path = self.config.iqa_cache_path
        if not path:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "image": str(record.image),
            "technical_signals": record.technical_signals.to_dict(),
            "timestamp": datetime.now(UTC).isoformat(),
        }
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload) + "\n")

    def _run_stage_iqa_only(self, images: List[Path]) -> None:
        logger.info(
            "Stage 1 IQA starting: %d image(s) on %s",
            len(images),
            "GPU" if self.config.iqa_device.lower() == "gpu" else "CPU",
        )
        extractor = TechnicalSignalExtractor(
            enable_nima=self.config.enable_nima,
            enable_musiq=self.config.enable_musiq,
            use_gpu=self.config.iqa_device.lower() == "gpu",
        )
        cache_path = self.config.iqa_cache_path
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as fh:
            for idx, image_path in enumerate(images, 1):
                signals = extractor.run(image_path)
                payload = {
                    "image": str(image_path.resolve()),
                    "technical_signals": signals.to_dict(),
                    "timestamp": datetime.now(UTC).isoformat(),
                }
                fh.write(json.dumps(payload) + "\n")
                fh.flush()
                self.progress.update(processed=idx, current_image=str(image_path))
                if idx == 1 or idx % 25 == 0 or idx == len(images):
                    logger.info(
                        "Stage 1 IQA progress: %d/%d images processed", idx, len(images)
                    )
        logger.info("Stage 1 IQA finished: %d image(s) processed", len(images))

    def _load_iqa_cache(self) -> Dict[str, TechnicalSignals]:
        path = self.config.iqa_cache_path
        if not path or not path.exists():
            raise FileNotFoundError(
                f"IQA cache not found at {path}. Run stage 'iqa' first or provide --iqa-cache."
            )
        cache: Dict[str, TechnicalSignals] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            image = payload.get("image")
            data = payload.get("technical_signals") or {}
            signals = TechnicalSignals(
                metrics=data.get("metrics") or {},
                notes=data.get("notes") or "",
                tonal_summary=data.get("tonal_summary"),
            )
            if image:
                cache[str(Path(image).resolve())] = signals
        return cache

    def _build_pairwise_plans(
        self, records: List[JudgeVisionRecord]
    ) -> List[_PairwiseCategoryPlan]:
        grouped: Dict[str, List[JudgeVisionRecord]] = {}
        for record in records:
            category = (
                record.competition_category or "Uncategorised"
            ).strip() or "Uncategorised"
            grouped.setdefault(category, []).append(record)

        plans: List[_PairwiseCategoryPlan] = []
        threshold = self.config.pairwise_threshold
        override = self.config.pairwise_rounds
        for category in sorted(grouped):
            cat_records = grouped[category]
            eligible = [
                rec
                for rec in cat_records
                if (rec.critique_total or 0) >= threshold and rec.image.exists()
            ]
            comparisons = override
            if comparisons is None:
                comparisons = recommended_pairwise_rounds(len(eligible))
            plans.append(
                _PairwiseCategoryPlan(
                    category=category,
                    records=cat_records,
                    eligible_count=len(eligible),
                    comparisons=int(comparisons or 0),
                )
            )
        return plans

    def _run_pairwise_plan(
        self, plan: _PairwiseCategoryPlan, *, model_name: str
    ) -> None:
        threshold = self.config.pairwise_threshold
        phase_label = f"Stage 3 – Pairwise ({plan.category})"

        if plan.eligible_count < 2 or plan.comparisons <= 0:
            self.progress.reset(total=1, phase=phase_label)
            if plan.eligible_count == 0:
                msg = f"{plan.category}: No eligible ≥{threshold} images"
            else:
                msg = f"{plan.category}: Need at least two eligible ≥{threshold} images"
            self.progress.update(processed=1, current_image=msg)
            self.progress.complete()
            return

        self.progress.reset(total=1, phase=phase_label)
        playoff_runner = PairwisePlayoffRunner(
            base_url=self.config.base_url,
            model_name=model_name,
            api_key=self.config.api_key,
            timeout=self.config.timeout,
            comparisons_per_image=plan.comparisons,
            threshold=threshold,
        )

        def _on_schedule(total_matches: int) -> None:
            if total_matches <= 0:
                return
            self.progress.reset(total=total_matches, phase=phase_label)

        def _on_progress(
            processed: int,
            total_matches: int,
            category: str,
            left: JudgeVisionRecord,
            right: JudgeVisionRecord,
        ) -> None:
            label = f"{left.image.name} vs {right.image.name} ({category})"
            self.progress.update(processed=processed, current_image=label)

        try:
            matches_run = playoff_runner.apply(
                plan.records,
                on_schedule=_on_schedule,
                on_progress=_on_progress,
            )
            if matches_run == 0:
                self.progress.reset(total=1, phase=phase_label)
                self.progress.update(
                    processed=1,
                    current_image=f"{plan.category}: No eligible ≥{threshold} images",
                )
            self.progress.complete()
        finally:
            playoff_runner.close()


def load_records_from_jsonl(path: Path) -> List[JudgeVisionRecord]:
    if not path.exists():
        raise FileNotFoundError(f"JSONL results not found at {path}")
    records: List[JudgeVisionRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            image_path = Path(payload.get("image"))
            record = JudgeVisionRecord(
                image=image_path,
                critique=payload.get("critique", ""),
                critique_title=payload.get("critique_title"),
                critique_category=payload.get("critique_category"),
                competition_category=payload.get("competition_category") or "Colour",
                style_inference=payload.get("style_inference"),
                image_title=payload.get("image_title"),
                critique_total_initial=payload.get("critique_total_initial"),
                critique_total=payload.get("critique_total"),
                critique_award=payload.get("critique_award"),
                critique_compliance_flag=payload.get("critique_compliance_flag"),
                backend=payload.get("backend", ""),
                model=payload.get("model", ""),
                duration_seconds=payload.get("duration_seconds", 0.0),
                pairwise_score_initial=payload.get("pairwise_score_initial"),
                pairwise_wins=payload.get("pairwise_wins"),
                pairwise_comparisons=payload.get("pairwise_comparisons"),
                pairwise_win_ratio=payload.get("pairwise_win_ratio"),
            )
            record.critique_subscores = RubricScores.from_dict(
                payload.get("critique_subscores")
            )
            record.compliance = ComplianceReport.from_dict(payload.get("compliance"))
            record.technical_signals = TechnicalSignals.from_dict(
                payload.get("technical_signals")
            )
            records.append(record)
    return records


__all__ = ["JudgeVisionRunner", "load_records_from_jsonl"]
