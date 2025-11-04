"""Execution harness for the Personal Tagger."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from imageworks.model_loader.metrics import BatchRunMetrics

from .config import PersonalTaggerConfig
from .inference import BaseInferenceEngine, create_inference_engine
from .judge_types import PairwiseReport
from .metadata_writer import PersonalMetadataWriter
from .models import PersonalTaggerRecord
from .pairwise import JudgeVisionEntry, run_pairwise_tournament

logger = logging.getLogger(__name__)


class PersonalTaggerRunner:
    """Coordinate discovery, inference, and metadata persistence."""

    def __init__(
        self,
        config: PersonalTaggerConfig,
        *,
        inference_engine: Optional[BaseInferenceEngine] = None,
        metadata_writer: Optional[PersonalMetadataWriter] = None,
    ) -> None:
        self.config = config
        self._inference_engine = inference_engine
        self._metadata_writer = metadata_writer

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def run(self) -> List[PersonalTaggerRecord]:
        logger.info("Starting personal tagger run")
        # Optional preflight before any heavy work
        if getattr(self.config, "preflight", False):
            try:
                self._run_preflight()
            except Exception:  # pragma: no cover - defensive; we log details
                logger.exception("Preflight checks failed; aborting run")
                raise
        images = self.discover_images()
        if not images:
            logger.warning("No images discovered for personal tagging")

        batch_metrics = BatchRunMetrics(
            model_name=self.config.description_model,
            backend=self.config.backend,
        )
        records: List[PersonalTaggerRecord] = []
        for image_path in images:
            stage_timing = batch_metrics.start_stage("image")
            record = self._process_image(image_path)
            batch_metrics.end_stage(stage_timing)
            records.append(record)

        pairwise_report = self._maybe_run_pairwise(records)
        if pairwise_report:
            for record in records:
                record.pairwise = pairwise_report

        batch_metrics.close_batch()
        self._write_jsonl(records)
        self._write_summary(records)
        self._write_batch_metrics(batch_metrics)

        logger.info(
            "Completed personal tagging for %d image(s) (dry_run=%s)",
            len(records),
            self.config.dry_run,
        )
        return records

    # ------------------------------------------------------------------
    # Discovery & processing
    # ------------------------------------------------------------------
    def discover_images(self) -> List[Path]:
        discovered: List[Path] = []
        for base in self.config.input_paths:
            if not base.exists():
                logger.warning("Input path does not exist: %s", base)
                continue
            if base.is_file():
                if self._matches_extension(base):
                    discovered.append(base)
                continue

            iterator: Iterable[Path]
            if self.config.recursive:
                iterator = base.rglob("*")
            else:
                iterator = base.glob("*")

            for candidate in iterator:
                if candidate.is_file() and self._matches_extension(candidate):
                    discovered.append(candidate)

        return sorted({path.resolve() for path in discovered})

    def _process_image(self, image_path: Path) -> PersonalTaggerRecord:
        logger.debug("Processing image %s", image_path)
        try:
            record = self._get_inference_engine().process(image_path)
        except Exception as exc:  # pragma: no cover - safety net for inference
            logger.exception("Inference failed for %s", image_path)
            record = PersonalTaggerRecord(
                image=image_path,
                backend=self.config.backend,
                notes=f"inference_failed: {exc}",
            )
            return record

        if self.config.dry_run or self.config.no_meta:
            reason = "dry-run" if self.config.dry_run else "no-meta"
            record.notes = (record.notes + f" | {reason}: metadata skipped").strip(" |")
            return record

        try:
            metadata_written = self._get_metadata_writer().write(image_path, record)
        except Exception as exc:  # pragma: no cover - resilient metadata handling
            logger.exception("Metadata write failed for %s", image_path)
            record.metadata_written = False
            record.notes = (record.notes + f" metadata_failed: {exc}").strip()
            return record

        record.metadata_written = metadata_written
        if not metadata_written and not record.notes:
            record.notes = "metadata unchanged (existing metadata preserved)"
        return record

    def _matches_extension(self, path: Path) -> bool:
        suffix = path.suffix.lower()
        return suffix in {ext.lower() for ext in self.config.image_extensions}

    def _maybe_run_pairwise(
        self, records: Sequence[PersonalTaggerRecord]
    ) -> Optional[PairwiseReport]:
        competition = getattr(self.config, "competition", None)
        if not competition:
            return None
        if getattr(self.config, "pairwise_rounds", 0) <= 0:
            return None
        if len(records) < 2:
            return None
        if not all(record.critique_total is not None for record in records):
            return None

        entries = [
            JudgeVisionEntry(
                image=str(record.image),
                total=float(record.critique_total or 0.0),
                rubric=record.critique_subscores,
            )
            for record in records
        ]
        report = run_pairwise_tournament(
            entries, rounds=int(self.config.pairwise_rounds)
        )

        for record in records:
            if record.critique_award is None:
                award = competition.score_bands.award_for(record.critique_total)
                if award:
                    record.critique_award = award
        return report

    def _write_batch_metrics(self, batch_metrics: BatchRunMetrics) -> None:
        try:
            out_dir = Path("outputs/metrics")
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / "personal_tagger_batch_metrics.json"
            existing: Dict[str, object] = {}
            if path.exists():
                try:
                    existing = json.loads(path.read_text())
                except Exception:  # noqa: BLE001
                    existing = {}
            summary = batch_metrics.summary()
            summary["timestamp"] = datetime.now(UTC).isoformat()
            history = existing.get("history") if isinstance(existing, dict) else None
            if not isinstance(history, list):
                history = []
            history.append(summary)
            payload = {"history": history, "last": summary}
            path.write_text(json.dumps(payload, indent=2))
        except Exception:  # pragma: no cover - metrics persistence is best-effort
            logger.debug("Failed to write batch metrics", exc_info=True)

    # ------------------------------------------------------------------
    # Preflight
    # ------------------------------------------------------------------
    def _run_preflight(self) -> None:
        """Validate basic backend availability and multimodal support.

        The preflight performs three lightweight probes:
          1. /v1/models listing to ensure the server responds.
          2. A minimal text-only chat completion.
          3. A minimal vision chat (1x1 PNG) to confirm image handling.

        Any failure raises a RuntimeError with actionable guidance.
        """
        import base64
        import requests
        import io
        import time
        from PIL import Image

        base_url = self.config.base_url.rstrip("/")
        model_name = self.config.caption_model
        headers = (
            {"Authorization": f"Bearer {self.config.api_key}"}
            if self.config.api_key
            else {}
        )

        # Align request timeout with backend spin-up expectations (vLLM autostart may take ~2 minutes).
        max_wait_seconds = max(int(self.config.timeout), 120)
        connect_timeout = min(15, max_wait_seconds)
        http_timeout = (connect_timeout, max_wait_seconds)
        logger.info(
            "Preflight: allowing up to %ds for chat proxy responses (connect≤%ds)",
            max_wait_seconds,
            connect_timeout,
        )

        # 1. Model list
        models_url = f"{base_url}/models"
        try:
            resp = requests.get(models_url, timeout=http_timeout, headers=headers)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Preflight: failed to connect to {models_url}: {exc}"
            ) from exc
        if resp.status_code != 200:
            raise RuntimeError(
                f"Preflight: model listing returned HTTP {resp.status_code}: {resp.text[:120]}"
            )
        if model_name not in json.dumps(resp.json()):
            logger.warning(
                "Preflight: model '%s' not explicitly listed; continuing but verify served-model-name",
                model_name,
            )

        # 2. Text-only chat
        chat_url = f"{base_url}/chat/completions"
        payload_text = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are a health check."},
                {"role": "user", "content": "Say READY once."},
            ],
            "max_tokens": 4,
        }
        text_deadline = time.monotonic() + max_wait_seconds
        while True:
            try:
                resp = requests.post(
                    chat_url, json=payload_text, timeout=http_timeout, headers=headers
                )
            except requests.exceptions.Timeout as exc:
                if time.monotonic() >= text_deadline:
                    raise RuntimeError(
                        "Preflight: text chat timed out waiting for chat proxy "
                        f"after {max_wait_seconds}s. Model may still be loading."
                    ) from exc
                logger.info(
                    "Preflight: text chat still waiting for backend warm-up; retrying..."
                )
                time.sleep(5)
                continue
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"Preflight: text chat failed: {exc}") from exc
            break
        if resp.status_code != 200:
            raise RuntimeError(
                f"Preflight: text chat failed HTTP {resp.status_code}: {resp.text[:160]}"
            )
        try:
            ready_text = (
                resp.json().get("choices", [{}])[0]["message"]["content"].lower()
            )
            if "ready" not in ready_text:
                logger.debug("Preflight: unexpected text response: %s", ready_text)
        except Exception:  # noqa: BLE001
            logger.debug("Preflight: could not parse text readiness response")

        # 3. Vision chat (generate tiny in-memory image)
        img = Image.new("RGB", (1, 1), (255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        payload_vision = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe the dominant color."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                        },
                    ],
                }
            ],
            "max_tokens": 8,
        }
        vision_deadline = time.monotonic() + max_wait_seconds
        while True:
            try:
                resp = requests.post(
                    chat_url,
                    json=payload_vision,
                    timeout=http_timeout,
                    headers=headers,
                )
            except requests.exceptions.Timeout as exc:
                if time.monotonic() >= vision_deadline:
                    raise RuntimeError(
                        "Preflight: vision chat timed out waiting for chat proxy "
                        f"after {max_wait_seconds}s. Model may still be loading."
                    ) from exc
                logger.info(
                    "Preflight: vision chat still waiting for backend warm-up; retrying..."
                )
                time.sleep(5)
                continue
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"Preflight: vision chat failed: {exc}") from exc
            break
        if resp.status_code != 200:
            raise RuntimeError(
                "Preflight: vision chat failed HTTP %s: %s"
                % (resp.status_code, resp.text[:160])
            )
        logger.info("Preflight succeeded for model '%s'", model_name)

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------
    def _write_jsonl(self, records: Sequence[PersonalTaggerRecord]) -> None:
        output_path = self.config.output_jsonl
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for record in records:
                payload = record.to_json(self.config.json_schema_version)
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        logger.info("Wrote JSONL results to %s", output_path)

    def _write_summary(self, records: Sequence[PersonalTaggerRecord]) -> None:
        summary_path = self.config.summary_path
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        total = len(records)
        lines: List[str] = [
            "# Personal Tagger Summary",
            f"Generated: {datetime.now(UTC).isoformat().replace('+00:00', 'Z')}",
            "",
            f"Processed {total} image(s).",
            f"Backend: {self.config.backend}",
            "",
        ]
        if self.config.dry_run:
            lines.append("_Dry run: metadata writes skipped (inference executed)._")
            lines.append("")
        elif self.config.no_meta:
            lines.append(
                "_No-meta mode: real AI inference used, metadata writes skipped._"
            )
            lines.append("")

        grouped: Dict[Path, List[PersonalTaggerRecord]] = defaultdict(list)
        for record in records:
            grouped[record.image.parent].append(record)

        for directory in sorted(grouped.keys()):
            entries = grouped[directory]
            lines.append(f"## {directory} ({len(entries)} image(s))")
            lines.append("")
            for record in entries:
                # Show all keywords without truncation
                all_keywords = [kw.title() for kw in record.keywords_as_list()]
                keywords = ", ".join(all_keywords) if all_keywords else "<none>"

                # Show full caption and description without truncation
                caption = record.caption.strip() if record.caption else "<none>"
                description = (
                    record.description.strip() if record.description else "<none>"
                )
                critique = record.critique.strip() if record.critique else "<none>"

                status = (
                    "metadata written"
                    if record.metadata_written
                    else "metadata pending"
                )
                if record.notes:
                    status = f"{status} ({record.notes})"
                lines.append(f"- **{record.image.name}** — {status}")
                lines.append(f"  - Keywords: {keywords}")
                lines.append(f"  - Caption: {caption}")
                lines.append(f"  - Description: {description}")
                if record.critique_title:
                    lines.append(f"  - Critique title: {record.critique_title}")
                if record.critique_category:
                    lines.append(f"  - Critique category: {record.critique_category}")
                if record.critique_total is not None:
                    lines.append(
                        f"  - Judge total: {record.critique_total:.1f}/20"
                    )
                elif record.critique_score is not None:
                    lines.append(f"  - Critique score: {record.critique_score}/20")
                scores_summary = record.critique_subscores.summary()
                if scores_summary and scores_summary != "<no scores>":
                    lines.append(f"  - Subscores: {scores_summary}")
                if record.critique_award:
                    lines.append(f"  - Award suggestion: {record.critique_award}")
                if record.critique_compliance_flag:
                    lines.append(
                        f"  - Compliance flag: {record.critique_compliance_flag}"
                    )
                elif record.compliance:
                    lines.append(
                        f"  - Compliance: {record.compliance.to_prompt()}"
                    )
                lines.append(
                    f"  - Technical signals: {record.technical_signals.to_prompt()}"
                )
                lines.append(f"  - Critique summary: {critique}")
            lines.append("")

        report = records[0].pairwise if records and records[0].pairwise else None
        if report:
            lines.append("## Pairwise Tournament")
            lines.append("")
            for match in report.rounds:
                left_name = Path(match.left_image).name
                right_name = Path(match.right_image).name
                winner_name = Path(match.winner).name
                lines.append(
                    f"- Round {match.round_index}: {left_name} vs {right_name} → **{winner_name}** ({match.reason})"
                )
            lines.append("")
            lines.append("### Final Rankings")
            for idx, entry in enumerate(report.final_rankings, start=1):
                lines.append(
                    f"{idx}. {Path(str(entry['image'])).name} — {entry['score']:.1f}"
                )
            if report.stability_metrics:
                metrics = ", ".join(
                    f"{key}={value}" for key, value in sorted(report.stability_metrics.items())
                )
                lines.append("")
                lines.append(f"Stability metrics: {metrics}")
            lines.append("")

        summary_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Wrote summary to %s", summary_path)

    # ------------------------------------------------------------------
    # Lazy accessors
    # ------------------------------------------------------------------
    def _get_inference_engine(self) -> BaseInferenceEngine:
        if self._inference_engine is None:
            self._inference_engine = create_inference_engine(self.config)
        return self._inference_engine

    def _get_metadata_writer(self) -> PersonalMetadataWriter:
        if self._metadata_writer is None:
            self._metadata_writer = PersonalMetadataWriter(
                backup_originals=self.config.backup_originals,
                overwrite_existing=self.config.overwrite_metadata,
            )
        return self._metadata_writer


def shorten_text(value: str, max_length: int) -> str:
    value = " ".join(value.split())
    if len(value) <= max_length:
        return value
    return value[: max_length - 1].rstrip() + "…"
