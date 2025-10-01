"""Execution harness for the Personal Tagger."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from .config import PersonalTaggerConfig
from .inference import BaseInferenceEngine, create_inference_engine
from .metadata_writer import PersonalMetadataWriter
from .models import PersonalTaggerRecord

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
        images = self.discover_images()
        if not images:
            logger.warning("No images discovered for personal tagging")

        records: List[PersonalTaggerRecord] = []
        for image_path in images:
            record = self._process_image(image_path)
            records.append(record)

        self._write_jsonl(records)
        self._write_summary(records)

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
            lines.append("_Dry run: fake test data used, metadata writes skipped._")
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
