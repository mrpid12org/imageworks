"""Metadata persistence for the personal tagger using ExifTool."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Iterable, List

from .models import PersonalTaggerRecord

logger = logging.getLogger(__name__)


class PersonalMetadataWriter:
    """Write personal tagger outputs using the ExifTool CLI."""

    def __init__(self, *, backup_originals: bool, overwrite_existing: bool) -> None:
        self.backup_originals = backup_originals
        self.overwrite_existing = overwrite_existing

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def write(self, image_path: Path, record: PersonalTaggerRecord) -> bool:
        if not record.caption and not record.keywords and not record.description:
            logger.debug("No metadata supplied for %s; skipping write", image_path)
            return False

        if not self.overwrite_existing and self._has_existing_metadata(image_path):
            logger.info(
                "Skipping metadata for %s (existing metadata present and overwrite disabled)",
                image_path,
            )
            return False

        command = self._build_command(image_path, record)
        logger.debug("ExifTool command for %s: %s", image_path, command)

        try:
            self._run_exiftool(command)
        except Exception as exc:  # noqa: BLE001
            logger.error("ExifTool write failed for %s: %s", image_path, exc)
            return False

        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_command(
        self, image_path: Path, record: PersonalTaggerRecord
    ) -> List[str]:
        cmd: List[str] = ["exiftool"]
        if not self.backup_originals or self.overwrite_existing:
            cmd.append("-overwrite_original")

        cmd.extend(self._title_params(record.caption))
        cmd.extend(self._keywords_params(record.keywords_as_list()))
        cmd.extend(self._description_params(record.description))
        cmd.append(str(image_path))
        return cmd

    @staticmethod
    def _title_params(caption: str) -> List[str]:
        if not caption:
            return []
        clean = " ".join(caption.strip().split())
        return [f"-XMP-dc:Title={clean}"]

    @staticmethod
    def _description_params(description: str) -> List[str]:
        if not description:
            return []
        clean = " ".join(description.strip().split())
        return [
            f"-XMP-dc:Description={clean}",
            f"-IPTC:Caption-Abstract={clean}",
        ]

    @staticmethod
    def _keywords_params(keywords: Iterable[str]) -> List[str]:
        kws = [kw.strip() for kw in keywords if kw.strip()]
        if not kws:
            return []

        params: List[str] = [
            "-XMP-dc:Subject=",
            "-IPTC:Keywords=",
            "-XMP-lr:HierarchicalSubject=",
        ]
        for keyword in kws:
            params.append(f"-XMP-dc:Subject+={keyword}")
            params.append(f"-IPTC:Keywords+={keyword}")
            params.append(f"-XMP-lr:HierarchicalSubject+={keyword}")
        return params

    def _has_existing_metadata(self, image_path: Path) -> bool:
        try:
            result = subprocess.run(
                [
                    "exiftool",
                    "-s",
                    "-s",
                    "-s",
                    "-XMP-dc:Subject",
                    "-XMP-dc:Description",
                    str(image_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:  # pragma: no cover - hard failure
            raise RuntimeError(
                "ExifTool not found in PATH. Install exiftool to enable metadata writing."
            ) from exc

        if result.returncode != 0:
            return False

        output = result.stdout.strip()
        return bool(output)

    def _run_exiftool(self, command: List[str]) -> None:
        try:
            subprocess.run(
                command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
        except FileNotFoundError as exc:  # pragma: no cover - hard failure
            raise RuntimeError(
                "ExifTool not found in PATH. Install exiftool to enable metadata writing."
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(exc.stderr.decode("utf-8", errors="ignore")) from exc
