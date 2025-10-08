"""Metadata helper for persisting similarity verdicts."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import List

from .models import CandidateSimilarity

logger = logging.getLogger(__name__)


class SimilarityMetadataWriter:
    """Write similarity outcomes to image metadata via ExifTool."""

    def __init__(self, *, backup_originals: bool, overwrite_existing: bool) -> None:
        self.backup_originals = backup_originals
        self.overwrite_existing = overwrite_existing

    def write(self, image_path: Path, result: CandidateSimilarity) -> bool:
        if not shutil.which("exiftool"):
            raise RuntimeError("ExifTool not found in PATH. Install exiftool to enable metadata writing.")

        keywords = self._build_keywords(result)
        if not keywords:
            logger.debug("No similarity metadata to write for %s", image_path)
            return False

        command = ["exiftool"]
        if not self.backup_originals or self.overwrite_existing:
            command.append("-overwrite_original")

        if self.overwrite_existing:
            command.extend(["-XMP-dc:Subject=", "-IPTC:Keywords=", "-XMP-lr:HierarchicalSubject="])

        for keyword in keywords:
            command.extend(
                [
                    f"-XMP-dc:Subject+={keyword}",
                    f"-IPTC:Keywords+={keyword}",
                    f"-XMP-lr:HierarchicalSubject+={keyword}",
                ]
            )

        command.append(str(image_path))

        try:
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError as exc:  # pragma: no cover - fatal error
            raise RuntimeError(
                "ExifTool not found in PATH. Install exiftool to enable metadata writing."
            ) from exc
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode("utf-8", errors="ignore")
            raise RuntimeError(stderr.strip() or "ExifTool metadata write failed") from exc

        return True

    @staticmethod
    def _build_keywords(result: CandidateSimilarity) -> List[str]:
        keywords = [
            "imageworks:similarity",
            f"similarity:verdict={result.verdict.value}",
            f"similarity:score={result.top_score:.3f}",
        ]
        best_match = result.best_match()
        if best_match is not None:
            keywords.append(f"similarity:match={best_match.reference.name}")
        strategies = best_match.extra.get("strategies") if best_match else {}
        if isinstance(strategies, dict):
            for name, score in strategies.items():
                keywords.append(f"similarity:{name}={float(score):.3f}")
        return keywords
