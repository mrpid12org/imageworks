"""Stage 0 compliance checks for Judge Vision."""

from __future__ import annotations

from pathlib import Path
from typing import List

from PIL import Image

from .competition import CompetitionConfig
from .judge_types import ComplianceReport


def evaluate_compliance(image_path: Path, competition: CompetitionConfig | None) -> ComplianceReport:
    """Evaluate simple compliance rules against an image."""

    issues: List[str] = []
    warnings: List[str] = []
    checks = {}

    try:
        with Image.open(image_path) as img:
            width, height = img.size
    except Exception as exc:  # noqa: BLE001
        return ComplianceReport(
            passed=False,
            issues=[f"Failed to open image: {exc}"],
            warnings=[],
            checks={"load_error": str(exc)},
        )

    checks["resolution"] = {"width": width, "height": height}

    if competition:
        rules = competition.rules
        if rules.max_width and width > rules.max_width:
            issues.append(f"Width {width}px exceeds limit {rules.max_width}px")
        if rules.max_height and height > rules.max_height:
            issues.append(f"Height {height}px exceeds limit {rules.max_height}px")
        if rules.min_width and width < rules.min_width:
            issues.append(f"Width {width}px below minimum {rules.min_width}px")
        if rules.min_height and height < rules.min_height:
            issues.append(f"Height {height}px below minimum {rules.min_height}px")
        if rules.borders:
            warnings.append(f"Border policy: {rules.borders}")
        if rules.manipulation:
            warnings.append(f"Manipulation policy: {rules.manipulation}")
        if not rules.watermark_allowed:
            warnings.append("Watermarks disallowed")
        else:
            warnings.append("Watermarks permitted")

    passed = not issues
    return ComplianceReport(passed=passed, issues=issues, warnings=warnings, checks=checks)

