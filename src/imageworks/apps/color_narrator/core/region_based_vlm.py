"""
Region-based VLM analysis for hallucination-resistant color description.

This module implements the approach suggested by the friend's advice:
- Uses structured JSON output for validation
- Avoids hallucination by grounding responses in technical data
- Provides uncertainty handling with confidence scores
- Eliminates priming examples that bias the model
"""

import json
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .vlm import VLMClient, VLMRequest
from .prompts import REGION_BASED_COLOR_ANALYSIS_TEMPLATE

logger = logging.getLogger(__name__)


@dataclass
class ColorRegion:
    """A color region detected by technical analysis."""

    index: int
    bbox_xywh: List[int]  # [x, y, width, height] in pixels
    centroid_norm: List[float]  # [x, y] normalized 0..1
    mean_L: float  # 0..100 lightness
    mean_cab: float  # chroma magnitude
    mean_hue_deg: float  # 0..360 hue angle
    hue_name: str  # human-readable color name
    area_pct: float  # percentage of image area


@dataclass
class RegionFinding:
    """A VLM finding for a specific region."""

    region_index: int
    object_part: str  # What the color appears on
    color_family: str  # Color description from VLM
    tonal_zone: str  # shadow/midtone/highlight
    location_phrase: str  # Optional spatial locator
    confidence: float  # 0.0-1.0 VLM confidence


@dataclass
class RegionBasedAnalysis:
    """Complete region-based analysis result."""

    file_name: str
    dominant_color: str
    dominant_hue_deg: float
    findings: List[RegionFinding]
    raw_response: str  # Full VLM output for debugging
    validation_errors: List[str]  # Any validation issues found


class RegionBasedVLMAnalyzer:
    """VLM analyzer using hallucination-resistant region-based prompts."""

    def __init__(self, vlm_client: VLMClient):
        self.vlm_client = vlm_client

    def analyze_regions(
        self,
        file_name: str,
        regions: List[ColorRegion],
        dominant_color: str,
        dominant_hue_deg: float,
        image_base64: str,
        overlay_hue_base64: Optional[str] = None,
        overlay_chroma_base64: Optional[str] = None,
    ) -> RegionBasedAnalysis:
        """Analyze color regions using hallucination-resistant prompts.

        Args:
            file_name: Name of the image file being analyzed
            regions: List of color regions from technical analysis
            dominant_color: Dominant color name (e.g., "yellow-green")
            dominant_hue_deg: Dominant hue angle in degrees
            image_base64: Original image as base64 string
            overlay_hue_base64: Optional hue overlay as base64 string
            overlay_chroma_base64: Optional chroma overlay as base64 string

        Returns:
            Complete analysis with findings and validation
        """
        # Convert regions to JSON format for prompt
        regions_json = json.dumps(
            [
                {
                    "index": r.index,
                    "bbox_xywh": r.bbox_xywh,
                    "centroid_norm": r.centroid_norm,
                    "mean_L": r.mean_L,
                    "mean_cab": r.mean_cab,
                    "mean_hue_deg": r.mean_hue_deg,
                    "hue_name": r.hue_name,
                    "area_pct": r.area_pct,
                }
                for r in regions
            ],
            indent=2,
        )

        # Build prompt
        prompt = REGION_BASED_COLOR_ANALYSIS_TEMPLATE.format(
            file_name=file_name,
            dominant_color=dominant_color,
            dominant_hue_deg=dominant_hue_deg,
            regions_json=regions_json,
        )

        # Build image content list
        content = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
            },
        ]

        # Add overlays if available
        if overlay_hue_base64:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{overlay_hue_base64}"},
                }
            )
        if overlay_chroma_base64:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{overlay_chroma_base64}"
                    },
                }
            )

        # Create VLM request with lower temperature for more deterministic output
        request = VLMRequest(
            messages=[{"role": "user", "content": content}],
            max_tokens=600,  # Compact bullets + JSON should be under this
            temperature=0.1,  # Low temperature for literal, grounded responses
        )

        # Get VLM response
        response = self.vlm_client.infer_single(request)

        # Parse the structured response
        findings, validation_errors = self._parse_vlm_response(
            response.content, regions
        )

        return RegionBasedAnalysis(
            file_name=file_name,
            dominant_color=dominant_color,
            dominant_hue_deg=dominant_hue_deg,
            findings=findings,
            raw_response=response.content,
            validation_errors=validation_errors,
        )

    def _parse_vlm_response(
        self, response_text: str, regions: List[ColorRegion]
    ) -> Tuple[List[RegionFinding], List[str]]:
        """Parse VLM response into structured findings with validation.

        Args:
            response_text: Raw VLM response text
            regions: Original regions for validation

        Returns:
            Tuple of (findings, validation_errors)
        """
        findings = []
        validation_errors = []

        try:
            # Find JSON block in response
            json_start = response_text.find('{\n  "findings"')
            if json_start == -1:
                json_start = response_text.find('{"findings"')
            if json_start == -1:
                validation_errors.append("No JSON block found in VLM response")
                return findings, validation_errors

            json_text = response_text[json_start:]

            # Handle case where there might be text after JSON
            brace_count = 0
            json_end = len(json_text)
            for i, char in enumerate(json_text):
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break

            json_text = json_text[:json_end]

            # Parse JSON
            parsed = json.loads(json_text)

            if "findings" not in parsed:
                validation_errors.append("Missing 'findings' key in JSON response")
                return findings, validation_errors

            # Process each finding
            region_indices = {r.index for r in regions}

            for finding_dict in parsed["findings"]:
                try:
                    # Extract and validate fields
                    region_index = finding_dict.get("region_index")
                    if region_index not in region_indices:
                        validation_errors.append(
                            f"Region index {region_index} not found in input regions"
                        )
                        continue

                    # Find corresponding region for validation
                    region = next(r for r in regions if r.index == region_index)

                    object_part = finding_dict.get("object_part", "")
                    color_family = finding_dict.get("color_family", "")
                    tonal_zone = finding_dict.get("tonal_zone", "")
                    location_phrase = finding_dict.get("location_phrase", "")
                    confidence = finding_dict.get("confidence", 0.0)

                    # Validate tonal zone against mean_L
                    expected_tonal_zone = self._compute_tonal_zone(region.mean_L)
                    if tonal_zone != expected_tonal_zone:
                        validation_errors.append(
                            f"Region {region_index}: VLM reported tonal zone '{tonal_zone}' "
                            f"but mean_L={region.mean_L:.1f} indicates '{expected_tonal_zone}'"
                        )
                        # Use computed value
                        tonal_zone = expected_tonal_zone

                    # Validate confidence range
                    if not (0.0 <= confidence <= 1.0):
                        validation_errors.append(
                            f"Region {region_index}: Invalid confidence {confidence}, "
                            "clamping to valid range"
                        )
                        confidence = max(0.0, min(1.0, confidence))

                    finding = RegionFinding(
                        region_index=region_index,
                        object_part=object_part,
                        color_family=color_family,
                        tonal_zone=tonal_zone,
                        location_phrase=location_phrase,
                        confidence=confidence,
                    )
                    findings.append(finding)

                except Exception as e:
                    validation_errors.append(f"Error processing finding: {e}")
                    continue

        except json.JSONDecodeError as e:
            validation_errors.append(f"JSON parsing error: {e}")
        except Exception as e:
            validation_errors.append(f"Unexpected error parsing VLM response: {e}")

        return findings, validation_errors

    def _compute_tonal_zone(self, mean_L: float) -> str:
        """Compute tonal zone from L* lightness value.

        Args:
            mean_L: L* lightness (0-100)

        Returns:
            Tonal zone: "shadow", "midtone", or "highlight"
        """
        if mean_L < 35:
            return "shadow"
        elif mean_L < 70:
            return "midtone"
        else:
            return "highlight"

    def generate_human_readable_summary(self, analysis: RegionBasedAnalysis) -> str:
        """Generate human-readable summary from region analysis.

        Args:
            analysis: Complete region-based analysis

        Returns:
            Human-readable summary string
        """
        if not analysis.findings:
            return f"No color contamination regions found in {analysis.file_name}."

        lines = [
            f"Color Analysis: {analysis.file_name}",
            f"Dominant color: {analysis.dominant_color} ({analysis.dominant_hue_deg:.1f}°)",
            "",
        ]

        # Group findings by confidence
        high_conf = [f for f in analysis.findings if f.confidence >= 0.8]
        med_conf = [f for f in analysis.findings if 0.5 <= f.confidence < 0.8]
        low_conf = [f for f in analysis.findings if f.confidence < 0.5]

        if high_conf:
            lines.append("**Clear color contamination:**")
            for f in high_conf:
                location = f" {f.location_phrase}" if f.location_phrase else ""
                lines.append(
                    f"• {f.color_family} on {f.object_part}{location} ({f.tonal_zone})"
                )

        if med_conf:
            lines.append("\n**Moderate color contamination:**")
            for f in med_conf:
                location = f" {f.location_phrase}" if f.location_phrase else ""
                lines.append(
                    f"• {f.color_family} on {f.object_part}{location} ({f.tonal_zone})"
                )

        if low_conf:
            lines.append("\n**Possible color contamination (low confidence):**")
            for f in low_conf:
                location = f" {f.location_phrase}" if f.location_phrase else ""
                lines.append(
                    f"• {f.color_family} on {f.object_part}{location} ({f.tonal_zone})"
                )

        if analysis.validation_errors:
            lines.append("\n**Technical Validation Issues:**")
            for error in analysis.validation_errors:
                lines.append(f"⚠️ {error}")

        return "\n".join(lines)


def create_demo_regions() -> List[ColorRegion]:
    """Create demo regions for testing when mono-checker regions aren't available.

    This function creates synthetic region data that matches the expected format
    until mono-checker is updated to provide actual region data.
    """
    return [
        ColorRegion(
            index=0,
            bbox_xywh=[120, 80, 60, 40],
            centroid_norm=[0.18, 0.22],
            mean_L=76.4,
            mean_cab=5.1,
            mean_hue_deg=88.0,
            hue_name="yellow-green",
            area_pct=4.9,
        ),
        ColorRegion(
            index=1,
            bbox_xywh=[300, 200, 45, 35],
            centroid_norm=[0.65, 0.41],
            mean_L=42.1,
            mean_cab=3.8,
            mean_hue_deg=195.5,
            hue_name="blue",
            area_pct=2.3,
        ),
    ]
