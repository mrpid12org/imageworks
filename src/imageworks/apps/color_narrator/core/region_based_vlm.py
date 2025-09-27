"""
Region-based VLM analysis supporting both technical regions and simple 3x3 grid.

This module implements hallucination-resistant color description using:
- Structured JSON output for validation
- Optional region data (technical or 3x3 grid)
- Uncertainty handling with confidence scores
- No priming examples to avoid bias
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import base64

from imageworks.apps.color_narrator.core.vlm import VLMClient
from imageworks.apps.color_narrator.core.grid_regions import (
    ImageGridAnalyzer,
)

logger = logging.getLogger(__name__)


@dataclass
class VLMAnalysisResult:
    """Result from VLM analysis with validation."""

    file_name: str
    dominant_color: str
    dominant_hue_deg: float
    findings: List[Dict[str, Any]]  # VLM findings
    raw_response: str
    validation_errors: List[str]
    region_type: str  # "grid", "technical", or "none"


class RegionBasedVLMAnalyzer:
    """VLM analyzer with optional region support (grid or technical)."""

    def __init__(self, vlm_client: VLMClient):
        self.vlm_client = vlm_client

    def analyze_with_regions(
        self,
        image_path: Path,
        mono_data: Dict[str, Any],
        region_data: Optional[Dict[str, Any]] = None,
        use_grid_regions: bool = True,
        image_dimensions: Optional[Tuple[int, int]] = None,
    ) -> VLMAnalysisResult:
        """Analyze image with VLM using mono-checker data and optional region information.

        Args:
            image_path: Path to the image file
            mono_data: Mono-checker analysis results
            region_data: Optional technical region analysis (luminance zones, etc.)
            use_grid_regions: Whether to use simple 3x3 grid regions
            image_dimensions: (width, height) for grid region calculation

        Returns:
            VLMAnalysisResult with structured analysis and confidence metrics
        """
        try:
            # Prepare region context
            region_context = "No region data provided. Analyze the entire image."
            region_type = "none"

            if use_grid_regions and image_dimensions:
                # Use simple 3x3 grid regions
                width, height = image_dimensions
                grid_regions = ImageGridAnalyzer.analyze_color_in_regions(
                    mono_data, width, height
                )
                if grid_regions:
                    region_json = ImageGridAnalyzer.regions_to_json(grid_regions)
                    region_context = (
                        f"Grid regions with color: {json.dumps(region_json, indent=2)}"
                    )
                    region_context += f"\\n\\nHuman-readable: {ImageGridAnalyzer.format_regions_for_human(grid_regions)}"
                    region_type = "grid"
                else:
                    region_context = (
                        "No significant color detected in any 3x3 grid regions."
                    )
                    region_type = "grid"

            elif region_data:
                # Use technical region data (legacy)
                region_context = self._format_region_data(region_data)
                region_type = "technical"

            # Format mono-checker data for VLM context
            mono_context = self._format_mono_data(mono_data)

            # Load and encode image
            image_base64 = self._encode_image(image_path)

            # Build prompt - simplified for optional regions
            if region_type == "none":
                prompt = self._build_simple_prompt(mono_data)
            else:
                prompt = f"""Analyze this monochrome competition image for color contamination.

TECHNICAL CONTEXT:
{mono_context}

REGION DATA:
{region_context}

INSTRUCTIONS:
1. Look at the image and identify any colored areas (non-grayscale regions)
2. If regions are provided, reference them in your analysis
3. Be specific about what objects/parts show color
4. Rate your confidence (0.0-1.0) for each observation
5. Format as JSON with this structure:
{{
  "findings": [
    {{
      "object_part": "description of what shows color",
      "color_family": "color description",
      "tonal_zone": "shadow/midtone/highlight",
      "location": "spatial description",
      "confidence": 0.0-1.0
    }}
  ]
}}

Respond ONLY with the JSON."""

            # Build VLM request
            content = [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                },
            ]

            # Send to VLM
            response = self.vlm_client.infer_openai_compatible(
                {
                    "messages": [{"role": "user", "content": content}],
                    "max_tokens": 1000,  # Increased for longer responses
                    "temperature": 0.1,
                }
            )

            # Parse response
            findings, validation_errors = self._parse_vlm_response(response.content)

            return VLMAnalysisResult(
                file_name=image_path.name,
                dominant_color=mono_data.get("dominant_color", "unknown"),
                dominant_hue_deg=mono_data.get("dominant_hue_deg", 0.0),
                findings=findings,
                raw_response=response.content,
                validation_errors=validation_errors,
                region_type=region_type,
            )

        except Exception as e:
            logger.error(f"VLM analysis failed for {image_path}: {e}")
            return VLMAnalysisResult(
                file_name=image_path.name,
                dominant_color=mono_data.get("dominant_color", "unknown"),
                dominant_hue_deg=mono_data.get("dominant_hue_deg", 0.0),
                findings=[],
                raw_response=f"Error: {e}",
                validation_errors=[f"Analysis failed: {e}"],
                region_type="error",
            )

    def _format_mono_data(self, mono_data: Dict[str, Any]) -> str:
        """Format mono-checker data for VLM context."""
        verdict = mono_data.get("verdict", "unknown")
        dominant_color = mono_data.get("dominant_color", "unknown")
        dominant_hue = mono_data.get("dominant_hue_deg", 0.0)
        chroma_score = mono_data.get("chroma_score", 0.0)

        return f"""Mono-checker verdict: {verdict}
Dominant color: {dominant_color} ({dominant_hue:.1f}°)
Overall chroma score: {chroma_score:.2f}"""

    def _format_region_data(self, region_data: Dict[str, Any]) -> str:
        """Format technical region data for VLM context."""
        # This would format complex technical region data
        # For now, just return a placeholder
        return f"Technical region data: {json.dumps(region_data, indent=2)}"

    def _build_simple_prompt(self, mono_data: Dict[str, Any]) -> str:
        """Build simple prompt without region data."""
        mono_context = self._format_mono_data(mono_data)

        return f"""Analyze this monochrome competition image for color contamination.

TECHNICAL CONTEXT:
{mono_context}

INSTRUCTIONS:
1. Look at the entire image for any colored areas (non-grayscale regions)
2. Be specific about what objects/parts show color
3. Describe the spatial location of any color
4. Rate your confidence (0.0-1.0) for each observation
5. Format as JSON with this structure:
{{
  "findings": [
    {{
      "object_part": "description of what shows color",
      "color_family": "color description",
      "tonal_zone": "shadow/midtone/highlight",
      "location": "spatial description",
      "confidence": 0.0-1.0
    }}
  ]
}}

Respond ONLY with the JSON."""

    def _encode_image(self, image_path: Path) -> str:
        """Encode image as base64."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _parse_vlm_response(
        self, response_text: str
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Parse VLM JSON response."""
        findings = []
        validation_errors = []

        try:
            # Find JSON in response
            json_start = response_text.find("{")
            if json_start == -1:
                validation_errors.append("No JSON found in VLM response")
                return findings, validation_errors

            json_text = response_text[json_start:]

            # Find end of JSON
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
            parsed = json.loads(json_text)

            if "findings" not in parsed:
                validation_errors.append("Missing 'findings' key in JSON")
                return findings, validation_errors

            # Validate each finding
            for finding in parsed["findings"]:
                if not isinstance(finding, dict):
                    validation_errors.append(f"Invalid finding format: {finding}")
                    continue

                # Validate required fields
                required_fields = ["object_part", "color_family", "confidence"]
                for field in required_fields:
                    if field not in finding:
                        validation_errors.append(f"Missing field '{field}' in finding")
                        finding[field] = "unknown" if field != "confidence" else 0.0

                # Validate confidence range
                confidence = finding.get("confidence", 0.0)
                if not isinstance(confidence, (int, float)) or not (
                    0.0 <= confidence <= 1.0
                ):
                    validation_errors.append(f"Invalid confidence: {confidence}")
                    finding["confidence"] = max(
                        0.0,
                        min(
                            1.0,
                            (
                                float(confidence)
                                if isinstance(confidence, (int, float))
                                else 0.0
                            ),
                        ),
                    )

                findings.append(finding)

        except json.JSONDecodeError as e:
            validation_errors.append(f"JSON parsing error: {e}")
        except Exception as e:
            validation_errors.append(f"Error parsing VLM response: {e}")

        return findings, validation_errors

    def generate_human_readable_summary(self, analysis: VLMAnalysisResult) -> str:
        """Generate human-readable summary."""
        if not analysis.findings:
            return f"No color contamination found in {analysis.file_name}."

        lines = [
            f"Color Analysis: {analysis.file_name}",
            f"Dominant color: {analysis.dominant_color} ({analysis.dominant_hue_deg:.1f}°)",
            f"Region analysis: {analysis.region_type}",
            "",
        ]

        # Group by confidence
        high_conf = [f for f in analysis.findings if f.get("confidence", 0) >= 0.8]
        med_conf = [f for f in analysis.findings if 0.5 <= f.get("confidence", 0) < 0.8]
        low_conf = [f for f in analysis.findings if f.get("confidence", 0) < 0.5]

        if high_conf:
            lines.append("**Clear color contamination:**")
            for f in high_conf:
                location = f" - {f.get('location', '')}" if f.get("location") else ""
                lines.append(
                    f"• {f.get('color_family', 'unknown')} on {f.get('object_part', 'unknown')}{location}"
                )

        if med_conf:
            lines.append("\\n**Moderate color contamination:**")
            for f in med_conf:
                location = f" - {f.get('location', '')}" if f.get("location") else ""
                lines.append(
                    f"• {f.get('color_family', 'unknown')} on {f.get('object_part', 'unknown')}{location}"
                )

        if low_conf:
            lines.append("\\n**Possible color contamination:**")
            for f in low_conf:
                location = f" - {f.get('location', '')}" if f.get("location") else ""
                lines.append(
                    f"• {f.get('color_family', 'unknown')} on {f.get('object_part', 'unknown')}{location}"
                )

        if analysis.validation_errors:
            lines.append("\\n**Validation Issues:**")
            for error in analysis.validation_errors[:3]:  # Limit to first 3 errors
                lines.append(f"⚠️ {error}")

        return "\\n".join(lines)


def create_demo_vlm_analysis() -> VLMAnalysisResult:
    """Create demo analysis for testing."""
    return VLMAnalysisResult(
        file_name="demo_image.jpg",
        dominant_color="yellow-green",
        dominant_hue_deg=88.0,
        findings=[
            {
                "object_part": "foliage in background",
                "color_family": "green",
                "tonal_zone": "midtone",
                "location": "top-right area",
                "confidence": 0.9,
            }
        ],
        raw_response='{"findings": [{"object_part": "foliage", "color_family": "green", "confidence": 0.9}]}',
        validation_errors=[],
        region_type="grid",
    )
