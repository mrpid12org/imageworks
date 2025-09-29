"""Main narrator module orchestrating VLM inference and metadata writing.

Coordinates data loading, VLM inference, and XMP metadata embedding for color narration.
Provides the primary processing pipeline for the color-narrator application.
"""

from typing import List, Dict, Optional, Any
from pathlib import Path
from dataclasses import dataclass
import logging
from datetime import datetime
import json

from .data_loader import ColorNarratorDataLoader, DataLoaderConfig, ColorNarratorItem
from .vlm import VLMBackend, VLMClient, VLMRequest, VLMResponse
from .metadata import XMPMetadataWriter, ColorNarrationMetadata
from . import prompts

logger = logging.getLogger(__name__)


@dataclass
class NarrationConfig:
    """Configuration for color narration processing."""

    # Data sources
    images_dir: Path
    overlays_dir: Path
    mono_jsonl: Path

    # VLM settings
    vlm_base_url: str = "http://localhost:24001/v1"
    vlm_model: str = "Qwen2.5-VL-7B-AWQ"
    vlm_timeout: int = 120
    vlm_backend: str = VLMBackend.LMDEPLOY.value
    vlm_api_key: str = "EMPTY"
    vlm_backend_options: Optional[Dict[str, Any]] = None

    # Processing settings
    batch_size: int = 4
    min_contamination_level: float = 0.1
    require_overlays: bool = True
    prompt_id: int = prompts.CURRENT_PROMPT_ID
    use_regions: bool = False
    allowed_verdicts: Optional[set[str]] = None

    # Output settings
    dry_run: bool = False
    debug: bool = False
    overwrite_existing: bool = False


@dataclass
class ProcessingResult:
    """Result from processing a single item."""

    item: ColorNarratorItem
    vlm_response: Optional[VLMResponse] = None
    metadata_written: bool = False
    error: Optional[str] = None
    processing_time: float = 0.0


class ColorNarrator:
    """Main color narrator processing engine."""

    def __init__(self, config: NarrationConfig):
        """Initialize color narrator with configuration.

        Args:
            config: Narration configuration
        """
        self.config = config

        # Initialize data loader
        allowed_verdicts = config.allowed_verdicts or {"fail", "pass_with_query"}
        self.allowed_verdicts = {v.lower() for v in allowed_verdicts}

        loader_config = DataLoaderConfig(
            images_dir=config.images_dir,
            overlays_dir=config.overlays_dir,
            mono_jsonl=config.mono_jsonl,
            min_contamination_level=config.min_contamination_level,
            require_overlays=config.require_overlays,
            allowed_verdicts=self.allowed_verdicts,
        )
        self.data_loader = ColorNarratorDataLoader(loader_config)

        # Initialize VLM client
        self.vlm_client = VLMClient(
            base_url=config.vlm_base_url,
            model_name=config.vlm_model,
            api_key=config.vlm_api_key,
            timeout=config.vlm_timeout,
            backend=config.vlm_backend,
            backend_options=config.vlm_backend_options,
        )

        # Initialize metadata writer
        self.metadata_writer = XMPMetadataWriter()

        # Prompt configuration
        self.prompt_definition = prompts.get_prompt_definition(config.prompt_id)
        self.use_regions = False
        if config.use_regions:
            if self.prompt_definition.supports_regions:
                self.use_regions = True
            else:
                logger.warning(
                    "Prompt %s does not support region context; ignoring --regions",
                    self.prompt_definition.name,
                )

    def process_all(self) -> List[ProcessingResult]:
        """Process all valid items for color narration.

        Returns:
            List of processing results for each item
        """
        logger.info("Starting color narration processing...")

        # Validate VLM server availability
        if not self.vlm_client.health_check():
            backend = self.config.vlm_backend
            hint = self.vlm_client.last_error or "backend did not respond"
            raise RuntimeError(f"VLM backend '{backend}' is not available: {hint}.")

        # Load and validate data
        self.data_loader.load()
        stats = self.data_loader.get_statistics()
        logger.info(f"Processing {stats['valid_items']} valid items")

        if self.config.debug:
            logger.info(f"Data statistics: {stats}")

        # Process in batches
        all_results = []
        for batch_items in self.data_loader.get_items(self.config.batch_size):
            batch_results = self._process_batch(batch_items)
            all_results.extend(batch_results)

            # Log progress
            processed_count = len(all_results)
            total_count = stats["valid_items"]
            logger.info(f"Progress: {processed_count}/{total_count} items processed")

        # Log summary
        successful = sum(1 for r in all_results if r.vlm_response and not r.error)
        failed = len(all_results) - successful
        logger.info(f"Processing complete: {successful} successful, {failed} failed")

        return all_results

    def _process_batch(self, items: List[ColorNarratorItem]) -> List[ProcessingResult]:
        """Process a batch of items through VLM inference and metadata writing.

        Args:
            items: Batch of items to process

        Returns:
            Processing results for each item
        """
        logger.debug(f"Processing batch of {len(items)} items")

        # Prepare VLM requests
        vlm_requests = []
        for item in items:
            if not self.config.overwrite_existing and item.has_existing_xmp:
                logger.debug(
                    f"Skipping {item.image_path.name} - already has XMP metadata"
                )
                continue

            prompt_payload = self._prepare_prompt_payload(item)
            prompt_template = self.prompt_definition.template

            vlm_request = VLMRequest(
                image_path=item.image_path,
                overlay_path=item.overlay_path,
                mono_data=prompt_payload,
                prompt_template=prompt_template,
            )
            vlm_requests.append((item, vlm_request))

        # Process through VLM
        results = []
        for item, vlm_request in vlm_requests:
            start_time = datetime.now()

            vlm_response = self.vlm_client.infer_single(vlm_request)
            metadata_written = False
            error = vlm_response.error

            if not self.config.dry_run and not vlm_response.error:
                try:
                    metadata = self._create_metadata(item, vlm_response)
                    self.metadata_writer.write_metadata(item.image_path, metadata)
                    metadata_written = True
                except Exception as e:
                    error = f"Metadata write error: {str(e)}"

            processing_time = (datetime.now() - start_time).total_seconds()

            result = ProcessingResult(
                item=item,
                vlm_response=vlm_response,
                metadata_written=metadata_written,
                error=error,
                processing_time=processing_time,
            )
            results.append(result)

            if self.config.debug:
                logger.debug(f"Processed {item.image_path.name}: {result}")

        return results

    def _get_prompt_template(self) -> str:
        """Get the prompt template for VLM inference."""
        return self.prompt_definition.template

    def _format_mono_context(self, mono_data: Dict[str, Any]) -> str:
        """Format mono-checker metrics for prompt context."""

        verdict = mono_data.get("verdict", "unknown")
        mode = mono_data.get("mode", "unknown")
        dominant_color = mono_data.get("dominant_color", "unknown")
        dominant_hue = mono_data.get("dominant_hue_deg", 0.0)
        colorfulness = mono_data.get("colorfulness", 0.0)
        chroma_max = mono_data.get("chroma_max", 0.0)
        hue_std = mono_data.get("hue_std_deg", 0.0)

        return (
            f"Mono-checker analysis: {verdict} ({mode})\n"
            f"Dominant color: {dominant_color} at {dominant_hue:.1f}°\n"
            f"Colorfulness: {colorfulness:.2f}, Max chroma: {chroma_max:.2f}\n"
            f"Hue variation: {hue_std:.1f}° standard deviation"
        )

    def _build_region_context(self, item: ColorNarratorItem) -> tuple[str, list[dict]]:
        """Build human-readable region context and JSON payload."""

        if not self.use_regions:
            return "", []

        try:
            from .grid_regions import ImageGridAnalyzer

            regions = ImageGridAnalyzer.create_grid_regions(
                item.image_path, item.mono_data
            )
            human_text = ImageGridAnalyzer.format_regions_for_human(regions)
            region_json = ImageGridAnalyzer.regions_to_json(regions)
            return human_text, region_json
        except Exception as exc:
            logger.debug("Region analysis unavailable for %s: %s", item.image_path, exc)
            return "Region analysis unavailable.", []

    def _prepare_prompt_payload(self, item: ColorNarratorItem) -> Dict[str, Any]:
        """Prepare template payload for the selected prompt definition."""

        payload = dict(item.mono_data)

        payload.setdefault("title", payload.get("title") or "Unknown")
        payload.setdefault("author", payload.get("author") or "Unknown")
        payload.setdefault("file_name", item.image_path.name)
        payload.setdefault("dominant_color", payload.get("dominant_color", "unknown"))
        payload.setdefault("dominant_hue_deg", payload.get("dominant_hue_deg", 0.0))
        payload.setdefault("top_colors", payload.get("top_colors", []))
        payload.setdefault("top_hues_deg", payload.get("top_hues_deg", []))
        payload.setdefault("top_weights", payload.get("top_weights", []))
        payload.setdefault("colorfulness", payload.get("colorfulness", 0.0))
        payload.setdefault("chroma_max", payload.get("chroma_max", 0.0))
        payload.setdefault("chroma_p95", payload.get("chroma_p95", 0.0))
        payload.setdefault(
            "chroma_ratio_2_4",
            (
                payload.get("chroma_ratio_4")
                if payload.get("chroma_ratio_4") is not None
                else payload.get("chroma_ratio_2", 0.0)
            ),
        )
        payload.setdefault(
            "contamination_level", payload.get("contamination_level", 0.0)
        )
        payload.setdefault(
            "delta_h_highs_shadows_deg", payload.get("delta_h_highs_shadows_deg")
        )
        payload.setdefault("verdict", payload.get("verdict", "unknown"))
        payload.setdefault("mode", payload.get("mode", "unknown"))
        payload.setdefault("reason_summary", payload.get("reason_summary", ""))

        definition = self.prompt_definition

        if definition.template in (
            prompts.ENHANCED_MONO_ANALYSIS_TEMPLATE_V6,
            prompts.ENHANCED_MONO_ANALYSIS_TEMPLATE_V5,
            prompts.ENHANCED_MONO_ANALYSIS_TEMPLATE_V4,
        ):
            payload["mono_context"] = self._format_mono_context(payload)
            region_text, region_json = self._build_region_context(item)
            payload["region_section"] = (
                f"\n\nSPATIAL ANALYSIS:\n{region_text}" if region_text else ""
            )
            if region_json:
                payload["regions_json"] = json.dumps(region_json, indent=2)

        elif definition.template == prompts.MONO_DESCRIPTION_ENHANCEMENT_TEMPLATE:
            payload.setdefault("chroma_max", payload.get("chroma_max", 0.0))

        elif definition.template == prompts.REGION_BASED_COLOR_ANALYSIS_TEMPLATE:
            region_text, region_json = self._build_region_context(item)
            payload["regions_json"] = json.dumps(region_json, indent=2)
            payload["region_text"] = region_text

        elif definition.template == prompts.TRIPTYCH_HUE_ANCHORED_TEMPLATE:
            # nothing extra beyond defaults; fields already populated
            pass

        elif definition.template == prompts.REGION_FIRST_TEMPLATE:
            region_text, _ = self._build_region_context(item)
            payload["region_guidance"] = region_text or "None"

        elif definition.template == prompts.TECHNICAL_ANALYST_TEMPLATE:
            analysis = {
                "verdict": payload.get("verdict"),
                "reason_summary": payload.get("reason_summary"),
                "dominant_color": payload.get("dominant_color"),
                "top_colors": payload.get("top_colors"),
                "delta_h_highs_shadows_deg": payload.get("delta_h_highs_shadows_deg"),
                "chroma_p95": payload.get("chroma_p95"),
                "chroma_ratio_4": payload.get("chroma_ratio_4"),
            }
            payload["analysis_json"] = json.dumps(
                analysis, indent=2, ensure_ascii=False
            )

        elif definition.template == prompts.FACTUAL_REPORTER_TEMPLATE:
            analysis = {
                "dominant_color": payload.get("dominant_color"),
                "top_colors": payload.get("top_colors"),
                "chroma_max": payload.get("chroma_max"),
                "reason_summary": payload.get("reason_summary"),
            }
            payload["analysis_json"] = json.dumps(
                analysis, indent=2, ensure_ascii=False
            )

        elif definition.template == prompts.TRIPTYCH_ANALYST_BRIEF_TEMPLATE:
            # nothing additional; base fields already set
            pass

        elif definition.template == prompts.REGION_HINT_BULLETS_TEMPLATE:
            region_text, _ = self._build_region_context(item)
            payload["region_guidance"] = region_text or "None"

        elif definition.template == prompts.FORENSIC_SPECIALIST_TEMPLATE:
            analysis = {
                "verdict": payload.get("verdict"),
                "reason_summary": payload.get("reason_summary"),
                "top_colors": payload.get("top_colors"),
                "delta_h_highs_shadows_deg": payload.get("delta_h_highs_shadows_deg"),
                "region_guidance": payload.get("region_guidance"),
            }
            payload["analysis_json"] = json.dumps(
                analysis, indent=2, ensure_ascii=False
            )

        elif definition.template == prompts.STRUCTURED_OBSERVER_TEMPLATE:
            analysis = {
                "top_colors": payload.get("top_colors"),
                "reason_summary": payload.get("reason_summary"),
            }
            payload["analysis_json"] = json.dumps(
                analysis, indent=2, ensure_ascii=False
            )

        return payload

    def _create_metadata(
        self, item: ColorNarratorItem, vlm_response: VLMResponse
    ) -> ColorNarrationMetadata:
        """Create XMP metadata from processing results.

        Args:
            item: Processed item
            vlm_response: VLM inference response

        Returns:
            Metadata object for XMP embedding
        """
        return ColorNarrationMetadata(
            description=vlm_response.description,
            confidence_score=vlm_response.confidence,
            color_regions=vlm_response.color_regions,
            processing_timestamp=datetime.now().isoformat(),
            mono_contamination_level=item.mono_data.get("contamination_level", 0.0),
            vlm_model=self.config.vlm_model,
            vlm_processing_time=vlm_response.processing_time,
        )

    def validate_existing(self, images_dir: Path) -> Dict[str, Any]:
        """Validate existing color narrations in a directory.

        Args:
            images_dir: Directory containing images to validate

        Returns:
            Validation results and statistics
        """
        logger.info(f"Validating existing color narrations in {images_dir}")

        # Find images with XMP metadata
        image_files = []
        for ext in [".jpg", ".jpeg", ".JPG", ".JPEG"]:
            image_files.extend(images_dir.glob(f"*{ext}"))

        validation_results = {
            "total_images": len(image_files),
            "with_metadata": 0,
            "valid_metadata": 0,
            "validation_errors": [],
        }

        for image_path in image_files:
            try:
                metadata = self.metadata_writer.read_metadata(image_path)
                if metadata:
                    validation_results["with_metadata"] += 1

                    # Validate metadata structure and content
                    if self._validate_metadata_content(metadata):
                        validation_results["valid_metadata"] += 1
                    else:
                        validation_results["validation_errors"].append(
                            f"{image_path.name}: Invalid metadata content"
                        )
            except Exception as e:
                validation_results["validation_errors"].append(
                    f"{image_path.name}: Error reading metadata - {str(e)}"
                )

        logger.info(f"Validation complete: {validation_results}")
        return validation_results

    def _validate_metadata_content(self, metadata: ColorNarrationMetadata) -> bool:
        """Validate the content of color narration metadata.

        Args:
            metadata: Metadata to validate

        Returns:
            True if metadata is valid
        """
        # Check required fields
        if not metadata.description or not metadata.description.strip():
            return False

        if metadata.confidence_score < 0.0 or metadata.confidence_score > 1.0:
            return False

        # Check description length (reasonable bounds)
        if len(metadata.description) < 10 or len(metadata.description) > 1000:
            return False

        return True
