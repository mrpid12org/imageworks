"""Main narrator module orchestrating VLM inference and metadata writing.

Coordinates data loading, VLM inference, and XMP metadata embedding for color narration.
Provides the primary processing pipeline for the color-narrator application.
"""

from typing import List, Dict, Optional, Any
from pathlib import Path
from dataclasses import dataclass
import logging
from datetime import datetime

from .data_loader import ColorNarratorDataLoader, DataLoaderConfig, ColorNarratorItem
from .vlm import VLMClient, VLMRequest, VLMResponse
from .metadata import XMPMetadataWriter, ColorNarrationMetadata

logger = logging.getLogger(__name__)


@dataclass
class NarrationConfig:
    """Configuration for color narration processing."""

    # Data sources
    images_dir: Path
    overlays_dir: Path
    mono_jsonl: Path

    # VLM settings
    vlm_base_url: str = "http://localhost:8000/v1"
    vlm_model: str = "Qwen/Qwen2-VL-7B-Instruct"
    vlm_timeout: int = 120

    # Processing settings
    batch_size: int = 4
    min_contamination_level: float = 0.1
    require_overlays: bool = True

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
        loader_config = DataLoaderConfig(
            images_dir=config.images_dir,
            overlays_dir=config.overlays_dir,
            mono_jsonl=config.mono_jsonl,
            min_contamination_level=config.min_contamination_level,
            require_overlays=config.require_overlays,
        )
        self.data_loader = ColorNarratorDataLoader(loader_config)

        # Initialize VLM client
        self.vlm_client = VLMClient(
            base_url=config.vlm_base_url,
            model_name=config.vlm_model,
            timeout=config.vlm_timeout,
        )

        # Initialize metadata writer
        self.metadata_writer = XMPMetadataWriter()

    def process_all(self) -> List[ProcessingResult]:
        """Process all valid items for color narration.

        Returns:
            List of processing results for each item
        """
        logger.info("Starting color narration processing...")

        # Validate VLM server availability
        if not self.vlm_client.health_check():
            raise RuntimeError(
                "VLM server is not available. Please start vLLM server first."
            )

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

            vlm_request = VLMRequest(
                image_path=item.image_path,
                overlay_path=item.overlay_path,
                mono_data=item.mono_data,
                prompt_template=self._get_prompt_template(),
            )
            vlm_requests.append((item, vlm_request))

        # Process through VLM
        results = []
        for item, vlm_request in vlm_requests:
            start_time = datetime.now()

            if self.config.dry_run:
                # Simulate processing for dry run
                vlm_response = VLMResponse(
                    description=f"[DRY RUN] Would generate color description for {item.image_path.name}",
                    confidence=0.8,
                    color_regions=["simulated"],
                    processing_time=0.1,
                )
                metadata_written = False
                error = None
            else:
                # Actual VLM inference
                vlm_response = self.vlm_client.infer_single(vlm_request)
                metadata_written = False
                error = vlm_response.error

                # Write metadata if inference successful
                if not vlm_response.error:
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
        # TODO: Make this configurable
        return """You are analyzing a photograph that should be monochrome (black and white) but contains some residual color.

The image has been analyzed and found to have:
- Hue distribution: {hue_analysis}
- Chroma levels: {chroma_analysis}
- Color contamination: {contamination_level}

Please describe in natural language where you observe residual color in this image. Focus on:
1. Specific regions or objects that show color
2. The type of color cast (warm/cool, specific hues)
3. Whether the color appears intentional or accidental

Provide a concise, professional description suitable for metadata (max 200 words)."""

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
