"""XMP metadata handling for color narration data.

Provides reading and writing of color narration metadata to JPEG XMP fields,
ensuring consistent metadata structure and proper embedding in image files.
"""

from typing import List, Dict, Optional, Any
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import json

try:  # pragma: no cover - exercised when libxmp & exempi are installed
    from libxmp import XMPFiles, XMPMeta, XMPError  # type: ignore[attr-defined]

    XMP_AVAILABLE = True
except Exception:  # pragma: no cover - ImportError or ExempiLoadError
    XMPFiles = XMPMeta = XMPError = None  # type: ignore[assignment]
    XMP_AVAILABLE = False

logger = logging.getLogger(__name__)

if not XMP_AVAILABLE:
    logger.info("XMP toolkit not available; using sidecar JSON metadata")


@dataclass
class ColorNarrationMetadata:
    """Structured metadata for color narration results."""

    description: str
    confidence_score: float
    color_regions: List[str]
    processing_timestamp: str
    mono_contamination_level: float
    vlm_model: str
    vlm_processing_time: float

    # Optional analysis details
    hue_analysis: Optional[str] = None
    chroma_analysis: Optional[str] = None
    validation_status: Optional[str] = None


class XMPMetadataWriter:
    """Writer for color narration XMP metadata."""

    # XMP namespace for color narration metadata
    CN_NAMESPACE = "http://imageworks.ai/color-narrator/1.0/"
    CN_PREFIX = "cn"

    def __init__(self, backup_original: bool = True):
        """Initialize XMP metadata writer.

        Args:
            backup_original: Whether to backup original files before modification
        """
        self.backup_original = backup_original

    def write_metadata(
        self, image_path: Path, metadata: ColorNarrationMetadata
    ) -> bool:
        """Write color narration metadata to image XMP.

        Args:
            image_path: Path to JPEG image file
            metadata: Color narration metadata to embed

        Returns:
            True if metadata was successfully written
        """
        try:
            logger.debug(f"Writing XMP metadata to {image_path}")

            # Backup original if requested
            if self.backup_original:
                self._backup_file(image_path)

            # Use actual XMP if available, otherwise fallback to sidecar
            if XMP_AVAILABLE:
                try:
                    self._write_xmp_metadata(image_path, metadata)
                except Exception as exc:
                    logger.warning(
                        "XMP write failed for %s (%s); falling back to sidecar JSON",
                        image_path,
                        exc,
                    )
                    self._write_sidecar_json(image_path, metadata)
            else:
                self._write_sidecar_json(image_path, metadata)

            logger.info(f"Successfully wrote color narration metadata to {image_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to write metadata to {image_path}: {e}")
            return False

    def read_metadata(self, image_path: Path) -> Optional[ColorNarrationMetadata]:
        """Read color narration metadata from image XMP.

        Args:
            image_path: Path to JPEG image file

        Returns:
            Color narration metadata if found, None otherwise
        """
        try:
            # Use actual XMP if available, otherwise fallback to sidecar
            if XMP_AVAILABLE:
                return self._read_xmp_metadata(image_path)
            else:
                return self._read_sidecar_json(image_path)

        except Exception as e:
            logger.debug(f"Failed to read metadata from {image_path}: {e}")
            return None

    def has_color_narration(self, image_path: Path) -> bool:
        """Check if image has color narration metadata.

        Args:
            image_path: Path to JPEG image file

        Returns:
            True if color narration metadata exists
        """
        metadata = self.read_metadata(image_path)
        return metadata is not None

    def remove_metadata(self, image_path: Path) -> bool:
        """Remove color narration metadata from image.

        Args:
            image_path: Path to JPEG image file

        Returns:
            True if metadata was successfully removed
        """
        try:
            logger.debug(f"Removing XMP metadata from {image_path}")

            # Backup original if requested
            if self.backup_original:
                self._backup_file(image_path)

            # TODO: Implement actual XMP metadata removal
            # For now, remove sidecar JSON file
            sidecar_path = self._get_sidecar_path(image_path)
            if sidecar_path.exists():
                sidecar_path.unlink()

            logger.debug(f"Successfully removed metadata from {image_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to remove metadata from {image_path}: {e}")
            return False

    def _backup_file(self, image_path: Path) -> Path:
        """Create backup copy of original image file.

        Args:
            image_path: Path to original image

        Returns:
            Path to backup file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = image_path.with_suffix(f".backup_{timestamp}{image_path.suffix}")

        if not backup_path.exists():
            backup_path.write_bytes(image_path.read_bytes())
            logger.debug(f"Created backup: {backup_path}")

        return backup_path

    def _get_sidecar_path(self, image_path: Path) -> Path:
        """Get path for sidecar JSON metadata file.

        Args:
            image_path: Path to original image

        Returns:
            Path to sidecar metadata file
        """
        return image_path.with_suffix(f"{image_path.suffix}.cn_metadata.json")

    def _write_sidecar_json(
        self, image_path: Path, metadata: ColorNarrationMetadata
    ) -> None:
        """Write metadata to sidecar JSON file (development implementation).

        Args:
            image_path: Path to original image
            metadata: Metadata to write
        """
        sidecar_path = self._get_sidecar_path(image_path)
        metadata_dict = asdict(metadata)

        # Add file information
        metadata_dict["source_file"] = str(image_path.name)
        metadata_dict["metadata_version"] = "1.0"

        with open(sidecar_path, "w") as f:
            json.dump(metadata_dict, f, indent=2)

        logger.debug(f"Wrote sidecar metadata to {sidecar_path}")

    def _read_sidecar_json(self, image_path: Path) -> Optional[ColorNarrationMetadata]:
        """Read metadata from sidecar JSON file (development implementation).

        Args:
            image_path: Path to original image

        Returns:
            Color narration metadata if found
        """
        sidecar_path = self._get_sidecar_path(image_path)

        if not sidecar_path.exists():
            return None

        try:
            with open(sidecar_path, "r") as f:
                metadata_dict = json.load(f)

            # Remove non-dataclass fields
            metadata_dict.pop("source_file", None)
            metadata_dict.pop("metadata_version", None)

            # Convert back to dataclass
            return ColorNarrationMetadata(**metadata_dict)

        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Invalid sidecar metadata in {sidecar_path}: {e}")
            return None

    def _write_xmp_metadata(
        self, image_path: Path, metadata: ColorNarrationMetadata
    ) -> None:
        """Write metadata to actual XMP fields.

        Args:
            image_path: Path to image file
            metadata: Metadata to embed
        """
        if not XMP_AVAILABLE:
            raise RuntimeError("XMP library not available")

        try:
            # Open the file for XMP modification
            xmpfile = XMPFiles()
            xmpfile.open_file(str(image_path), open_flags=1)  # XMP_OPEN_FOR_UPDATE

            # Get existing XMP or create new
            xmp = xmpfile.get_xmp()
            if xmp is None:
                xmp = XMPMeta()

            # Register our namespace
            xmp.register_namespace(self.CN_NAMESPACE, self.CN_PREFIX)

            logger.debug(f"Writing metadata fields for {image_path}")
            logger.debug(
                f"Description: '{metadata.description}' (type: {type(metadata.description)})"
            )

            # Write core metadata fields with extensive validation
            desc = str(metadata.description or "")
            conf = float(metadata.confidence_score or 0.0)
            timestamp = str(metadata.processing_timestamp or "")
            contam = float(metadata.mono_contamination_level or 0.0)
            model = str(metadata.vlm_model or "")
            proc_time = float(metadata.vlm_processing_time or 0.0)

            logger.debug(
                f"Validated values: desc='{desc}', conf={conf}, timestamp='{timestamp}'"
            )

            xmp.set_property(self.CN_NAMESPACE, "description", desc)
            xmp.set_property_float(self.CN_NAMESPACE, "confidenceScore", conf)
            xmp.set_property(self.CN_NAMESPACE, "processingTimestamp", timestamp)
            xmp.set_property_float(self.CN_NAMESPACE, "monoContaminationLevel", contam)
            xmp.set_property(self.CN_NAMESPACE, "vlmModel", model)
            xmp.set_property_float(self.CN_NAMESPACE, "vlmProcessingTime", proc_time)

            # Write optional fields
            if metadata.hue_analysis:
                xmp.set_property(
                    self.CN_NAMESPACE, "hueAnalysis", str(metadata.hue_analysis)
                )
            if metadata.chroma_analysis:
                xmp.set_property(
                    self.CN_NAMESPACE, "chromaAnalysis", str(metadata.chroma_analysis)
                )
            if metadata.validation_status:
                xmp.set_property(
                    self.CN_NAMESPACE,
                    "validationStatus",
                    str(metadata.validation_status),
                )

            # Write color regions as an array
            if metadata.color_regions:
                # Create array property first
                try:
                    xmp.set_property(
                        self.CN_NAMESPACE,
                        "colorRegions",
                        None,
                        prop_value_is_array=True,
                    )
                    for i, region in enumerate(metadata.color_regions, 1):
                        xmp.append_array_item(self.CN_NAMESPACE, "colorRegions", region)
                except XMPError as e:
                    logger.debug(f"Array handling error (non-fatal): {e}")
                    # Try alternative approach - just store as comma-separated string
                    region_str = ", ".join(metadata.color_regions)
                    xmp.set_property(self.CN_NAMESPACE, "colorRegionsText", region_str)

            # Write updated XMP back to file
            xmpfile.put_xmp(xmp)
            xmpfile.close_file()

            logger.debug(f"Successfully wrote XMP metadata to {image_path}")

        except XMPError as e:
            raise RuntimeError(f"XMP error writing to {image_path}: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error writing XMP to {image_path}: {e}")

    def _read_xmp_metadata(self, image_path: Path) -> Optional[ColorNarrationMetadata]:
        """Read metadata from actual XMP fields.

        Args:
            image_path: Path to image file

        Returns:
            Color narration metadata if found
        """
        if not XMP_AVAILABLE:
            raise RuntimeError("XMP library not available")

        try:
            xmpfile = XMPFiles()
            xmpfile.open_file(str(image_path), open_flags=0)  # XMP_OPEN_FOR_READ

            xmp = xmpfile.get_xmp()
            if xmp is None:
                xmpfile.close_file()
                return None

            # Check if our namespace exists
            if not xmp.does_property_exist(self.CN_NAMESPACE, "description"):
                xmpfile.close_file()
                return None

            # Read core fields
            description = xmp.get_property(self.CN_NAMESPACE, "description")
            confidence_score = xmp.get_property_float(
                self.CN_NAMESPACE, "confidenceScore"
            )
            processing_timestamp = xmp.get_property(
                self.CN_NAMESPACE, "processingTimestamp"
            )
            mono_contamination_level = xmp.get_property_float(
                self.CN_NAMESPACE, "monoContaminationLevel"
            )
            vlm_model = xmp.get_property(self.CN_NAMESPACE, "vlmModel")
            vlm_processing_time = xmp.get_property_float(
                self.CN_NAMESPACE, "vlmProcessingTime"
            )

            # Read optional fields
            hue_analysis = None
            chroma_analysis = None
            validation_status = None

            try:
                hue_analysis = xmp.get_property(self.CN_NAMESPACE, "hueAnalysis")
            except Exception:
                pass
            try:
                chroma_analysis = xmp.get_property(self.CN_NAMESPACE, "chromaAnalysis")
            except Exception:
                pass
            try:
                validation_status = xmp.get_property(
                    self.CN_NAMESPACE, "validationStatus"
                )
            except Exception:
                pass

            # Read color regions array
            color_regions = []
            try:
                # Try to read as array first
                count = xmp.count_array_items(self.CN_NAMESPACE, "colorRegions")
                for i in range(1, count + 1):
                    region = xmp.get_array_item(self.CN_NAMESPACE, "colorRegions", i)
                    color_regions.append(region)
            except Exception:
                # Fallback to text format
                try:
                    regions_text = xmp.get_property(
                        self.CN_NAMESPACE, "colorRegionsText"
                    )
                    if regions_text:
                        color_regions = [r.strip() for r in regions_text.split(",")]
                except Exception:
                    pass

            xmpfile.close_file()

            return ColorNarrationMetadata(
                description=description,
                confidence_score=confidence_score,
                color_regions=color_regions,
                processing_timestamp=processing_timestamp,
                mono_contamination_level=mono_contamination_level,
                vlm_model=vlm_model,
                vlm_processing_time=vlm_processing_time,
                hue_analysis=hue_analysis,
                chroma_analysis=chroma_analysis,
                validation_status=validation_status,
            )

        except XMPError as e:
            logger.debug(f"XMP error reading from {image_path}: {e}")
            return None
        except Exception as e:
            logger.debug(f"Unexpected error reading XMP from {image_path}: {e}")
            return None


class XMPMetadataBatch:
    """Batch operations for XMP metadata."""

    def __init__(self, writer: XMPMetadataWriter):
        """Initialize batch processor.

        Args:
            writer: XMP metadata writer instance
        """
        self.writer = writer

    def batch_write(
        self, metadata_pairs: List[tuple[Path, ColorNarrationMetadata]]
    ) -> Dict[str, Any]:
        """Write metadata to multiple images in batch.

        Args:
            metadata_pairs: List of (image_path, metadata) tuples

        Returns:
            Batch processing results
        """
        results = {
            "total": len(metadata_pairs),
            "successful": 0,
            "failed": 0,
            "errors": [],
        }

        for image_path, metadata in metadata_pairs:
            try:
                if self.writer.write_metadata(image_path, metadata):
                    results["successful"] += 1
                else:
                    results["failed"] += 1
                    results["errors"].append(f"{image_path.name}: Write failed")
            except Exception as e:
                results["failed"] += 1
                results["errors"].append(f"{image_path.name}: {str(e)}")

        return results

    def batch_read(
        self, image_paths: List[Path]
    ) -> Dict[Path, Optional[ColorNarrationMetadata]]:
        """Read metadata from multiple images in batch.

        Args:
            image_paths: List of image paths to read

        Returns:
            Dictionary mapping paths to metadata (None if not found)
        """
        results = {}

        for image_path in image_paths:
            try:
                metadata = self.writer.read_metadata(image_path)
                results[image_path] = metadata
            except Exception as e:
                logger.debug(f"Failed to read metadata from {image_path}: {e}")
                results[image_path] = None

        return results
