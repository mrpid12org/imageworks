"""Data loader module for color-narrator processing.

Handles loading and validation of JPEG originals, lab overlay PNGs, and mono-checker JSONL data.
Provides batching and filtering capabilities for VLM processing pipeline.
"""

from typing import List, Dict, Optional, Iterator, Any, Iterable
from pathlib import Path
from dataclasses import dataclass
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class ColorNarratorItem:
    """Single item for color narration processing."""

    image_path: Path
    overlay_path: Path
    mono_data: Dict[str, Any]
    has_existing_xmp: bool = False


@dataclass
class DataLoaderConfig:
    """Configuration for data loading and filtering."""

    images_dir: Path
    overlays_dir: Path
    mono_jsonl: Path
    image_extensions: List[str] = None
    overlay_extensions: List[str] = None
    min_contamination_level: float = 0.1
    require_overlays: bool = True
    allowed_verdicts: Optional[Iterable[str]] = None

    def __post_init__(self):
        if self.image_extensions is None:
            self.image_extensions = [".jpg", ".jpeg", ".JPG", ".JPEG"]
        if self.overlay_extensions is None:
            self.overlay_extensions = [
                ".png",
                ".PNG",
                ".jpg",
                ".JPG",
                ".jpeg",
                ".JPEG",
            ]
        if self.allowed_verdicts is not None:
            self.allowed_verdicts = {v.lower() for v in self.allowed_verdicts}


class ColorNarratorDataLoader:
    """Data loader for color-narrator VLM processing."""

    def __init__(self, config: DataLoaderConfig):
        """Initialize data loader with configuration.

        Args:
            config: Data loader configuration
        """
        self.config = config
        self._mono_data = {}
        self._loaded = False

    def load(self) -> None:
        """Load and validate all data sources."""
        logger.info("Loading color-narrator data sources...")

        # Validate directories exist
        if not self.config.images_dir.exists():
            raise FileNotFoundError(
                f"Images directory not found: {self.config.images_dir}"
            )
        if not self.config.overlays_dir.exists():
            raise FileNotFoundError(
                f"Overlays directory not found: {self.config.overlays_dir}"
            )
        if not self.config.mono_jsonl.exists():
            raise FileNotFoundError(
                f"Mono JSONL file not found: {self.config.mono_jsonl}"
            )

        # Load mono-checker results
        self._load_mono_data()

        self._loaded = True
        logger.info(f"Loaded {len(self._mono_data)} mono-checker results")

    def _load_mono_data(self) -> None:
        """Load mono-checker JSONL data."""
        self._mono_data = {}

        try:
            with open(self.config.mono_jsonl, "r") as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        filename = data.get("filename")
                        if not filename:
                            source_path = data.get("path")
                            if source_path:
                                filename = Path(str(source_path)).name
                        if filename:
                            self._mono_data[filename] = data
                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"Skipping malformed JSON at line {line_num}: {e}"
                        )
                        continue

        except Exception as e:
            raise RuntimeError(f"Failed to load mono JSONL data: {e}")

    def get_items(
        self, batch_size: Optional[int] = None
    ) -> Iterator[List[ColorNarratorItem]]:
        """Get items for processing in batches.

        Args:
            batch_size: Size of each batch (None for all items)

        Yields:
            Batches of ColorNarratorItem objects
        """
        if not self._loaded:
            self.load()

        items = list(self._iter_valid_items())

        if batch_size is None:
            yield items
        else:
            for i in range(0, len(items), batch_size):
                yield items[i : i + batch_size]

    def _iter_valid_items(self) -> Iterator[ColorNarratorItem]:
        """Iterate over valid items that meet filtering criteria."""
        # Find all image files (recursive to support nested competition folders)
        image_files: List[Path] = []
        for ext in self.config.image_extensions:
            image_files.extend(self.config.images_dir.rglob(f"*{ext}"))

        for image_path in image_files:
            # Check if we have mono data for this image
            mono_data = self._mono_data.get(image_path.name)
            if not mono_data:
                logger.debug(f"No mono data for {image_path.name}, skipping")
                continue

            verdict = str(mono_data.get("verdict", "")).lower()
            if (
                self.config.allowed_verdicts
                and verdict not in self.config.allowed_verdicts
            ):
                logger.debug(
                    "Verdict %s not allowed for %s, skipping",
                    verdict,
                    image_path.name,
                )
                continue

            # Check contamination level filter
            contamination = mono_data.get("contamination_level")
            if contamination is None:
                contamination = mono_data.get("chroma_max", 0.0)
            if contamination < self.config.min_contamination_level:
                logger.debug(
                    f"Contamination {contamination} below threshold for {image_path.name}"
                )
                continue

            # Find corresponding overlay
            overlay_path = self._find_overlay_for_image(image_path)
            if self.config.require_overlays and not overlay_path:
                logger.debug(f"No overlay found for {image_path.name}, skipping")
                continue

            # Check if XMP already exists
            has_existing_xmp = self._has_existing_color_narration(image_path)

            yield ColorNarratorItem(
                image_path=image_path,
                overlay_path=overlay_path,
                mono_data=mono_data,
                has_existing_xmp=has_existing_xmp,
            )

    def _find_overlay_for_image(self, image_path: Path) -> Optional[Path]:
        """Find corresponding overlay PNG for an image."""
        # Try different naming patterns
        base_name = image_path.stem

        search_dirs = [self.config.overlays_dir]
        if image_path.parent not in search_dirs:
            search_dirs.insert(0, image_path.parent)

        def _find_with_suffix(suffix: str = "") -> Optional[Path]:
            for directory in search_dirs:
                for ext in self.config.overlay_extensions:
                    candidate = directory / f"{base_name}{suffix}{ext}"
                    if candidate.exists() and candidate != image_path:
                        return candidate
            return None

        overlay_path = _find_with_suffix()
        if overlay_path:
            return overlay_path

        overlay_path = _find_with_suffix("_lab_chroma")
        if overlay_path:
            return overlay_path

        overlay_path = _find_with_suffix("_lab_residual")
        if overlay_path:
            return overlay_path

        return None

    def _has_existing_color_narration(self, image_path: Path) -> bool:
        """Check if image already has color narration XMP metadata."""
        try:
            # TODO: Implement XMP metadata checking
            # For now, return False to always process
            return False
        except Exception as e:
            logger.debug(f"Error checking XMP for {image_path}: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about loaded data."""
        if not self._loaded:
            self.load()

        # Count valid items
        valid_items = list(self._iter_valid_items())

        def _count_files(root: Path, extensions: List[str]) -> int:
            return sum(1 for ext in extensions for _ in root.rglob(f"*{ext}"))

        # Calculate statistics
        total_images = _count_files(
            self.config.images_dir, self.config.image_extensions
        )
        total_overlays = _count_files(
            self.config.overlays_dir, self.config.overlay_extensions
        )

        contamination_levels = [
            item.mono_data.get("contamination_level", 0.0) for item in valid_items
        ]

        return {
            "total_images": total_images,
            "total_overlays": total_overlays,
            "mono_results": len(self._mono_data),
            "valid_items": len(valid_items),
            "avg_contamination": (
                sum(contamination_levels) / len(contamination_levels)
                if contamination_levels
                else 0.0
            ),
            "max_contamination": (
                max(contamination_levels) if contamination_levels else 0.0
            ),
            "items_with_xmp": sum(1 for item in valid_items if item.has_existing_xmp),
        }

    def validate_data_consistency(self) -> List[str]:
        """Validate data consistency and return list of issues found."""
        if not self._loaded:
            self.load()

        issues = []

        # Check for orphaned mono results
        image_names = {
            path.name
            for ext in self.config.image_extensions
            for path in self.config.images_dir.rglob(f"*{ext}")
        }
        for filename in self._mono_data.keys():
            if filename not in image_names:
                issues.append(f"Orphaned mono result: {filename}")

        # Check for missing overlays when required
        if self.config.require_overlays:
            valid_items = list(self._iter_valid_items())
            for item in valid_items:
                if not item.overlay_path or not item.overlay_path.exists():
                    issues.append(f"Missing overlay for: {item.image_path.name}")

        return issues
