"""
Simple 3x3 grid region system for human-readable spatial references.

This module provides a simple, intuitive spatial grid system that divides
images into 9 regions that humans can easily understand and reference.
"""

from dataclasses import dataclass
from typing import List, Tuple
from enum import Enum
from pathlib import Path


class GridRegion(Enum):
    """Simple 3x3 grid regions for spatial reference."""

    TOP_LEFT = "top-left"
    TOP_MIDDLE = "top-middle"
    TOP_RIGHT = "top-right"
    CENTER_LEFT = "center-left"
    CENTER = "center"
    CENTER_RIGHT = "center-right"
    BOTTOM_LEFT = "bottom-left"
    BOTTOM_MIDDLE = "bottom-middle"
    BOTTOM_RIGHT = "bottom-right"


@dataclass
class GridColorRegion:
    """A color region within a 3x3 grid system."""

    grid_region: GridRegion
    dominant_color: str
    hue_deg: float
    area_pct: float  # Percentage of the grid cell with this color
    mean_lightness: float  # 0-100 L* value

    @property
    def tonal_zone(self) -> str:
        """Compute tonal zone from lightness."""
        if self.mean_lightness < 35:
            return "shadow"
        elif self.mean_lightness < 70:
            return "midtone"
        else:
            return "highlight"


class ImageGridAnalyzer:
    """Analyzes images using simple 3x3 grid regions."""

    @staticmethod
    def create_grid_regions(image_path: Path, mono_data: dict) -> List[GridColorRegion]:
        """Create 3x3 grid regions with color analysis from image and mono data.

        Args:
            image_path: Path to the image file
            mono_data: Mono-checker analysis results

        Returns:
            List of GridColorRegion with color information
        """
        from PIL import Image

        # Load image to get dimensions
        with Image.open(image_path) as img:
            image_width, image_height = img.size

        return ImageGridAnalyzer.analyze_color_in_regions(
            mono_data, image_width, image_height
        )

    @staticmethod
    def create_grid_regions_from_dimensions(
        image_width: int, image_height: int
    ) -> List[Tuple[GridRegion, Tuple[int, int, int, int]]]:
        """Create 3x3 grid regions with bounding boxes from dimensions.

        Args:
            image_width: Image width in pixels
            image_height: Image height in pixels

        Returns:
            List of (GridRegion, (x, y, width, height)) tuples
        """
        # Calculate grid dimensions
        cell_width = image_width // 3
        cell_height = image_height // 3

        regions = []

        # Create 3x3 grid
        for row in range(3):
            for col in range(3):
                # Calculate bounding box
                x = col * cell_width
                y = row * cell_height

                # Handle remainder pixels in rightmost/bottom cells
                width = cell_width if col < 2 else image_width - x
                height = cell_height if row < 2 else image_height - y

                # Map to grid region enum
                region_map = [
                    [GridRegion.TOP_LEFT, GridRegion.TOP_MIDDLE, GridRegion.TOP_RIGHT],
                    [
                        GridRegion.CENTER_LEFT,
                        GridRegion.CENTER,
                        GridRegion.CENTER_RIGHT,
                    ],
                    [
                        GridRegion.BOTTOM_LEFT,
                        GridRegion.BOTTOM_MIDDLE,
                        GridRegion.BOTTOM_RIGHT,
                    ],
                ]

                grid_region = region_map[row][col]
                bbox = (x, y, width, height)

                regions.append((grid_region, bbox))

        return regions

    @staticmethod
    def analyze_color_in_regions(
        mono_data: dict,
        image_width: int,
        image_height: int,
        min_color_threshold: float = 2.0,  # Minimum chroma to be considered "colored"
    ) -> List[GridColorRegion]:
        """Analyze color contamination within 3x3 grid regions.

        This is a simplified version that would need integration with mono-checker's
        actual pixel-level analysis. For now, it creates demo data.

        Args:
            mono_data: Mono-checker analysis results
            image_width: Image width in pixels
            image_height: Image height in pixels
            min_color_threshold: Minimum chroma level to include

        Returns:
            List of grid regions with significant color contamination
        """
        # TODO: Integrate with actual mono-checker pixel analysis
        # For now, create demo regions based on mono-checker data

        colored_regions = []

        # Extract key info from mono data
        dominant_color = mono_data.get("dominant_color", "unknown")
        dominant_hue = mono_data.get("dominant_hue_deg", 0.0)

        # Demo: assume color appears in a few regions
        # In real implementation, this would analyze actual pixel data per grid cell
        if mono_data.get("verdict") in ["fail", "pass_with_query"]:
            # Simulate color contamination in 1-3 regions
            demo_regions = [
                GridColorRegion(
                    grid_region=GridRegion.TOP_LEFT,
                    dominant_color=dominant_color,
                    hue_deg=dominant_hue,
                    area_pct=15.2,  # Percentage of this grid cell affected
                    mean_lightness=65.0,
                ),
                GridColorRegion(
                    grid_region=GridRegion.CENTER,
                    dominant_color=dominant_color,
                    hue_deg=dominant_hue,
                    area_pct=8.7,
                    mean_lightness=45.0,
                ),
            ]
            colored_regions.extend(demo_regions)

        return colored_regions

    @staticmethod
    def regions_to_json(regions: List[GridColorRegion]) -> List[dict]:
        """Convert grid regions to JSON format for VLM prompts."""
        return [
            {
                "grid_region": region.grid_region.value,
                "dominant_color": region.dominant_color,
                "hue_deg": region.hue_deg,
                "area_pct": region.area_pct,
                "mean_lightness": region.mean_lightness,
                "tonal_zone": region.tonal_zone,
            }
            for region in regions
        ]

    @staticmethod
    def format_regions_for_human(regions: List[GridColorRegion]) -> str:
        """Format regions in human-readable text."""
        if not regions:
            return "No significant color regions detected."

        lines = ["Color contamination detected in:"]
        for region in regions:
            lines.append(
                f"• {region.grid_region.value}: {region.dominant_color} "
                f"({region.area_pct:.1f}% affected, {region.tonal_zone})"
            )

        return "\n".join(lines)


def create_demo_grid_regions() -> List[GridColorRegion]:
    """Create demo grid regions for testing."""
    return [
        GridColorRegion(
            grid_region=GridRegion.TOP_RIGHT,
            dominant_color="yellow-green",
            hue_deg=88.0,
            area_pct=12.5,
            mean_lightness=72.0,  # highlight
        ),
        GridColorRegion(
            grid_region=GridRegion.CENTER_LEFT,
            dominant_color="blue",
            hue_deg=210.0,
            area_pct=6.8,
            mean_lightness=38.0,  # shadow
        ),
    ]
