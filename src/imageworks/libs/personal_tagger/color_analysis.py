"""Color analysis utilities for personal tagger applications.

Provides color space conversions, statistical analysis, and color distribution
calculations for integration with mono-checker and color-narrator workflows.
"""

from typing import List, Dict, Optional, Any
import numpy as np
from pathlib import Path
from PIL import Image
import cv2


class ColorAnalyzer:
    """Utility class for color space analysis and conversions."""

    @staticmethod
    def rgb_to_lab(rgb_array: np.ndarray) -> np.ndarray:
        """Convert RGB array to LAB color space.

        Args:
            rgb_array: RGB image array (H, W, 3)

        Returns:
            LAB image array (H, W, 3)
        """
        # Ensure input is float32 in range [0, 1]
        if rgb_array.dtype == np.uint8:
            rgb_array = rgb_array.astype(np.float32) / 255.0

        # Convert RGB to LAB using OpenCV
        lab_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2LAB)
        return lab_array

    @staticmethod
    def lab_to_rgb(lab_array: np.ndarray) -> np.ndarray:
        """Convert LAB array to RGB color space.

        Args:
            lab_array: LAB image array (H, W, 3)

        Returns:
            RGB image array (H, W, 3) as uint8
        """
        # Convert LAB to RGB using OpenCV
        rgb_array = cv2.cvtColor(lab_array, cv2.COLOR_LAB2RGB)

        # Convert to uint8
        rgb_array = (rgb_array * 255).astype(np.uint8)
        return rgb_array

    @staticmethod
    def calculate_chroma(lab_array: np.ndarray) -> np.ndarray:
        """Calculate chroma from LAB array.

        Args:
            lab_array: LAB image array (H, W, 3)

        Returns:
            Chroma array (H, W)
        """
        a_channel = lab_array[:, :, 1]
        b_channel = lab_array[:, :, 2]

        # Calculate chroma as sqrt(a^2 + b^2)
        chroma = np.sqrt(a_channel**2 + b_channel**2)
        return chroma

    @staticmethod
    def calculate_hue(lab_array: np.ndarray) -> np.ndarray:
        """Calculate hue angle from LAB array.

        Args:
            lab_array: LAB image array (H, W, 3)

        Returns:
            Hue angle array in degrees (H, W)
        """
        a_channel = lab_array[:, :, 1]
        b_channel = lab_array[:, :, 2]

        # Calculate hue angle in degrees
        hue = np.degrees(np.arctan2(b_channel, a_channel))

        # Normalize to [0, 360)
        hue = np.where(hue < 0, hue + 360, hue)
        return hue

    @staticmethod
    def analyze_color_distribution(
        image_path: Path, mask_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Analyze color distribution in an image.

        Args:
            image_path: Path to input image
            mask_path: Optional mask to limit analysis region

        Returns:
            Dictionary with color distribution statistics
        """
        # Load image
        image = Image.open(image_path).convert("RGB")
        rgb_array = np.array(image)

        # Load mask if provided
        mask = None
        if mask_path and mask_path.exists():
            mask_img = Image.open(mask_path).convert("L")
            mask = np.array(mask_img) > 128  # Binary mask

        # Convert to LAB
        lab_array = ColorAnalyzer.rgb_to_lab(rgb_array)

        # Calculate chroma and hue
        chroma = ColorAnalyzer.calculate_chroma(lab_array)
        hue = ColorAnalyzer.calculate_hue(lab_array)

        # Apply mask if provided
        if mask is not None:
            lab_array = lab_array[mask]
            chroma = chroma[mask]
            hue = hue[mask]
        else:
            lab_array = lab_array.reshape(-1, 3)
            chroma = chroma.flatten()
            hue = hue.flatten()

        # Calculate statistics
        analysis = {
            "mean_lightness": float(np.mean(lab_array[:, 0])),
            "std_lightness": float(np.std(lab_array[:, 0])),
            "mean_chroma": float(np.mean(chroma)),
            "std_chroma": float(np.std(chroma)),
            "max_chroma": float(np.max(chroma)),
            "mean_hue": float(np.mean(hue)),
            "hue_std": float(np.std(hue)),
            "total_pixels": len(chroma),
            "high_chroma_pixels": int(
                np.sum(chroma > 10)
            ),  # Threshold for "colorful" pixels
            "contamination_level": float(
                np.sum(chroma > 5) / len(chroma)
            ),  # Basic contamination metric
        }

        return analysis

    @staticmethod
    def create_color_overlay(
        image_path: Path,
        chroma_threshold: float = 5.0,
        output_path: Optional[Path] = None,
    ) -> np.ndarray:
        """Create color contamination overlay visualization.

        Args:
            image_path: Path to input image
            chroma_threshold: Minimum chroma level to highlight
            output_path: Optional path to save overlay image

        Returns:
            Overlay array showing color contamination regions
        """
        # Load image
        image = Image.open(image_path).convert("RGB")
        rgb_array = np.array(image)

        # Convert to LAB and calculate chroma
        lab_array = ColorAnalyzer.rgb_to_lab(rgb_array)
        chroma = ColorAnalyzer.calculate_chroma(lab_array)

        # Create overlay based on chroma threshold
        overlay = np.zeros_like(rgb_array)

        # Highlight regions above threshold in red
        color_mask = chroma > chroma_threshold
        overlay[color_mask] = [255, 0, 0]  # Red overlay

        # Blend with original image
        alpha = 0.3
        blended = ((1 - alpha) * rgb_array + alpha * overlay).astype(np.uint8)

        # Save if output path provided
        if output_path:
            overlay_image = Image.fromarray(blended)
            overlay_image.save(output_path)

        return blended


class ColorRegionAnalyzer:
    """Analyzer for color regions and patterns in images."""

    @staticmethod
    def detect_color_regions(
        image_path: Path, chroma_threshold: float = 5.0, min_region_size: int = 100
    ) -> List[Dict[str, Any]]:
        """Detect distinct color contamination regions.

        Args:
            image_path: Path to input image
            chroma_threshold: Minimum chroma to consider as colored
            min_region_size: Minimum pixels for a valid region

        Returns:
            List of color region descriptions
        """
        # Load and analyze image
        image = Image.open(image_path).convert("RGB")
        rgb_array = np.array(image)
        lab_array = ColorAnalyzer.rgb_to_lab(rgb_array)

        chroma = ColorAnalyzer.calculate_chroma(lab_array)
        hue = ColorAnalyzer.calculate_hue(lab_array)

        # Create binary mask for colored regions
        color_mask = (chroma > chroma_threshold).astype(np.uint8)

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            color_mask
        )

        regions = []
        for i in range(1, num_labels):  # Skip background (0)
            if stats[i, cv2.CC_STAT_AREA] < min_region_size:
                continue

            # Get region pixels
            region_mask = labels == i
            region_chroma = chroma[region_mask]
            region_hue = hue[region_mask]

            # Analyze region characteristics
            centroid_y, centroid_x = centroids[i]
            bbox = stats[i]

            region_info = {
                "area": int(stats[i, cv2.CC_STAT_AREA]),
                "centroid": (float(centroid_x), float(centroid_y)),
                "bbox": {
                    "left": int(bbox[cv2.CC_STAT_LEFT]),
                    "top": int(bbox[cv2.CC_STAT_TOP]),
                    "width": int(bbox[cv2.CC_STAT_WIDTH]),
                    "height": int(bbox[cv2.CC_STAT_HEIGHT]),
                },
                "mean_chroma": float(np.mean(region_chroma)),
                "mean_hue": float(np.mean(region_hue)),
                "dominant_color_family": ColorRegionAnalyzer._classify_hue(
                    np.mean(region_hue)
                ),
                "intensity": "high" if np.mean(region_chroma) > 15 else "low",
            }

            regions.append(region_info)

        # Sort by area (largest first)
        regions.sort(key=lambda x: x["area"], reverse=True)
        return regions

    @staticmethod
    def _classify_hue(hue_degrees: float) -> str:
        """Classify hue angle into color family.

        Args:
            hue_degrees: Hue angle in degrees

        Returns:
            Color family name
        """
        # Normalize hue to [0, 360)
        hue = hue_degrees % 360

        if 0 <= hue < 30 or 330 <= hue < 360:
            return "red"
        elif 30 <= hue < 90:
            return "yellow"
        elif 90 <= hue < 150:
            return "green"
        elif 150 <= hue < 210:
            return "cyan"
        elif 210 <= hue < 270:
            return "blue"
        elif 270 <= hue < 330:
            return "magenta"
        else:
            return "unknown"

    @staticmethod
    def describe_color_pattern(regions: List[Dict[str, Any]]) -> str:
        """Generate natural language description of color patterns.

        Args:
            regions: List of color regions from detect_color_regions

        Returns:
            Natural language description
        """
        if not regions:
            return "No significant color contamination detected"

        # Analyze overall pattern
        total_colored_area = sum(r["area"] for r in regions)
        dominant_colors = [r["dominant_color_family"] for r in regions[:3]]  # Top 3

        description_parts = []

        # Start with contamination level
        if len(regions) == 1:
            description_parts.append("Localized color contamination")
        elif len(regions) <= 3:
            description_parts.append("Multiple areas of color contamination")
        else:
            description_parts.append("Widespread color contamination")

        # Describe dominant colors
        color_counts = {}
        for color in dominant_colors:
            color_counts[color] = color_counts.get(color, 0) + 1

        if color_counts:
            primary_color = max(color_counts.keys(), key=lambda k: color_counts[k])
            if color_counts[primary_color] > 1:
                description_parts.append(f"predominantly {primary_color} tones")
            else:
                color_list = ", ".join(set(dominant_colors))
                description_parts.append(f"with {color_list} tones")

        # Describe distribution
        largest_region = regions[0]
        if largest_region["area"] > total_colored_area * 0.6:
            description_parts.append("concentrated in one main area")
        else:
            description_parts.append("distributed across multiple regions")

        return " ".join(description_parts) + "."


class ColorStatistics:
    """Statistical analysis utilities for color data."""

    @staticmethod
    def calculate_color_moments(
        lab_array: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate color moments for LAB image.

        Args:
            lab_array: LAB image array (H, W, 3)
            mask: Optional binary mask

        Returns:
            Dictionary of color moments
        """
        if mask is not None:
            pixels = lab_array[mask]
        else:
            pixels = lab_array.reshape(-1, 3)

        moments = {}
        for i, channel in enumerate(["L", "A", "B"]):
            channel_data = pixels[:, i]
            moments[f"{channel}_mean"] = float(np.mean(channel_data))
            moments[f"{channel}_std"] = float(np.std(channel_data))
            moments[f"{channel}_skew"] = float(
                ColorStatistics._calculate_skewness(channel_data)
            )
            moments[f"{channel}_kurtosis"] = float(
                ColorStatistics._calculate_kurtosis(channel_data)
            )

        return moments

    @staticmethod
    def _calculate_skewness(data: np.ndarray) -> float:
        """Calculate skewness of data distribution."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0

        skewness = np.mean(((data - mean) / std) ** 3)
        return skewness

    @staticmethod
    def _calculate_kurtosis(data: np.ndarray) -> float:
        """Calculate kurtosis of data distribution."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0

        kurtosis = np.mean(((data - mean) / std) ** 4) - 3
        return kurtosis

    @staticmethod
    def compare_color_distributions(
        dist1: Dict[str, float], dist2: Dict[str, float]
    ) -> Dict[str, float]:
        """Compare two color distributions.

        Args:
            dist1: First distribution statistics
            dist2: Second distribution statistics

        Returns:
            Comparison metrics
        """
        comparison = {}

        for key in dist1:
            if key in dist2:
                diff = abs(dist1[key] - dist2[key])
                comparison[f"{key}_diff"] = diff

        # Calculate overall similarity score
        mean_diffs = [v for k, v in comparison.items() if "_mean" in k]
        if mean_diffs:
            comparison["similarity_score"] = 1.0 / (1.0 + np.mean(mean_diffs))

        return comparison
