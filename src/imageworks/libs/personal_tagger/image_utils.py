"""Image processing utilities for personal tagger applications.

Provides common image operations, format conversions, and preprocessing
utilities for color analysis and VLM processing workflows.
"""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import logging

logger = logging.getLogger(__name__)


class ImageLoader:
    """Utility for loading and validating image files."""

    SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    @staticmethod
    def load_image(
        image_path: Path,
        target_size: Optional[Tuple[int, int]] = None,
        color_mode: str = "RGB",
    ) -> Image.Image:
        """Load and preprocess image from path.

        Args:
            image_path: Path to image file
            target_size: Optional (width, height) for resizing
            color_mode: Color mode for conversion ('RGB', 'L', etc.)

        Returns:
            PIL Image object
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        if image_path.suffix.lower() not in ImageLoader.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported image format: {image_path.suffix}")

        try:
            image = Image.open(image_path)

            # Convert color mode if needed
            if image.mode != color_mode:
                image = image.convert(color_mode)

            # Resize if requested
            if target_size:
                image = image.resize(target_size, Image.Resampling.LANCZOS)

            return image

        except Exception as e:
            raise RuntimeError(f"Failed to load image {image_path}: {e}")

    @staticmethod
    def load_image_array(
        image_path: Path,
        target_size: Optional[Tuple[int, int]] = None,
        normalize: bool = False,
    ) -> np.ndarray:
        """Load image as numpy array.

        Args:
            image_path: Path to image file
            target_size: Optional (width, height) for resizing
            normalize: Whether to normalize to [0, 1] range

        Returns:
            Image as numpy array
        """
        image = ImageLoader.load_image(image_path, target_size, "RGB")
        array = np.array(image)

        if normalize:
            array = array.astype(np.float32) / 255.0

        return array

    @staticmethod
    def validate_image_batch(image_paths: List[Path]) -> Dict[str, List[Path]]:
        """Validate a batch of image paths.

        Args:
            image_paths: List of image paths to validate

        Returns:
            Dictionary with 'valid' and 'invalid' path lists
        """
        valid = []
        invalid = []

        for path in image_paths:
            try:
                if (
                    path.exists()
                    and path.is_file()
                    and path.suffix.lower() in ImageLoader.SUPPORTED_FORMATS
                ):

                    # Try to open image briefly to verify it's valid
                    with Image.open(path) as img:
                        img.verify()
                    valid.append(path)
                else:
                    invalid.append(path)
            except Exception:
                invalid.append(path)

        return {"valid": valid, "invalid": invalid}


class ImageProcessor:
    """Image processing operations for analysis workflows."""

    @staticmethod
    def resize_maintain_aspect(image: Image.Image, max_size: int = 1024) -> Image.Image:
        """Resize image maintaining aspect ratio.

        Args:
            image: PIL Image to resize
            max_size: Maximum dimension size

        Returns:
            Resized PIL Image
        """
        width, height = image.size

        if max(width, height) <= max_size:
            return image

        if width > height:
            new_width = max_size
            new_height = int((height * max_size) / width)
        else:
            new_height = max_size
            new_width = int((width * max_size) / height)

        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    @staticmethod
    def create_thumbnail(
        image: Image.Image,
        size: Tuple[int, int] = (256, 256),
        maintain_aspect: bool = True,
    ) -> Image.Image:
        """Create thumbnail of image.

        Args:
            image: PIL Image for thumbnail creation
            size: Target thumbnail size
            maintain_aspect: Whether to maintain aspect ratio

        Returns:
            Thumbnail PIL Image
        """
        if maintain_aspect:
            image.thumbnail(size, Image.Resampling.LANCZOS)
            return image
        else:
            return image.resize(size, Image.Resampling.LANCZOS)

    @staticmethod
    def enhance_image(
        image: Image.Image,
        brightness: float = 1.0,
        contrast: float = 1.0,
        saturation: float = 1.0,
        sharpness: float = 1.0,
    ) -> Image.Image:
        """Apply enhancement filters to image.

        Args:
            image: PIL Image to enhance
            brightness: Brightness factor (1.0 = no change)
            contrast: Contrast factor (1.0 = no change)
            saturation: Saturation factor (1.0 = no change)
            sharpness: Sharpness factor (1.0 = no change)

        Returns:
            Enhanced PIL Image
        """
        enhanced = image

        if brightness != 1.0:
            enhancer = ImageEnhance.Brightness(enhanced)
            enhanced = enhancer.enhance(brightness)

        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(contrast)

        if saturation != 1.0:
            enhancer = ImageEnhance.Color(enhanced)
            enhanced = enhancer.enhance(saturation)

        if sharpness != 1.0:
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(sharpness)

        return enhanced

    @staticmethod
    def apply_blur(
        image: Image.Image, blur_type: str = "gaussian", radius: float = 2.0
    ) -> Image.Image:
        """Apply blur filter to image.

        Args:
            image: PIL Image to blur
            blur_type: Type of blur ('gaussian', 'box', 'motion')
            radius: Blur radius

        Returns:
            Blurred PIL Image
        """
        if blur_type == "gaussian":
            return image.filter(ImageFilter.GaussianBlur(radius))
        elif blur_type == "box":
            return image.filter(ImageFilter.BoxBlur(radius))
        elif blur_type == "motion":
            # Motion blur approximation
            return image.filter(ImageFilter.BLUR)
        else:
            raise ValueError(f"Unsupported blur type: {blur_type}")


class ImageComparison:
    """Utilities for comparing images and detecting differences."""

    @staticmethod
    def calculate_mse(image1: np.ndarray, image2: np.ndarray) -> float:
        """Calculate Mean Squared Error between two images.

        Args:
            image1: First image array
            image2: Second image array

        Returns:
            MSE value
        """
        if image1.shape != image2.shape:
            raise ValueError("Images must have the same dimensions")

        mse = np.mean((image1.astype(np.float32) - image2.astype(np.float32)) ** 2)
        return float(mse)

    @staticmethod
    def calculate_psnr(image1: np.ndarray, image2: np.ndarray) -> float:
        """Calculate Peak Signal-to-Noise Ratio between two images.

        Args:
            image1: First image array
            image2: Second image array

        Returns:
            PSNR value in dB
        """
        mse = ImageComparison.calculate_mse(image1, image2)

        if mse == 0:
            return float("inf")

        max_pixel = 255.0 if image1.dtype == np.uint8 else 1.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return float(psnr)

    @staticmethod
    def calculate_ssim(image1: np.ndarray, image2: np.ndarray) -> float:
        """Calculate Structural Similarity Index between two images.

        Args:
            image1: First image array (grayscale)
            image2: Second image array (grayscale)

        Returns:
            SSIM value (0-1)
        """
        # Convert to grayscale if needed
        if len(image1.shape) == 3:
            image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
        if len(image2.shape) == 3:
            image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)

        # Calculate SSIM using OpenCV
        score = cv2.matchTemplate(image1, image2, cv2.TM_CCOEFF_NORMED)[0][0]
        return float(score)

    @staticmethod
    def find_differences(
        image1: np.ndarray, image2: np.ndarray, threshold: int = 30
    ) -> np.ndarray:
        """Find pixel differences between two images.

        Args:
            image1: First image array
            image2: Second image array
            threshold: Difference threshold

        Returns:
            Binary mask of differences
        """
        if image1.shape != image2.shape:
            raise ValueError("Images must have the same dimensions")

        # Calculate absolute difference
        diff = cv2.absdiff(image1, image2)

        # Convert to grayscale if color
        if len(diff.shape) == 3:
            diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)

        # Apply threshold
        _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

        return mask


class ImageBatch:
    """Utilities for batch image processing operations."""

    def __init__(
        self,
        image_paths: List[Path],
        batch_size: int = 32,
        target_size: Optional[Tuple[int, int]] = None,
    ):
        """Initialize image batch processor.

        Args:
            image_paths: List of image paths to process
            batch_size: Size of processing batches
            target_size: Optional target size for all images
        """
        self.image_paths = image_paths
        self.batch_size = batch_size
        self.target_size = target_size

    def process_batches(
        self, processor_func: callable, **processor_kwargs
    ) -> List[Any]:
        """Process images in batches with custom function.

        Args:
            processor_func: Function to apply to each batch
            **processor_kwargs: Additional arguments for processor

        Returns:
            List of processing results
        """
        results = []

        for i in range(0, len(self.image_paths), self.batch_size):
            batch_paths = self.image_paths[i : i + self.batch_size]

            # Load batch images
            batch_images = []
            for path in batch_paths:
                try:
                    image = ImageLoader.load_image(path, self.target_size)
                    batch_images.append(image)
                except Exception as e:
                    logger.warning(f"Failed to load {path}: {e}")
                    batch_images.append(None)

            # Process batch
            batch_results = processor_func(batch_images, **processor_kwargs)
            results.extend(batch_results)

        return results

    def load_as_arrays(
        self, normalize: bool = False, color_mode: str = "RGB"
    ) -> List[Optional[np.ndarray]]:
        """Load all images as numpy arrays.

        Args:
            normalize: Whether to normalize pixel values
            color_mode: Color mode for loading

        Returns:
            List of image arrays (None for failed loads)
        """
        arrays = []

        for path in self.image_paths:
            try:
                image = ImageLoader.load_image(path, self.target_size, color_mode)
                array = np.array(image)

                if normalize:
                    array = array.astype(np.float32) / 255.0

                arrays.append(array)
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")
                arrays.append(None)

        return arrays

    def save_processed_images(
        self,
        images: List[Image.Image],
        output_dir: Path,
        suffix: str = "_processed",
        format: str = "JPEG",
    ) -> List[Path]:
        """Save processed images to output directory.

        Args:
            images: List of processed PIL Images
            output_dir: Output directory
            suffix: Suffix to add to filenames
            format: Output image format

        Returns:
            List of saved file paths
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_paths = []

        for i, (original_path, processed_image) in enumerate(
            zip(self.image_paths, images)
        ):
            if processed_image is None:
                saved_paths.append(None)
                continue

            # Generate output filename
            output_name = original_path.stem + suffix + f".{format.lower()}"
            output_path = output_dir / output_name

            try:
                processed_image.save(output_path, format=format)
                saved_paths.append(output_path)
            except Exception as e:
                logger.error(f"Failed to save {output_path}: {e}")
                saved_paths.append(None)

        return saved_paths
