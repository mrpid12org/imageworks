"""Image viewer component with caching."""

import streamlit as st
from PIL import Image
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional


@st.cache_data(show_spinner="Loading image...")
def load_image(image_path: str, max_size: int = 800) -> np.ndarray:
    """
    Load and resize image (CACHED).

    Args:
        image_path: Path to image
        max_size: Maximum dimension for resizing

    Returns:
        Image as numpy array
    """
    img = Image.open(image_path)

    # Convert to RGB if necessary
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Resize if too large
    if max(img.size) > max_size:
        ratio = max_size / max(img.size)
        new_size = tuple(int(dim * ratio) for dim in img.size)
        img = img.resize(new_size, Image.Resampling.LANCZOS)

    return np.array(img)


@st.cache_data(show_spinner="Loading overlay...")
def load_overlay(overlay_path: str, max_size: int = 800) -> Optional[np.ndarray]:
    """
    Load overlay image (CACHED).

    Args:
        overlay_path: Path to overlay image
        max_size: Maximum dimension

    Returns:
        Overlay as numpy array or None
    """
    if not Path(overlay_path).exists():
        return None

    try:
        img = Image.open(overlay_path)

        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = tuple(int(dim * ratio) for dim in img.size)
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        return np.array(img)
    except Exception:
        return None


def render_image_grid(
    images: List[Dict[str, Any]],
    key_prefix: str,
    columns: int = 3,
    show_overlays: bool = False,
    overlay_key: str = "overlay_path",
    max_images: int = 100,
) -> Optional[str]:
    """
    Render grid of images with optional overlays.

    Args:
        images: List of image metadata dicts
        key_prefix: Unique prefix for widgets
        columns: Number of columns in grid
        show_overlays: Whether to show overlay toggle
        overlay_key: Key in metadata dict for overlay path
        max_images: Maximum images to show

    Returns:
        Selected image path or None
    """

    if not images:
        st.info("No images to display")
        return None

    # Limit number of images
    if len(images) > max_images:
        st.warning(f"⚠️ Showing first {max_images} of {len(images)} images")
        images = images[:max_images]

    # Overlay toggle
    show_overlay = False
    if show_overlays:
        show_overlay = st.checkbox(
            "Show overlays", value=False, key=f"{key_prefix}_show_overlay"
        )

    # Pagination
    items_per_page = columns * 4  # 4 rows per page
    total_pages = (len(images) + items_per_page - 1) // items_per_page

    if total_pages > 1:
        page = st.number_input(
            "Page",
            min_value=1,
            max_value=total_pages,
            value=1,
            key=f"{key_prefix}_page",
        )
        start_idx = (page - 1) * items_per_page
        end_idx = start_idx + items_per_page
        page_images = images[start_idx:end_idx]
    else:
        page_images = images

    # Render grid
    selected_image = None

    for i in range(0, len(page_images), columns):
        cols = st.columns(columns)

        for j, col in enumerate(cols):
            if i + j >= len(page_images):
                break

            img_data = page_images[i + j]
            img_path = img_data["path"]

            with col:
                try:
                    # Load image
                    img_array = load_image(img_path, max_size=400)

                    # Load overlay if enabled
                    if show_overlay and overlay_key in img_data:
                        overlay_path = img_data[overlay_key]
                        overlay_array = load_overlay(overlay_path, max_size=400)

                        if overlay_array is not None:
                            # Blend images
                            img_array = (img_array * 0.6 + overlay_array * 0.4).astype(
                                np.uint8
                            )

                    st.image(img_array, use_container_width=True)

                    # Image info
                    st.caption(img_data["name"])

                    # Select button
                    if st.button("📋 Select", key=f"{key_prefix}_select_{i}_{j}"):
                        selected_image = img_path

                except Exception as e:
                    st.error(f"Failed to load: {e}")

    return selected_image


def render_image_detail(
    image_path: str,
    overlay_path: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    show_overlay_toggle: bool = True,
) -> None:
    """
    Render detailed image view with metadata.

    Args:
        image_path: Path to image
        overlay_path: Optional overlay path
        metadata: Optional metadata dict
        show_overlay_toggle: Whether to show overlay toggle
    """

    col1, col2 = st.columns([2, 1])

    with col1:
        # Load and display image
        try:
            img_array = load_image(image_path, max_size=1200)

            # Overlay toggle
            show_overlay = False
            if show_overlay_toggle and overlay_path:
                show_overlay = st.checkbox("Show overlay", value=False)

            if show_overlay and overlay_path:
                overlay_array = load_overlay(overlay_path, max_size=1200)
                if overlay_array is not None:
                    # Blend images
                    alpha = st.slider("Overlay opacity", 0.0, 1.0, 0.5)
                    img_array = (
                        (1 - alpha) * img_array + alpha * overlay_array
                    ).astype(np.uint8)

            st.image(img_array, use_container_width=True)

        except Exception as e:
            st.error(f"Failed to load image: {e}")

    with col2:
        st.subheader("Image Info")

        # Basic info
        path_obj = Path(image_path)
        st.text(f"Name: {path_obj.name}")
        st.text(f"Path: {path_obj.parent}")

        if path_obj.exists():
            size_mb = path_obj.stat().st_size / (1024 * 1024)
            st.text(f"Size: {size_mb:.2f} MB")

        # Metadata
        if metadata:
            st.markdown("---")
            st.subheader("Metadata")

            for key, value in metadata.items():
                if key not in ["path", "name"]:
                    st.text(f"{key}: {value}")


def render_side_by_side(
    image1_path: str,
    image2_path: str,
    title1: str = "Image 1",
    title2: str = "Image 2",
) -> None:
    """
    Render two images side by side for comparison.

    Args:
        image1_path: Path to first image
        image2_path: Path to second image
        title1: Title for first image
        title2: Title for second image
    """

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(title1)
        try:
            img1 = load_image(image1_path, max_size=800)
            st.image(img1, use_container_width=True)
            st.caption(Path(image1_path).name)
        except Exception as e:
            st.error(f"Failed to load: {e}")

    with col2:
        st.subheader(title2)
        try:
            img2 = load_image(image2_path, max_size=800)
            st.image(img2, use_container_width=True)
            st.caption(Path(image2_path).name)
        except Exception as e:
            st.error(f"Failed to load: {e}")
