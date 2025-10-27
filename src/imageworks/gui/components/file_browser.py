"""File browser component with caching."""

import streamlit as st
from pathlib import Path
from typing import List, Dict, Any, Optional
from imageworks.gui.config import IMAGE_EXTENSIONS, RAW_EXTENSIONS, ALL_IMAGE_EXTENSIONS


@st.cache_data(ttl=300, show_spinner="Scanning directory...")
def scan_directory(
    directory: str, recursive: bool = True, file_types: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Scan directory for image files (CACHED).

    Args:
        directory: Path to scan
        recursive: Whether to scan subdirectories
        file_types: List of extensions to include (e.g., ['.jpg', '.png'])

    Returns:
        List of file metadata dicts
    """
    if file_types is None:
        file_types = ALL_IMAGE_EXTENSIONS

    path = Path(directory)
    if not path.exists():
        return []

    files = []
    pattern = "**/*" if recursive else "*"

    for file_path in path.glob(pattern):
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() in file_types:
            try:
                stat = file_path.stat()
                files.append(
                    {
                        "path": str(file_path),
                        "name": file_path.name,
                        "size": stat.st_size,
                        "modified": stat.st_mtime,
                        "parent": str(file_path.parent),
                    }
                )
            except Exception:
                continue

    return sorted(files, key=lambda x: x["name"])


def render_file_browser(
    key_prefix: str,
    default_dir: Optional[str] = None,
    file_types: Optional[List[str]] = None,
    show_recursive: bool = True,
) -> Dict[str, Any]:
    """
    Render file browser component.

    Args:
        key_prefix: Unique prefix for widget keys
        default_dir: Default directory path
        file_types: File extensions to filter
        show_recursive: Whether to show recursive checkbox

    Returns:
        Dict with 'directory', 'files', 'recursive' keys
    """

    # Directory input
    directory = st.text_input(
        "📁 Directory",
        value=default_dir or str(Path.home()),
        key=f"{key_prefix}_directory",
        help="Enter the path to scan for images",
    )

    # Recursive option
    if show_recursive:
        recursive = st.checkbox(
            "Include subdirectories", value=True, key=f"{key_prefix}_recursive"
        )
    else:
        recursive = False

    # File type filter
    col1, col2, col3 = st.columns(3)
    with col1:
        include_jpg = st.checkbox("JPG/PNG", value=True, key=f"{key_prefix}_jpg")
    with col2:
        include_raw = st.checkbox("RAW", value=False, key=f"{key_prefix}_raw")
    with col3:
        if st.button("🔄 Refresh", key=f"{key_prefix}_refresh"):
            scan_directory.clear()
            st.rerun()

    # Build file type list
    selected_types = []
    if include_jpg:
        selected_types.extend(IMAGE_EXTENSIONS)
    if include_raw:
        selected_types.extend(RAW_EXTENSIONS)

    if not selected_types:
        selected_types = IMAGE_EXTENSIONS

    # Scan directory (cached)
    files = []
    if Path(directory).exists():
        files = scan_directory(directory, recursive, selected_types)

        if files:
            st.success(f"✅ Found {len(files)} images")
        else:
            st.info("No images found in this directory")
    else:
        st.error(f"❌ Directory does not exist: {directory}")

    return {
        "directory": directory,
        "files": files,
        "recursive": recursive,
        "file_types": selected_types,
    }


def render_simple_directory_picker(
    label: str,
    default_dir: Optional[str] = None,
    key: Optional[str] = None,
) -> str:
    """
    Simple directory picker without file scanning.

    Args:
        label: Label for the input
        default_dir: Default directory path
        key: Widget key

    Returns:
        Selected directory path
    """
    directory = st.text_input(
        label,
        value=default_dir or str(Path.home()),
        key=key,
        help="Enter directory path",
    )

    if not Path(directory).exists():
        st.warning(f"⚠️ Directory does not exist: {directory}")

    return directory
