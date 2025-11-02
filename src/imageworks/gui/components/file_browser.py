"""File browser component with caching."""

from pathlib import Path
from typing import List, Dict, Any, Optional

import streamlit as st

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


def _streamlit_rerun() -> None:
    """Trigger a rerun using the stable API if available, otherwise fall back."""

    rerun = getattr(st, "rerun", None)
    if callable(rerun):
        rerun()
        return

    legacy_rerun = getattr(st, "experimental_rerun", None)
    if callable(legacy_rerun):
        legacy_rerun()


def render_path_browser(
    key_prefix: str,
    start_path: Optional[str] = None,
    allow_file_selection: bool = False,
    file_types: Optional[List[str]] = None,
    initial_file: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    """
    Interactive filesystem browser for directories (and optional file selection).

    Args:
        key_prefix: Unique key namespace for Streamlit widgets
        start_path: Initial directory to open
        allow_file_selection: Whether to allow choosing a single file
        file_types: Optional list of file extensions to show (lowercase)
        initial_file: Optional path to preselect when allow_file_selection is True

    Returns:
        Dict with keys:
            - current_dir: Directory currently being viewed
            - selected_dir: Directory selection (same as current view)
            - selected_file: Selected file path when `allow_file_selection` is True
    """

    session_key = f"{key_prefix}_current_dir"
    if session_key not in st.session_state:
        default_start = Path(start_path).expanduser() if start_path else Path.home()
        st.session_state[session_key] = str(default_start.resolve())

    current_dir = Path(st.session_state[session_key]).expanduser()
    if not current_dir.exists():
        fallback = Path(start_path or Path.home()).expanduser()
        st.session_state[session_key] = str(fallback.resolve())
        current_dir = fallback

    st.markdown(f"**Current Directory:** `{current_dir}`")

    jump_key = f"{key_prefix}_jump"
    dir_select_widget_key = f"{key_prefix}_dir_select_widget"
    file_select_key = f"{key_prefix}_file_select"
    jump_input_key = f"{jump_key}_value"

    pending_override_key = f"{jump_input_key}_override"
    if pending_override_key in st.session_state:
        st.session_state[jump_input_key] = st.session_state.pop(pending_override_key)

    if jump_input_key not in st.session_state:
        st.session_state[jump_input_key] = str(current_dir)

    col_home, col_up, col_refresh = st.columns([1, 1, 1])
    home_key = f"{key_prefix}_home_button"
    up_key = f"{key_prefix}_up_button"
    refresh_key = f"{key_prefix}_refresh_button"

    if col_home.button("🏠 Home", key=home_key):
        new_path = Path.home().resolve()
        st.session_state[session_key] = str(new_path)
        st.session_state[pending_override_key] = str(new_path)
        st.session_state.pop(dir_select_widget_key, None)
        st.session_state.pop(file_select_key, None)
        _streamlit_rerun()

    if col_up.button("⬆️ Up", key=up_key):
        parent = current_dir.parent
        if parent != current_dir:
            new_path = parent.resolve()
            st.session_state[session_key] = str(new_path)
            st.session_state[pending_override_key] = str(new_path)
            st.session_state.pop(dir_select_widget_key, None)
            st.session_state.pop(file_select_key, None)
            _streamlit_rerun()

    if col_refresh.button("🔄 Refresh", key=refresh_key):
        scan_directory.clear()

    jump_col, go_col = st.columns([3, 1])

    jump_value = jump_col.text_input(
        "Jump to path",
        value=st.session_state[jump_input_key],
        key=jump_input_key,
        help="Enter a path and click Go to navigate directly.",
    )

    if go_col.button("Go", key=f"{key_prefix}_jump_go"):
        target = Path(jump_value).expanduser()
        if target.exists():
            resolved = target.resolve()
            st.session_state[session_key] = str(resolved)
            st.session_state[pending_override_key] = str(resolved)
            st.session_state.pop(dir_select_widget_key, None)
            st.session_state.pop(file_select_key, None)
            _streamlit_rerun()
        else:
            st.warning(f"⚠️ Path does not exist: {target}")

    directories: List[Path] = []
    files: List[Path] = []
    try:
        for entry in sorted(current_dir.iterdir(), key=lambda p: p.name.lower()):
            if entry.is_dir():
                directories.append(entry)
            elif allow_file_selection:
                if file_types:
                    if entry.suffix.lower() in file_types:
                        files.append(entry)
                else:
                    files.append(entry)
    except PermissionError:
        st.error(f"Permission denied accessing {current_dir}")

    if directories:
        options = ["(select directory)"] + [d.name for d in directories]
        choice = st.selectbox(
            "Subdirectories",
            options=options,
            key=dir_select_widget_key,
            index=0,
            label_visibility="visible",
        )
        if choice and choice != "(select directory)":
            selected = next((d for d in directories if d.name == choice), None)
            if selected is not None:
                resolved = selected.resolve()
                st.session_state[session_key] = str(resolved)
                st.session_state[pending_override_key] = str(resolved)
                st.session_state.pop(dir_select_widget_key, None)
                st.session_state.pop(file_select_key, None)
                _streamlit_rerun()
    else:
        st.caption("No subdirectories.")

    selected_file: Optional[str] = None
    if allow_file_selection:
        if files:
            default_index = 0
            if initial_file:
                for idx, file_path in enumerate(files):
                    if str(file_path.resolve()) == str(Path(initial_file).resolve()):
                        default_index = idx
                        break

            st.radio(
                "Files",
                options=list(range(len(files))),
                format_func=lambda idx: files[idx].name,
                key=file_select_key,
                index=default_index,
            )
            selected_idx = st.session_state.get(file_select_key, default_index)
            try:
                selected_file = str(files[selected_idx].resolve())
            except (IndexError, TypeError):
                selected_file = None
        else:
            st.caption("No files match the selected filters.")
    return {
        "current_dir": str(current_dir.resolve()),
        "selected_dir": str(current_dir.resolve()),
        "selected_file": selected_file,
    }
