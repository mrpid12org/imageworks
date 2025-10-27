"""GUI configuration and constants."""

from pathlib import Path

# Paths
GUI_ROOT = Path(__file__).parent
PROJECT_ROOT = GUI_ROOT.parent.parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CONFIGS_DIR = PROJECT_ROOT / "configs"
LOGS_DIR = PROJECT_ROOT / "logs"

# Registry paths
MODEL_REGISTRY_PATH = CONFIGS_DIR / "model_registry.json"
MODEL_REGISTRY_DISCOVERED_PATH = CONFIGS_DIR / "model_registry.discovered.json"

# GUI-specific paths
GUI_STATE_FILE = OUTPUTS_DIR / "gui_state.json"
JOB_HISTORY_FILE = OUTPUTS_DIR / "gui_job_history.json"

# Default input directory (from pyproject.toml [tool.imageworks.mono])
DEFAULT_INPUT_DIR = "/mnt/d/Proper Photos/photos/ccc competition images"

# Mono Checker defaults (from pyproject.toml [tool.imageworks.mono])
MONO_DEFAULT_OUTPUT_JSONL = OUTPUTS_DIR / "results" / "mono_results.jsonl"
MONO_DEFAULT_SUMMARY_PATH = OUTPUTS_DIR / "summaries" / "mono_summary.md"
MONO_DEFAULT_OVERLAYS_DIR = OUTPUTS_DIR / "overlays"

# Personal Tagger defaults (from pyproject.toml [tool.imageworks.personal_tagger])
TAGGER_DEFAULT_OUTPUT_JSONL = OUTPUTS_DIR / "results" / "personal_tagger.jsonl"
TAGGER_DEFAULT_SUMMARY_PATH = OUTPUTS_DIR / "summaries" / "personal_tagger_summary.md"

# Image Similarity defaults (from pyproject.toml [tool.imageworks.image_similarity_checker])
SIMILARITY_DEFAULT_LIBRARY_ROOT = "/mnt/d/Proper Photos/photos/ccc competition images"
SIMILARITY_DEFAULT_OUTPUT_JSONL = OUTPUTS_DIR / "results" / "similarity_results.jsonl"
SIMILARITY_DEFAULT_SUMMARY_PATH = OUTPUTS_DIR / "summaries" / "similarity_summary.md"
SIMILARITY_DEFAULT_CACHE_DIR = OUTPUTS_DIR / "cache" / "similarity"

# Color Narrator defaults (from pyproject.toml [tool.imageworks.color_narrator])
NARRATOR_DEFAULT_IMAGES_DIR = "/mnt/d/Proper Photos/photos/ccc competition images"
NARRATOR_DEFAULT_OVERLAYS_DIR = "/mnt/d/Proper Photos/photos/ccc competition images"

# Zip Extract defaults (from pyproject.toml [tool.imageworks.zip-extract])
ZIP_DEFAULT_ZIP_DIR = "/mnt/c/Users/stewa/Downloads/CCC comp zips"
ZIP_DEFAULT_EXTRACT_ROOT = "/mnt/d/Proper Photos/photos/ccc competition images"
ZIP_DEFAULT_SUMMARY_OUTPUT = OUTPUTS_DIR / "zip_extract_summary.md"

# Legacy aliases for backward compatibility
DEFAULT_OUTPUT_JSONL = MONO_DEFAULT_OUTPUT_JSONL
DEFAULT_SUMMARY_PATH = MONO_DEFAULT_SUMMARY_PATH
DEFAULT_OVERLAYS_DIR = MONO_DEFAULT_OVERLAYS_DIR

# Backend URLs
# Note: vLLM runs inside chat_proxy docker container, not as separate service
DEFAULT_BACKENDS = {
    "chat_proxy": "http://localhost:8100/v1",
    "ollama": "http://localhost:11434",
}

# File type filters
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"]
RAW_EXTENSIONS = [".cr2", ".cr3", ".nef", ".arw", ".dng", ".orf", ".rw2"]
ALL_IMAGE_EXTENSIONS = IMAGE_EXTENSIONS + RAW_EXTENSIONS

# UI Configuration
PAGE_SIZE_JSONL = 50  # Items per page in JSONL viewer
MAX_IMAGES_GRID = 100  # Max images to show in grid view
CACHE_TTL_SECONDS = 300  # 5 minutes default cache TTL
CACHE_TTL_HEALTH_CHECK = 10  # 10 seconds for backend health checks


def ensure_directories():
    """Ensure all required directories exist."""
    OUTPUTS_DIR.mkdir(exist_ok=True)
    (OUTPUTS_DIR / "results").mkdir(exist_ok=True)
    (OUTPUTS_DIR / "summaries").mkdir(exist_ok=True)
    (OUTPUTS_DIR / "cache").mkdir(exist_ok=True)
    (OUTPUTS_DIR / "metrics").mkdir(exist_ok=True)
    DEFAULT_OVERLAYS_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)


def get_app_setting(session_state, app_key: str, setting_name: str, default_value):
    """
    Get per-app setting with fallback to default.

    Args:
        session_state: Streamlit session state
        app_key: App identifier (e.g., 'mono', 'tagger', 'similarity')
        setting_name: Setting name (e.g., 'input_dir', 'output_jsonl')
        default_value: Default value if not overridden

    Returns:
        Setting value (override or default)
    """
    # Initialize app settings dict if not exists
    if "app_settings" not in session_state:
        session_state.app_settings = {}

    if app_key not in session_state.app_settings:
        session_state.app_settings[app_key] = {}

    # Return override if exists, otherwise default
    return session_state.app_settings[app_key].get(setting_name, default_value)


def set_app_setting(session_state, app_key: str, setting_name: str, value):
    """
    Set per-app setting override.

    Args:
        session_state: Streamlit session state
        app_key: App identifier
        setting_name: Setting name
        value: Value to set
    """
    if "app_settings" not in session_state:
        session_state.app_settings = {}

    if app_key not in session_state.app_settings:
        session_state.app_settings[app_key] = {}

    session_state.app_settings[app_key][setting_name] = value


def reset_app_settings(session_state, app_key: str):
    """
    Reset per-app settings to defaults.

    Args:
        session_state: Streamlit session state
        app_key: App identifier
    """
    if "app_settings" not in session_state:
        session_state.app_settings = {}

    session_state.app_settings[app_key] = {}
