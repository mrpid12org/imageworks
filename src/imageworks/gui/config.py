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

# Default output paths
DEFAULT_OUTPUT_JSONL = OUTPUTS_DIR / "results" / "latest_results.jsonl"
DEFAULT_SUMMARY_PATH = OUTPUTS_DIR / "summaries" / "latest_summary.md"
DEFAULT_OVERLAYS_DIR = OUTPUTS_DIR / "tmp_test_images"

# Backend URLs
DEFAULT_BACKENDS = {
    "chat_proxy": "http://localhost:8100/v1",
    "vllm": "http://localhost:24001/v1",
    "lmdeploy": "http://localhost:24001/v1",
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
