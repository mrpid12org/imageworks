"""Settings and configuration page."""

import streamlit as st
import json

from imageworks.gui.state import init_session_state
from imageworks.gui.config import (
    PROJECT_ROOT,
    MODEL_REGISTRY_PATH,
    DEFAULT_BACKENDS,
    OUTPUTS_DIR,
    LOGS_DIR,
)
from imageworks.gui.utils.config_manager import (
    load_pyproject_config,
    save_pyproject_config,
    get_tool_config,
    update_tool_config,
)


def main():
    """Settings page."""
    st.set_page_config(layout="wide")
    init_session_state()

    # Apply wide layout CSS (ensures consistency on page refresh)
    st.markdown(
        """
        <style>
        .main .block-container {
            max-width: 95% !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }
        section.main > div {
            max-width: none !important;
        }
        .stDataFrame {
            width: 100% !important;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    st.title("⚙️ Settings")
    st.markdown("Configure ImageWorks GUI preferences and defaults")

    # Tabs for different settings
    tabs = st.tabs(
        [
            "🔧 General",
            "📁 Paths",
            "� Default Directories",
            "�🔌 Backends",
            "🎨 Appearance",
            "ℹ️ About",
        ]
    )

    # === GENERAL SETTINGS ===
    with tabs[0]:
        st.markdown("### General Settings")

        # Auto-save settings
        auto_save = st.checkbox(
            "Auto-save job history",
            value=True,
            help="Automatically save job history after each run",
        )

        # Cache settings
        st.markdown("#### Cache Settings")

        col1, col2 = st.columns(2)

        with col1:
            cache_ttl = st.number_input(
                "Cache TTL (seconds)",
                min_value=60,
                max_value=3600,
                value=300,
                step=60,
                help="Time-to-live for cached data",
            )

        with col2:
            st.number_input(
                "Max Cache Size (MB)",
                min_value=100,
                max_value=5000,
                value=1000,
                step=100,
                help="Maximum size for cache storage",
            )

        # Clear cache button
        if st.button("🗑️ Clear All Caches"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("✅ All caches cleared")

        # Default module settings
        st.markdown("---")
        st.markdown("#### Default Module Settings")

        default_dry_run = st.checkbox(
            "Enable Dry Run by default",
            value=True,
            help="Start all modules in dry-run mode for safety",
        )

        default_backup = st.checkbox(
            "Enable Backup Originals by default",
            value=True,
            help="Create backups before modifying files",
        )

    # === PATHS SETTINGS ===
    with tabs[1]:
        st.markdown("### Path Configuration")

        # Show current paths
        st.markdown("#### Current Paths")

        paths = {
            "Project Root": PROJECT_ROOT,
            "Outputs Directory": OUTPUTS_DIR,
            "Logs Directory": LOGS_DIR,
            "Model Registry": MODEL_REGISTRY_PATH,
        }

        for name, path in paths.items():
            col1, col2 = st.columns([3, 1])

            with col1:
                st.text_input(
                    name,
                    value=str(path),
                    disabled=True,
                    key=f"path_{name.lower().replace(' ', '_')}",
                )

            with col2:
                if st.button("📁 Open", key=f"open_{name.lower().replace(' ', '_')}"):
                    if path.exists():
                        st.code(f"# Path:\n{path}")
                    else:
                        st.warning(f"Path does not exist: {path}")

        # Custom paths
        st.markdown("---")
        st.markdown("#### Custom Paths")

        custom_download_path = st.text_input(
            "Custom Model Download Path",
            value="",
            help="Override default model download location",
        )

        custom_output_path = st.text_input(
            "Custom Output Path",
            value=str(OUTPUTS_DIR),
            help="Override default output directory",
        )

    # === DEFAULT DIRECTORIES ===
    with tabs[2]:
        st.markdown("### Default Directories Management")
        st.markdown(
            "Edit global default paths used by all modules. Changes are saved to `pyproject.toml`."
        )

        # Load current config
        try:
            pyproject_config = load_pyproject_config(PROJECT_ROOT)
        except Exception as e:
            st.error(f"Failed to load pyproject.toml: {e}")
            pyproject_config = {}

        # Track changes
        if "config_changes" not in st.session_state:
            st.session_state.config_changes = {}

        # Tabs for each module
        module_tabs = st.tabs(
            [
                "🖼️ Mono Checker",
                "🏷️ Personal Tagger",
                "🔍 Image Similarity",
                "🎨 Color Narrator",
            ]
        )

        # Mono Checker defaults
        with module_tabs[0]:
            st.markdown("#### Mono Checker Defaults")
            mono_config = get_tool_config(pyproject_config, "imageworks.mono")

            default_folder = st.text_input(
                "Default Input Folder",
                value=mono_config.get("default_folder", ""),
                key="mono_default_folder",
                help="Default directory for mono checker input",
            )

            default_jsonl = st.text_input(
                "Default Output JSONL",
                value=mono_config.get(
                    "default_jsonl", "outputs/results/mono_results.jsonl"
                ),
                key="mono_default_jsonl",
                help="Default path for mono results JSONL",
            )

            default_summary = st.text_input(
                "Default Summary Path",
                value=mono_config.get(
                    "default_summary", "outputs/summaries/mono_summary.md"
                ),
                key="mono_default_summary",
                help="Default path for mono summary markdown",
            )

            if st.button("💾 Save Mono Defaults", key="save_mono"):
                try:
                    updates = {
                        "default_folder": default_folder,
                        "default_jsonl": default_jsonl,
                        "default_summary": default_summary,
                    }
                    pyproject_config = update_tool_config(
                        pyproject_config, "imageworks.mono", updates
                    )
                    save_pyproject_config(PROJECT_ROOT, pyproject_config)
                    st.success("✅ Mono defaults saved to pyproject.toml")
                    st.cache_data.clear()  # Clear cache to reload config
                except Exception as e:
                    st.error(f"Failed to save: {e}")

        # Personal Tagger defaults
        with module_tabs[1]:
            st.markdown("#### Personal Tagger Defaults")
            tagger_config = get_tool_config(
                pyproject_config, "imageworks.personal_tagger"
            )

            tagger_output_jsonl = st.text_input(
                "Default Output JSONL",
                value=tagger_config.get(
                    "default_output_jsonl", "outputs/results/personal_tagger.jsonl"
                ),
                key="tagger_default_output_jsonl",
            )

            tagger_summary = st.text_input(
                "Default Summary Path",
                value=tagger_config.get(
                    "default_summary_path",
                    "outputs/summaries/personal_tagger_summary.md",
                ),
                key="tagger_default_summary",
            )

            if st.button("💾 Save Tagger Defaults", key="save_tagger"):
                try:
                    updates = {
                        "default_output_jsonl": tagger_output_jsonl,
                        "default_summary_path": tagger_summary,
                    }
                    pyproject_config = update_tool_config(
                        pyproject_config, "imageworks.personal_tagger", updates
                    )
                    save_pyproject_config(PROJECT_ROOT, pyproject_config)
                    st.success("✅ Tagger defaults saved to pyproject.toml")
                    st.cache_data.clear()
                except Exception as e:
                    st.error(f"Failed to save: {e}")

        # Image Similarity defaults
        with module_tabs[2]:
            st.markdown("#### Image Similarity Defaults")
            similarity_config = get_tool_config(
                pyproject_config, "imageworks.image_similarity_checker"
            )

            similarity_library = st.text_input(
                "Default Library Root",
                value=similarity_config.get("default_library_root", ""),
                key="similarity_default_library",
            )

            similarity_output_jsonl = st.text_input(
                "Default Output JSONL",
                value=similarity_config.get(
                    "default_output_jsonl", "outputs/results/similarity_results.jsonl"
                ),
                key="similarity_default_output_jsonl",
            )

            similarity_summary = st.text_input(
                "Default Summary Path",
                value=similarity_config.get(
                    "default_summary_path", "outputs/summaries/similarity_summary.md"
                ),
                key="similarity_default_summary",
            )

            similarity_cache = st.text_input(
                "Default Cache Directory",
                value=similarity_config.get(
                    "default_cache_dir", "outputs/cache/similarity"
                ),
                key="similarity_default_cache",
            )

            if st.button("💾 Save Similarity Defaults", key="save_similarity"):
                try:
                    updates = {
                        "default_library_root": similarity_library,
                        "default_output_jsonl": similarity_output_jsonl,
                        "default_summary_path": similarity_summary,
                        "default_cache_dir": similarity_cache,
                    }
                    pyproject_config = update_tool_config(
                        pyproject_config, "imageworks.image_similarity_checker", updates
                    )
                    save_pyproject_config(PROJECT_ROOT, pyproject_config)
                    st.success("✅ Similarity defaults saved to pyproject.toml")
                    st.cache_data.clear()
                except Exception as e:
                    st.error(f"Failed to save: {e}")

        # Color Narrator defaults
        with module_tabs[3]:
            st.markdown("#### Color Narrator Defaults")
            narrator_config = get_tool_config(
                pyproject_config, "imageworks.color_narrator"
            )

            narrator_images_dir = st.text_input(
                "Default Images Directory",
                value=narrator_config.get("default_images_dir", ""),
                key="narrator_default_images",
            )

            narrator_overlays_dir = st.text_input(
                "Default Overlays Directory",
                value=narrator_config.get("default_overlays_dir", ""),
                key="narrator_default_overlays",
            )

            if st.button("💾 Save Narrator Defaults", key="save_narrator"):
                try:
                    updates = {
                        "default_images_dir": narrator_images_dir,
                        "default_overlays_dir": narrator_overlays_dir,
                    }
                    pyproject_config = update_tool_config(
                        pyproject_config, "imageworks.color_narrator", updates
                    )
                    save_pyproject_config(PROJECT_ROOT, pyproject_config)
                    st.success("✅ Narrator defaults saved to pyproject.toml")
                    st.cache_data.clear()
                except Exception as e:
                    st.error(f"Failed to save: {e}")

        st.markdown("---")
        st.info("💡 After saving, restart the GUI or reload pages to see changes.")

    # === BACKENDS SETTINGS ===
    with tabs[3]:
        st.markdown("### Backend Configuration")

        st.markdown("#### Default Backend URLs")

        # Show current backends
        for name, url in DEFAULT_BACKENDS.items():
            new_url = st.text_input(
                name.replace("_", " ").title(), value=url, key=f"backend_{name}"
            )

            # Test connection button
            if st.button(f"🔍 Test {name}", key=f"test_{name}"):
                from imageworks.gui.components.backend_monitor import (
                    check_backend_health,
                )

                with st.spinner(f"Testing {name}..."):
                    health = check_backend_health(new_url)

                    if health["status"] == "healthy":
                        st.success(f"✅ {name} is healthy")
                    else:
                        st.error(
                            f"❌ {name} is {health['status']}: {health.get('error', 'Unknown')}"
                        )

        # Backend preferences
        st.markdown("---")
        st.markdown("#### Backend Preferences")

        st.selectbox(
            "Preferred VLM Backend",
            options=["vllm", "lmdeploy", "ollama"],
            index=0,
            help="Default backend for VLM operations",
        )

        st.selectbox(
            "Preferred Embedding Backend",
            options=["siglip", "open_clip", "simple", "remote"],
            index=0,
            help="Default backend for embedding computations",
        )

    # === APPEARANCE SETTINGS ===
    with tabs[4]:
        st.markdown("### Appearance")

        st.info("ℹ️ Appearance settings are controlled by Streamlit themes")

        # Show current theme
        st.markdown("#### Current Theme")
        st.write("To change theme, use Streamlit's built-in theme selector:")
        st.code("Settings (☰) → Theme → Choose theme")

        # Display preferences
        st.markdown("#### Display Preferences")

        items_per_page = st.slider(
            "Items per page (JSONL viewer)",
            min_value=10,
            max_value=100,
            value=50,
            step=10,
            help="Number of items to show per page",
        )

        max_images_grid = st.slider(
            "Max images in grid",
            min_value=20,
            max_value=200,
            value=100,
            step=20,
            help="Maximum images to display in grid view",
        )

        show_debug = st.checkbox(
            "Show debug information",
            value=st.session_state.get("debug_mode", False),
            help="Display debug panels throughout the GUI",
        )
        st.session_state["debug_mode"] = show_debug

    # === ABOUT ===
    with tabs[5]:
        st.markdown("### About ImageWorks GUI")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### Version Information")
            st.write("**Version:** 1.0.0")
            st.write("**Framework:** Streamlit")
            st.write("**Python:** 3.9+")

        with col2:
            st.markdown("#### Components")
            st.write("- Mono Checker")
            st.write("- Image Similarity")
            st.write("- Personal Tagger")
            st.write("- Color Narrator")
            st.write("- Model Manager")

        st.markdown("---")
        st.markdown("#### Documentation")

        doc_links = {
            "User Guide": "docs/index.md",
            "Mono Checker Docs": "docs/domains/mono/mono-overview.md",
            "Image Similarity Guide": "docs/guides/image-similarity-checker.md",
            "Personal Tagger Docs": "docs/domains/personal-tagger/overview.md",
            "Color Narrator Docs": "docs/domains/color-narrator/reference.md",
        }

        for name, path in doc_links.items():
            doc_path = PROJECT_ROOT / path
            if doc_path.exists():
                st.write(f"- [{name}]({doc_path})")
            else:
                st.write(f"- {name} (not found)")

        st.markdown("---")
        st.markdown("#### System Information")

        import platform
        import sys

        st.write(f"**OS:** {platform.system()} {platform.release()}")
        st.write(f"**Python:** {sys.version.split()[0]}")
        st.write(f"**Architecture:** {platform.machine()}")

        # GPU info
        try:
            from imageworks.libs.hardware.gpu_detector import GPUDetector

            gpus = GPUDetector.detect_gpus()
            if gpus:
                st.write(f"**GPU:** {gpus[0].name}")
                st.write(f"**VRAM:** {gpus[0].vram_total_mb} MB")
        except Exception:
            st.write("**GPU:** Not detected")

        st.markdown("---")

        # Export configuration
        st.markdown("#### Export Configuration")

        if st.button("📋 Export Settings"):
            settings = {
                "version": "1.0.0",
                "general": {
                    "auto_save": auto_save,
                    "cache_ttl": cache_ttl,
                    "default_dry_run": default_dry_run,
                    "default_backup": default_backup,
                },
                "paths": {
                    "custom_download_path": custom_download_path,
                    "custom_output_path": custom_output_path,
                },
                "backends": DEFAULT_BACKENDS,
                "appearance": {
                    "items_per_page": items_per_page,
                    "max_images_grid": max_images_grid,
                },
            }

            st.json(settings)
            st.download_button(
                "💾 Download Settings",
                data=json.dumps(settings, indent=2),
                file_name="imageworks_gui_settings.json",
                mime="application/json",
            )


if __name__ == "__main__":
    main()
