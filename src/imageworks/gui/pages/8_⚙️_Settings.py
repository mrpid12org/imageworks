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


def main():
    """Settings page."""
    init_session_state()

    st.title("⚙️ Settings")
    st.markdown("Configure ImageWorks GUI preferences and defaults")

    # Tabs for different settings
    tabs = st.tabs(
        ["🔧 General", "📁 Paths", "🔌 Backends", "🎨 Appearance", "ℹ️ About"]
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

    # === BACKENDS SETTINGS ===
    with tabs[2]:
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
    with tabs[3]:
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
    with tabs[4]:
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
