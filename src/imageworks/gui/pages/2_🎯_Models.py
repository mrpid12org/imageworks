"""Models management hub page."""

import streamlit as st
from pathlib import Path

from imageworks.gui.state import init_session_state
from imageworks.gui.components.registry_table import (
    render_registry_table,
    render_registry_stats,
)
from imageworks.gui.components.backend_monitor import (
    render_backend_monitor,
    render_system_resources,
)
from imageworks.gui.components.process_runner import execute_command
from imageworks.gui.utils.cli_wrapper import build_downloader_command
from imageworks.gui.config import MODEL_REGISTRY_PATH, DEFAULT_BACKENDS


def render_download_tab():
    """Render model download tab."""
    st.markdown("### Download Models from HuggingFace")

    col1, col2 = st.columns([2, 1])

    with col1:
        model_input = st.text_input(
            "Model URL or owner/repo",
            value="",
            key="download_model_input",
            help="Enter HuggingFace URL or owner/repo (e.g., Qwen/Qwen2.5-VL-7B-AWQ)",
        )

    with col2:
        # Format preferences
        formats = st.multiselect(
            "Preferred Formats",
            options=["GGUF", "AWQ", "GPTQ", "safetensors"],
            default=["AWQ", "GGUF"],
            key="download_formats",
        )

    # Location selection
    location = st.selectbox(
        "Download Location",
        options=["linux_wsl", "windows_lmstudio", "custom"],
        key="download_location",
        help="Predefined locations or custom path",
    )

    custom_path = ""
    if location == "custom":
        custom_path = st.text_input("Custom Path", value="", key="download_custom_path")

    # Additional options
    col1, col2 = st.columns(2)

    with col1:
        resume = st.checkbox(
            "Resume interrupted downloads", value=True, key="download_resume"
        )

    with col2:
        update_registry = st.checkbox(
            "Update registry", value=True, key="download_update_registry"
        )

    # Download button
    if st.button("📥 Download Model", type="primary", disabled=not model_input):
        config = {
            "model": model_input,
            "format": formats,
            "location": location if location != "custom" else None,
            "output_dir": custom_path if custom_path else None,
            "resume": resume,
            "update_registry": update_registry,
        }

        with st.spinner(f"Downloading {model_input}..."):
            command = build_downloader_command(config)
            result = execute_command(command, timeout=3600)  # 1 hour timeout

            if result["success"]:
                st.success(f"✅ Successfully downloaded {model_input}")
                st.text(result["stdout"])
            else:
                st.error("❌ Download failed")
                st.text(result["stderr"])

    # Download history
    st.markdown("---")
    st.markdown("### Recent Downloads")

    # TODO: Implement download history tracking
    st.info("Download history coming soon")


def render_backends_tab():
    """Render backends monitoring tab."""
    st.markdown("### Backend Status")

    # Render backend monitor
    render_backend_monitor(backends=DEFAULT_BACKENDS, key_prefix="models_backends")

    # System resources
    st.markdown("---")
    render_system_resources()

    # Backend management (placeholder)
    st.markdown("---")
    st.markdown("### Backend Management")
    st.info("ℹ️ Start/stop controls coming in Phase 6")

    # Show commands for manual starting
    with st.expander("📝 Manual Start Commands", expanded=False):
        st.code(
            """
# Start vLLM
uv run vllm serve <model_name> --port 24001

# Start LMDeploy
uv run lmdeploy serve api <model_name> --port 24001

# Start Ollama
ollama serve

# Start Chat Proxy
uv run imageworks-chat-proxy
        """,
            language="bash",
        )


def render_profiles_tab():
    """Render deployment profiles tab."""
    st.markdown("### Deployment Profiles")

    st.info("ℹ️ Deployment profile visualization coming in Phase 6")

    # Show current pyproject.toml role assignments
    st.markdown("### Current Role Assignments")

    from imageworks.gui.config import PROJECT_ROOT

    pyproject_path = PROJECT_ROOT / "pyproject.toml"

    if pyproject_path.exists():
        try:
            try:
                import tomllib
            except ImportError:
                import tomli as tomllib

            with open(pyproject_path, "rb") as f:
                config = tomllib.load(f)

            # Look for imageworks configuration
            imageworks_config = config.get("tool", {}).get("imageworks", {})

            if imageworks_config:
                st.json(imageworks_config)
            else:
                st.warning("No ImageWorks configuration found in pyproject.toml")

        except Exception as e:
            st.error(f"Failed to load pyproject.toml: {e}")
    else:
        st.error("pyproject.toml not found")

    # Role assignment editor (placeholder)
    st.markdown("---")
    st.markdown("### Edit Role Assignments")
    st.info("ℹ️ Role assignment editor coming in Phase 6")


def main():
    """Models management hub page."""
    init_session_state()

    st.title("🎯 Models Management")
    st.markdown("Manage models, backends, and deployment profiles")

    # Tabs for different functions
    tabs = st.tabs(["📚 Registry", "📥 Download", "🔌 Backends", "⚙️ Profiles"])

    # === REGISTRY TAB ===
    with tabs[0]:
        st.markdown("### Model Registry")

        # Registry stats
        render_registry_stats()

        st.markdown("---")

        # Registry table with filters
        selected_model = render_registry_table(
            key_prefix="models_registry", filterable_columns=["format", "quantization"]
        )

        if selected_model:
            st.markdown("---")
            st.markdown("### Selected Model Details")

            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Name:** {selected_model.get('name', 'Unknown')}")
                st.write(f"**Format:** {selected_model.get('format', 'Unknown')}")
                st.write(
                    f"**Quantization:** {selected_model.get('quantization', 'None')}"
                )

            with col2:
                st.write(f"**Path:** {selected_model.get('path', 'Unknown')}")

                # Show size if available
                model_path = Path(selected_model.get("path", ""))
                if model_path.exists():
                    if model_path.is_file():
                        size_mb = model_path.stat().st_size / (1024 * 1024)
                        st.write(f"**Size:** {size_mb:.1f} MB")
                    elif model_path.is_dir():
                        total_size = sum(
                            f.stat().st_size
                            for f in model_path.rglob("*")
                            if f.is_file()
                        )
                        size_mb = total_size / (1024 * 1024)
                        st.write(f"**Size:** {size_mb:.1f} MB")

            # Full metadata
            with st.expander("Full Metadata", expanded=False):
                st.json(selected_model)

        # Registry management
        st.markdown("---")
        st.markdown("### Registry Management")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🔄 Refresh Registry"):
                from imageworks.gui.components.registry_table import load_registry

                load_registry.clear()
                st.rerun()

        with col2:
            if st.button("📝 Edit Registry"):
                st.info("Open registry file in editor")
                st.code(str(MODEL_REGISTRY_PATH))

        with col3:
            if st.button("🔗 Sync from Downloader"):
                st.info("Sync command: uv run imageworks-download sync")

    # === DOWNLOAD TAB ===
    with tabs[1]:
        render_download_tab()

    # === BACKENDS TAB ===
    with tabs[2]:
        render_backends_tab()

    # === PROFILES TAB ===
    with tabs[3]:
        render_profiles_tab()


if __name__ == "__main__":
    main()
