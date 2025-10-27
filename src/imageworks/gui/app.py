"""ImageWorks GUI Control Center - Main Application."""

import sys
from pathlib import Path

import streamlit as st

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from imageworks.gui.components.help_docs import (  # noqa: E402
    render_help_dialog,
    render_help_sidebar,
)
from imageworks.gui.config import ensure_directories  # noqa: E402
from imageworks.gui.state import init_session_state  # noqa: E402


def main():
    """Main application entry point."""

    # Ensure directories exist
    ensure_directories()

    # Initialize session state
    init_session_state()

    # Page configuration
    st.set_page_config(
        page_title="ImageWorks Control Center",
        page_icon="🖼️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Sidebar branding and status
    with st.sidebar:
        st.title("🖼️ ImageWorks")
        st.markdown("**Control Center v1.0**")
        st.markdown("---")

        # System status
        st.markdown("### System Status")

        # GPU info
        try:
            from imageworks.libs.hardware.gpu_detector import GPUDetector

            gpu_info = GPUDetector.detect_gpus()
            if gpu_info:
                gpu = gpu_info[0]
                st.success(f"✅ {gpu.name}")
                st.caption(f"🎮 {gpu.vram_total_mb} MB VRAM")
                if gpu.vram_free_mb is not None:
                    st.caption(f"📊 {gpu.vram_free_mb} MB free")
            else:
                st.warning("⚠️ No GPU detected")
        except Exception as e:
            st.error("❌ GPU detection failed")
            if st.session_state.get("debug_mode"):
                st.caption(str(e))

        st.markdown("---")

        # Backend status (placeholder for now)
        st.markdown("### Backend Status")
        st.caption("🔌 Chat Proxy: :8100")
        st.caption("🔌 vLLM: :24001")
        st.caption("🔌 Ollama: :11434")

        st.markdown("---")

        # Quick actions
        st.markdown("### Quick Actions")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()
        with col2:
            debug = st.checkbox(
                "🐛 Debug", value=st.session_state.get("debug_mode", False)
            )
            st.session_state["debug_mode"] = debug

        st.markdown("---")

        # Help button
        render_help_sidebar()

    # Main content area
    st.title("Welcome to ImageWorks Control Center")

    # Show help dialog if activated
    render_help_dialog()

    st.markdown(
        """
    ### 🚀 Getting Started

    Use the sidebar navigation to access different modules:

    - **🏠 Dashboard**: System overview and recent jobs
    - **🎯 Models**: Manage models, backends, and deployment profiles
    - **🖼️ Workflows**: Run analysis tools
      - Mono Checker: Detect non-monochrome images
      - Image Similarity: Find duplicate/similar images
      - Personal Tagger: Generate captions and keywords
      - Color Narrator: Describe color contamination
    - **📊 Results**: Browse outputs and metrics
    - **⚙️ Settings**: Configure defaults and presets

    ### 📚 Quick Tips

    - **Presets**: Most workflows offer Quick/Standard/Thorough presets
    - **Advanced Options**: Click "Show Advanced Options" for expert controls
    - **Dry Run**: Always enabled by default - review before committing changes
    - **Job History**: Re-run previous configurations from the Results page

    ### 🔗 Documentation

    For detailed information, see the [ImageWorks Documentation](../docs/index.md).
    """
    )

    # Debug panel
    if st.session_state.get("debug_mode"):
        with st.expander("🐛 Debug Information", expanded=False):
            st.write("**Session State:**")
            st.json(
                {
                    k: str(v)[:100] if isinstance(v, (str, list, dict)) else str(v)
                    for k, v in st.session_state.items()
                }
            )


if __name__ == "__main__":
    main()
