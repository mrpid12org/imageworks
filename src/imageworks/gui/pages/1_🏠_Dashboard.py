"""Dashboard page - System overview and recent activity."""

import streamlit as st
from datetime import datetime
import json

from imageworks.gui.state import init_session_state
from imageworks.gui.config import LOGS_DIR, OUTPUTS_DIR
from imageworks.gui.components.sidebar_footer import render_sidebar_footer


def main():
    """Dashboard page."""
    st.set_page_config(layout="wide")
    init_session_state()
    with st.sidebar:
        render_sidebar_footer()

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

    st.title("🏠 Dashboard")
    st.markdown("System overview and recent activity")

    # System overview
    st.header("System Status")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="📊 Total Jobs",
            value=len(st.session_state.get("job_history", [])),
            delta=None,
        )

    with col2:
        # Count recent outputs
        results_dir = OUTPUTS_DIR / "results"
        if results_dir.exists():
            recent_files = list(results_dir.glob("*.jsonl"))
            st.metric(label="📁 Recent Results", value=len(recent_files), delta=None)
        else:
            st.metric(label="📁 Recent Results", value=0)

    with col3:
        # Log file size
        log_file = LOGS_DIR / "chat_proxy.jsonl"
        if log_file.exists():
            size_mb = log_file.stat().st_size / (1024 * 1024)
            st.metric(label="📝 Log Size", value=f"{size_mb:.1f} MB", delta=None)
        else:
            st.metric(label="📝 Log Size", value="0 MB")

    st.markdown("---")

    # Recent Jobs
    st.header("Recent Jobs")

    job_history = st.session_state.get("job_history", [])

    if not job_history:
        st.info("No jobs run yet. Navigate to a workflow page to get started!")
    else:
        # Show last 5 jobs
        for job in reversed(job_history[-5:]):
            with st.container():
                col1, col2, col3 = st.columns([2, 1, 1])

                with col1:
                    st.write(f"**{job.get('module', 'Unknown')}**")
                    st.caption(job.get("description", "No description"))

                with col2:
                    timestamp = job.get("timestamp", "Unknown")
                    if timestamp != "Unknown":
                        try:
                            dt = datetime.fromisoformat(timestamp)
                            st.caption(dt.strftime("%Y-%m-%d %H:%M"))
                        except (ValueError, TypeError):
                            st.caption(timestamp)
                    else:
                        st.caption(timestamp)

                with col3:
                    status = job.get("status", "unknown")
                    if status == "success":
                        st.success("✅ Success")
                    elif status == "failed":
                        st.error("❌ Failed")
                    else:
                        st.info("⏳ Running")

                st.markdown("---")

    # Quick Stats
    st.header("Quick Stats")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Storage")
        if OUTPUTS_DIR.exists():
            total_size = sum(
                f.stat().st_size for f in OUTPUTS_DIR.rglob("*") if f.is_file()
            )
            st.write(f"Outputs directory: **{total_size / (1024*1024):.1f} MB**")
        else:
            st.write("Outputs directory: **0 MB**")

    with col2:
        st.subheader("Models")
        from imageworks.gui.config import MODEL_REGISTRY_PATH

        if MODEL_REGISTRY_PATH.exists():
            try:
                with open(MODEL_REGISTRY_PATH) as f:
                    registry = json.load(f)
                    model_count = len(registry.get("models", []))
                    st.write(f"Registered models: **{model_count}**")
            except Exception:
                st.write("Registered models: **Unknown**")
        else:
            st.write("Registered models: **0**")

    # Action buttons
    st.header("Quick Actions")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🖼️ Run Mono Checker", use_container_width=True):
            st.switch_page("pages/3_🖼️_Mono_Checker.py")

    with col2:
        if st.button("🔍 Check Similarity", use_container_width=True):
            st.switch_page("pages/4_🖼️_Image_Similarity.py")

    with col3:
        if st.button("🏷️ Tag Images", use_container_width=True):
            st.switch_page("pages/5_🖼️_Personal_Tagger.py")


if __name__ == "__main__":
    main()
