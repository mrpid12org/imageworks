"""Job history component."""

import streamlit as st
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
from imageworks.gui.config import JOB_HISTORY_FILE


def save_job_history(jobs: List[Dict[str, Any]]) -> None:
    """
    Save job history to file.

    Args:
        jobs: List of job dicts
    """
    try:
        JOB_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

        with open(JOB_HISTORY_FILE, "w") as f:
            json.dump(jobs, f, indent=2)

    except Exception as e:
        st.error(f"Failed to save job history: {e}")


def load_job_history() -> List[Dict[str, Any]]:
    """
    Load job history from file.

    Returns:
        List of job dicts
    """
    if not JOB_HISTORY_FILE.exists():
        return []

    try:
        with open(JOB_HISTORY_FILE, "r") as f:
            return json.load(f)

    except Exception as e:
        st.error(f"Failed to load job history: {e}")
        return []


def add_job_to_history(
    module: str,
    description: str,
    config: Dict[str, Any],
    command: Optional[str] = None,
    status: str = "success",
) -> None:
    """
    Add job to history.

    Args:
        module: Module name (mono, similarity, tagger, narrator)
        description: Human-readable description
        config: Configuration dict
        command: Command that was run
        status: Job status (success, failed, running)
    """
    job = {
        "timestamp": datetime.now().isoformat(),
        "module": module,
        "description": description,
        "config": config,
        "command": command,
        "status": status,
    }

    # Add to session state
    if "job_history" not in st.session_state:
        st.session_state["job_history"] = []

    st.session_state["job_history"].append(job)

    # Save to file
    save_job_history(st.session_state["job_history"])


def render_job_history(
    key_prefix: str = "job_history",
    max_jobs: int = 50,
    module_filter: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Render job history with re-run capability.

    Args:
        key_prefix: Unique prefix for widgets
        max_jobs: Maximum jobs to display
        module_filter: Filter by module name

    Returns:
        Selected job dict or None
    """

    st.subheader("📜 Job History")

    # Load history
    jobs = st.session_state.get("job_history", [])

    if not jobs:
        # Try loading from file
        jobs = load_job_history()
        if jobs:
            st.session_state["job_history"] = jobs

    if not jobs:
        st.info("No jobs run yet")
        return None

    # Filter by module if specified
    if module_filter:
        jobs = [j for j in jobs if j.get("module") == module_filter]

    # Limit number of jobs
    if len(jobs) > max_jobs:
        jobs = jobs[-max_jobs:]

    st.write(f"Showing last {len(jobs)} jobs")

    # Module filter (if not already filtered)
    if not module_filter:
        modules = sorted(set(j.get("module", "unknown") for j in jobs))

        selected_module = st.multiselect(
            "Filter by module", options=modules, key=f"{key_prefix}_module_filter"
        )

        if selected_module:
            jobs = [j for j in jobs if j.get("module") in selected_module]

    # Status filter
    statuses = ["success", "failed", "running"]
    selected_status = st.multiselect(
        "Filter by status", options=statuses, key=f"{key_prefix}_status_filter"
    )

    if selected_status:
        jobs = [j for j in jobs if j.get("status") in selected_status]

    # Display jobs (most recent first)
    selected_job = None

    for i, job in enumerate(reversed(jobs)):
        with st.expander(
            f"{job.get('timestamp', 'Unknown time')} - {job.get('module', 'Unknown')}: {job.get('description', 'No description')}",
            expanded=False,
        ):
            col1, col2 = st.columns([3, 1])

            with col1:
                st.write(f"**Module:** {job.get('module', 'Unknown')}")
                st.write(f"**Status:** {job.get('status', 'Unknown')}")
                st.write(f"**Time:** {job.get('timestamp', 'Unknown')}")

                if job.get("command"):
                    st.code(job["command"], language="bash")

            with col2:
                if st.button("🔄 Re-run", key=f"{key_prefix}_rerun_{i}"):
                    selected_job = job

                if st.button("📋 Copy Config", key=f"{key_prefix}_copy_{i}"):
                    st.session_state[f"{job.get('module')}_config"] = job.get(
                        "config", {}
                    )
                    st.success("Configuration copied!")

            # Show configuration
            if job.get("config"):
                with st.expander("Configuration", expanded=False):
                    st.json(job["config"])

    # Clear history button
    if st.button("🗑️ Clear History", key=f"{key_prefix}_clear"):
        st.session_state["job_history"] = []
        save_job_history([])
        st.rerun()

    return selected_job


def render_job_comparison(
    job1: Dict[str, Any],
    job2: Dict[str, Any],
) -> None:
    """
    Render comparison between two jobs.

    Args:
        job1: First job dict
        job2: Second job dict
    """

    st.subheader("🔍 Job Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Job 1")
        st.write(f"**Module:** {job1.get('module')}")
        st.write(f"**Time:** {job1.get('timestamp')}")
        st.write(f"**Status:** {job1.get('status')}")

        if job1.get("config"):
            with st.expander("Configuration"):
                st.json(job1["config"])

    with col2:
        st.markdown("### Job 2")
        st.write(f"**Module:** {job2.get('module')}")
        st.write(f"**Time:** {job2.get('timestamp')}")
        st.write(f"**Status:** {job2.get('status')}")

        if job2.get("config"):
            with st.expander("Configuration"):
                st.json(job2["config"])

    # Config diff (simplified)
    if job1.get("config") and job2.get("config"):
        st.markdown("### Configuration Differences")

        config1 = job1["config"]
        config2 = job2["config"]

        all_keys = set(config1.keys()) | set(config2.keys())

        differences = []
        for key in sorted(all_keys):
            val1 = config1.get(key, "❌ Not set")
            val2 = config2.get(key, "❌ Not set")

            if val1 != val2:
                differences.append(
                    {
                        "Setting": key,
                        "Job 1": str(val1),
                        "Job 2": str(val2),
                    }
                )

        if differences:
            import pandas as pd

            df = pd.DataFrame(differences)
            st.dataframe(df, use_container_width=True)
        else:
            st.success("✅ Configurations are identical")
