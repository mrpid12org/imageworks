"""Session state management utilities."""

import streamlit as st
from typing import Any, Dict, Optional


def init_session_state():
    """Initialize session state variables if they don't exist."""

    # Configuration states
    if "config" not in st.session_state:
        st.session_state["config"] = {}

    # Results caches
    if "mono_results" not in st.session_state:
        st.session_state["mono_results"] = None
    if "similarity_results" not in st.session_state:
        st.session_state["similarity_results"] = None
    if "tagger_results" not in st.session_state:
        st.session_state["tagger_results"] = None
    if "narrator_results" not in st.session_state:
        st.session_state["narrator_results"] = None

    # Execution states
    if "last_command" not in st.session_state:
        st.session_state["last_command"] = None
    if "last_exit_code" not in st.session_state:
        st.session_state["last_exit_code"] = None
    if "last_stdout" not in st.session_state:
        st.session_state["last_stdout"] = ""
    if "last_stderr" not in st.session_state:
        st.session_state["last_stderr"] = ""

    # UI states
    if "show_logs" not in st.session_state:
        st.session_state["show_logs"] = False
    if "debug_mode" not in st.session_state:
        st.session_state["debug_mode"] = False

    # Job history
    if "job_history" not in st.session_state:
        st.session_state["job_history"] = []


def store_result(module: str, result: Any):
    """Store module result in session state."""
    key = f"{module}_results"
    st.session_state[key] = result


def get_result(module: str) -> Optional[Any]:
    """Retrieve module result from session state."""
    key = f"{module}_results"
    return st.session_state.get(key)


def clear_results(module: Optional[str] = None):
    """Clear results for specific module or all modules."""
    if module:
        key = f"{module}_results"
        if key in st.session_state:
            st.session_state[key] = None
    else:
        # Clear all results
        for key in [
            "mono_results",
            "similarity_results",
            "tagger_results",
            "narrator_results",
        ]:
            st.session_state[key] = None


def get_config(module: str) -> Dict[str, Any]:
    """Get configuration for specific module."""
    if module not in st.session_state["config"]:
        st.session_state["config"][module] = {}
    return st.session_state["config"][module]


def update_config(module: str, config: Dict[str, Any]):
    """Update configuration for specific module."""
    st.session_state["config"][module] = config
