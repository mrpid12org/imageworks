"""Global sidebar controls shared across pages."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import streamlit as st

_RESTART_CONFIRM_KEY = "sidebar_restart_pending"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _restart_command() -> list[str]:
    root = _project_root()
    # Run stop/start sequence via bash so it can continue after this process exits.
    return [
        "/bin/bash",
        "-lc",
        f"cd {root} && sleep 1 && ./scripts/stop_gui_bg.sh >/dev/null 2>&1 && ./scripts/start_gui_bg.sh >/dev/null 2>&1",
    ]


def _trigger_restart() -> None:
    try:
        subprocess.Popen(
            _restart_command(),
            start_new_session=True,
            env=os.environ.copy(),
        )
        st.success("Restart initiated… the GUI will reload shortly.")
    except FileNotFoundError:
        st.error(
            "Restart scripts not found. Please run ./scripts/start_gui_bg.sh manually."
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Unable to restart GUI: {exc}")
    finally:
        st.session_state.pop(_RESTART_CONFIRM_KEY, None)


def render_sidebar_footer() -> None:
    """Render restart controls at the bottom of the sidebar."""

    st.markdown("---")
    if st.session_state.get(_RESTART_CONFIRM_KEY):
        st.warning(
            "Restart the GUI service? This will stop the current session and start a new one."
        )
        col1, col2 = st.columns(2)
        with col1:
            if st.button(
                "✅ Confirm", use_container_width=True, key="confirm_restart_button"
            ):
                _trigger_restart()
        with col2:
            if st.button(
                "Cancel", use_container_width=True, key="cancel_restart_button"
            ):
                st.session_state.pop(_RESTART_CONFIRM_KEY, None)
    else:
        if st.button(
            "🔁 Restart GUI", use_container_width=True, key="restart_gui_button"
        ):
            st.session_state[_RESTART_CONFIRM_KEY] = True
