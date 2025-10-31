"""Process runner component for executing CLI commands."""

import streamlit as st
import subprocess
from typing import List, Dict, Any, Optional
from datetime import datetime


def execute_command(
    command: List[str],
    capture_output: bool = True,
    timeout: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Execute command and return results.

    Args:
        command: Command as list of strings
        capture_output: Whether to capture stdout/stderr
        timeout: Timeout in seconds

    Returns:
        Dict with stdout, stderr, exit_code, command
    """
    try:
        result = subprocess.run(
            command,
            capture_output=capture_output,
            text=True,
            timeout=timeout,
        )

        return {
            "command": " ".join(command),
            "exit_code": result.returncode,
            "stdout": result.stdout if capture_output else "",
            "stderr": result.stderr if capture_output else "",
            "success": result.returncode == 0,
            "timestamp": datetime.now().isoformat(),
        }
    except subprocess.TimeoutExpired:
        return {
            "command": " ".join(command),
            "exit_code": -1,
            "stdout": "",
            "stderr": f"Command timed out after {timeout} seconds",
            "success": False,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        return {
            "command": " ".join(command),
            "exit_code": -1,
            "stdout": "",
            "stderr": str(e),
            "success": False,
            "timestamp": datetime.now().isoformat(),
        }


def render_process_runner(
    button_label: str,
    command_builder: callable,
    config: Dict[str, Any],
    key_prefix: str,
    result_key: str,
    show_command: bool = True,
    timeout: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """
    Render process runner with execute button.

    Args:
        button_label: Label for execute button
        command_builder: Function that takes config and returns command list
        config: Configuration dict to pass to command_builder
        key_prefix: Unique prefix for widgets
        result_key: Session state key to store results
        show_command: Whether to display the command being run
        timeout: Command timeout in seconds

    Returns:
        Execution result dict if command was run, None otherwise
    """

    # Build command
    try:
        command = command_builder(config)
    except Exception as e:
        st.error(f"❌ Failed to build command: {e}")
        return None

    # Show command preview
    if show_command:
        with st.expander("🔍 Command Preview", expanded=False):
            st.code(" ".join(command), language="bash")

    # Execute button
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        execute = st.button(
            button_label,
            key=f"{key_prefix}_execute",
            type="primary",
            use_container_width=True,
        )

    with col2:
        dry_run = st.checkbox(
            "Dry Run",
            value=config.get("dry_run", True),
            key=f"{key_prefix}_dry_run_toggle",
        )
        config["dry_run"] = dry_run

    with col3:
        if result_key in st.session_state and st.session_state[result_key]:
            if st.button("🗑️ Clear", key=f"{key_prefix}_clear"):
                st.session_state[result_key] = None
                st.rerun()

    # Execute command
    if execute:
        with st.status(
            "⏳ Running command...", state="running", expanded=True
        ) as status:
            result = execute_command(command, capture_output=True, timeout=timeout)
            if result["success"]:
                status.update(label="✅ Command completed", state="complete")
            else:
                status.update(
                    label=f"❌ Command failed (exit {result['exit_code']})",
                    state="error",
                )

        # Store in session state
        st.session_state[result_key] = result
        st.session_state["last_command"] = result["command"]
        st.session_state["last_exit_code"] = result["exit_code"]

        # Add to job history
        job_entry = {
            "timestamp": result["timestamp"],
            "module": key_prefix.split("_")[0] if "_" in key_prefix else key_prefix,
            "command": result["command"],
            "config": config.copy(),
            "status": "success" if result["success"] else "failed",
            "description": button_label,
        }

        if "job_history" not in st.session_state:
            st.session_state["job_history"] = []
        st.session_state["job_history"].append(job_entry)

        return result

    # Display previous result if exists
    if result_key in st.session_state and st.session_state[result_key]:
        result = st.session_state[result_key]

        if result["success"]:
            st.success("✅ Command executed successfully")
        else:
            st.error(f"❌ Command failed with exit code {result['exit_code']}")

        # Show output
        if result["stdout"]:
            with st.expander("📄 Output", expanded=True):
                st.text(result["stdout"])

        if result["stderr"]:
            with st.expander("⚠️ Errors", expanded=result["success"] is False):
                st.text(result["stderr"])

    return None


def render_streaming_runner(
    button_label: str,
    command: List[str],
    key_prefix: str,
) -> bool:
    """
    Render streaming command runner (for long-running processes).

    Args:
        button_label: Label for execute button
        command: Command to execute
        key_prefix: Unique prefix for widgets

    Returns:
        True if command is currently running
    """

    # Check if process is running
    is_running = st.session_state.get(f"{key_prefix}_running", False)

    if not is_running:
        if st.button(button_label, key=f"{key_prefix}_start", type="primary"):
            st.session_state[f"{key_prefix}_running"] = True
            st.session_state[f"{key_prefix}_output"] = []
            st.rerun()
    else:
        st.warning("⏳ Process is running...")

        if st.button("⏹️ Stop", key=f"{key_prefix}_stop"):
            st.session_state[f"{key_prefix}_running"] = False
            st.rerun()

        # Show accumulated output
        output = st.session_state.get(f"{key_prefix}_output", [])
        if output:
            st.text_area("Output", value="\n".join(output), height=400, disabled=True)

    return is_running
