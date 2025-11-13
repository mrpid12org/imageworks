"""Process runner component for executing CLI commands."""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import streamlit as st


def _mark_judge_run_inactive() -> None:
    if "judge_run_active" in st.session_state:
        st.session_state["judge_run_active"] = False


_ERROR_PATTERN = re.compile(
    r"\b(error|warning|traceback|exception|critical)\b",
    re.IGNORECASE,
)


def _extract_error_lines(stderr_text: str) -> List[str]:
    """Return only lines that look like warnings/errors for display."""
    lines: List[str] = []
    for line in stderr_text.splitlines():
        if _ERROR_PATTERN.search(line):
            lines.append(line)
    return lines


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
    on_execute: Optional[Callable[[], None]] = None,
    *,
    async_mode: bool = False,
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
        on_execute: Optional callback invoked before the command starts

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

    process_state_key = f"{key_prefix}_process_state"
    process_state = st.session_state.get(process_state_key)

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
        if on_execute:
            try:
                on_execute()
            except Exception as exc:  # pragma: no cover - defensive UI
                st.error(f"❌ Unable to start command: {exc}")
                return None
        if async_mode:
            stdout_fd, stdout_path = tempfile.mkstemp(
                prefix=f"{key_prefix}_stdout_", suffix=".log"
            )
            stderr_fd, stderr_path = tempfile.mkstemp(
                prefix=f"{key_prefix}_stderr_", suffix=".log"
            )
            os.close(stdout_fd)
            os.close(stderr_fd)
            stdout_handle = open(stdout_path, "w", encoding="utf-8")
            stderr_handle = open(stderr_path, "w", encoding="utf-8")
            try:
                process = subprocess.Popen(
                    command,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    text=True,
                )
            except Exception as exc:  # noqa: BLE001
                stdout_handle.close()
                stderr_handle.close()
                Path(stdout_path).unlink(missing_ok=True)
                Path(stderr_path).unlink(missing_ok=True)
                st.error(f"❌ Failed to start command: {exc}")
                _mark_judge_run_inactive()
                return None
            finally:
                stdout_handle.close()
                stderr_handle.close()

            st.session_state[process_state_key] = {
                "process": process,
                "stdout_path": stdout_path,
                "stderr_path": stderr_path,
                "command": " ".join(command),
                "started": datetime.now().isoformat(),
                "timeout": timeout,
                "config": config.copy(),
                "button_label": button_label,
            }
            st.session_state.pop(result_key, None)
            st.session_state["last_command"] = " ".join(command)
            st.session_state["last_exit_code"] = None
            st.rerun()
            return None
        else:
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
            _mark_judge_run_inactive()

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

    if async_mode and process_state:
        process = process_state["process"]
        retcode = process.poll()
        stdout_path = process_state["stdout_path"]
        stderr_path = process_state["stderr_path"]
        command_str = process_state["command"]

        def _read_output(path: str, limit: int = 4000) -> str:
            try:
                data = Path(path).read_text()
            except Exception:
                return ""
            return data[-limit:]

        col_run, col_stop = st.columns([3, 1])
        with col_run:
            st.info("⏳ Command is running in the background.")
        with col_stop:
            if st.button("🛑 Stop Run", key=f"{key_prefix}_stop"):
                process.terminate()
                st.warning("Termination signal sent.")

        if retcode is None:
            st.caption("Progress updates will appear below while the run continues.")
            return None

        stdout = Path(stdout_path).read_text(encoding="utf-8", errors="ignore")
        stderr = Path(stderr_path).read_text(encoding="utf-8", errors="ignore")
        Path(stdout_path).unlink(missing_ok=True)
        Path(stderr_path).unlink(missing_ok=True)
        st.session_state.pop(process_state_key, None)
        _mark_judge_run_inactive()

        result_payload = {
            "command": command_str,
            "exit_code": retcode,
            "stdout": stdout,
            "stderr": stderr,
            "success": retcode == 0,
            "timestamp": datetime.now().isoformat(),
        }
        st.session_state[result_key] = result_payload
        st.session_state["last_command"] = command_str
        st.session_state["last_exit_code"] = retcode

        job_entry = {
            "timestamp": result_payload["timestamp"],
            "module": key_prefix.split("_")[0] if "_" in key_prefix else key_prefix,
            "command": command_str,
            "config": process_state.get("config", {}).copy(),
            "status": "success" if retcode == 0 else "failed",
            "description": process_state.get("button_label", button_label),
        }
        if "job_history" not in st.session_state:
            st.session_state["job_history"] = []
        st.session_state["job_history"].append(job_entry)

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
            error_lines = _extract_error_lines(result["stderr"])
            has_errors = bool(error_lines)
            label = (
                f"⚠️ Errors ({len(error_lines)})"
                if has_errors
                else "✅ Errors · No warnings or errors detected"
            )
            with st.expander(
                label, expanded=has_errors and (result["success"] is False)
            ):
                if has_errors:
                    st.text("\n".join(error_lines))
                else:
                    st.caption("No warnings or errors detected in stderr output.")

            with st.expander("📄 Full stderr", expanded=False):
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
