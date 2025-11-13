"""Backend monitoring component."""

from __future__ import annotations

import subprocess
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import psutil
import requests
import streamlit as st

from imageworks.gui.config import DEFAULT_BACKENDS


_BACKEND_METADATA: Dict[str, Dict[str, str]] = {
    "chat_proxy": {
        "label": "Chat Proxy",
        "role": "Orchestrator",
        "icon": "🧠",
        "type": "chat_proxy",
        "order": "0",
    },
    "vllm_executor": {
        "label": "vLLM Container",
        "role": "LLM Backend",
        "icon": "⚙️",
        "type": "vllm_executor",
        "order": "1",
    },
    "tf_iqa": {
        "label": "TF IQA Service",
        "role": "Vision Backend",
        "icon": "🧪",
        "type": "tf_iqa",
        "order": "2",
    },
    "ollama": {
        "label": "Ollama",
        "role": "LLM Backend",
        "icon": "🤖",
        "type": "ollama",
        "order": "3",
    },
}


def _normalized_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def _pretty_label(name: str, metadata: Optional[Dict[str, str]]) -> str:
    if metadata and metadata.get("label"):
        return metadata["label"]
    cleaned = name.replace("_", " ").title()
    return cleaned


def _build_status_record(status: str, responsive: bool, **extra: Any) -> Dict[str, Any]:
    payload = {
        "status": status,
        "responsive": responsive,
        "checked_at": datetime.now().isoformat(),
    }
    payload.update(extra)
    return payload


def _check_simple_health(
    url: str, timeout: int, path: str = "health"
) -> Dict[str, Any]:
    base = url.rstrip("/")
    target = f"{base}/{path.lstrip('/')}" if path else base
    try:
        response = requests.get(target, timeout=timeout)
        if response.status_code == 200:
            try:
                data = response.json()
            except ValueError:
                data = {}
            return _build_status_record(
                "healthy",
                True,
                details=data,
            )
        return _build_status_record(
            "unhealthy",
            True,
            error=f"HTTP {response.status_code}",
        )
    except requests.Timeout:
        return _build_status_record("timeout", False, error="Request timed out")
    except requests.ConnectionError:
        return _build_status_record("offline", False, error="Connection refused")
    except Exception as exc:  # noqa: BLE001
        return _build_status_record("error", False, error=str(exc))


def _check_chat_proxy_health(url: str, timeout: int) -> Dict[str, Any]:
    base = url.rstrip("/")
    if base.endswith("/health"):
        health_path = ""
    elif base.endswith("/v1"):
        health_path = "health"
    else:
        base = f"{base}/v1"
        health_path = "health"
    result = _check_simple_health(base, timeout, health_path)
    if result["status"] == "healthy":
        # Add uptime in friendly format if available
        details = result.get("details") or {}
        uptime = details.get("uptime_seconds")
        if isinstance(uptime, (int, float)):
            hours = uptime / 3600.0
            result["uptime_hours"] = hours
    return result


def _check_ollama_health(url: str, timeout: int) -> Dict[str, Any]:
    base = url.rstrip("/")
    tags_url = f"{base}/api/tags"
    try:
        response = requests.get(tags_url, timeout=timeout)
        if response.status_code == 200:
            data = response.json()
            models = []
            if isinstance(data, dict):
                if isinstance(data.get("models"), list):
                    models = [m.get("name", "unknown") for m in data["models"]]
            return _build_status_record(
                "healthy",
                True,
                models=models,
            )
        return _build_status_record(
            "unhealthy",
            True,
            error=f"HTTP {response.status_code}",
        )
    except requests.Timeout:
        return _build_status_record("timeout", False, error="Request timed out")
    except requests.ConnectionError:
        return _build_status_record("offline", False, error="Connection refused")
    except Exception as exc:  # noqa: BLE001
        return _build_status_record("error", False, error=str(exc))


def _check_openai_backend(url: str, timeout: int) -> Dict[str, Any]:
    base = url.rstrip("/")
    models_url = f"{base}/models"
    try:
        response = requests.get(models_url, timeout=timeout)
        if response.status_code == 200:
            data = response.json()
            models: List[str] = []
            if isinstance(data, dict) and isinstance(data.get("data"), list):
                models = [m.get("id", "unknown") for m in data["data"]]
            return _build_status_record(
                "healthy",
                True,
                models=models,
            )
        return _build_status_record(
            "unhealthy",
            True,
            error=f"HTTP {response.status_code}",
        )
    except requests.Timeout:
        return _build_status_record("timeout", False, error="Request timed out")
    except requests.ConnectionError:
        return _build_status_record("offline", False, error="Connection refused")
    except Exception as exc:  # noqa: BLE001
        return _build_status_record("error", False, error=str(exc))


@st.cache_data(ttl=10, show_spinner=False)
def check_backend_health(
    url: str,
    backend_name: str = "",
    timeout: int = 2,
    backend_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Check backend health (CACHED for 10 seconds).

    Args:
        url: Backend URL
        backend_name: Backend name for special handling
        timeout: Request timeout
        backend_type: Optional explicit backend type hint

    Returns:
        Dict with health status
    """

    backend_key = _normalized_name(backend_type or backend_name)

    if backend_key == "chat_proxy":
        return _check_chat_proxy_health(url, timeout)
    if backend_key == "vllm_executor":
        return _check_simple_health(url, timeout, "health")
    if backend_key == "ollama":
        return _check_ollama_health(url, timeout)
    if backend_key == "tf_iqa":
        return _check_simple_health(url, timeout, "health")

    # Default to OpenAI-compatible probing
    return _check_openai_backend(url, timeout)


def render_backend_card(
    name: str,
    url: str,
    key_prefix: str,
    metadata: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Render single backend status card.

    Args:
        name: Backend name
        url: Backend URL
        key_prefix: Unique prefix for widgets

    Returns:
        Health status dict
    """

    display_name = _pretty_label(name, metadata)
    role = (metadata or {}).get("role")
    icon = (metadata or {}).get("icon", "")
    backend_type = (metadata or {}).get("type") or name

    with st.container():
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            title = f"{icon} {display_name}".strip()
            st.markdown(f"**{title}**")
            if role:
                st.caption(role)
            st.caption(url)

        with col2:
            # Check health with backend name for special handling
            health = check_backend_health(
                url, backend_name=name, backend_type=backend_type
            )

            status = health["status"]
            if status == "healthy":
                st.success("✅ Healthy")
            elif status == "unhealthy":
                st.warning("⚠️ Unhealthy")
            elif status == "timeout":
                st.error("⏱️ Timeout")
            elif status == "offline":
                st.error("🔴 Offline")
            else:
                st.error("❌ Error")

        with col3:
            if st.button("🔄 Check", key=f"{key_prefix}_check_{name}"):
                check_backend_health.clear()
                st.rerun()

        # Show models if available
        if health.get("models"):
            with st.expander("Models", expanded=False):
                for model in health["models"]:
                    st.caption(f"• {model}")

        # Show error if any
        if health.get("error"):
            st.caption(f"⚠️ {health['error']}")
        elif health.get("details"):
            details = health["details"]
            if isinstance(details, dict):
                detail_bits = []
                uptime_hours = health.get("uptime_hours")
                if isinstance(uptime_hours, (int, float)):
                    detail_bits.append(f"Uptime: {uptime_hours:,.2f} h")
                status_detail = details.get("status")
                if status_detail and status_detail != "ok":
                    detail_bits.append(f"Status: {status_detail}")
                if detail_bits:
                    st.caption(" • ".join(detail_bits))

        st.markdown("---")

    return health


def render_backend_monitor(
    backends: Optional[Dict[str, str]] = None,
    key_prefix: str = "backends",
) -> Dict[str, Dict[str, Any]]:
    """
    Render backend monitoring dashboard.

    Args:
        backends: Dict of backend_name -> url
        key_prefix: Unique prefix for widgets

    Returns:
        Dict of backend_name -> health status
    """

    if backends is None:
        backends = DEFAULT_BACKENDS

    st.subheader("🔌 Backend Status")

    # Auto-refresh toggle
    auto_refresh = st.checkbox(
        "Auto-refresh (10s)", value=False, key=f"{key_prefix}_auto_refresh"
    )

    if auto_refresh:
        import time

        time.sleep(10)
        st.rerun()

    # Check all backends (sorted by declared priority to keep orchestrators on top)
    health_status = {}

    sorted_backends = sorted(
        backends.items(),
        key=lambda item: _BACKEND_METADATA.get(_normalized_name(item[0]), {}).get(
            "order", "9"
        ),
    )

    for name, url in sorted_backends:
        meta = _BACKEND_METADATA.get(_normalized_name(name))
        health = render_backend_card(name, url, key_prefix, metadata=meta)
        health_status[name] = health

    # Summary
    st.markdown("### Summary")

    healthy_count = sum(1 for h in health_status.values() if h["status"] == "healthy")
    total_count = len(health_status)

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Healthy Backends", f"{healthy_count}/{total_count}")

    with col2:
        if healthy_count == total_count:
            st.success("✅ All backends operational")
        elif healthy_count > 0:
            st.warning(f"⚠️ {total_count - healthy_count} backend(s) down")
        else:
            st.error("❌ All backends offline")

    return health_status


def render_system_resources() -> None:
    """Render system resource usage."""

    st.subheader("💻 System Resources")

    col1, col2, col3 = st.columns(3)

    with col1:
        cpu_percent = psutil.cpu_percent(interval=1)
        st.metric("CPU Usage", f"{cpu_percent:.1f}%")

    with col2:
        memory = psutil.virtual_memory()
        st.metric("Memory Usage", f"{memory.percent:.1f}%")
        st.caption(f"{memory.used / (1024**3):.1f} / {memory.total / (1024**3):.1f} GB")

    with col3:
        disk = psutil.disk_usage("/")
        st.metric("Disk Usage", f"{disk.percent:.1f}%")
        st.caption(f"{disk.used / (1024**3):.1f} / {disk.total / (1024**3):.1f} GB")


@st.cache_data(ttl=5, show_spinner=False)
def get_gpu_info() -> List[Dict[str, Any]]:
    """
    Get GPU information from nvidia-smi (CACHED for 5 seconds).

    Returns:
        List of GPU info dicts with usage details
    """
    try:
        # Get GPU utilization and memory info
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )

        gpus = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue

            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 7:
                continue

            index, name, mem_total, mem_used, mem_free, util, temp = parts

            gpus.append(
                {
                    "index": int(index),
                    "name": name,
                    "memory_total_mb": int(float(mem_total)),
                    "memory_used_mb": int(float(mem_used)),
                    "memory_free_mb": int(float(mem_free)),
                    "memory_percent": (
                        (int(float(mem_used)) / int(float(mem_total)) * 100)
                        if float(mem_total) > 0
                        else 0
                    ),
                    "utilization_percent": int(float(util)) if util else 0,
                    "temperature_c": int(float(temp)) if temp else 0,
                }
            )

        return gpus

    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return []


@st.cache_data(ttl=5, show_spinner=False)
def get_gpu_processes() -> List[Dict[str, Any]]:
    """
    Get processes running on GPU (CACHED for 5 seconds).

    Returns:
        List of process info dicts
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )

        processes = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue

            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue

            pid, process_name, used_mem = parts

            # Try to get command line for more context
            try:
                cmd_result = subprocess.run(
                    ["ps", "-p", pid, "-o", "args="],
                    capture_output=True,
                    text=True,
                    timeout=1,
                )
                cmd_line = (
                    cmd_result.stdout.strip()
                    if cmd_result.returncode == 0
                    else process_name
                )
            except Exception:
                cmd_line = process_name

            processes.append(
                {
                    "pid": int(pid),
                    "name": process_name,
                    "memory_mb": int(float(used_mem)),
                    "command": cmd_line,
                }
            )

        return processes

    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return []


def render_gpu_monitor() -> None:
    """Render GPU monitoring dashboard."""

    st.subheader("🎮 GPU Status")

    gpus = get_gpu_info()

    if not gpus:
        st.warning("⚠️ No GPU detected or nvidia-smi not available")
        return

    # Show each GPU
    for gpu in gpus:
        with st.container():
            st.markdown(f"**GPU {gpu['index']}: {gpu['name']}**")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "VRAM Used",
                    f"{gpu['memory_used_mb']:,} MB",
                    delta=f"{gpu['memory_percent']:.1f}%",
                )
                st.caption(f"Free: {gpu['memory_free_mb']:,} MB")

            with col2:
                st.metric("Total VRAM", f"{gpu['memory_total_mb']:,} MB")

            with col3:
                util_color = (
                    "🟢"
                    if gpu["utilization_percent"] < 70
                    else "🟡" if gpu["utilization_percent"] < 90 else "🔴"
                )
                st.metric(
                    "GPU Utilization", f"{util_color} {gpu['utilization_percent']}%"
                )

            with col4:
                temp_color = (
                    "🟢"
                    if gpu["temperature_c"] < 70
                    else "🟡" if gpu["temperature_c"] < 85 else "🔴"
                )
                st.metric("Temperature", f"{temp_color} {gpu['temperature_c']}°C")

            # Progress bar for memory usage
            st.progress(
                gpu["memory_percent"] / 100,
                text=f"VRAM: {gpu['memory_percent']:.1f}% used",
            )

            st.markdown("---")

    # Show GPU processes
    processes = get_gpu_processes()

    if processes:
        st.markdown("**🔧 Processes Using GPU**")

        with st.expander(f"💻 {len(processes)} active process(es)", expanded=True):
            for proc in processes:
                col1, col2, col3 = st.columns([1, 3, 1])

                with col1:
                    st.caption(f"**PID {proc['pid']}**")

                with col2:
                    # Extract relevant parts of command for display
                    cmd = proc["command"]
                    if "vllm" in cmd.lower():
                        st.caption("🚀 vLLM Server")
                    elif "python" in cmd.lower():
                        # Try to show the script name
                        parts = cmd.split()
                        script = next(
                            (p for p in parts if p.endswith(".py")), proc["name"]
                        )
                        st.caption(f"🐍 {script}")
                    else:
                        st.caption(f"⚙️ {proc['name']}")

                    # Show full command in very small text
                    if len(cmd) > 80:
                        st.caption(f"`{cmd[:80]}...`")
                    else:
                        st.caption(f"`{cmd}`")

                with col3:
                    st.caption(f"**{proc['memory_mb']:,} MB**")
    else:
        st.info("ℹ️ No GPU processes detected")

    render_gpu_lease_panel()


@st.cache_data(ttl=5, show_spinner=False)
def get_gpu_lease_status() -> Dict[str, Any]:
    """Fetch GPU lease status from the chat proxy."""

    base = DEFAULT_BACKENDS.get("chat_proxy", "http://127.0.0.1:8100").rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    status_url = f"{base}/gpu/status"
    try:
        response = requests.get(status_url, timeout=3)
        if response.status_code == 200:
            return response.json()
        return {"error": f"HTTP {response.status_code}"}
    except requests.Timeout:
        return {"error": "timeout"}
    except requests.ConnectionError:
        return {"error": "connection-refused"}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def render_gpu_lease_panel() -> None:
    """Show current GPU lease owner and copyable token."""

    st.subheader("🔒 GPU Lease Status")
    data = get_gpu_lease_status()

    if not data or data.get("error"):
        st.warning(f"⚠️ Unable to retrieve lease status: {data.get('error', 'unknown')}")
        return

    if not data.get("leased"):
        st.success("GPU is free – no active lease.")
        st.caption("Run Judge Vision or other GPU-exclusive jobs to acquire a lease.")
        return

    lease = data.get("lease") or {}
    owner = lease.get("owner") or "unknown"
    reason = lease.get("reason") or "unspecified"
    token = lease.get("token") or "N/A"
    granted_at = lease.get("granted_at")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Owner", owner)
    with col2:
        st.metric("Reason", reason)

    if granted_at:
        age_seconds = max(0.0, time.time() - float(granted_at))
        minutes, seconds = divmod(int(age_seconds), 60)
        st.caption(f"Lease age: {minutes}m {seconds}s")

    st.info("Use this header when calling the chat proxy while the lease is active:")
    st.code(f"X-GPU-Lease-Token: {token}", language="http")
