"""Backend monitoring component."""

import streamlit as st
import requests
from typing import Dict, Any, Optional, List
from datetime import datetime
import psutil
import subprocess
from imageworks.gui.config import DEFAULT_BACKENDS


@st.cache_data(ttl=10, show_spinner=False)
def check_backend_health(
    url: str, backend_name: str = "", timeout: int = 2
) -> Dict[str, Any]:
    """
    Check backend health (CACHED for 10 seconds).

    Args:
        url: Backend URL
        backend_name: Backend name for special handling
        timeout: Request timeout

    Returns:
        Dict with health status
    """
    try:
        # Ollama uses different API endpoint
        if "11434" in url or backend_name.lower() == "ollama":
            models_url = f"{url}/api/tags"
        else:
            # OpenAI-compatible APIs (chat_proxy, vllm, lmdeploy)
            models_url = f"{url}/models" if not url.endswith("/v1") else f"{url}/models"

        response = requests.get(models_url, timeout=timeout)

        if response.status_code == 200:
            data = response.json()
            models = []

            # Handle different response formats
            if "data" in data:
                # OpenAI-compatible format
                models = [m.get("id", "unknown") for m in data["data"]]
            elif "models" in data and isinstance(data["models"], list):
                # Ollama format
                models = [m.get("name", "unknown") for m in data["models"]]
            elif "models" in data:
                # Generic list format
                models = data["models"]

            return {
                "status": "healthy",
                "responsive": True,
                "models": models,
                "checked_at": datetime.now().isoformat(),
            }
        else:
            return {
                "status": "unhealthy",
                "responsive": True,
                "error": f"HTTP {response.status_code}",
                "checked_at": datetime.now().isoformat(),
            }

    except requests.Timeout:
        return {
            "status": "timeout",
            "responsive": False,
            "error": "Request timed out",
            "checked_at": datetime.now().isoformat(),
        }

    except requests.ConnectionError:
        return {
            "status": "offline",
            "responsive": False,
            "error": "Connection refused",
            "checked_at": datetime.now().isoformat(),
        }

    except Exception as e:
        return {
            "status": "error",
            "responsive": False,
            "error": str(e),
            "checked_at": datetime.now().isoformat(),
        }


def render_backend_card(
    name: str,
    url: str,
    key_prefix: str,
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

    with st.container():
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            st.markdown(f"**{name}**")
            st.caption(url)

        with col2:
            # Check health with backend name for special handling
            health = check_backend_health(url, backend_name=name)

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

    # Check all backends
    health_status = {}

    for name, url in backends.items():
        health = render_backend_card(name, url, key_prefix)
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
