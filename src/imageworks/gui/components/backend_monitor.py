"""Backend monitoring component."""

import streamlit as st
import requests
from typing import Dict, Any, Optional
from datetime import datetime
import psutil
from imageworks.gui.config import DEFAULT_BACKENDS


@st.cache_data(ttl=10, show_spinner=False)
def check_backend_health(url: str, timeout: int = 2) -> Dict[str, Any]:
    """
    Check backend health (CACHED for 10 seconds).

    Args:
        url: Backend URL
        timeout: Request timeout

    Returns:
        Dict with health status
    """
    try:
        # Try to get models endpoint (common for OpenAI-compatible APIs)
        models_url = f"{url}/models" if not url.endswith("/v1") else f"{url}/models"

        response = requests.get(models_url, timeout=timeout)

        if response.status_code == 200:
            data = response.json()
            models = []

            if "data" in data:
                models = [m.get("id", "unknown") for m in data["data"]]
            elif "models" in data:
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
            # Check health
            health = check_backend_health(url)

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
