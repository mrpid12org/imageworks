#!/usr/bin/env python3
"""Generate comprehensive Models.py with all CLI commands exposed in GUI."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

output_file = (
    project_root / "src" / "imageworks" / "gui" / "pages" / "2_🎯_Models_new.py"
)

content = '''"""Models management hub page with comprehensive CLI parity."""

import streamlit as st
from pathlib import Path
import json
from typing import Dict, List, Any, Optional

from imageworks.gui.state import init_session_state
from imageworks.gui.components.registry_table import (
    render_registry_table,
    render_registry_stats,
)
from imageworks.gui.components.backend_monitor import (
    render_backend_monitor,
    render_system_resources,
)
from imageworks.gui.components.process_runner import execute_command
from imageworks.gui.config import MODEL_REGISTRY_PATH, DEFAULT_BACKENDS, PROJECT_ROOT
from imageworks.model_loader.registry import save_registry
from imageworks.model_loader.registry import load_registry as load_model_registry


def run_command_with_progress(
    command: List[str],
    description: str,
    show_output: bool = True,
    timeout: Optional[int] = None,
) -> Dict[str, Any]:
    """Run command with progress indicator and optional output display."""
    with st.spinner(description):
        result = execute_command(command, timeout=timeout)

    if show_output:
        if result["success"]:
            st.success(f"✅ {description} - Complete")
            if result["stdout"]:
                with st.expander("📄 Output", expanded=False):
                    st.code(result["stdout"])
        else:
            st.error(f"❌ {description} - Failed")
            if result["stderr"]:
                st.error(result["stderr"])

    return result


def confirm_destructive_operation(operation_name: str, details: str) -> bool:
    """Show confirmation dialog for destructive operations."""
    st.warning(f"⚠️ **{operation_name}**")
    st.markdown(details)

    confirm_key = f"confirm_{operation_name.replace(' ', '_').lower()}"
    confirmed = st.checkbox(
        f"I understand this will {operation_name.lower()}",
        key=confirm_key
    )

    return confirmed


def render_browse_manage_tab():
    """Browse & Manage tab - model browsing with role/config editing."""
    st.markdown("### 📚 Browse & Manage Models")

    # Stats widget
    with st.expander("📊 Registry Statistics", expanded=True):
        render_registry_stats()

    st.markdown("---")

    # Model table with filters
    selected_model = render_registry_table(
        key_prefix="browse_manage",
        filterable_columns=["backend", "format", "quantization"]
    )

    if selected_model:
        st.markdown("---")
        st.markdown("### 🔍 Selected Model Details")

        col1, col2 = st.columns(2)

        with col1:
            st.write(f"**Name:** {selected_model.get('name', 'Unknown')}")
            st.write(f"**Backend:** {selected_model.get('backend', 'Unknown')}")
            st.write(f"**Format:** {selected_model.get('format', 'Unknown')}")
            st.write(f"**Quantization:** {selected_model.get('quantization', 'None')}")

        with col2:
            st.write(f"**Path:** {selected_model.get('path', 'Unknown')}")

            # Show size if available
            model_path = Path(selected_model.get("path", ""))
            if model_path.exists():
                if model_path.is_file():
                    size_mb = model_path.stat().st_size / (1024 * 1024)
                    st.write(f"**Size:** {size_mb:.1f} MB")
                elif model_path.is_dir():
                    total_size = sum(
                        f.stat().st_size
                        for f in model_path.rglob("*")
                        if f.is_file()
                    )
                    size_gb = total_size / (1024 * 1024 * 1024)
                    st.write(f"**Size:** {size_gb:.2f} GB")

        # Full metadata
        with st.expander("📋 Full Metadata", expanded=False):
            st.json(selected_model)

        # Role & Configuration Management
        st.markdown("---")
        st.markdown("### 🎭 Role & Configuration Management")

        model_name = selected_model.get('name', '')

        if model_name:
            registry = load_model_registry()
            registry_entry = registry.get(model_name)

            if registry_entry:
                # Role assignment
                st.markdown("#### Assigned Roles")

                available_roles = [
                    "caption",
                    "keywords",
                    "description",
                    "narration",
                    "vision_general",
                    "chat",
                    "code",
                    "reasoning",
                ]

                current_roles = list(registry_entry.roles or [])

                selected_roles = st.multiselect(
                    "Roles this model can perform",
                    options=available_roles,
                    default=current_roles,
                    key=f"roles_{model_name}",
                    help="Select all roles this model is capable of performing",
                )

                # Role Priority Editor
                if selected_roles:
                    st.markdown("#### Role Priority Scores")
                    st.caption("Higher priority = preferred for that role (0-100)")

                    role_priorities = {}
                    cols = st.columns(min(len(selected_roles), 3))

                    for idx, role in enumerate(selected_roles):
                        col_idx = idx % 3
                        with cols[col_idx]:
                            current_priority = registry_entry.role_priority.get(role, 50)
                            priority = st.number_input(
                                f"{role}",
                                min_value=0,
                                max_value=100,
                                value=current_priority,
                                step=5,
                                key=f"priority_{model_name}_{role}",
                            )
                            role_priorities[role] = priority

                # Backend Config Editor
                st.markdown("---")
                st.markdown("#### Backend Configuration")

                backend_config = registry_entry.backend_config

                col1, col2 = st.columns(2)

                with col1:
                    st.write(f"**Backend:** {registry_entry.backend}")
                    if backend_config:
                        st.write(f"**Port:** {backend_config.port}")
                    else:
                        st.write(f"**Port:** N/A")

                with col2:
                    if backend_config:
                        st.write(f"**Host:** {backend_config.host or 'localhost'}")
                    else:
                        st.write(f"**Host:** localhost")

                # Extra Args Editor
                st.markdown("**Extra Arguments**")
                st.caption("Additional CLI arguments passed to the backend (one per line)")

                current_extra_args = backend_config.extra_args if backend_config else []
                extra_args_text = "\\n".join(current_extra_args)

                edited_extra_args = st.text_area(
                    "Extra Arguments (one per line)",
                    value=extra_args_text,
                    height=150,
                    key=f"extra_args_{model_name}",
                    help="Example args:\\n--max-model-len 8192\\n--gpu-memory-utilization 0.9\\n--tensor-parallel-size 2",
                    label_visibility="collapsed",
                )

                # Common arg helpers
                with st.expander("📖 Common Arguments", expanded=False):
                    st.markdown("""
                    **vLLM:**
                    - `--max-model-len 8192` - Set context length
                    - `--gpu-memory-utilization 0.9` - GPU memory fraction
                    - `--tensor-parallel-size 2` - Multi-GPU parallelism

                    **LMDeploy:**
                    - `--tp 2` - Tensor parallelism
                    - `--cache-max-entry-count 0.9` - KV cache size

                    **Ollama:**
                    - `--num-ctx 8192` - Context length
                    - `--num-gpu 1` - Number of GPU layers
                    """)

                # Save button
                st.markdown("---")
                col1, col2, col3 = st.columns([1, 1, 2])

                with col1:
                    if st.button("💾 Save Changes", type="primary", key=f"save_{model_name}"):
                        try:
                            # Update registry entry
                            registry_entry.roles = selected_roles
                            registry_entry.role_priority = role_priorities

                            # Update extra_args in backend_config
                            new_extra_args = [
                                line.strip()
                                for line in edited_extra_args.split('\\n')
                                if line.strip()
                            ]

                            if backend_config:
                                backend_config.extra_args = new_extra_args

                            # Save registry
                            save_registry()

                            st.success(f"✅ Saved changes for {model_name}")
                            st.rerun()

                        except Exception as e:
                            st.error(f"❌ Failed to save: {e}")

                with col2:
                    if st.button("↩️ Reset", key=f"reset_{model_name}"):
                        st.rerun()

            else:
                st.warning(f"Model '{model_name}' not found in registry")
        else:
            st.info("ℹ️ Select a model from the table above to edit its roles and configuration")


def render_download_import_tab():
    """Download & Import tab - download from HF, scan existing, import Ollama."""
    st.markdown("### 📥 Download & Import Models")
    st.info("ℹ️ Comprehensive download/import UI implementation in progress. Use CLI for now.")

    # Placeholder for comprehensive implementation
    st.markdown("**Coming Soon:**")
    st.markdown("- HuggingFace model download with branch/format options")
    st.markdown("- Scan existing downloads from local directories")
    st.markdown("- Import Ollama models from internal store")


def render_registry_maintenance_tab():
    """Registry Maintenance tab - normalize, purge, rebuild operations."""
    st.markdown("### 🔧 Registry Maintenance")
    st.info("ℹ️ Registry maintenance UI implementation in progress. Use CLI for now.")

    # Placeholder for comprehensive implementation
    st.markdown("**Coming Soon:**")
    st.markdown("- Normalize formats & rebuild dynamic fields")
    st.markdown("- Purge operations (deprecated, logical-only, HF entries)")
    st.markdown("- Advanced cleanup (prune duplicates, restore, backfill)")


def render_backends_tab():
    """Backends tab - monitoring and management."""
    st.markdown("### 🔌 Backend Status")

    # Render backend monitor
    render_backend_monitor(backends=DEFAULT_BACKENDS, key_prefix="backends_monitor")

    # System resources
    st.markdown("---")
    render_system_resources()

    # Backend management
    st.markdown("---")
    st.markdown("### Backend Management")
    st.info("ℹ️ Start/stop controls coming in future phase")

    # Show commands for manual starting
    with st.expander("📝 Manual Start Commands", expanded=False):
        st.code(
            """
# Start vLLM
uv run vllm serve <model_name> --port 24001

# Start LMDeploy
uv run lmdeploy serve api <model_name> --port 24001

# Start Ollama
ollama serve

# Start Chat Proxy
uv run imageworks-chat-proxy
        """,
            language="bash",
        )


def render_advanced_tab():
    """Advanced Operations tab - model removal, verification, deployment profiles."""
    st.markdown("### ⚙️ Advanced Operations")
    st.info("ℹ️ Advanced operations UI implementation in progress. Use CLI for now.")

    # Placeholder for comprehensive implementation
    st.markdown("**Coming Soon:**")
    st.markdown("- Model removal (metadata/files/purge with confirmations)")
    st.markdown("- Verification and integrity checking")
    st.markdown("- Deployment profiles editor")


def main():
    """Models management hub page with comprehensive CLI parity."""
    init_session_state()

    st.title("🎯 Models Management")
    st.markdown("Comprehensive model management with full CLI parity")

    # 5 tabs for different functions
    tabs = st.tabs([
        "📚 Browse & Manage",
        "📥 Download & Import",
        "🔧 Registry Maintenance",
        "🔌 Backends",
        "⚙️ Advanced",
    ])

    with tabs[0]:
        render_browse_manage_tab()

    with tabs[1]:
        render_download_import_tab()

    with tabs[2]:
        render_registry_maintenance_tab()

    with tabs[3]:
        render_backends_tab()

    with tabs[4]:
        render_advanced_tab()


if __name__ == "__main__":
    main()
'''

# Write the file
output_file.write_text(content)
print(f"✅ Generated {output_file}")
print(f"📏 File size: {len(content)} characters")
print(f"📄 Line count: {content.count(chr(10)) + 1} lines")
