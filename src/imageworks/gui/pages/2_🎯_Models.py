"""Models management hub page with comprehensive CLI parity."""

import streamlit as st
from pathlib import Path
from typing import Dict, List, Any, Optional

from imageworks.gui.state import init_session_state
from imageworks.gui.components.backend_monitor import (
    render_backend_monitor,
    render_system_resources,
    render_gpu_monitor,
)
from imageworks.gui.components.process_runner import execute_command
from imageworks.gui.config import DEFAULT_BACKENDS, PROJECT_ROOT
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
        f"I understand this will {operation_name.lower()}", key=confirm_key
    )

    return confirmed


def render_browse_manage_tab():
    """Browse & Manage tab - model browsing with role/config editing."""
    st.markdown("### 📚 Browse & Manage Models")

    # Stats widget - show only downloaded models (matching CLI)
    with st.expander("📊 Registry Statistics", expanded=True):
        from imageworks.model_loader.download_adapter import list_downloads

        # Get downloaded models only (matching CLI default behavior)
        downloads = list(list_downloads(only_installed=False))

        # Filter out logical-only entries (no download_path or synthetic ollama://)
        installed_downloads = []
        for d in downloads:
            dp = d.download_path
            if dp is None:
                continue
            if isinstance(dp, str) and dp.startswith("ollama://"):
                continue
            installed_downloads.append(d)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Models", len(installed_downloads))

        with col2:
            formats = {}
            for d in installed_downloads:
                fmt = d.download_format or "unknown"
                formats[fmt] = formats.get(fmt, 0) + 1

            st.metric("Formats", len(formats))
            for fmt, count in sorted(formats.items()):
                # Map format names to match CLI display
                fmt_display = fmt
                if fmt == "safetensors":
                    fmt_display = "safetensors"
                elif fmt == "gguf":
                    fmt_display = "gguf"
                elif fmt == "ollama":
                    fmt_display = "ollama"
                st.caption(f"{fmt_display}: {count}")

        with col3:
            total_size = sum(d.download_size_bytes or 0 for d in installed_downloads)
            st.metric("Total Size", f"{total_size / (1024**3):.1f} GB")

    st.markdown("---")

    # Model table with filters - use download_adapter to match CLI
    st.markdown("### 📋 Downloaded Models")

    from imageworks.model_loader.download_adapter import list_downloads

    # Get downloaded models (filter out logical-only)
    downloads = list(list_downloads(only_installed=False))
    installed_downloads = []
    for d in downloads:
        dp = d.download_path
        if dp is None:
            continue
        if isinstance(dp, str) and dp.startswith("ollama://"):
            continue
        installed_downloads.append(d)

    if not installed_downloads:
        st.info("No downloaded models found. Use Download & Import tab to add models.")
        return

    st.success(f"✅ Loaded {len(installed_downloads)} models")

    # Filters
    st.markdown("**Filters**")
    col1, col2, col3 = st.columns(3)

    with col1:
        backends = sorted(set(d.backend for d in installed_downloads))
        backend_filter = st.selectbox(
            "Backend", options=["All"] + backends, key="browse_backend_filter"
        )

    with col2:
        formats = sorted(
            set(d.download_format for d in installed_downloads if d.download_format)
        )
        format_filter = st.selectbox(
            "Format", options=["All"] + formats, key="browse_format_filter"
        )

    with col3:
        quants = sorted(
            set(d.quantization for d in installed_downloads if d.quantization)
        )
        quant_filter = st.selectbox(
            "Quantization", options=["All"] + quants, key="browse_quant_filter"
        )

    # Apply filters
    filtered_downloads = installed_downloads
    if backend_filter != "All":
        filtered_downloads = [
            d for d in filtered_downloads if d.backend == backend_filter
        ]
    if format_filter != "All":
        filtered_downloads = [
            d for d in filtered_downloads if d.download_format == format_filter
        ]
    if quant_filter != "All":
        filtered_downloads = [
            d for d in filtered_downloads if d.quantization == quant_filter
        ]

    # Display as table matching CLI output
    if not filtered_downloads:
        st.warning("No models match the current filters")
        return

    # Create table data matching CLI columns
    table_data = []
    for d in filtered_downloads:
        # Get display name (simplified like CLI)
        display_name = d.display_name or d.name

        # Format abbreviation (matching CLI)
        fmt = d.download_format or ""
        fmt_display = "ST" if fmt == "safetensors" else fmt.upper() if fmt else ""

        # Quantization
        quant_display = d.quantization or ""

        # Producer (from family or metadata)
        producer = d.family or ""
        if hasattr(d, "metadata") and d.metadata:
            producer = d.metadata.get("producer", producer)

        # Capabilities (match CLI logic exactly)
        caps_dict = d.capabilities or {}
        caps_tokens = []
        if caps_dict.get("vision"):
            caps_tokens.append("V")
        if caps_dict.get("embedding"):
            caps_tokens.append("E")
        if caps_dict.get("audio"):
            caps_tokens.append("A")
        if caps_dict.get("thinking"):
            caps_tokens.append("R")  # Reasoning/Thinking
        if caps_dict.get("tools"):
            caps_tokens.append("T")  # Tool/function calling
        caps = "".join(caps_tokens) or "-"

        # Size
        size_gb = (d.download_size_bytes or 0) / (1024**3)
        size_display = f"{size_gb:.1f} GB"

        table_data.append(
            {
                "Model": display_name,
                "Fmt": fmt_display,
                "Quant": quant_display,
                "Producer": producer,
                "Caps": caps,
                "Size": size_display,
                "Backend": d.backend,
            }
        )

    # Display table
    import pandas as pd

    df = pd.DataFrame(table_data)

    # Make it selectable
    selected_idx = st.selectbox(
        "Select model to edit",
        options=range(len(filtered_downloads)),
        format_func=lambda i: filtered_downloads[i].display_name
        or filtered_downloads[i].name,
        key="browse_model_select",
    )

    # Show table
    st.dataframe(df, use_container_width=True, hide_index=True)

    selected_model = (
        filtered_downloads[selected_idx] if selected_idx is not None else None
    )

    if selected_model:
        st.markdown("---")
        st.markdown("### 🔍 Selected Model Details")

        col1, col2 = st.columns(2)

        with col1:
            st.write(f"**Name:** {selected_model.name}")
            st.write(f"**Backend:** {selected_model.backend}")
            st.write(f"**Format:** {selected_model.download_format or 'Unknown'}")
            st.write(f"**Quantization:** {selected_model.quantization or 'None'}")

        with col2:
            st.write(f"**Path:** {selected_model.download_path or 'Unknown'}")

            # Show size if available
            if selected_model.download_path:
                model_path = Path(selected_model.download_path)
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
            elif selected_model.download_size_bytes:
                size_gb = selected_model.download_size_bytes / (1024 * 1024 * 1024)
                st.write(f"**Size:** {size_gb:.2f} GB")

        # Full metadata
        with st.expander("📋 Full Metadata", expanded=False):
            # Convert RegistryEntry to dict for display
            metadata_dict = {
                "name": selected_model.name,
                "display_name": selected_model.display_name,
                "backend": selected_model.backend,
                "download_format": selected_model.download_format,
                "quantization": selected_model.quantization,
                "download_path": selected_model.download_path,
                "download_size_bytes": selected_model.download_size_bytes,
                "roles": list(selected_model.roles) if selected_model.roles else [],
                "capabilities": (
                    list(selected_model.capabilities)
                    if selected_model.capabilities
                    else []
                ),
                "family": selected_model.family,
                "served_model_id": selected_model.served_model_id,
            }
            st.json(metadata_dict)

        # Role & Configuration Management
        st.markdown("---")
        st.markdown("### 🎭 Role & Configuration Management")

        model_name = selected_model.name

        if model_name:
            registry = load_model_registry()
            registry_entry = registry.get(model_name)

            if registry_entry:
                # Role assignment
                st.markdown("#### Assigned Roles")

                # Common roles found in registry
                available_roles = [
                    "caption",
                    "keywords",
                    "description",
                    "narration",
                    "object_detection",
                    "vision_general",
                    "chat",
                    "code",
                    "reasoning",
                    "ocr",
                    "embedding",
                ]

                current_roles = list(registry_entry.roles or [])
                # Filter to only include roles that exist in available_roles
                valid_current_roles = [r for r in current_roles if r in available_roles]

                selected_roles = st.multiselect(
                    "Roles this model can perform",
                    options=available_roles,
                    default=valid_current_roles,
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
                            current_priority = registry_entry.role_priority.get(
                                role, 50
                            )
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
                        st.write("**Port:** N/A")

                with col2:
                    if backend_config:
                        st.write(f"**Host:** {backend_config.host or 'localhost'}")
                    else:
                        st.write("**Host:** localhost")

                # Extra Args Editor
                st.markdown("**Extra Arguments**")
                st.caption(
                    "Additional CLI arguments passed to the backend (one per line)"
                )

                current_extra_args = backend_config.extra_args if backend_config else []
                extra_args_text = "\n".join(current_extra_args)

                edited_extra_args = st.text_area(
                    "Extra Arguments (one per line)",
                    value=extra_args_text,
                    height=150,
                    key=f"extra_args_{model_name}",
                    help="Example args:\n--max-model-len 8192\n--gpu-memory-utilization 0.9\n--tensor-parallel-size 2",
                    label_visibility="collapsed",
                )

                # Common arg helpers
                with st.expander("📖 Common Arguments", expanded=False):
                    st.markdown(
                        """
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
                    """
                    )

                # Save button
                st.markdown("---")
                col1, col2, col3 = st.columns([1, 1, 2])

                with col1:
                    if st.button(
                        "💾 Save Changes", type="primary", key=f"save_{model_name}"
                    ):
                        try:
                            # Update registry entry
                            registry_entry.roles = selected_roles
                            registry_entry.role_priority = role_priorities

                            # Update extra_args in backend_config
                            new_extra_args = [
                                line.strip()
                                for line in edited_extra_args.split("\n")
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
            st.info(
                "ℹ️ Select a model from the table above to edit its roles and configuration"
            )


def render_download_import_tab():
    """Download & Import tab - download from HF, scan existing, import Ollama."""
    st.markdown("### 📥 Download & Import Models")

    # Sub-sections with tabs
    subtabs = st.tabs(
        ["🌐 Download from HuggingFace", "📁 Scan Existing", "🦙 Import Ollama"]
    )

    # === DOWNLOAD FROM HUGGINGFACE ===
    with subtabs[0]:
        st.markdown("#### Download from HuggingFace")

        col1, col2 = st.columns([2, 1])

        with col1:
            model_input = st.text_input(
                "Model identifier",
                value="",
                key="hf_model_input",
                help="Enter: owner/repo, owner/repo@branch, or full URL",
                placeholder="e.g., Qwen/Qwen2.5-VL-7B-AWQ",
            )

        with col2:
            branch = st.text_input(
                "Branch (optional)",
                value="",
                key="hf_branch",
                help="Use @branch in identifier or specify here",
            )

        col1, col2 = st.columns(2)

        with col1:
            formats = st.multiselect(
                "Preferred Formats",
                options=["gguf", "awq", "gptq", "safetensors"],
                default=["awq"],
                key="hf_formats",
            )

            location = st.selectbox(
                "Location",
                options=["linux_wsl", "windows_lmstudio", "custom"],
                key="hf_location",
            )

        with col2:
            include_optional = st.checkbox("Include optional files", key="hf_optional")
            force_redownload = st.checkbox("Force re-download", key="hf_force")
            non_interactive = st.checkbox(
                "Non-interactive", value=True, key="hf_nonint"
            )

        custom_path = ""
        if location == "custom":
            custom_path = st.text_input("Custom Path", value="", key="hf_custom_path")

        with st.expander("🔍 Command Preview", expanded=False):
            cmd_parts = ["uv", "run", "imageworks-download", "download"]
            if model_input:
                full_model = f"{model_input}@{branch}" if branch else model_input
                cmd_parts.append(f'"{full_model}"')
            if formats:
                cmd_parts.extend(["--format", ",".join(formats)])
            if location != "custom":
                cmd_parts.extend(["--location", location])
            elif custom_path:
                cmd_parts.extend(["--location", custom_path])
            if include_optional:
                cmd_parts.append("--include-optional")
            if force_redownload:
                cmd_parts.append("--force")
            if non_interactive:
                cmd_parts.append("--non-interactive")
            st.code(" ".join(cmd_parts), language="bash")

        if st.button(
            "📥 Download", type="primary", disabled=not model_input, key="hf_dl_btn"
        ):
            full_model = f"{model_input}@{branch}" if branch else model_input
            command = ["uv", "run", "imageworks-download", "download", full_model]
            if formats:
                command.extend(["--format", ",".join(formats)])
            if location != "custom":
                command.extend(["--location", location])
            elif custom_path:
                command.extend(["--location", custom_path])
            if include_optional:
                command.append("--include-optional")
            if force_redownload:
                command.append("--force")
            if non_interactive:
                command.append("--non-interactive")

            run_command_with_progress(
                command, f"Downloading {full_model}", timeout=3600
            )

    # === SCAN EXISTING ===
    with subtabs[1]:
        st.markdown("#### Scan Existing Downloads")
        st.caption("Import previously downloaded models")

        scan_base = st.text_input(
            "Base directory",
            value=str(Path.home() / "ai-models" / "weights"),
            key="scan_base",
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            scan_dry_run = st.checkbox("Dry run", value=True, key="scan_dry")
        with col2:
            scan_update = st.checkbox("Update existing", key="scan_update")
        with col3:
            scan_testing = st.checkbox("Include testing", key="scan_testing")

        scan_format = st.selectbox(
            "Fallback format",
            options=["", "awq", "gptq", "safetensors", "gguf"],
            key="scan_format",
        )

        with st.expander("🔍 Command Preview", expanded=False):
            cmd_parts = [
                "uv",
                "run",
                "imageworks-download",
                "scan",
                "--base",
                scan_base,
            ]
            if scan_dry_run:
                cmd_parts.append("--dry-run")
            if scan_update:
                cmd_parts.append("--update-existing")
            if scan_testing:
                cmd_parts.append("--include-testing")
            if scan_format:
                cmd_parts.extend(["--format", scan_format])
            st.code(" ".join(cmd_parts), language="bash")

        if st.button("📁 Scan Directory", type="primary", key="scan_btn"):
            command = ["uv", "run", "imageworks-download", "scan", "--base", scan_base]
            if scan_dry_run:
                command.append("--dry-run")
            if scan_update:
                command.append("--update-existing")
            if scan_testing:
                command.append("--include-testing")
            if scan_format:
                command.extend(["--format", scan_format])

            run_command_with_progress(command, "Scanning directory", timeout=300)

    # === IMPORT OLLAMA ===
    with subtabs[2]:
        st.markdown("#### Import Ollama Models")
        st.caption("Import from Ollama's internal store")

        col1, col2 = st.columns(2)
        with col1:
            ollama_dry = st.checkbox("Dry run", value=True, key="ollama_dry")
        with col2:
            ollama_deprecate = st.checkbox("Deprecate placeholders", key="ollama_dep")

        ollama_backend = st.text_input("Backend", value="ollama", key="ollama_backend")
        ollama_location = st.selectbox(
            "Location",
            options=["linux_wsl", "windows_lmstudio"],
            key="ollama_loc",
        )

        with st.expander("🔍 Command Preview", expanded=False):
            cmd_parts = ["uv", "run", "python", "scripts/import_ollama_models.py"]
            if ollama_dry:
                cmd_parts.append("--dry-run")
            if ollama_deprecate:
                cmd_parts.append("--deprecate-placeholders")
            cmd_parts.extend(
                ["--backend", ollama_backend, "--location", ollama_location]
            )
            st.code(" ".join(cmd_parts), language="bash")

        if st.button("🦙 Import", type="primary", key="ollama_btn"):
            command = ["uv", "run", "python", "scripts/import_ollama_models.py"]
            if ollama_dry:
                command.append("--dry-run")
            if ollama_deprecate:
                command.append("--deprecate-placeholders")
            command.extend(["--backend", ollama_backend, "--location", ollama_location])

            run_command_with_progress(command, "Importing Ollama models", timeout=300)


def render_registry_maintenance_tab():
    """Registry Maintenance tab - normalize, purge, rebuild operations."""
    st.markdown("### 🔧 Registry Maintenance")
    st.warning(
        "⚠️ These operations modify the registry. Always review dry-run output first."
    )

    # Sub-sections
    subtabs = st.tabs(["🔄 Normalize", "🗑️ Purge", "🔨 Cleanup"])

    # === NORMALIZE ===
    with subtabs[0]:
        st.markdown("#### Normalize Formats & Rebuild")
        st.caption("Re-detect formats/quantization and rebuild metadata")

        col1, col2 = st.columns(2)
        with col1:
            norm_dry = st.checkbox("Dry run", value=True, key="norm_dry")
            norm_rebuild = st.checkbox("Rebuild dynamic fields", key="norm_rebuild")
        with col2:
            norm_prune = st.checkbox("Prune missing", key="norm_prune")
            norm_backup = st.checkbox("Create backup", value=True, key="norm_backup")

        with st.expander("🔍 Command Preview", expanded=False):
            cmd_parts = ["uv", "run", "imageworks-download", "normalize-formats"]
            if norm_dry:
                cmd_parts.append("--dry-run")
            else:
                cmd_parts.append("--apply")
            if norm_rebuild:
                cmd_parts.append("--rebuild")
            if norm_prune:
                cmd_parts.append("--prune-missing")
            if not norm_backup:
                cmd_parts.append("--no-backup")
            st.code(" ".join(cmd_parts), language="bash")

        btn_label = "👁️ Preview" if norm_dry else "🔄 Apply"
        if st.button(btn_label, type="primary", key="norm_btn"):
            command = ["uv", "run", "imageworks-download", "normalize-formats"]
            if norm_dry:
                command.append("--dry-run")
            else:
                command.append("--apply")
            if norm_rebuild:
                command.append("--rebuild")
            if norm_prune:
                command.append("--prune-missing")
            if not norm_backup:
                command.append("--no-backup")

            run_command_with_progress(command, "Normalizing registry", timeout=600)

    # === PURGE ===
    with subtabs[1]:
        st.markdown("#### Purge Operations")

        purge_op = st.selectbox(
            "Operation",
            options=[
                "purge-deprecated",
                "purge-logical-only",
                "purge-hf",
                "reset-discovered",
            ],
            key="purge_op",
        )

        if purge_op == "purge-deprecated":
            st.markdown("**Remove deprecated entries**")
            placeholders_only = st.checkbox(
                "Legacy placeholders only", key="purge_dep_plc"
            )
            purge_dry = st.checkbox("Dry run", value=True, key="purge_dep_dry")

            if st.button("🗑️ Purge Deprecated", type="primary", key="purge_dep_btn"):
                command = ["uv", "run", "imageworks-download", "purge-deprecated"]
                if placeholders_only:
                    command.append("--placeholders-only")
                if purge_dry:
                    command.append("--dry-run")
                run_command_with_progress(command, "Purging deprecated", timeout=60)

        elif purge_op == "purge-logical-only":
            st.markdown("**Remove logical-only entries (no download_path)**")
            include_curated = st.checkbox("Include curated", key="purge_log_cur")
            purge_dry = st.checkbox("Dry run", value=True, key="purge_log_dry")

            if st.button("🗑️ Purge Logical", type="primary", key="purge_log_btn"):
                command = ["uv", "run", "imageworks-download", "purge-logical-only"]
                if include_curated:
                    command.append("--include-curated")
                else:
                    command.append("--discovered-only")
                if purge_dry:
                    command.append("--dry-run")
                else:
                    command.append("--apply")
                run_command_with_progress(command, "Purging logical-only", timeout=60)

        elif purge_op == "purge-hf":
            st.markdown("**Remove HF entries from weights root**")
            weights_root = st.text_input(
                "Weights root",
                value=str(Path.home() / "ai-models" / "weights"),
                key="purge_hf_root",
            )
            backend_filter = st.text_input(
                "Backend filter (optional)", key="purge_hf_backend"
            )
            purge_dry = st.checkbox("Dry run", value=True, key="purge_hf_dry")

            if st.button("🗑️ Purge HF", type="primary", key="purge_hf_btn"):
                command = [
                    "uv",
                    "run",
                    "imageworks-download",
                    "purge-hf",
                    "--weights-root",
                    weights_root,
                ]
                if backend_filter:
                    command.extend(["--backend", backend_filter])
                if purge_dry:
                    command.append("--dry-run")
                run_command_with_progress(command, "Purging HF entries", timeout=60)

        elif purge_op == "reset-discovered":
            st.markdown("**Reset discovered layer**")
            reset_backend = st.selectbox(
                "Backend",
                options=["all", "ollama", "vllm", "lmdeploy"],
                key="reset_backend",
            )
            purge_dry = st.checkbox("Dry run", value=True, key="reset_dry")

            if st.button("🔄 Reset", type="primary", key="reset_btn"):
                command = [
                    "uv",
                    "run",
                    "imageworks-download",
                    "reset-discovered",
                    "--backend",
                    reset_backend,
                ]
                if purge_dry:
                    command.append("--dry-run")
                run_command_with_progress(
                    command, f"Resetting {reset_backend}", timeout=60
                )

    # === CLEANUP ===
    with subtabs[2]:
        st.markdown("#### Advanced Cleanup")

        cleanup_op = st.selectbox(
            "Operation",
            options=["prune-duplicates", "restore-ollama", "backfill-ollama-paths"],
            key="cleanup_op",
        )

        if cleanup_op == "prune-duplicates":
            st.markdown("**Remove duplicates (keep richest metadata)**")
            cleanup_backend = st.text_input(
                "Backend filter (optional)", key="cleanup_backend"
            )
            cleanup_dry = st.checkbox("Dry run", value=True, key="cleanup_dry")

            if st.button("🧹 Prune Duplicates", type="primary", key="prune_dup_btn"):
                command = ["uv", "run", "imageworks-download", "prune-duplicates"]
                if cleanup_backend:
                    command.extend(["--backend", cleanup_backend])
                if cleanup_dry:
                    command.append("--dry-run")
                run_command_with_progress(command, "Pruning duplicates", timeout=60)

        elif cleanup_op == "restore-ollama":
            st.markdown("**Restore Ollama entries from backup**")
            backup_file = st.text_input("Backup file (optional)", key="restore_backup")
            restore_deprecated = st.checkbox("Include deprecated", key="restore_dep")
            restore_dry = st.checkbox("Dry run", value=True, key="restore_dry")

            if st.button("↩️ Restore", type="primary", key="restore_btn"):
                command = ["uv", "run", "imageworks-download", "restore-ollama"]
                if backup_file:
                    command.extend(["--backup", backup_file])
                if restore_deprecated:
                    command.append("--include-deprecated")
                if restore_dry:
                    command.append("--dry-run")
                run_command_with_progress(command, "Restoring Ollama", timeout=60)

        elif cleanup_op == "backfill-ollama-paths":
            st.markdown("**Backfill synthetic paths for legacy entries**")
            backfill_dry = st.checkbox("Dry run", value=True, key="backfill_dry")

            if st.button("🔧 Backfill", type="primary", key="backfill_btn"):
                command = ["uv", "run", "imageworks-download", "backfill-ollama-paths"]
                if backfill_dry:
                    command.append("--dry-run")
                run_command_with_progress(command, "Backfilling paths", timeout=60)


def render_backends_tab():
    """Backends tab - monitoring and management."""

    # Render backend monitor (includes its own "Backend Status" heading)
    render_backend_monitor(backends=DEFAULT_BACKENDS, key_prefix="backends_monitor")

    # GPU monitoring
    st.markdown("---")
    render_gpu_monitor()

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
    st.error("⚠️ **DANGEROUS OPERATIONS** - Can delete files permanently")

    # Sub-sections
    subtabs = st.tabs(["🗑️ Remove Models", "✅ Verify", "📊 Profiles"])

    # === REMOVE MODELS ===
    with subtabs[0]:
        st.markdown("#### Remove Model Variants")

        # Get list of models
        registry = load_model_registry()
        model_names = sorted(registry.keys())

        if not model_names:
            st.warning("No models found in registry")
        else:
            selected_variant = st.selectbox(
                "Select model to remove",
                options=model_names,
                key="remove_variant",
            )

            if selected_variant:
                entry = registry[selected_variant]

                # Show details
                st.markdown("**Selected Model:**")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.write(f"**Name:** {selected_variant}")
                    st.write(f"**Backend:** {entry.backend}")

                with col2:
                    st.write(f"**Format:** {entry.download_format or 'N/A'}")
                    st.write(f"**Quant:** {entry.quantization or 'None'}")

                with col3:
                    download_path = entry.download_path
                    if download_path:
                        st.write(f"**Path:** {download_path}")
                        exists = Path(download_path).exists()
                        st.write(f"**Exists:** {'✅' if exists else '❌'}")

                st.markdown("---")

                # Removal options
                st.markdown("**Removal Options:**")

                removal_mode = st.radio(
                    "What to remove?",
                    options=[
                        "Metadata only (keep files & logical entry)",
                        "Files only (keep logical entry)",
                        "Purge entirely (delete entry + files)",
                    ],
                    key="removal_mode",
                )

                delete_files = removal_mode in [
                    "Files only (keep logical entry)",
                    "Purge entirely (delete entry + files)",
                ]

                purge = removal_mode == "Purge entirely (delete entry + files)"

                # Confirmation
                st.markdown("---")

                if delete_files:
                    confirmed = confirm_destructive_operation(
                        "Delete Files" if not purge else "Purge Model",
                        f"This will **permanently delete** files at: `{download_path}`\\n\\n"
                        + (
                            "Registry entry will also be removed."
                            if purge
                            else "Registry entry will remain (logical entry only)."
                        ),
                    )
                else:
                    confirmed = st.checkbox(
                        "I understand this will clear download metadata",
                        key="confirm_metadata",
                    )

                # Command preview
                with st.expander("🔍 Command Preview", expanded=False):
                    cmd_parts = [
                        "uv",
                        "run",
                        "imageworks-download",
                        "remove",
                        f'"{selected_variant}"',
                    ]
                    if delete_files:
                        cmd_parts.append("--delete-files")
                    if purge:
                        cmd_parts.append("--purge")
                    cmd_parts.append("--force")
                    st.code(" ".join(cmd_parts), language="bash")

                # Remove button
                if st.button(
                    "🗑️ Remove", type="primary", disabled=not confirmed, key="remove_btn"
                ):
                    command = [
                        "uv",
                        "run",
                        "imageworks-download",
                        "remove",
                        selected_variant,
                    ]
                    if delete_files:
                        command.append("--delete-files")
                    if purge:
                        command.append("--purge")
                    command.append("--force")

                    run_command_with_progress(
                        command, f"Removing {selected_variant}", timeout=60
                    )

                    # Refresh
                    from imageworks.gui.components.registry_table import load_registry

                    load_registry.clear()
                    st.rerun()

    # === VERIFY ===
    with subtabs[1]:
        st.markdown("#### Verify Model Integrity")
        st.caption("Check directories exist and checksums match")

        verify_all = st.checkbox("Verify all models", value=True, key="verify_all")

        verify_variant = None
        if not verify_all:
            model_names = sorted(load_model_registry().keys())
            verify_variant = st.selectbox(
                "Select model", options=model_names, key="verify_variant"
            )

        verify_fix = st.checkbox(
            "Auto-fix missing",
            value=False,
            key="verify_fix",
            help="Clear download metadata for missing entries",
        )

        with st.expander("🔍 Command Preview", expanded=False):
            cmd_parts = ["uv", "run", "imageworks-download", "verify"]
            if not verify_all and verify_variant:
                cmd_parts.append(f'"{verify_variant}"')
            if verify_fix:
                cmd_parts.append("--fix-missing")
            st.code(" ".join(cmd_parts), language="bash")

        if st.button("✅ Verify", type="primary", key="verify_btn"):
            command = ["uv", "run", "imageworks-download", "verify"]
            if not verify_all and verify_variant:
                command.append(verify_variant)
            if verify_fix:
                command.append("--fix-missing")

            run_command_with_progress(command, "Verifying integrity", timeout=300)

    # === PROFILES ===
    with subtabs[2]:
        st.markdown("#### Deployment Profiles")

        st.info("ℹ️ Visual profile editor coming in future phase")

        # Show current pyproject.toml
        st.markdown("### Current Configuration")

        pyproject_path = PROJECT_ROOT / "pyproject.toml"

        if pyproject_path.exists():
            try:
                try:
                    import tomllib
                except ImportError:
                    import tomli as tomllib  # type: ignore[import-not-found]

                with open(pyproject_path, "rb") as f:
                    config = tomllib.load(f)

                imageworks_config = config.get("tool", {}).get("imageworks", {})

                if imageworks_config:
                    st.json(imageworks_config)
                else:
                    st.warning("No ImageWorks configuration found")

            except Exception as e:
                st.error(f"Failed to load pyproject.toml: {e}")
        else:
            st.error("pyproject.toml not found")


def main():
    """Models management hub page with comprehensive CLI parity."""
    init_session_state()

    # Custom CSS for wider layout and better table display
    st.markdown(
        """
        <style>
        /* Force wider main content area - override Streamlit defaults */
        .main .block-container {
            max-width: 100% !important;
            padding-left: 3rem !important;
            padding-right: 3rem !important;
        }

        /* Remove max-width constraint from app view */
        section.main > div {
            max-width: 100% !important;
            padding-left: 3rem !important;
            padding-right: 3rem !important;
        }

        /* Make dataframes use full available width */
        .stDataFrame {
            width: 100% !important;
        }

        /* Ensure tables don't truncate unnecessarily */
        div[data-testid="stDataFrame"] > div {
            width: 100% !important;
            overflow-x: auto !important;
        }

        /* Expand dataframe container */
        .element-container:has(div[data-testid="stDataFrame"]) {
            width: 100% !important;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    st.title("🎯 Models Management")
    st.markdown("Comprehensive model management with full CLI parity")

    # 5 tabs for different functions
    tabs = st.tabs(
        [
            "📚 Browse & Manage",
            "📥 Download & Import",
            "🔧 Registry Maintenance",
            "🔌 Backends",
            "⚙️ Advanced",
        ]
    )

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
