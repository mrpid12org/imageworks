"""Models management hub page with comprehensive CLI parity."""

import json
import shlex
import streamlit as st
from pathlib import Path
from typing import Dict, List, Any, Optional
import re

from imageworks.gui.state import init_session_state
from imageworks.gui.components.backend_monitor import (
    render_backend_monitor,
    render_system_resources,
    render_gpu_monitor,
)
from imageworks.gui.config import (
    DEFAULT_BACKENDS,
    MODEL_REGISTRY_DISCOVERED_PATH,
    PROJECT_ROOT,
)
from imageworks.model_loader.registry import save_registry
from imageworks.model_loader.registry import load_registry as load_model_registry


_QUANT_TOKEN_PATTERN = re.compile(
    r"(iq\d+_[a-z]+|q\d+_[a-z]+|q\d+|fp\d+|bf16|int\d+)", re.IGNORECASE
)


def _format_bytes(num: int) -> str:
    """Convert bytes to human-readable string."""
    if num <= 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} TB"


def _guess_quant_label(filename: str) -> Optional[str]:
    match = _QUANT_TOKEN_PATTERN.findall(filename)
    if match:
        return match[-1].upper()
    return None


def _classify_discovered_provider(entry: Dict[str, Any]) -> str:
    """Classify a discovered entry into provider buckets used in purge previews."""

    backend = (entry.get("backend") or "").lower()
    download_path = (entry.get("download_path") or "").lower()
    metadata = entry.get("metadata") or {}
    source = entry.get("source") or {}
    provider = (
        (entry.get("source_provider") or source.get("provider") or "").strip().lower()
    )

    if (
        backend == "ollama"
        or download_path.startswith("ollama:")
        or provider == "ollama"
    ):
        return "ollama"
    if provider in {"hf", "huggingface"} or metadata.get("created_from_download"):
        return "hf"
    if provider:
        return provider
    return "other"


def summarize_discovered_layer() -> Optional[Dict[str, int]]:
    """Return counts of discovered entries per provider for registry maintenance hints."""

    try:
        if not MODEL_REGISTRY_DISCOVERED_PATH.exists():
            return None
        data = json.loads(MODEL_REGISTRY_DISCOVERED_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None

    if not isinstance(data, list):
        return None

    summary: Dict[str, int] = {"ollama": 0, "hf": 0, "other": 0, "total": len(data)}
    for entry in data:
        if not isinstance(entry, dict):
            continue
        bucket = _classify_discovered_provider(entry)
        summary[bucket] = summary.get(bucket, 0) + 1
    summary["total"] = len(data)
    return summary


def run_command_with_progress(
    command: List[str],
    description: str,
    show_output: bool = True,
    timeout: Optional[int] = None,
) -> Dict[str, Any]:
    """Run command with progress indicator and live output display."""
    import subprocess

    # Create placeholders for live updates
    status_placeholder = st.empty()
    output_placeholder = st.empty()

    status_placeholder.info(f"⏳ {description}...")

    output_lines = []

    try:
        # Run process with streaming output
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        # Stream output line by line
        for line in process.stdout:
            output_lines.append(line.rstrip())
            # Show last 20 lines in real-time
            if show_output:
                recent_output = "\n".join(output_lines[-20:])
                output_placeholder.code(recent_output, language="bash")

        process.wait(timeout=timeout)

        # Final result
        result = {
            "command": " ".join(command),
            "exit_code": process.returncode,
            "stdout": "\n".join(output_lines),
            "stderr": "",
            "success": process.returncode == 0,
        }

        if result["success"]:
            status_placeholder.success(f"✅ {description} - Complete")
            if show_output and output_lines:
                with output_placeholder.container():
                    with st.expander("📄 Full Output", expanded=True):
                        st.code("\n".join(output_lines), language="bash")
        else:
            status_placeholder.error(f"❌ {description} - Failed")
            if show_output and output_lines:
                output_placeholder.error("\n".join(output_lines[-50:]))

        return result

    except subprocess.TimeoutExpired:
        status_placeholder.error(f"❌ {description} - Timed out after {timeout}s")
        return {
            "command": " ".join(command),
            "exit_code": -1,
            "stdout": "\n".join(output_lines),
            "stderr": f"Timeout after {timeout}s",
            "success": False,
        }
    except Exception as e:
        status_placeholder.error(f"❌ {description} - Error: {e}")
        return {
            "command": " ".join(command),
            "exit_code": -1,
            "stdout": "\n".join(output_lines),
            "stderr": str(e),
            "success": False,
        }


def confirm_destructive_operation(operation_name: str, details: str) -> bool:
    """Show confirmation dialog for destructive operations."""
    st.warning(f"⚠️ **{operation_name}**")
    st.markdown(details)

    confirm_key = f"confirm_{operation_name.replace(' ', '_').lower()}"
    confirmed = st.checkbox(
        f"I understand this will {operation_name.lower()}", key=confirm_key
    )

    return confirmed


def extract_known_flags(extra_args: List[str]) -> Dict[str, Optional[str]]:
    """Extract known model loading flags from extra_args list.

    Returns dict with flag names (without --) as keys and their values.
    Returns None for flags not present.
    """
    known_flags = {
        "max-model-len": None,
        "gpu-memory-utilization": None,
        "tensor-parallel-size": None,
        "kv-cache-dtype": None,
        "cache-max-entry-count": None,  # LMDeploy
        "tp": None,  # LMDeploy tensor parallel
        "num-ctx": None,  # Ollama
        "num-gpu": None,  # Ollama
    }

    i = 0
    while i < len(extra_args):
        arg = extra_args[i]

        # Handle --flag=value format
        if "=" in arg:
            flag_part, value_part = arg.split("=", 1)
            flag_name = flag_part.lstrip("-")
            if flag_name in known_flags:
                known_flags[flag_name] = value_part
            i += 1
            continue

        # Handle --flag value format
        if arg.startswith("--"):
            flag_name = arg.lstrip("-")
            if flag_name in known_flags:
                # Get next arg as value if it exists and doesn't start with --
                if i + 1 < len(extra_args) and not extra_args[i + 1].startswith("--"):
                    known_flags[flag_name] = extra_args[i + 1]
                    i += 2
                    continue

        i += 1

    return known_flags


def update_extra_args(
    extra_args: List[str], updates: Dict[str, Optional[str]]
) -> List[str]:
    """Update extra_args list with new flag values.

    - If value is None, removes the flag
    - If flag exists, updates its value
    - If flag doesn't exist, appends it
    - Preserves unknown flags
    """
    result = []
    skip_next = False
    processed_flags = set()

    i = 0
    while i < len(extra_args):
        if skip_next:
            skip_next = False
            i += 1
            continue

        arg = extra_args[i]

        # Handle --flag=value format
        if "=" in arg and arg.startswith("--"):
            flag_part, value_part = arg.split("=", 1)
            flag_name = flag_part.lstrip("-")

            if flag_name in updates:
                processed_flags.add(flag_name)
                # Update or remove
                if updates[flag_name] is not None:
                    result.append(f"--{flag_name}={updates[flag_name]}")
                # else: skip (remove)
            else:
                # Keep unknown flag
                result.append(arg)
            i += 1
            continue

        # Handle --flag value format
        if arg.startswith("--"):
            flag_name = arg.lstrip("-")

            if flag_name in updates:
                processed_flags.add(flag_name)
                # Update or remove
                if updates[flag_name] is not None:
                    result.append(f"--{flag_name}")
                    result.append(updates[flag_name])
                # else: skip both flag and value
                if i + 1 < len(extra_args) and not extra_args[i + 1].startswith("--"):
                    skip_next = True
            else:
                # Keep unknown flag
                result.append(arg)
        else:
            # Keep non-flag args
            result.append(arg)

        i += 1

    # Add new flags that weren't already present
    for flag_name, value in updates.items():
        if flag_name not in processed_flags and value is not None:
            result.append(f"--{flag_name}")
            result.append(value)

    return result


def analyze_hf_repository(model_id: str) -> Dict[str, Any]:
    """Analyze HuggingFace repository and return structured results."""
    from imageworks.tools.model_downloader.url_analyzer import URLAnalyzer

    analyzer = URLAnalyzer()
    analysis = analyzer.analyze_url(model_id)

    # Check critical files
    critical_status = check_critical_files(analysis)

    return {
        "analysis": analysis,
        "critical_status": critical_status,
        "warnings": generate_warnings(analysis, critical_status),
    }


def check_critical_files(analysis) -> Dict[str, Any]:
    """Check which critical files are present/missing."""
    files = analysis.files
    critical = {
        "config.json": None,
        "tokenizer_config.json": None,
        "tokenizer.json or tokenizer.model": None,
        "generation_config.json": None,
        "chat_template": None,
    }

    # Get file lists
    config_files = [f.path for f in files.get("config", [])]
    tokenizer_files = [f.path for f in files.get("tokenizer", [])]

    # Check config.json
    if any("config.json" in f for f in config_files):
        critical["config.json"] = "found"
    else:
        critical["config.json"] = "missing"

    # Check tokenizer_config.json
    if any("tokenizer_config.json" in f for f in config_files):
        critical["tokenizer_config.json"] = "found"
    else:
        critical["tokenizer_config.json"] = "missing"

    # Check tokenizer.json or tokenizer.model
    if any("tokenizer.json" in f for f in tokenizer_files):
        critical["tokenizer.json or tokenizer.model"] = "tokenizer.json"
    elif any("tokenizer.model" in f for f in tokenizer_files):
        critical["tokenizer.json or tokenizer.model"] = "tokenizer.model"
    else:
        critical["tokenizer.json or tokenizer.model"] = "missing"

    # Check generation_config.json
    if any("generation_config.json" in f for f in config_files):
        critical["generation_config.json"] = "found"
    else:
        critical["generation_config.json"] = "missing"

    # Check chat_template
    chat_template_files = [
        f
        for f in config_files
        if "chat_template" in f.lower()
        and (f.endswith(".json") or f.endswith(".jinja"))
    ]
    if chat_template_files:
        critical["chat_template"] = "standalone"
    else:
        critical["chat_template"] = "check_embedded"

    return critical


def generate_warnings(analysis, critical_status: Dict[str, Any]) -> List[str]:
    """Generate warnings based on analysis results."""
    warnings = []

    # Critical file warnings
    if critical_status["config.json"] == "missing":
        warnings.append("❌ config.json is MISSING - download will likely fail")

    if critical_status["tokenizer.json or tokenizer.model"] == "missing":
        warnings.append("❌ Tokenizer file is MISSING - model cannot be loaded")

    if critical_status["chat_template"] == "check_embedded":
        warnings.append(
            "⚠️ No standalone chat_template.json - may be embedded in tokenizer_config"
        )

    if critical_status["generation_config.json"] == "missing":
        warnings.append(
            "⚠️ generation_config.json missing - may need manual configuration"
        )

    # Format detection warnings
    if not analysis.formats:
        warnings.append("❌ No model format detected - download may fail")
    elif analysis.formats[0].confidence < 0.5:
        warnings.append("⚠️ Low confidence format detection - verify before downloading")

    # Size warnings
    if analysis.total_size > 50 * 1024**3:  # > 50GB
        warnings.append(
            f"⚠️ Large download: {analysis.total_size / (1024**3):.1f} GB - ensure sufficient disk space"
        )

    return warnings


def render_browse_manage_tab():
    """Browse & Manage tab - model browsing with role/config editing."""
    st.markdown("### 📚 Browse & Manage Models")

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

    # Show table first
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Summary line (CLI-style)
    total_size = sum(d.download_size_bytes or 0 for d in filtered_downloads)
    st.caption(
        f"📊 {len(filtered_downloads)} models • {total_size / (1024**3):.1f} GB total"
    )

    # Model selector below table
    st.markdown("---")

    # Check if we have unsaved changes and warn before changing selection
    prev_selection_key = "browse_prev_model_select"
    if prev_selection_key not in st.session_state:
        st.session_state[prev_selection_key] = 0

    # Check if any model has unsaved changes
    has_any_unsaved = False
    if st.session_state.get(prev_selection_key) is not None:
        prev_model = filtered_downloads[st.session_state[prev_selection_key]]
        prev_model_name = prev_model.name
        has_changes_key = f"params_has_changes_{prev_model_name}"
        edit_mode_key = f"params_edit_mode_{prev_model_name}"
        if st.session_state.get(edit_mode_key) and st.session_state.get(
            has_changes_key
        ):
            has_any_unsaved = True

    if has_any_unsaved:
        st.warning(
            "⚠️ You have unsaved changes. Please save or cancel before selecting a different model."
        )

    selected_idx = st.selectbox(
        "Select model to view/edit",
        options=range(len(filtered_downloads)),
        format_func=lambda i: filtered_downloads[i].display_name
        or filtered_downloads[i].name,
        key="browse_model_select",
        disabled=has_any_unsaved,
    )

    # Update previous selection
    st.session_state[prev_selection_key] = selected_idx

    selected_model = (
        filtered_downloads[selected_idx] if selected_idx is not None else None
    )

    if selected_model:
        st.markdown("---")
        st.markdown("### 🔍 Selected Model Details")

        # Single line - just name and path
        st.write(
            f"**Name:** {selected_model.name} • **Path:** {selected_model.download_path or 'Unknown'}"
        )

        # Load from registry for complete info
        model_name = selected_model.name
        registry = load_model_registry()
        registry_entry = registry.get(model_name)

        # Generation & Runtime Parameters
        st.markdown("---")
        st.markdown("### ⚙️ Generation & Runtime Parameters")
        st.caption("💡 Common model parameters for generation and model loading.")

        if model_name and registry_entry:
            # Edit mode tracking
            edit_mode_key = f"params_edit_mode_{model_name}"
            has_changes_key = f"params_has_changes_{model_name}"

            # Initialize session state
            if edit_mode_key not in st.session_state:
                st.session_state[edit_mode_key] = False
            if has_changes_key not in st.session_state:
                st.session_state[has_changes_key] = False

            is_editing = st.session_state[edit_mode_key]

            # Get current values
            gen_defaults = registry_entry.generation_defaults
            backend_config = registry_entry.backend_config
            current_extra_args = backend_config.extra_args if backend_config else []

            # Extract known flags from extra_args
            known_flags = extract_known_flags(current_extra_args)

            # Edit/Save/Cancel buttons at top
            col_btn1, col_btn2, col_btn3, col_spacer = st.columns([1, 1, 1, 3])

            with col_btn1:
                if not is_editing:
                    if st.button("✏️ Edit", key=f"edit_params_{model_name}"):
                        st.session_state[edit_mode_key] = True
                        st.rerun()
                else:
                    if st.button(
                        "💾 Save",
                        type="primary",
                        key=f"save_params_{model_name}",
                    ):
                        # Will handle save below
                        pass

            with col_btn2:
                if is_editing:
                    if st.button("❌ Cancel", key=f"cancel_params_{model_name}"):
                        st.session_state[edit_mode_key] = False
                        st.session_state[has_changes_key] = False
                        st.rerun()

            with col_btn3:
                if is_editing and st.session_state[has_changes_key]:
                    st.warning("⚠️ Unsaved changes")

            col1, col2 = st.columns(2)

            # LEFT COLUMN - Generation Parameters
            with col1:
                st.markdown("**🎲 Generation Parameters**")

                temperature = st.number_input(
                    "Temperature",
                    min_value=0.0,
                    max_value=2.0,
                    value=(
                        gen_defaults.temperature
                        if gen_defaults.temperature is not None
                        else 0.7
                    ),
                    step=0.1,
                    key=f"gen_temp_{model_name}",
                    help="Controls randomness (0.0-0.3: deterministic, 0.4-0.7: balanced, 0.8-1.2: creative)",
                    disabled=not is_editing,
                )

                top_p = st.number_input(
                    "Top-p (Nucleus Sampling)",
                    min_value=0.0,
                    max_value=1.0,
                    value=gen_defaults.top_p if gen_defaults.top_p is not None else 0.9,
                    step=0.05,
                    key=f"gen_top_p_{model_name}",
                    help="Cumulative probability threshold (0.8-0.9 recommended)",
                    disabled=not is_editing,
                )

                top_k = st.number_input(
                    "Top-k",
                    min_value=0,
                    max_value=200,
                    value=gen_defaults.top_k if gen_defaults.top_k is not None else 40,
                    step=5,
                    key=f"gen_top_k_{model_name}",
                    help="Limits sampling to top k tokens (20-50 typical, 0 to disable)",
                    disabled=not is_editing,
                )

                max_tokens = st.number_input(
                    "Max Tokens",
                    min_value=1,
                    max_value=32000,
                    value=(
                        gen_defaults.max_tokens
                        if gen_defaults.max_tokens is not None
                        else 512
                    ),
                    step=64,
                    key=f"gen_max_tokens_{model_name}",
                    help="Maximum tokens to generate",
                    disabled=not is_editing,
                )

                context_window = st.number_input(
                    "Context Window",
                    min_value=512,
                    max_value=128000,
                    value=(
                        gen_defaults.context_window
                        if gen_defaults.context_window is not None
                        else 4096
                    ),
                    step=512,
                    key=f"gen_context_{model_name}",
                    help="Model's context window size",
                    disabled=not is_editing,
                )

                frequency_penalty = st.number_input(
                    "Frequency Penalty",
                    min_value=0.0,
                    max_value=2.0,
                    value=(
                        gen_defaults.frequency_penalty
                        if gen_defaults.frequency_penalty is not None
                        else 0.0
                    ),
                    step=0.1,
                    key=f"gen_freq_penalty_{model_name}",
                    help="Reduces repetition based on frequency (0.0-0.5 typical)",
                    disabled=not is_editing,
                )

                presence_penalty = st.number_input(
                    "Presence Penalty",
                    min_value=0.0,
                    max_value=2.0,
                    value=(
                        gen_defaults.presence_penalty
                        if gen_defaults.presence_penalty is not None
                        else 0.0
                    ),
                    step=0.1,
                    key=f"gen_pres_penalty_{model_name}",
                    help="Encourages new topics (0.0-0.5 typical)",
                    disabled=not is_editing,
                )

                stop_sequences_text = st.text_area(
                    "Stop Sequences",
                    value=(
                        ", ".join(gen_defaults.stop_sequences)
                        if gen_defaults.stop_sequences
                        else ""
                    ),
                    height=60,
                    key=f"gen_stop_{model_name}",
                    help="Comma-separated stop sequences",
                    disabled=not is_editing,
                )

            # RIGHT COLUMN - Model Loading Parameters
            with col2:
                st.markdown("**🔧 Model Loading Parameters**")

                # Read-only backend info at top
                st.text_input(
                    "Backend",
                    value=registry_entry.backend,
                    key=f"load_backend_{model_name}",
                    disabled=True,
                    help="Backend serving this model",
                )

                st.number_input(
                    "Port",
                    value=backend_config.port if backend_config else 8000,
                    key=f"load_port_{model_name}",
                    disabled=True,
                    help="Port where backend is listening",
                )

                st.markdown("**Editable Parameters:**")

                max_model_len = st.number_input(
                    "Max Model Length (vLLM)",
                    min_value=512,
                    max_value=128000,
                    value=(
                        int(known_flags["max-model-len"])
                        if known_flags["max-model-len"]
                        else 4096
                    ),
                    step=512,
                    key=f"load_max_model_len_{model_name}",
                    help="Maximum sequence length for vLLM (overrides model config)",
                    disabled=not is_editing,
                )

                gpu_mem = st.number_input(
                    "GPU Memory Utilization (vLLM)",
                    min_value=0.1,
                    max_value=1.0,
                    value=(
                        float(known_flags["gpu-memory-utilization"])
                        if known_flags["gpu-memory-utilization"]
                        else 0.90
                    ),
                    step=0.05,
                    key=f"load_gpu_mem_{model_name}",
                    help="Fraction of GPU memory to use (0.85-0.95 recommended)",
                    disabled=not is_editing,
                )

                tensor_parallel = st.number_input(
                    "Tensor Parallel Size (vLLM)",
                    min_value=1,
                    max_value=8,
                    value=(
                        int(known_flags["tensor-parallel-size"])
                        if known_flags["tensor-parallel-size"]
                        else 1
                    ),
                    step=1,
                    key=f"load_tp_{model_name}",
                    help="Number of GPUs for tensor parallelism",
                    disabled=not is_editing,
                )

                kv_cache = st.text_input(
                    "KV Cache Dtype (vLLM)",
                    value=known_flags["kv-cache-dtype"] or "auto",
                    key=f"load_kv_cache_{model_name}",
                    help="KV cache data type (auto, fp8, fp16)",
                    disabled=not is_editing,
                )

                # Ollama-specific
                num_ctx_ollama = st.number_input(
                    "Context Length (Ollama)",
                    min_value=512,
                    max_value=128000,
                    value=(
                        int(known_flags["num-ctx"]) if known_flags["num-ctx"] else 4096
                    ),
                    step=512,
                    key=f"load_num_ctx_{model_name}",
                    help="Context window for Ollama",
                    disabled=not is_editing,
                )

                num_gpu_ollama = st.number_input(
                    "GPU Layers (Ollama)",
                    min_value=0,
                    max_value=999,
                    value=int(known_flags["num-gpu"]) if known_flags["num-gpu"] else 35,
                    step=1,
                    key=f"load_num_gpu_{model_name}",
                    help="Number of layers to offload to GPU",
                    disabled=not is_editing,
                )

            # Track changes (check if values differ from stored values)
            if is_editing:
                has_changes = (
                    temperature != (gen_defaults.temperature or 0.7)
                    or top_p != (gen_defaults.top_p or 0.9)
                    or top_k != (gen_defaults.top_k or 40)
                    or max_tokens != (gen_defaults.max_tokens or 512)
                    or context_window != (gen_defaults.context_window or 4096)
                    or frequency_penalty != (gen_defaults.frequency_penalty or 0.0)
                    or presence_penalty != (gen_defaults.presence_penalty or 0.0)
                )
                st.session_state[has_changes_key] = has_changes

            # Handle save action
            if is_editing and st.session_state.get(f"save_params_{model_name}"):
                try:
                    # Update generation_defaults
                    gen_defaults.temperature = temperature
                    gen_defaults.top_p = top_p
                    gen_defaults.top_k = top_k if top_k > 0 else None
                    gen_defaults.max_tokens = max_tokens
                    gen_defaults.context_window = context_window
                    gen_defaults.frequency_penalty = (
                        frequency_penalty if frequency_penalty > 0 else None
                    )
                    gen_defaults.presence_penalty = (
                        presence_penalty if presence_penalty > 0 else None
                    )
                    gen_defaults.stop_sequences = [
                        s.strip() for s in stop_sequences_text.split(",") if s.strip()
                    ]

                    # Update extra_args with model loading parameters
                    updates = {}

                    # vLLM parameters
                    if registry_entry.backend == "vllm":
                        updates["max-model-len"] = str(max_model_len)
                        updates["gpu-memory-utilization"] = str(gpu_mem)
                        if tensor_parallel > 1:
                            updates["tensor-parallel-size"] = str(tensor_parallel)
                        if kv_cache and kv_cache != "auto":
                            updates["kv-cache-dtype"] = kv_cache

                    # Ollama parameters
                    elif registry_entry.backend == "ollama":
                        updates["num-ctx"] = str(num_ctx_ollama)
                        updates["num-gpu"] = str(num_gpu_ollama)

                    # Update extra_args
                    if backend_config:
                        backend_config.extra_args = update_extra_args(
                            current_extra_args, updates
                        )

                    # Save registry
                    save_registry()

                    # Reset edit mode
                    st.session_state[edit_mode_key] = False
                    st.session_state[has_changes_key] = False

                    st.success(f"✅ Saved parameters for {model_name}")
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Failed to save parameters: {e}")

            # Full Metadata expander at bottom of this section
            st.markdown("---")
            with st.expander("📋 Full Metadata", expanded=False):
                # Show actual capability values (True/False), not just keys
                capabilities_dict = {}
                if registry_entry.capabilities:
                    # Only show capabilities that are True
                    for key, value in registry_entry.capabilities.items():
                        if value:  # Only include True capabilities
                            capabilities_dict[key] = value

                # Build comprehensive metadata dict
                metadata_dict = {
                    "name": registry_entry.name,
                    "display_name": registry_entry.display_name,
                    "backend": registry_entry.backend,
                    "backend_config": {
                        "port": registry_entry.backend_config.port,
                        "host": registry_entry.backend_config.host,
                        "model_path": registry_entry.backend_config.model_path,
                        "extra_args": registry_entry.backend_config.extra_args,
                    },
                    "download_format": registry_entry.download_format,
                    "quantization": registry_entry.quantization,
                    "download_path": registry_entry.download_path,
                    "download_size_bytes": registry_entry.download_size_bytes,
                    "roles": list(registry_entry.roles) if registry_entry.roles else [],
                    "capabilities": capabilities_dict,  # Only True values
                    "generation_defaults": {
                        "temperature": registry_entry.generation_defaults.temperature,
                        "top_p": registry_entry.generation_defaults.top_p,
                        "top_k": registry_entry.generation_defaults.top_k,
                        "max_tokens": registry_entry.generation_defaults.max_tokens,
                        "context_window": registry_entry.generation_defaults.context_window,
                        "frequency_penalty": registry_entry.generation_defaults.frequency_penalty,
                        "presence_penalty": registry_entry.generation_defaults.presence_penalty,
                        "stop_sequences": registry_entry.generation_defaults.stop_sequences,
                    },
                    "family": registry_entry.family,
                    "served_model_id": registry_entry.served_model_id,
                }
                st.json(metadata_dict)

        else:
            if not model_name:
                st.info(
                    "ℹ️ Select a model from the dropdown above to view/edit parameters"
                )
            else:
                st.warning(f"Model '{model_name}' not found in registry")

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

                # Save button for roles
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

        weight_options: List[str] = []
        support_repo_value = st.session_state.get("hf_support_repo", "").strip()

        # Initialize session state for analysis
        if "hf_analysis" not in st.session_state:
            st.session_state.hf_analysis = None
        if "hf_analyzed_model" not in st.session_state:
            st.session_state.hf_analyzed_model = None

        col1, col2, col3 = st.columns([3, 2, 1])

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

        with col3:
            st.write("")  # spacing
            st.write("")  # spacing
            analyze_btn = st.button(
                "🔍 Analyze",
                disabled=not model_input,
                key="hf_analyze_btn",
                use_container_width=True,
            )

        # Clear analysis if model changed
        if st.session_state.get("hf_analyzed_model") != model_input:
            st.session_state.hf_analysis = None
            st.session_state.hf_analyzed_model = None

        # Handle analyze button click
        if analyze_btn and model_input:
            full_model = f"{model_input}@{branch}" if branch else model_input
            with st.spinner("🔍 Analyzing repository..."):
                try:
                    result = analyze_hf_repository(full_model)
                    st.session_state.hf_analysis = result
                    st.session_state.hf_analyzed_model = model_input
                    st.success("✅ Analysis complete!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Analysis failed: {e}")
                    st.session_state.hf_analysis = None

        # Display analysis results
        if st.session_state.hf_analysis:
            result = st.session_state.hf_analysis
            analysis = result["analysis"]
            critical = result["critical_status"]
            warnings_list = result["warnings"]

            st.markdown("---")
            st.markdown("### 📊 Repository Analysis")

            # Repository info
            repo = analysis.repository
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Repository", f"{repo.owner}/{repo.repo}")
            with col2:
                if repo.model_type:
                    st.metric("Model Type", repo.model_type)
            with col3:
                if repo.library_name:
                    st.metric("Library", repo.library_name)

            # Available formats
            st.markdown("#### 🎯 Available Formats")
            if analysis.formats:
                for fmt in analysis.formats:
                    confidence = f"{fmt.confidence:.0%}"
                    quant_str = ""
                    if fmt.quantization_details:
                        details = ", ".join(
                            f"{k}={v}" for k, v in fmt.quantization_details.items()
                        )
                        quant_str = f" `[{details}]`"

                    # Color code by confidence
                    if fmt.confidence >= 0.8:
                        confidence_icon = "🟢"
                    elif fmt.confidence >= 0.5:
                        confidence_icon = "🟡"
                    else:
                        confidence_icon = "🔴"

                    st.markdown(
                        f"{confidence_icon} **{fmt.format_type.upper()}** ({confidence}){quant_str}"
                    )

                    # Show evidence in expander
                    if fmt.evidence:
                        with st.expander(
                            f"Evidence for {fmt.format_type}", expanded=False
                        ):
                            for evidence in fmt.evidence[:5]:
                                st.caption(f"• {evidence}")
                            if len(fmt.evidence) > 5:
                                st.caption(
                                    f"... and {len(fmt.evidence) - 5} more signals"
                                )
            else:
                st.warning("⚠️ No formats detected")

            # Critical files status
            st.markdown("#### 🔍 Critical Files Status")

            col1, col2 = st.columns([3, 1])

            with col1:
                # Config files
                if critical["config.json"] == "found":
                    st.success("✅ config.json")
                else:
                    st.error("❌ config.json - MISSING (REQUIRED)")

                if critical["tokenizer_config.json"] == "found":
                    st.success("✅ tokenizer_config.json")
                else:
                    st.warning("⚠️ tokenizer_config.json - Missing (recommended)")

                # Tokenizer
                if critical["tokenizer.json or tokenizer.model"] in [
                    "tokenizer.json",
                    "tokenizer.model",
                ]:
                    st.success(f"✅ {critical['tokenizer.json or tokenizer.model']}")
                else:
                    st.error("❌ Tokenizer file - MISSING (REQUIRED)")

                # Generation config
                if critical["generation_config.json"] == "found":
                    st.success("✅ generation_config.json")
                else:
                    st.warning(
                        "⚠️ generation_config.json - Missing (may affect generation)"
                    )

                # Chat template
                if critical["chat_template"] == "standalone":
                    st.success("✅ chat_template (standalone file)")
                elif critical["chat_template"] == "check_embedded":
                    st.info(
                        "ℹ️ chat_template - Not found as standalone (may be embedded in tokenizer_config)"
                    )
                else:
                    st.warning("⚠️ chat_template - Status unknown")

            with col2:
                # File counts
                files = analysis.files
                st.metric("Weights", len(files.get("model_weights", [])))
                st.metric("Config", len(files.get("config", [])))
                st.metric("Tokenizer", len(files.get("tokenizer", [])))

            # Size breakdown
            st.markdown("#### 💾 Download Size")

            required_size = sum(
                f.size
                for f in files.get("model_weights", [])
                + files.get("config", [])
                + files.get("tokenizer", [])
            )
            optional_size = sum(f.size for f in files.get("optional", []))

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Required Files", f"{required_size / (1024**3):.2f} GB")
            with col2:
                st.metric("Optional Files", f"{optional_size / (1024**2):.2f} MB")

            # Weight variants selection
            weight_files = (
                analysis.files.get("model_weights", []) if analysis.files else []
            )
            weight_options = [f.path for f in weight_files]
            repo_key = f"{analysis.repository.owner}/{analysis.repository.repo}@{analysis.repository.branch}"
            if st.session_state.get("hf_weight_variants_repo") != repo_key:
                st.session_state["hf_weight_variants_repo"] = repo_key
                st.session_state["hf_weight_variants_state"] = (
                    weight_options[:1] if weight_options else []
                )
            st.session_state.setdefault(
                "hf_weight_variants_state",
                weight_options[:1] if weight_options else [],
            )

            if weight_files:
                st.markdown("#### 🎚️ Weight Variants")
                st.caption(
                    "Select specific quantized weights to download. Leave empty to download all variants."
                )
                weight_labels = {
                    option: f"{Path(option).name} ({_format_bytes(file_info.size)})"
                    + (
                        f" • {_guess_quant_label(Path(option).name)}"
                        if _guess_quant_label(Path(option).name)
                        else ""
                    )
                    for option, file_info in zip(weight_options, weight_files)
                }
                col_weights, col_weight_actions = st.columns([3, 1])
                with col_weights:
                    st.multiselect(
                        "Weight files",
                        options=weight_options,
                        key="hf_weight_variants_state",
                        format_func=lambda opt: weight_labels.get(opt, opt),
                    )
                with col_weight_actions:
                    st.write("")
                    if st.button("Select all", key="hf_weight_variants_all"):
                        st.session_state["hf_weight_variants_state"] = list(
                            weight_options
                        )
                        st.rerun()
                    if st.button("Select none", key="hf_weight_variants_none"):
                        st.session_state["hf_weight_variants_state"] = []
                        st.rerun()
            else:
                st.info("No weight files detected in the repository analysis.")

            st.markdown("#### 🧩 Support Repository (Optional)")
            st.caption(
                "Specify the original repository containing config/tokenizer files if this quantization repo omits them."
            )
            support_repo_value = st.text_input(
                "Support repository",
                value=st.session_state.get("hf_support_repo", ""),
                key="hf_support_repo",
                placeholder="owner/repo or owner/repo@branch",
            ).strip()
            support_repo_value = st.session_state.get("hf_support_repo", "").strip()

            # Warnings
            if warnings_list:
                st.markdown("#### ⚠️ Warnings")
                for warning in warnings_list:
                    if warning.startswith("❌"):
                        st.error(warning)
                    else:
                        st.warning(warning)

            st.markdown("---")

        selected_weights = st.session_state.get("hf_weight_variants_state", [])

        # Download options
        col1, col2 = st.columns(2)

        with col1:
            # Suggest format from analysis if available
            default_format = ["awq"]
            if st.session_state.hf_analysis:
                analysis = st.session_state.hf_analysis["analysis"]
                if analysis.formats:
                    suggested = analysis.formats[0].format_type
                    if suggested in ["gguf", "awq", "gptq", "safetensors"]:
                        default_format = [suggested]

            formats = st.multiselect(
                "Preferred Formats",
                options=["gguf", "awq", "gptq", "safetensors"],
                default=default_format,
                key="hf_formats",
                help="Format suggested from analysis if available",
            )

            location = st.selectbox(
                "Location",
                options=["linux_wsl", "windows_lmstudio", "custom"],
                key="hf_location",
            )

        with col2:
            include_optional = st.checkbox(
                "Include optional files", value=True, key="hf_optional"
            )
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
            if weight_options:
                if selected_weights and 0 < len(selected_weights) < len(weight_options):
                    cmd_parts.extend(["--weights", ",".join(selected_weights)])
            if support_repo_value:
                cmd_parts.extend(["--support-repo", support_repo_value])
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
            if weight_options:
                if selected_weights and 0 < len(selected_weights) < len(weight_options):
                    command.extend(["--weights", ",".join(selected_weights)])
            if support_repo_value:
                command.extend(["--support-repo", support_repo_value])
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

    st.markdown("#### ♻️ Drop & Rebuild Shortcuts")
    st.caption(
        "Hard reset the discovered layer for a provider, then repopulate it using the "
        "adapter-backed tooling. Run a dry-run first to understand the impact."
    )

    col_hf, col_ollama = st.columns(2)

    # --- Hugging Face rebuild ---
    with col_hf:
        st.markdown("##### Hugging Face (local weights)")
        hf_root_default = str(Path.home() / "ai-models" / "weights")
        hf_root = st.text_input(
            "Weights root",
            value=hf_root_default,
            key="rebuild_hf_root",
            help="Root directory containing <owner>/<repo> folders downloaded from Hugging Face.",
        )
        hf_backend = st.selectbox(
            "Assign backend",
            options=["vllm", "ollama"],
            index=0,
            key="rebuild_hf_backend",
            help="Backend to associate with the regenerated entries.",
        )
        hf_backup = st.checkbox(
            "Backup discovered layer before purge",
            value=True,
            key="rebuild_hf_backup",
        )
        hf_dry = st.checkbox(
            "Dry run only (no writes)", value=True, key="rebuild_hf_dry"
        )

        purge_hf_cmd = [
            "uv",
            "run",
            "imageworks-loader",
            "purge-imported",
            "--providers",
            "hf",
        ]
        if hf_dry:
            purge_hf_cmd.append("--dry-run")
        else:
            purge_hf_cmd.append("--apply")
        if not hf_backup:
            purge_hf_cmd.append("--no-backup")

        ingest_hf_cmd = [
            "uv",
            "run",
            "imageworks-loader",
            "ingest-local-hf",
            "--root",
            hf_root,
            "--backend",
            hf_backend,
        ]
        if hf_dry:
            ingest_hf_cmd.append("--dry-run")

        with st.expander("🔍 Command Preview (HF)", expanded=False):
            preview = "\n\n".join(
                [
                    " ".join(shlex.quote(part) for part in purge_hf_cmd),
                    " ".join(shlex.quote(part) for part in ingest_hf_cmd),
                ]
            )
            st.code(preview, language="bash")

        hf_confirmed = True
        if not hf_dry:
            hf_confirmed = confirm_destructive_operation(
                "Drop & Rebuild Hugging Face entries",
                (
                    "This will remove all discovered Hugging Face records, then "
                    f"re-ingest models from `{hf_root}` using the `{hf_backend}` backend."
                ),
            )

        if st.button(
            "♻️ Drop & Rebuild HF",
            type="primary",
            key="rebuild_hf_btn",
            disabled=not hf_confirmed,
        ):
            commands = [
                (purge_hf_cmd, "Purging discovered Hugging Face entries"),
                (ingest_hf_cmd, "Re-ingesting local Hugging Face models"),
            ]
            overall_success = True
            for cmd, description in commands:
                result = run_command_with_progress(cmd, description, timeout=900)
                if not result["success"]:
                    overall_success = False
                    break
            if overall_success and not hf_dry:
                load_model_registry(force=True)
                st.success("Hugging Face registry entries rebuilt.")

    # --- Ollama rebuild ---
    with col_ollama:
        st.markdown("##### Ollama (container service)")

        discovered_summary = summarize_discovered_layer() or {}
        ollama_count = discovered_summary.get("ollama", 0)
        hf_count = discovered_summary.get("hf", 0)
        other_count = discovered_summary.get("other", 0)
        if discovered_summary:
            keep_count = hf_count + other_count
            message = (
                f"Discovered layer currently tracks **{ollama_count}** Ollama entr"
                f"{'y' if ollama_count == 1 else 'ies'}."
            )
            if keep_count:
                message += (
                    f" {keep_count} other entries (HF/local={hf_count}, other={other_count}) "
                    "remain untouched — shown as keep=N in the purge dry run."
                )
            else:
                message += " Purge dry runs will report keep=0 for other providers."
            st.info(message)

        ollama_location = st.text_input(
            "Location label",
            value="linux_wsl",
            key="rebuild_ollama_location",
            help="Populates the location field on imported entries.",
        )
        ollama_backup = st.checkbox(
            "Backup discovered layer before purge",
            value=True,
            key="rebuild_ollama_backup",
        )
        ollama_show = st.checkbox(
            "Show imported list after rebuild",
            value=False,
            key="rebuild_ollama_show",
        )
        ollama_dry = st.checkbox(
            "Dry run only (no writes)", value=True, key="rebuild_ollama_dry"
        )

        purge_ollama_cmd = [
            "uv",
            "run",
            "imageworks-loader",
            "purge-imported",
            "--providers",
            "ollama",
        ]
        if ollama_dry:
            purge_ollama_cmd.append("--dry-run")
        else:
            purge_ollama_cmd.append("--apply")
        if not ollama_backup:
            purge_ollama_cmd.append("--no-backup")

        rebuild_ollama_cmd = [
            "uv",
            "run",
            "imageworks-loader",
            "rebuild-ollama",
            "--location",
            ollama_location,
        ]
        if ollama_dry:
            rebuild_ollama_cmd.append("--dry-run")
        if not ollama_show:
            rebuild_ollama_cmd.append("--no-show")

        with st.expander("🔍 Command Preview (Ollama)", expanded=False):
            preview = "\n\n".join(
                [
                    " ".join(shlex.quote(part) for part in purge_ollama_cmd),
                    " ".join(shlex.quote(part) for part in rebuild_ollama_cmd),
                ]
            )
            st.code(preview, language="bash")

        ollama_confirmed = True
        if not ollama_dry:
            ollama_confirmed = confirm_destructive_operation(
                "Drop & Rebuild Ollama entries",
                (
                    "This will remove all discovered Ollama entries and repopulate them "
                    f"by querying the running Ollama service (location `{ollama_location}`)."
                ),
            )

        if st.button(
            "♻️ Drop & Rebuild Ollama",
            type="primary",
            key="rebuild_ollama_btn",
            disabled=not ollama_confirmed,
        ):
            commands = [(purge_ollama_cmd, "Purging discovered Ollama entries")]
            if ollama_dry:
                preview_cmd = [
                    "uv",
                    "run",
                    "python",
                    "scripts/import_ollama_models.py",
                    "--dry-run",
                    "--location",
                    ollama_location,
                ]
                commands.append(
                    (preview_cmd, "Dry-run: querying Ollama for import preview")
                )
            else:
                commands.append((rebuild_ollama_cmd, "Re-importing models from Ollama"))

            overall_success = True
            for idx, (cmd, description) in enumerate(commands):
                result = run_command_with_progress(cmd, description, timeout=900)
                if not result["success"]:
                    overall_success = False
                    break
                if ollama_dry and idx == 0 and discovered_summary:
                    keep_count = hf_count + other_count
                    st.caption(
                        f"Dry-run summary: would remove {ollama_count} Ollama entries "
                        f"from the discovered layer and keep {keep_count} other entries "
                        f"(HF/local={hf_count}, other={other_count})."
                    )
            if overall_success and not ollama_dry:
                load_model_registry(force=True)
                st.success("Ollama registry entries rebuilt.")

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
    st.info(
        "ℹ️ Use the controls below to restart the bundled chat proxy service or review manual commands."
    )

    with st.expander("♻️ Restart Chat Proxy", expanded=False):
        st.markdown(
            "Restarting the proxy frees GPU memory and reloads the latest registry metadata. "
            "Existing downloads or Ollama pulls continue running."
        )
        confirm_restart = st.checkbox(
            "I understand the API will briefly go offline during the restart",
            key="restart_proxy_confirm",
        )
        if st.button(
            "🔄 Restart Chat Proxy",
            type="primary",
            key="restart_proxy_btn",
            disabled=not confirm_restart,
        ):
            run_command_with_progress(
                ["docker", "restart", "imageworks-chat-proxy"],
                "Restarting chat proxy",
                timeout=120,
            )

    # Show commands for manual starting
    with st.expander("📝 Manual Start Commands", expanded=False):
        st.code(
            """
# Start vLLM (local process)
uv run vllm serve <model_name> --port 24001

# Start LMDeploy (local process)
uv run lmdeploy serve api <model_name> --port 24001

# Start Ollama (docker compose service)
docker compose -f docker-compose.chat-proxy.yml up -d imageworks-ollama

# Open Ollama shell inside the container
docker exec -it imageworks-ollama ollama ps

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

        show_installed_only = st.checkbox(
            "Show installed variants only",
            value=False,
            key="remove_show_installed_only",
            help="Filters to entries whose download directory exists on disk.",
        )

        filtered_items = [
            (name, entry)
            for name, entry in registry.items()
            if not show_installed_only
            or (entry.download_path and Path(entry.download_path).exists())
        ]
        model_names = [name for name, _ in sorted(filtered_items, key=lambda x: x[0])]

        if not model_names:
            st.warning(
                "No models match the current filter. Disable the installer-only toggle to view logical entries."
            )
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

                    # Reset selection & confirmation toggles to avoid accidental repeat deletions
                    st.session_state.pop("remove_variant", None)
                    st.session_state["removal_mode"] = (
                        "Metadata only (keep files & logical entry)"
                    )
                    if delete_files:
                        confirm_key = (
                            "confirm_purge_model" if purge else "confirm_delete_files"
                        )
                        st.session_state.pop(confirm_key, None)
                    else:
                        st.session_state.pop("confirm_metadata", None)
                    # Also clear the generic destructive confirmation cache key if present
                    st.session_state.pop("confirm_purge_model", None)
                    st.session_state.pop("confirm_delete_files", None)
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
    st.set_page_config(layout="wide")
    init_session_state()

    # Apply wide layout CSS (consistent with global settings)
    st.markdown(
        """
        <style>
        /* Force wider main content area - consistent with app.py */
        .main .block-container {
            max-width: 95% !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }

        /* Remove Streamlit's default max-width constraint */
        section.main > div {
            max-width: none !important;
        }

        /* Make dataframes use full available width */
        .stDataFrame {
            width: 100% !important;
        }

        /* Ensure tables don't get cut off */
        div[data-testid="stDataFrame"] > div {
            width: 100% !important;
            overflow-x: auto !important;
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
