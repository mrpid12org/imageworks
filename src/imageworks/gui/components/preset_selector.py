"""Preset selector component."""

import streamlit as st
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field


@dataclass
class PresetConfig:
    """Single preset configuration."""

    name: str
    description: str
    flags: Dict[str, Any]
    hidden_flags: List[str] = field(default_factory=list)
    common_overrides: List[str] = field(default_factory=list)


@dataclass
class ModulePresets:
    """All presets for a module."""

    module_name: str
    default_preset: str
    presets: Dict[str, PresetConfig]

    def get_preset(self, name: str) -> PresetConfig:
        """Get preset by name."""
        return self.presets[name]

    def get_flags_with_overrides(
        self, preset_name: str, overrides: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge preset flags with user overrides."""
        preset = self.get_preset(preset_name)
        flags = preset.flags.copy()
        flags.update(overrides)
        return flags


def _render_flag_widget(
    flag_name: str,
    default_value: Any,
    key: str,
    help_text: Optional[str] = None,
) -> Any:
    """
    Render appropriate widget for a flag based on its type.

    Args:
        flag_name: Name of the flag
        default_value: Default value (determines widget type)
        key: Widget key
        help_text: Optional help text

    Returns:
        Widget value
    """
    label = flag_name.replace("_", " ").title()

    if isinstance(default_value, bool):
        return st.checkbox(label, value=default_value, key=key, help=help_text)

    elif isinstance(default_value, int):
        return st.number_input(label, value=default_value, key=key, help=help_text)

    elif isinstance(default_value, float):
        # Check if it's a threshold (0-1 range)
        if 0 <= default_value <= 1:
            return st.slider(
                label,
                min_value=0.0,
                max_value=1.0,
                value=default_value,
                step=0.01,
                key=key,
                help=help_text,
            )
        else:
            return st.number_input(label, value=default_value, key=key, help=help_text)

    elif isinstance(default_value, list):
        # For lists, show as multiselect or text area
        if len(default_value) > 0 and isinstance(default_value[0], str):
            # String list - show as multiselect if options are known
            return st.text_area(
                label,
                value="\n".join(default_value),
                key=key,
                help=help_text or "One item per line",
            ).split("\n")
        else:
            return st.text_area(
                label, value=str(default_value), key=key, help=help_text
            )

    elif isinstance(default_value, str):
        # Check if it's a path
        if "path" in flag_name.lower() or "dir" in flag_name.lower():
            return st.text_input(label, value=default_value, key=key, help=help_text)
        # Check if it's a selection from known options
        elif flag_name in ["backend", "embedding_backend", "similarity_metric"]:
            options = {
                "backend": ["vllm", "lmdeploy", "ollama", "triton"],
                "embedding_backend": ["simple", "open_clip", "siglip", "remote"],
                "similarity_metric": ["cosine", "euclidean", "manhattan"],
            }
            return st.selectbox(
                label,
                options=options.get(flag_name, [default_value]),
                index=(
                    options.get(flag_name, [default_value]).index(default_value)
                    if default_value in options.get(flag_name, [])
                    else 0
                ),
                key=key,
                help=help_text,
            )
        else:
            return st.text_input(label, value=default_value, key=key, help=help_text)

    else:
        # Fallback to text input
        return st.text_input(label, value=str(default_value), key=key, help=help_text)


def render_preset_selector(
    module_presets: ModulePresets,
    session_key_prefix: str,
    custom_override_renderer: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Render preset selector with common overrides and advanced expander.

    Args:
        module_presets: Module preset configuration
        session_key_prefix: Prefix for session state keys
        custom_override_renderer: Optional function to render custom overrides

    Returns:
        Dict of all flags (preset + overrides + advanced)
    """

    # Preset selection
    st.subheader("⚙️ Configuration")

    preset_names = list(module_presets.presets.keys())
    default_index = (
        preset_names.index(module_presets.default_preset)
        if module_presets.default_preset in preset_names
        else 0
    )

    preset_name = st.radio(
        "Preset",
        options=preset_names,
        index=default_index,
        format_func=lambda x: module_presets.presets[x].name,
        key=f"{session_key_prefix}_preset",
        horizontal=True,
    )

    preset = module_presets.get_preset(preset_name)
    st.info(f"ℹ️ {preset.description}")

    # Initialize overrides dict
    overrides = {}

    # Common overrides section
    if preset.common_overrides:
        st.markdown("### Common Settings")

        # Use custom renderer if provided
        if custom_override_renderer:
            custom_overrides = custom_override_renderer(preset, session_key_prefix)
            overrides.update(custom_overrides)
        else:
            # Default rendering
            for flag_name in preset.common_overrides:
                if flag_name in preset.flags:
                    override_value = _render_flag_widget(
                        flag_name,
                        preset.flags[flag_name],
                        key=f"{session_key_prefix}_override_{flag_name}",
                    )
                    if override_value != preset.flags[flag_name]:
                        overrides[flag_name] = override_value

    # Advanced options expander
    advanced_flags = {
        k: v
        for k, v in preset.flags.items()
        if k not in preset.common_overrides and k not in preset.hidden_flags
    }

    if advanced_flags:
        with st.expander("🔧 Advanced Options", expanded=False):
            st.markdown("⚠️ **Expert settings** - modify with caution")

            for flag_name, flag_value in advanced_flags.items():
                advanced_value = _render_flag_widget(
                    flag_name,
                    flag_value,
                    key=f"{session_key_prefix}_advanced_{flag_name}",
                )
                if advanced_value != flag_value:
                    overrides[flag_name] = advanced_value

    # Get final configuration
    final_config = module_presets.get_flags_with_overrides(preset_name, overrides)

    # Store in session state
    if f"{session_key_prefix}_config" not in st.session_state:
        st.session_state[f"{session_key_prefix}_config"] = {}

    st.session_state[f"{session_key_prefix}_config"] = final_config
    st.session_state[f"{session_key_prefix}_preset_name"] = preset_name

    return final_config
