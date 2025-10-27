"""Personal Tagger page with preview and edit workflow."""

import streamlit as st
from pathlib import Path

from imageworks.gui.state import init_session_state
from imageworks.gui.presets import PERSONAL_TAGGER_PRESETS
from imageworks.gui.components.preset_selector import render_preset_selector
from imageworks.gui.components.process_runner import (
    render_process_runner,
    execute_command,
)
from imageworks.gui.components.metadata_editor import (
    render_metadata_editor,
    render_compact_tag_list,
)
from imageworks.gui.components.results_viewer import parse_jsonl
from imageworks.gui.utils.cli_wrapper import build_tagger_command
from imageworks.gui.config import (
    DEFAULT_INPUT_DIR,
    TAGGER_DEFAULT_OUTPUT_JSONL,
    TAGGER_DEFAULT_SUMMARY_PATH,
    get_app_setting,
    set_app_setting,
    reset_app_settings,
)
from imageworks.model_loader.role_selection import list_models_for_role, select_by_role


def render_custom_overrides(preset, session_key_prefix):
    """Custom override renderer for personal tagger."""
    overrides = {}

    # Reset button
    col_reset, col_spacer = st.columns([1, 3])
    with col_reset:
        if st.button(
            "🔄 Reset to Defaults",
            key=f"{session_key_prefix}_reset",
            help="Reset all paths to global defaults",
        ):
            reset_app_settings(st.session_state, "tagger")
            st.success("✅ Reset to defaults")
            st.rerun()

    # Input directory
    current_input = get_app_setting(
        st.session_state, "tagger", "input_dir", DEFAULT_INPUT_DIR
    )
    input_dir = st.text_input(
        "Input Directory",
        value=current_input,
        key=f"{session_key_prefix}_input",
        help="Directory containing images to tag",
    )
    if input_dir and input_dir != current_input:
        set_app_setting(st.session_state, "tagger", "input_dir", input_dir)
    if input_dir:
        overrides["input"] = [input_dir]

    # Output paths
    col1, col2 = st.columns(2)

    with col1:
        current_jsonl = get_app_setting(
            st.session_state, "tagger", "output_jsonl", str(TAGGER_DEFAULT_OUTPUT_JSONL)
        )
        output_jsonl = st.text_input(
            "Output JSONL",
            value=current_jsonl,
            key=f"{session_key_prefix}_output_jsonl",
        )
        if output_jsonl and output_jsonl != current_jsonl:
            set_app_setting(st.session_state, "tagger", "output_jsonl", output_jsonl)
        if output_jsonl:
            overrides["output_jsonl"] = output_jsonl

    with col2:
        current_summary = get_app_setting(
            st.session_state, "tagger", "summary_path", str(TAGGER_DEFAULT_SUMMARY_PATH)
        )
        summary_path = st.text_input(
            "Summary Markdown",
            value=current_summary,
            key=f"{session_key_prefix}_summary",
        )
        if summary_path and summary_path != current_summary:
            set_app_setting(st.session_state, "tagger", "summary_path", summary_path)
        if summary_path:
            overrides["summary"] = summary_path

    # Role toggles (if in common overrides)
    if any(
        role in preset.common_overrides
        for role in ["caption_role", "keyword_role", "description_role"]
    ):
        st.markdown("#### Tag Types")
        col1, col2, col3 = st.columns(3)

        with col1:
            enable_caption = st.checkbox(
                "Caption",
                value=preset.flags.get("caption_role") is not None,
                key=f"{session_key_prefix}_enable_caption",
            )
            if enable_caption:
                overrides["caption_role"] = "caption"
            else:
                overrides["caption_role"] = None

        with col2:
            enable_keywords = st.checkbox(
                "Keywords",
                value=preset.flags.get("keyword_role") is not None,
                key=f"{session_key_prefix}_enable_keywords",
            )
            if enable_keywords:
                overrides["keyword_role"] = "keywords"
            else:
                overrides["keyword_role"] = None

        with col3:
            enable_description = st.checkbox(
                "Description",
                value=preset.flags.get("description_role") is not None,
                key=f"{session_key_prefix}_enable_description",
            )
            if enable_description:
                overrides["description_role"] = "description"
            else:
                overrides["description_role"] = None

    # Dry run toggle
    dry_run = st.checkbox(
        "Dry Run (preview only)",
        value=preset.flags.get("dry_run", True),
        key=f"{session_key_prefix}_dry_run_main",
    )
    overrides["dry_run"] = dry_run

    # Advanced VLM/Backend Settings
    with st.expander("🔧 Advanced Settings (VLM/Backend)", expanded=False):
        st.markdown("##### Model Selection Mode")

        # Mode selection
        use_registry = st.checkbox(
            "Use Registry (Auto)",
            value=preset.flags.get("use_registry", True),
            key=f"{session_key_prefix}_use_registry",
            help="Automatically select models by role from registry. Uncheck for manual backend/model selection.",
        )
        overrides["use_registry"] = use_registry

        if use_registry:
            # AUTO MODE: Show what models will be used for each role
            st.markdown("**Registry-based model selection enabled**")
            st.info(
                "Models will be automatically selected from the registry based on role priority scores. "
                "The system will choose the highest-priority model for each task."
            )

            # Show which specific models will be selected
            st.markdown("**Models that will be used:**")
            try:
                if overrides.get("caption_role"):
                    caption_model = select_by_role(overrides["caption_role"])
                    st.markdown(f"- 📝 **Caption**: `{caption_model}`")
                if overrides.get("keyword_role"):
                    keyword_model = select_by_role(overrides["keyword_role"])
                    st.markdown(f"- 🏷️ **Keywords**: `{keyword_model}`")
                if overrides.get("description_role"):
                    description_model = select_by_role(overrides["description_role"])
                    st.markdown(f"- 📄 **Description**: `{description_model}`")

                if not any(
                    [
                        overrides.get("caption_role"),
                        overrides.get("keyword_role"),
                        overrides.get("description_role"),
                    ]
                ):
                    st.warning("No roles enabled. Enable at least one tag type above.")
            except Exception as e:
                st.error(f"Error loading models from registry: {e}")

        else:
            # MANUAL MODE: Full backend/model control
            st.markdown("**Manual backend/model configuration**")

            # Backend selection
            backend = st.selectbox(
                "Backend",
                options=["ollama", "vllm", "lmdeploy"],
                index=0,  # Default to ollama
                key=f"{session_key_prefix}_backend",
                help="Inference backend to use",
            )
            overrides["backend"] = backend

            # Base URL based on backend
            default_urls = {
                "ollama": "http://localhost:11434/v1",
                "vllm": "http://localhost:8000/v1",
                "lmdeploy": "http://localhost:23333/v1",
            }
            base_url = st.text_input(
                "Base URL",
                value=default_urls.get(backend, "http://localhost:11434/v1"),
                key=f"{session_key_prefix}_base_url",
                help="API endpoint for the selected backend",
            )
            overrides["base_url"] = base_url

            # Model selection
            st.markdown("**Model Selection**")

            # Get available models for each role
            try:
                caption_models = (
                    list_models_for_role("caption")
                    if overrides.get("caption_role")
                    else []
                )
                keyword_models = (
                    list_models_for_role("keywords")
                    if overrides.get("keyword_role")
                    else []
                )
                description_models = (
                    list_models_for_role("description")
                    if overrides.get("description_role")
                    else []
                )

                # Check if all roles can use same models
                all_models = (
                    set(caption_models) & set(keyword_models) & set(description_models)
                    if all([caption_models, keyword_models, description_models])
                    else []
                )

            except Exception as e:
                st.error(f"Error loading available models: {e}")
                caption_models = keyword_models = description_models = []
                all_models = []

            use_same_model = st.checkbox(
                "Use same model for all roles",
                value=True,
                key=f"{session_key_prefix}_use_same_model",
                help="Select one model for all enabled roles (model must support all enabled roles)",
            )

            if use_same_model:
                # Show models that support ALL enabled roles
                if all_models:
                    model = st.selectbox(
                        "Model",
                        options=sorted(all_models),
                        index=0,
                        key=f"{session_key_prefix}_model",
                        help="Model must support all enabled roles",
                    )
                    overrides["model"] = model
                else:
                    st.warning(
                        "No models support all enabled roles. Use per-role selection or enable fewer roles."
                    )
                    # Fallback to text input
                    model = st.text_input(
                        "Model (manual entry)",
                        value="qwen2.5vl:7b",
                        key=f"{session_key_prefix}_model_manual",
                        help="Manually specify model name",
                    )
                    overrides["model"] = model
            else:
                # Per-role model selection with dropdowns
                col1, col2, col3 = st.columns(3)

                with col1:
                    if overrides.get("caption_role") and caption_models:
                        caption_model = st.selectbox(
                            "Caption Model",
                            options=sorted(caption_models),
                            index=0,
                            key=f"{session_key_prefix}_caption_model",
                            help="Models capable of caption generation",
                        )
                        overrides["caption_model"] = caption_model

                with col2:
                    if overrides.get("keyword_role") and keyword_models:
                        keyword_model = st.selectbox(
                            "Keyword Model",
                            options=sorted(keyword_models),
                            index=0,
                            key=f"{session_key_prefix}_keyword_model",
                            help="Models capable of keyword extraction",
                        )
                        overrides["keyword_model"] = keyword_model

                with col3:
                    if overrides.get("description_role") and description_models:
                        description_model = st.selectbox(
                            "Description Model",
                            options=sorted(description_models),
                            index=0,
                            key=f"{session_key_prefix}_description_model",
                            help="Models capable of description generation",
                        )
                        overrides["description_model"] = description_model

    return overrides


def main():
    """Personal tagger page."""
    st.set_page_config(layout="wide")
    init_session_state()

    # Apply wide layout CSS (ensures consistency on page refresh)
    st.markdown(
        """
        <style>
        .main .block-container {
            max-width: 95% !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }
        section.main > div {
            max-width: none !important;
        }
        .stDataFrame {
            width: 100% !important;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    st.title("🏷️ Personal Tagger")
    st.markdown("Generate captions, keywords, and descriptions for your images")

    # Workflow tabs
    tabs = st.tabs(["⚙️ Configure", "▶️ Preview", "✏️ Edit", "✅ Commit"])

    # === CONFIGURE TAB ===
    with tabs[0]:
        st.markdown("### Configuration")

        # Render preset selector
        config = render_preset_selector(
            PERSONAL_TAGGER_PRESETS,
            session_key_prefix="tagger",
            custom_override_renderer=render_custom_overrides,
        )

        # Validate
        if not config.get("input"):
            st.warning("⚠️ Please specify an input directory")
        else:
            input_path = Path(config["input"][0])
            if not input_path.exists():
                st.error(f"❌ Directory does not exist: {input_path}")

        # Check if any roles are enabled
        has_roles = any(
            config.get(role)
            for role in ["caption_role", "keyword_role", "description_role"]
        )
        if not has_roles:
            st.warning(
                "⚠️ No tag types enabled. Please enable at least one: Caption, Keywords, or Description"
            )

        # Configuration summary
        with st.expander("📋 Configuration Summary", expanded=False):
            st.json(config)

    # === PREVIEW TAB ===
    with tabs[1]:
        st.markdown("### Generate Tag Preview")

        config = st.session_state.get("tagger_config", {})

        if not config.get("input"):
            st.error("❌ No input specified. Configure in the Configure tab.")
        else:
            st.markdown(f"**Input:** {config['input'][0]}")
            st.markdown(
                f"**Preset:** {st.session_state.get('tagger_preset_name', 'unknown')}"
            )

            # Ensure dry-run is enabled for preview
            preview_config = config.copy()
            preview_config["dry_run"] = True

            # Render process runner
            result = render_process_runner(
                button_label="🔍 Generate Preview (Dry Run)",
                command_builder=build_tagger_command,
                config=preview_config,
                key_prefix="tagger_preview",
                result_key="tagger_preview_results",
                show_command=True,
                timeout=1800,  # 30 minutes
            )

            if result and result.get("success"):
                st.success("✅ Preview generated!")

                # Parse results
                output_jsonl = preview_config.get("output_jsonl")
                if output_jsonl and Path(output_jsonl).exists():
                    results = parse_jsonl(output_jsonl)

                    if results:
                        # Store in session state for editing
                        st.session_state["tagger_preview_data"] = results

                        st.info(
                            f"Generated tags for {len(results)} images. Switch to Edit tab to review."
                        )

                        # Show sample
                        st.markdown("#### Sample Result")
                        st.json(results[0])

    # === EDIT TAB ===
    with tabs[2]:
        st.markdown("### Edit Tags")

        # Get preview data
        preview_data = st.session_state.get("tagger_preview_data", [])

        if not preview_data:
            st.info("Generate a preview first to edit tags")
        else:
            # Prepare data for editor
            images_with_tags = []

            for result in preview_data:
                img_data = {
                    "path": result.get("image_path", ""),
                    "caption": result.get("caption", ""),
                    "keywords": result.get("keywords", []),
                    "description": result.get("description", ""),
                    "approved": True,  # Default to approved
                }
                images_with_tags.append(img_data)

            # Render metadata editor
            edited_data = render_metadata_editor(
                images_with_tags, key_prefix="tagger_editor"
            )

            # Update session state
            st.session_state["tagger_edited_data"] = edited_data

            # Show summary
            st.markdown("---")
            render_compact_tag_list(edited_data, show_approved_only=False)

    # === COMMIT TAB ===
    with tabs[3]:
        st.markdown("### Write Tags to Files")

        edited_data = st.session_state.get("tagger_edited_data", [])

        if not edited_data:
            st.info("Edit tags first before committing")
        else:
            # Show what will be written
            approved_images = [img for img in edited_data if img.get("approved", True)]

            st.markdown("#### Ready to Write")
            st.write(f"**{len(approved_images)}** images approved for tagging")
            st.write(f"**{len(edited_data) - len(approved_images)}** images rejected")

            # Summary of approved tags
            render_compact_tag_list(edited_data, show_approved_only=True)

            st.markdown("---")

            # Commit button
            col1, col2 = st.columns([1, 1])

            with col1:
                if st.button(
                    "✅ Write to Metadata", type="primary", use_container_width=True
                ):
                    if not approved_images:
                        st.error("No images approved for writing")
                    else:
                        # Build modified config with dry_run=False
                        config = st.session_state.get("tagger_config", {}).copy()
                        config["dry_run"] = False

                        # TODO: Filter input to only approved images
                        # For now, we'll write all and rely on the tool's behavior

                        with st.spinner("Writing metadata..."):
                            command = build_tagger_command(config)
                            result = execute_command(command, timeout=1800)

                            if result["success"]:
                                st.success(
                                    f"✅ Successfully wrote tags to {len(approved_images)} images!"
                                )
                                st.text(result["stdout"])

                                # Clear preview data
                                st.session_state["tagger_preview_data"] = []
                                st.session_state["tagger_edited_data"] = []
                            else:
                                st.error("❌ Failed to write metadata")
                                st.text(result["stderr"])

            with col2:
                if st.button("🗑️ Clear Preview", use_container_width=True):
                    st.session_state["tagger_preview_data"] = []
                    st.session_state["tagger_edited_data"] = []
                    st.success("Preview cleared. Generate new preview in Preview tab.")
                    st.rerun()


if __name__ == "__main__":
    main()
