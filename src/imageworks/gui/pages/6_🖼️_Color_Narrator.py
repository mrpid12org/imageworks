"""Color Narrator page with pipeline mode."""

import streamlit as st
from pathlib import Path

from imageworks.gui.state import init_session_state
from imageworks.gui.presets import COLOR_NARRATOR_PRESETS
from imageworks.gui.components.preset_selector import render_preset_selector
from imageworks.gui.components.process_runner import render_process_runner
from imageworks.gui.components.results_viewer import (
    render_unified_results_browser,
    parse_jsonl,
)
from imageworks.gui.components.image_viewer import render_image_detail
from imageworks.gui.utils.cli_wrapper import build_narrator_command
from imageworks.gui.config import (
    OUTPUTS_DIR,
    MONO_DEFAULT_OUTPUT_JSONL,
    MONO_DEFAULT_OVERLAYS_DIR,
    get_app_setting,
    set_app_setting,
    reset_app_settings,
)


def render_custom_overrides(preset, session_key_prefix):
    """Custom override renderer for color narrator."""
    overrides = {}

    # Reset button
    col_reset, col_spacer = st.columns([1, 3])
    with col_reset:
        if st.button(
            "🔄 Reset to Defaults",
            key=f"{session_key_prefix}_reset",
            help="Reset all paths to global defaults",
        ):
            reset_app_settings(st.session_state, "narrator")
            st.success("✅ Reset to defaults")
            st.rerun()

    # Import mono results option
    use_mono_results = st.checkbox(
        "📊 Import from Mono Checker Results",
        value=False,
        key=f"{session_key_prefix}_use_mono",
        help="Automatically import images from mono checker output",
    )

    if use_mono_results:
        # Mono JSONL path - use per-app setting
        current_mono_jsonl = get_app_setting(
            st.session_state, "narrator", "mono_jsonl", str(MONO_DEFAULT_OUTPUT_JSONL)
        )
        mono_jsonl = st.text_input(
            "Mono Results JSONL",
            value=current_mono_jsonl,
            key=f"{session_key_prefix}_mono_jsonl",
        )
        if mono_jsonl and mono_jsonl != current_mono_jsonl:
            set_app_setting(st.session_state, "narrator", "mono_jsonl", mono_jsonl)
        if mono_jsonl:
            overrides["mono_jsonl"] = mono_jsonl

        # Overlays directory - use per-app setting
        current_overlays = get_app_setting(
            st.session_state, "narrator", "overlays_dir", str(MONO_DEFAULT_OVERLAYS_DIR)
        )
        overlays_dir = st.text_input(
            "Overlays Directory",
            value=current_overlays,
            key=f"{session_key_prefix}_overlays",
        )
        if overlays_dir and overlays_dir != current_overlays:
            set_app_setting(st.session_state, "narrator", "overlays_dir", overlays_dir)
        if overlays_dir:
            overrides["overlays"] = overlays_dir

        # Filter by verdict
        filter_verdicts = st.multiselect(
            "Filter by Verdict",
            options=["fail", "query", "pass"],
            default=["fail", "query"],
            key=f"{session_key_prefix}_filter_verdicts",
            help="Only process images with these verdicts",
        )

        # Will be handled in the execute phase
        st.session_state[f"{session_key_prefix}_filter_verdicts"] = filter_verdicts

    else:
        # Manual image selection
        images_input = st.text_area(
            "Image Paths (one per line)",
            value="",
            key=f"{session_key_prefix}_images_input",
            help="Enter image paths to narrate",
        )

        images = [p.strip() for p in images_input.split("\n") if p.strip()]
        if images:
            overrides["images"] = images

        current_overlays_manual = get_app_setting(
            st.session_state, "narrator", "overlays_dir_manual", ""
        )
        overlays_dir = st.text_input(
            "Overlays Directory (optional)",
            value=current_overlays_manual,
            key=f"{session_key_prefix}_overlays_manual",
        )
        if overlays_dir and overlays_dir != current_overlays_manual:
            set_app_setting(
                st.session_state, "narrator", "overlays_dir_manual", overlays_dir
            )
        if overlays_dir:
            overrides["overlays"] = overlays_dir

    # Output path
    summary_path = st.text_input(
        "Summary Markdown",
        value=str(OUTPUTS_DIR / "summaries" / "narrator_summary.md"),
        key=f"{session_key_prefix}_summary",
    )
    if summary_path:
        overrides["summary"] = summary_path

    # Dry run
    dry_run = st.checkbox(
        "Dry Run",
        value=preset.flags.get("dry_run", False),
        key=f"{session_key_prefix}_dry_run_main",
    )
    overrides["dry_run"] = dry_run

    return overrides


def main():
    """Color narrator page."""
    init_session_state()

    st.title("🎨 Color Narrator")
    st.markdown("Generate natural language descriptions of color contamination")

    # Pipeline mode indicator
    if st.session_state.get("mono_results"):
        st.info("📊 Pipeline Mode: Mono results available for automatic import")

    # Workflow tabs
    tabs = st.tabs(["⚙️ Configure", "▶️ Execute", "📊 Results", "🔍 Review"])

    # === CONFIGURE TAB ===
    with tabs[0]:
        st.markdown("### Configuration")

        # Render preset selector
        config = render_preset_selector(
            COLOR_NARRATOR_PRESETS,
            session_key_prefix="narrator",
            custom_override_renderer=render_custom_overrides,
        )

        # Validate
        has_input = config.get("images") or config.get("mono_jsonl")

        if not has_input:
            st.warning("⚠️ Please specify images to process")

        # Show pipeline status
        if config.get("mono_jsonl"):
            st.success("✅ Pipeline mode enabled - will import from mono results")

            # Check if file exists
            mono_jsonl_path = Path(config["mono_jsonl"])
            if mono_jsonl_path.exists():
                # Count images by verdict
                results = parse_jsonl(str(mono_jsonl_path))
                verdicts = {}
                for r in results:
                    verdict = r.get("verdict", "unknown")
                    verdicts[verdict] = verdicts.get(verdict, 0) + 1

                st.write("**Mono Results Summary:**")
                cols = st.columns(len(verdicts))
                for i, (verdict, count) in enumerate(sorted(verdicts.items())):
                    with cols[i]:
                        st.metric(verdict.upper(), count)
            else:
                st.error(f"❌ Mono results not found: {mono_jsonl_path}")

        # Configuration summary
        with st.expander("📋 Configuration Summary", expanded=False):
            st.json(config)

    # === EXECUTE TAB ===
    with tabs[1]:
        st.markdown("### Generate Narrations")

        config = st.session_state.get("narrator_config", {})

        has_input = config.get("images") or config.get("mono_jsonl")

        if not has_input:
            st.error("❌ No input specified. Configure in the Configure tab.")
        else:
            # Show what will be processed
            if config.get("mono_jsonl"):
                mono_jsonl_path = Path(config["mono_jsonl"])
                if mono_jsonl_path.exists():
                    results = parse_jsonl(str(mono_jsonl_path))

                    # Filter by verdict if specified
                    filter_verdicts = st.session_state.get(
                        "narrator_filter_verdicts", ["fail", "query"]
                    )
                    filtered = [
                        r for r in results if r.get("verdict") in filter_verdicts
                    ]

                    st.write(
                        f"**Will process {len(filtered)} images** (filtered from {len(results)} total)"
                    )
                    st.caption(f"Verdicts: {', '.join(filter_verdicts)}")

                    # Update config with filtered image paths
                    image_paths = [
                        r.get("image_path") for r in filtered if r.get("image_path")
                    ]
                    config = config.copy()
                    config["images"] = image_paths
            else:
                images = config.get("images", [])
                st.write(f"**Will process {len(images)} images**")

            # Backend check
            backend = config.get("backend", "unknown")
            base_url = config.get("vlm_base_url", "unknown")
            st.write(f"**Backend:** {backend} ({base_url})")

            # Render process runner
            result = render_process_runner(
                button_label="▶️ Generate Narrations",
                command_builder=build_narrator_command,
                config=config,
                key_prefix="narrator",
                result_key="narrator_results",
                show_command=True,
                timeout=3600,  # 1 hour timeout
            )

            if result and result.get("success"):
                st.success("✅ Narrations generated!")

    # === RESULTS TAB ===
    with tabs[2]:
        st.markdown("### Results")

        config = st.session_state.get("narrator_config", {})

        # Render unified results browser
        render_unified_results_browser(
            key_prefix="narrator_results_view",
            default_jsonl=None,  # Color narrator doesn't output JSONL by default
            default_markdown=config.get("summary"),
        )

    # === REVIEW TAB ===
    with tabs[3]:
        st.markdown("### Review Narrations")

        config = st.session_state.get("narrator_config", {})

        # If we have mono results, show images with narrations
        if config.get("mono_jsonl"):
            mono_jsonl_path = Path(config["mono_jsonl"])
            if mono_jsonl_path.exists():
                results = parse_jsonl(str(mono_jsonl_path))

                # Filter by verdict
                filter_verdicts = st.session_state.get(
                    "narrator_filter_verdicts", ["fail", "query"]
                )
                filtered = [r for r in results if r.get("verdict") in filter_verdicts]

                if filtered:
                    # Select image to review
                    image_names = [Path(r.get("image_path", "")).name for r in filtered]
                    selected_idx = st.selectbox(
                        "Select image",
                        range(len(image_names)),
                        format_func=lambda i: image_names[i],
                        key="narrator_review_select",
                    )

                    if selected_idx is not None:
                        result = filtered[selected_idx]
                        img_path = result.get("image_path")

                        if img_path and Path(img_path).exists():
                            # Find overlay
                            overlays_dir = config.get("overlays")
                            overlay_path = None

                            if overlays_dir:
                                overlay_file = Path(overlays_dir) / Path(img_path).name
                                if overlay_file.exists():
                                    overlay_path = str(overlay_file)

                            # Show image with overlay
                            render_image_detail(
                                img_path,
                                overlay_path=overlay_path,
                                metadata={
                                    "Verdict": result.get("verdict", "unknown"),
                                    "Contamination %": f"{result.get('contamination_percent', 0):.2f}%",
                                },
                                show_overlay_toggle=True,
                            )

                            # Show narration if available
                            # Note: The actual narration would be written to metadata
                            # This is a placeholder for display
                            st.markdown("---")
                            st.markdown("### Narration")
                            st.info(
                                "Narration has been written to image metadata. "
                                "View in Lightroom or use ExifTool to read."
                            )
                else:
                    st.info("No images match the selected verdict filter")
        else:
            st.info("Review feature works best when using mono checker pipeline mode")


if __name__ == "__main__":
    main()
