"""Mono Checker page."""

import streamlit as st
from pathlib import Path

from imageworks.gui.state import init_session_state
from imageworks.gui.presets import MONO_CHECKER_PRESETS
from imageworks.gui.components.preset_selector import render_preset_selector
from imageworks.gui.components.process_runner import render_process_runner
from imageworks.gui.components.results_viewer import (
    render_unified_results_browser,
    parse_jsonl,
)
from imageworks.gui.components.image_viewer import (
    render_image_grid,
    render_image_detail,
)
from imageworks.gui.utils.cli_wrapper import build_mono_command
from imageworks.gui.config import (
    DEFAULT_INPUT_DIR,
    DEFAULT_OVERLAYS_DIR,
    DEFAULT_OUTPUT_JSONL,
    DEFAULT_SUMMARY_PATH,
    get_app_setting,
    set_app_setting,
    reset_app_settings,
)


def render_custom_overrides(preset, session_key_prefix):
    """Custom override renderer for mono checker."""
    overrides = {}

    # Reset button at the top
    col_reset, col_spacer = st.columns([1, 3])
    with col_reset:
        if st.button(
            "🔄 Reset to Defaults",
            key=f"{session_key_prefix}_reset",
            help="Reset all paths to global defaults",
        ):
            reset_app_settings(st.session_state, "mono")
            st.success("✅ Reset to defaults")
            st.rerun()

    # Input directory - use per-app setting with fallback to default
    current_input = get_app_setting(
        st.session_state, "mono", "input_dir", DEFAULT_INPUT_DIR
    )
    input_dir = st.text_input(
        "Input Directory",
        value=current_input,
        key=f"{session_key_prefix}_input",
        help="Directory containing images to check",
    )
    if input_dir and input_dir != current_input:
        set_app_setting(st.session_state, "mono", "input_dir", input_dir)
    if input_dir:
        overrides["input"] = [input_dir]

    # Output paths
    col1, col2 = st.columns(2)

    with col1:
        current_overlays = get_app_setting(
            st.session_state, "mono", "overlays_dir", str(DEFAULT_OVERLAYS_DIR)
        )
        overlays_dir = st.text_input(
            "Overlays Output",
            value=current_overlays,
            key=f"{session_key_prefix}_overlays",
        )
        if overlays_dir and overlays_dir != current_overlays:
            set_app_setting(st.session_state, "mono", "overlays_dir", overlays_dir)
        if overlays_dir:
            overrides["overlays"] = overlays_dir

        current_jsonl = get_app_setting(
            st.session_state, "mono", "output_jsonl", str(DEFAULT_OUTPUT_JSONL)
        )
        output_jsonl = st.text_input(
            "Output JSONL",
            value=current_jsonl,
            key=f"{session_key_prefix}_output_jsonl",
        )
        if output_jsonl and output_jsonl != current_jsonl:
            set_app_setting(st.session_state, "mono", "output_jsonl", output_jsonl)
        if output_jsonl:
            overrides["output_jsonl"] = output_jsonl

    with col2:
        current_summary = get_app_setting(
            st.session_state, "mono", "summary_path", str(DEFAULT_SUMMARY_PATH)
        )
        summary_path = st.text_input(
            "Summary Markdown",
            value=current_summary,
            key=f"{session_key_prefix}_summary",
        )
        if summary_path and summary_path != current_summary:
            set_app_setting(st.session_state, "mono", "summary_path", summary_path)
        if summary_path:
            overrides["summary"] = summary_path

        dry_run = st.checkbox(
            "Dry Run",
            value=preset.flags.get("dry_run", False),
            key=f"{session_key_prefix}_dry_run_main",
        )
        overrides["dry_run"] = dry_run

    # Threshold overrides
    if "rgb_delta_threshold" in preset.common_overrides:
        with st.expander("🎚️ Threshold Adjustments", expanded=False):
            rgb_delta = st.slider(
                "RGB Delta Threshold",
                min_value=5.0,
                max_value=20.0,
                value=preset.flags.get("rgb_delta_threshold", 10.0),
                step=0.5,
                key=f"{session_key_prefix}_rgb_delta",
                help="Maximum allowed difference between RGB channels",
            )
            if rgb_delta != preset.flags.get("rgb_delta_threshold"):
                overrides["rgb_delta_threshold"] = rgb_delta

            chroma = st.slider(
                "Chroma Threshold",
                min_value=2.0,
                max_value=15.0,
                value=preset.flags.get("chroma_threshold", 5.0),
                step=0.5,
                key=f"{session_key_prefix}_chroma",
                help="Maximum allowed chroma (color saturation)",
            )
            if chroma != preset.flags.get("chroma_threshold"):
                overrides["chroma_threshold"] = chroma

    return overrides


def main():
    """Mono checker page."""
    init_session_state()

    st.title("🖼️ Mono Checker")
    st.markdown("Detect non-monochrome images in competition submissions")

    # Tabs for workflow
    tab_config, tab_execute, tab_results, tab_review = st.tabs(
        ["⚙️ Configure", "▶️ Execute", "📊 Results", "🔍 Review Images"]
    )

    # === CONFIGURE TAB ===
    with tab_config:
        st.markdown("### Configuration")

        # Render preset selector
        config = render_preset_selector(
            MONO_CHECKER_PRESETS,
            session_key_prefix="mono",
            custom_override_renderer=render_custom_overrides,
        )

        # Validate configuration
        if not config.get("input"):
            st.warning("⚠️ Please specify an input directory")
        else:
            input_path = Path(config["input"][0])
            if not input_path.exists():
                st.error(f"❌ Directory does not exist: {input_path}")
            elif input_path.is_file():
                st.info(f"ℹ️ Will check single file: {input_path.name}")
            else:
                st.info(f"ℹ️ Will check directory: {input_path}")

        # Show current configuration
        with st.expander("📋 Configuration Summary", expanded=False):
            st.json(config)

    # === EXECUTE TAB ===
    with tab_execute:
        st.markdown("### Run Mono Checker")

        config = st.session_state.get("mono_config", {})

        if not config.get("input"):
            st.error("❌ No input specified. Please configure in the Configure tab.")
        else:
            st.markdown(f"**Checking:** {config['input'][0]}")
            st.markdown(
                f"**Preset:** {st.session_state.get('mono_preset_name', 'balanced')}"
            )

            # Render process runner
            result = render_process_runner(
                button_label="▶️ Run Mono Check",
                command_builder=build_mono_command,
                config=config,
                key_prefix="mono",
                result_key="mono_results",
                show_command=True,
                timeout=600,
            )

            if result and result.get("success"):
                st.success("✅ Mono check completed!")

                # Show quick stats
                output_jsonl = config.get("output_jsonl")
                if output_jsonl and Path(output_jsonl).exists():
                    results = parse_jsonl(output_jsonl)

                    if results:
                        verdicts = {}
                        for r in results:
                            verdict = r.get("verdict", "unknown")
                            verdicts[verdict] = verdicts.get(verdict, 0) + 1

                        st.markdown("#### Quick Stats")
                        cols = st.columns(len(verdicts))
                        for i, (verdict, count) in enumerate(sorted(verdicts.items())):
                            with cols[i]:
                                if verdict.lower() == "pass":
                                    st.metric(verdict.upper(), count, delta=None)
                                elif verdict.lower() == "fail":
                                    st.metric(verdict.upper(), count, delta=None)
                                else:
                                    st.metric(verdict.upper(), count, delta=None)

    # === RESULTS TAB ===
    with tab_results:
        st.markdown("### Results")

        config = st.session_state.get("mono_config", {})

        # Render unified results browser
        render_unified_results_browser(
            key_prefix="mono_results_view",
            default_jsonl=config.get("output_jsonl"),
            default_markdown=config.get("summary"),
        )

    # === REVIEW IMAGES TAB ===
    with tab_review:
        st.markdown("### Review Images")

        config = st.session_state.get("mono_config", {})
        output_jsonl = config.get("output_jsonl")
        overlays_dir = config.get("overlays")

        if not output_jsonl or not Path(output_jsonl).exists():
            st.info("Run mono check to review results")
        else:
            results = parse_jsonl(output_jsonl)

            if not results:
                st.warning("No results found")
            else:
                # Verdict filter
                st.markdown("#### Filter by Verdict")
                verdicts = sorted(set(r.get("verdict", "unknown") for r in results))

                # Safe default: only include verdicts that actually exist
                default_verdicts = [v for v in ["fail", "query"] if v in verdicts]

                selected_verdicts = st.multiselect(
                    "Show verdicts",
                    options=verdicts,
                    default=default_verdicts,
                    key="mono_verdict_filter",
                )

                if selected_verdicts:
                    filtered_results = [
                        r for r in results if r.get("verdict") in selected_verdicts
                    ]
                else:
                    filtered_results = results

                st.info(f"Showing {len(filtered_results)} of {len(results)} images")

                if filtered_results:
                    # Prepare image data for grid
                    images = []
                    for r in filtered_results:
                        img_path = r.get("image_path")
                        if img_path:
                            # Find corresponding overlay
                            overlay_path = None
                            if overlays_dir:
                                img_name = Path(img_path).name
                                potential_overlay = Path(overlays_dir) / img_name
                                if potential_overlay.exists():
                                    overlay_path = str(potential_overlay)

                            images.append(
                                {
                                    "path": img_path,
                                    "name": Path(img_path).name,
                                    "verdict": r.get("verdict", "unknown"),
                                    "overlay_path": overlay_path,
                                    "contamination_percent": r.get(
                                        "contamination_percent", 0
                                    ),
                                }
                            )

                    # Render grid
                    if images:
                        selected_image = render_image_grid(
                            images,
                            key_prefix="mono_review",
                            columns=3,
                            show_overlays=True,
                            overlay_key="overlay_path",
                            max_images=100,
                        )

                        # Show detail view if image selected
                        if selected_image:
                            st.markdown("---")
                            st.markdown("### Image Detail")

                            # Find the result data
                            result_data = next(
                                (
                                    r
                                    for r in filtered_results
                                    if r.get("image_path") == selected_image
                                ),
                                None,
                            )

                            if result_data:
                                overlay_path = next(
                                    (
                                        img["overlay_path"]
                                        for img in images
                                        if img["path"] == selected_image
                                    ),
                                    None,
                                )

                                render_image_detail(
                                    selected_image,
                                    overlay_path=overlay_path,
                                    metadata={
                                        "Verdict": result_data.get(
                                            "verdict", "unknown"
                                        ),
                                        "Contamination %": f"{result_data.get('contamination_percent', 0):.2f}%",
                                        "Contaminated Pixels": result_data.get(
                                            "contaminated_pixels", 0
                                        ),
                                        "RGB Delta Max": f"{result_data.get('rgb_delta_max', 0):.2f}",
                                        "Chroma Max": f"{result_data.get('chroma_max', 0):.2f}",
                                    },
                                    show_overlay_toggle=True,
                                )


if __name__ == "__main__":
    main()
