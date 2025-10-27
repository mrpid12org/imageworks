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
    ZIP_DEFAULT_ZIP_DIR,
    ZIP_DEFAULT_EXTRACT_ROOT,
    ZIP_DEFAULT_SUMMARY_OUTPUT,
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

    # LAB threshold overrides
    if "lab_neutral_chroma" in preset.common_overrides:
        with st.expander("🎚️ LAB Threshold Adjustments", expanded=False):
            lab_neutral_chroma = st.slider(
                "LAB Neutral Chroma (C*)",
                min_value=0.5,
                max_value=5.0,
                value=preset.flags.get("lab_neutral_chroma", 2.0),
                step=0.1,
                key=f"{session_key_prefix}_lab_neutral_chroma",
                help="Neutral chroma threshold in LAB color space (lower = stricter)",
            )
            if lab_neutral_chroma != preset.flags.get("lab_neutral_chroma"):
                overrides["lab_neutral_chroma"] = lab_neutral_chroma

            lab_toned_pass = st.slider(
                "LAB Toned Pass (degrees)",
                min_value=5.0,
                max_value=25.0,
                value=preset.flags.get("lab_toned_pass", 10.0),
                step=0.5,
                key=f"{session_key_prefix}_lab_toned_pass",
                help="LAB hue standard deviation threshold for PASS verdict",
            )
            if lab_toned_pass != preset.flags.get("lab_toned_pass"):
                overrides["lab_toned_pass"] = lab_toned_pass

            lab_toned_query = st.slider(
                "LAB Toned Query (degrees)",
                min_value=8.0,
                max_value=30.0,
                value=preset.flags.get("lab_toned_query", 14.0),
                step=0.5,
                key=f"{session_key_prefix}_lab_toned_query",
                help="LAB hue standard deviation threshold for QUERY verdict",
            )
            if lab_toned_query != preset.flags.get("lab_toned_query"):
                overrides["lab_toned_query"] = lab_toned_query

    return overrides


def main():
    """Mono checker page."""
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

    st.title("🖼️ Mono Checker")
    st.markdown("Detect non-monochrome images in competition submissions")

    # Tabs for workflow (preprocessing first, following the workflow order)
    tab_preprocess, tab_config, tab_execute, tab_results, tab_review = st.tabs(
        [
            "📦 Preprocessing",
            "⚙️ Configure",
            "▶️ Execute",
            "📊 Results",
            "🔍 Review Images",
        ]
    )

    # === PREPROCESSING TAB ===
    with tab_preprocess:
        st.markdown("### Stage 1: Zip Extraction & Preprocessing")
        st.info(
            "📋 **Workflow**: Extract competition zip files → Lightroom auto-imports → Run mono checker"
        )

        # Reset button for preprocessing settings
        col_reset, col_spacer = st.columns([1, 3])
        with col_reset:
            if st.button(
                "🔄 Reset to Defaults",
                key="zip_reset",
                help="Reset all paths to global defaults",
            ):
                reset_app_settings(st.session_state, "zip_extract")
                st.success("✅ Reset to defaults")
                st.rerun()

        # Zip directory input
        current_zip_dir = get_app_setting(
            st.session_state, "zip_extract", "zip_dir", ZIP_DEFAULT_ZIP_DIR
        )
        zip_dir = st.text_input(
            "Zip Files Directory",
            value=current_zip_dir,
            key="zip_dir_input",
            help="Directory containing competition ZIP files to extract",
        )
        if zip_dir and zip_dir != current_zip_dir:
            set_app_setting(st.session_state, "zip_extract", "zip_dir", zip_dir)

        # Extract root directory
        current_extract_root = get_app_setting(
            st.session_state, "zip_extract", "extract_root", ZIP_DEFAULT_EXTRACT_ROOT
        )
        extract_root = st.text_input(
            "Extract To (Lightroom Watched Folder)",
            value=current_extract_root,
            key="extract_root_input",
            help="Destination directory (Lightroom auto-import folder)",
        )
        if extract_root and extract_root != current_extract_root:
            set_app_setting(
                st.session_state, "zip_extract", "extract_root", extract_root
            )

        # Summary output path
        current_summary = get_app_setting(
            st.session_state,
            "zip_extract",
            "summary_output",
            str(ZIP_DEFAULT_SUMMARY_OUTPUT),
        )
        summary_output = st.text_input(
            "Summary Output File",
            value=current_summary,
            key="zip_summary_output",
            help="Path for extraction summary markdown file",
        )
        if summary_output and summary_output != current_summary:
            set_app_setting(
                st.session_state, "zip_extract", "summary_output", summary_output
            )

        # Options
        col1, col2 = st.columns(2)

        with col1:
            include_xmp = st.checkbox(
                "Include XMP Files",
                value=False,
                key="zip_include_xmp",
                help="Also extract .xmp sidecar files (normally not needed)",
            )

        with col2:
            update_all_metadata = st.checkbox(
                "Update All Metadata",
                value=False,
                key="zip_update_all",
                help="Re-update metadata for existing files (not just new extractions)",
            )

        # Show what will be processed
        zip_path = Path(zip_dir) if zip_dir else None
        if zip_path and zip_path.exists():
            zip_files = list(zip_path.glob("*.zip"))
            if zip_files:
                st.success(f"✅ Found {len(zip_files)} zip file(s) to process")
                with st.expander("📋 Zip Files", expanded=False):
                    for zf in zip_files:
                        st.text(f"  • {zf.name}")
            else:
                st.warning(f"⚠️ No .zip files found in {zip_dir}")
        elif zip_path:
            st.error(f"❌ Directory does not exist: {zip_dir}")

        # Execute button
        st.markdown("---")
        if st.button(
            "▶️ Extract Competition Zips",
            type="primary",
            key="run_zip_extract",
            disabled=not (zip_path and zip_path.exists()),
        ):
            import subprocess

            cmd = ["imageworks-zip", "run"]

            if zip_dir:
                cmd.extend(["--zip-dir", zip_dir])
            if extract_root:
                cmd.extend(["--extract-root", extract_root])
            if summary_output:
                cmd.extend(["--output-file", summary_output])
            if include_xmp:
                cmd.append("--include-xmp")
            if update_all_metadata:
                cmd.extend(["--metadata"])

            with st.spinner("Extracting competition zips..."):
                try:
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        cwd=str(Path.cwd()),
                        timeout=600,
                    )

                    if result.returncode == 0:
                        st.success("✅ Zip extraction completed!")

                        # Show output
                        with st.expander("📋 Extraction Details", expanded=True):
                            st.code(result.stdout)

                        # Show summary file if it exists
                        summary_path = Path(summary_output)
                        if summary_path.exists():
                            with st.expander("📄 Summary File", expanded=False):
                                st.markdown(summary_path.read_text())

                        st.info(
                            "💡 **Next Steps**:\n"
                            "1. Lightroom will auto-import the extracted images\n"
                            "2. Switch to '⚙️ Configure' tab to set up mono checking\n"
                            "3. Use the extracted folder as input for mono checker"
                        )
                    else:
                        st.error(
                            f"❌ Extraction failed (exit code: {result.returncode})"
                        )
                        if result.stderr:
                            st.code(result.stderr)
                        if result.stdout:
                            st.code(result.stdout)

                except subprocess.TimeoutExpired:
                    st.error("❌ Extraction timed out (> 10 minutes)")
                except Exception as e:
                    st.error(f"❌ Error running extraction: {e}")

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
