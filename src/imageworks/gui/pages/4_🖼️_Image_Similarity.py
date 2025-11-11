"""Image Similarity Checker page."""

import streamlit as st
from pathlib import Path

from imageworks.gui.state import init_session_state
from imageworks.gui.presets import IMAGE_SIMILARITY_PRESETS
from imageworks.gui.components.preset_selector import render_preset_selector
from imageworks.gui.components.process_runner import render_process_runner
from imageworks.gui.components.results_viewer import render_unified_results_browser
from imageworks.gui.components.image_viewer import render_side_by_side
from imageworks.gui.utils.cli_wrapper import build_similarity_command
from imageworks.gui.config import (
    SIMILARITY_DEFAULT_LIBRARY_ROOT,
    SIMILARITY_DEFAULT_OUTPUT_JSONL,
    SIMILARITY_DEFAULT_SUMMARY_PATH,
    get_app_setting,
    set_app_setting,
    reset_app_settings,
)
from imageworks.gui.components.sidebar_footer import render_sidebar_footer


def render_custom_overrides(preset, session_key_prefix):
    """Custom override renderer for similarity checker."""
    overrides = {}

    # Reset button
    col_reset, col_spacer = st.columns([1, 3])
    with col_reset:
        if st.button(
            "🔄 Reset to Defaults",
            key=f"{session_key_prefix}_reset",
            help="Reset all paths to global defaults",
        ):
            reset_app_settings(st.session_state, "similarity")
            st.success("✅ Reset to defaults")
            st.rerun()

    col1, col2 = st.columns(2)

    with col1:
        # Candidates
        candidates_input = st.text_area(
            "Candidate Images/Directories",
            value="",
            key=f"{session_key_prefix}_candidates",
            help="Enter paths to check for similarity, one per line",
        )
        candidates = [p.strip() for p in candidates_input.split("\n") if p.strip()]
        if candidates:
            overrides["candidates"] = candidates

        # Library root - use per-app setting
        current_library = get_app_setting(
            st.session_state,
            "similarity",
            "library_root",
            SIMILARITY_DEFAULT_LIBRARY_ROOT,
        )
        library_root = st.text_input(
            "Library Root",
            value=current_library,
            key=f"{session_key_prefix}_library_root",
            help="Root directory of image library to compare against",
        )
        if library_root and library_root != current_library:
            set_app_setting(
                st.session_state, "similarity", "library_root", library_root
            )
        if library_root:
            overrides["library_root"] = library_root

    with col2:
        # Thresholds
        if "fail_threshold" in preset.common_overrides:
            fail_threshold = st.slider(
                "Fail Threshold",
                min_value=0.70,
                max_value=1.0,
                value=preset.flags.get("fail_threshold", 0.92),
                step=0.01,
                key=f"{session_key_prefix}_fail_threshold",
                help="Similarity above this marks as duplicate",
            )
            if fail_threshold != preset.flags.get("fail_threshold"):
                overrides["fail_threshold"] = fail_threshold

        if "query_threshold" in preset.common_overrides:
            query_threshold = st.slider(
                "Query Threshold",
                min_value=0.60,
                max_value=0.95,
                value=preset.flags.get("query_threshold", 0.82),
                step=0.01,
                key=f"{session_key_prefix}_query_threshold",
                help="Similarity above this marks as potential match",
            )
            if query_threshold != preset.flags.get("query_threshold"):
                overrides["query_threshold"] = query_threshold

    # Output paths - use per-app settings
    current_jsonl = get_app_setting(
        st.session_state,
        "similarity",
        "output_jsonl",
        str(SIMILARITY_DEFAULT_OUTPUT_JSONL),
    )
    output_jsonl = st.text_input(
        "Output JSONL",
        value=current_jsonl,
        key=f"{session_key_prefix}_output_jsonl",
    )
    if output_jsonl and output_jsonl != current_jsonl:
        set_app_setting(st.session_state, "similarity", "output_jsonl", output_jsonl)
    if output_jsonl:
        overrides["output_jsonl"] = output_jsonl

    current_summary = get_app_setting(
        st.session_state,
        "similarity",
        "summary_path",
        str(SIMILARITY_DEFAULT_SUMMARY_PATH),
    )
    summary_path = st.text_input(
        "Summary Markdown",
        value=current_summary,
        key=f"{session_key_prefix}_summary",
    )
    if summary_path and summary_path != current_summary:
        set_app_setting(st.session_state, "similarity", "summary_path", summary_path)
    if summary_path:
        overrides["summary"] = summary_path

    # Dry run toggle
    dry_run = st.checkbox(
        "Dry Run (recommended)",
        value=preset.flags.get("dry_run", True),
        key=f"{session_key_prefix}_dry_run_main",
    )
    overrides["dry_run"] = dry_run

    return overrides


def main():
    """Image similarity checker page."""
    st.set_page_config(layout="wide")
    init_session_state()
    with st.sidebar:
        render_sidebar_footer()

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

    st.title("🔍 Image Similarity Checker")
    st.markdown("Find duplicate and similar images in your library")

    # Tabs for workflow stages
    tab_config, tab_execute, tab_results = st.tabs(
        ["⚙️ Configure", "▶️ Execute", "📊 Results"]
    )

    # === CONFIGURE TAB ===
    with tab_config:
        st.markdown("### Configuration")

        # Render preset selector with custom overrides
        config = render_preset_selector(
            IMAGE_SIMILARITY_PRESETS,
            session_key_prefix="similarity",
            custom_override_renderer=render_custom_overrides,
        )

        # Validate configuration
        if not config.get("candidates"):
            st.warning("⚠️ Please specify at least one candidate image or directory")

        if not config.get("library_root"):
            st.info(
                "ℹ️ Library root not specified - will only check candidates against each other"
            )

        # Show current configuration summary
        with st.expander("📋 Configuration Summary", expanded=False):
            st.json(config)

    # === EXECUTE TAB ===
    with tab_execute:
        st.markdown("### Execute Similarity Check")

        # Get configuration from session state
        config = st.session_state.get("similarity_config", {})

        if not config.get("candidates"):
            st.error(
                "❌ No candidates specified. Please configure in the Configure tab."
            )
        else:
            # Show what will be checked
            st.markdown("#### Checking:")
            candidates = config.get("candidates", [])
            for candidate in candidates[:10]:  # Show first 10
                st.text(f"  • {candidate}")

            if len(candidates) > 10:
                st.caption(f"  ... and {len(candidates) - 10} more")

            if config.get("library_root"):
                st.markdown(f"**Against library:** {config['library_root']}")

            # Render process runner
            result = render_process_runner(
                button_label="▶️ Run Similarity Check",
                command_builder=build_similarity_command,
                config=config,
                key_prefix="similarity",
                result_key="similarity_results",
                show_command=True,
                timeout=600,  # 10 minutes timeout
            )

            if result and result.get("success"):
                st.success("✅ Similarity check completed!")

                # Show quick stats if JSONL exists
                output_jsonl = config.get("output_jsonl")
                if output_jsonl and Path(output_jsonl).exists():
                    from imageworks.gui.components.results_viewer import parse_jsonl

                    results = parse_jsonl(output_jsonl)

                    if results:
                        # Count verdicts
                        verdicts = {}
                        for r in results:
                            verdict = r.get("verdict", "unknown")
                            verdicts[verdict] = verdicts.get(verdict, 0) + 1

                        st.markdown("#### Quick Stats")
                        cols = st.columns(len(verdicts))
                        for i, (verdict, count) in enumerate(sorted(verdicts.items())):
                            with cols[i]:
                                st.metric(verdict.upper(), count)

    # === RESULTS TAB ===
    with tab_results:
        st.markdown("### Results")

        config = st.session_state.get("similarity_config", {})

        # Render unified results browser
        render_unified_results_browser(
            key_prefix="similarity_results",
            default_jsonl=config.get("output_jsonl"),
            default_markdown=config.get("summary"),
        )

        # Side-by-side comparison for matches
        st.markdown("---")
        st.markdown("### View Matches")

        output_jsonl = config.get("output_jsonl")
        if output_jsonl and Path(output_jsonl).exists():
            from imageworks.gui.components.results_viewer import parse_jsonl

            results = parse_jsonl(output_jsonl)

            # Filter to failed/query images (potential matches)
            matches = [r for r in results if r.get("verdict") in ["fail", "query"]]

            if matches:
                st.info(f"Found {len(matches)} potential matches")

                # Select image to view
                image_names = [Path(m.get("candidate_path", "")).name for m in matches]
                selected_idx = st.selectbox(
                    "Select image",
                    range(len(image_names)),
                    format_func=lambda i: image_names[i],
                    key="similarity_match_select",
                )

                if selected_idx is not None:
                    match = matches[selected_idx]
                    candidate_path = match.get("candidate_path")

                    # Show match details
                    st.markdown(f"**Verdict:** {match.get('verdict', 'Unknown')}")
                    st.markdown(
                        f"**Best Match Score:** {match.get('best_match_score', 0):.4f}"
                    )

                    # Get best match path
                    best_matches = match.get("best_matches", [])
                    if best_matches and len(best_matches) > 0:
                        best_match_path = best_matches[0].get("library_path")

                        if candidate_path and best_match_path:
                            if (
                                Path(candidate_path).exists()
                                and Path(best_match_path).exists()
                            ):
                                render_side_by_side(
                                    candidate_path,
                                    best_match_path,
                                    title1="Candidate",
                                    title2="Best Match",
                                )
                            else:
                                st.error("❌ Image files not found")
            else:
                st.success("✅ No duplicate or similar images found!")
        else:
            st.info("Run similarity check to view results")


if __name__ == "__main__":
    main()
