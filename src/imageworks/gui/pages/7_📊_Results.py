"""Unified results browser page."""

import streamlit as st
from pathlib import Path

from imageworks.gui.state import init_session_state
from imageworks.gui.components.results_viewer import (
    render_unified_results_browser,
    parse_jsonl,
)
from imageworks.gui.components.job_history import render_job_history
from imageworks.gui.config import OUTPUTS_DIR


def main():
    """Results browser page."""
    init_session_state()

    st.title("📊 Results Browser")
    st.markdown("View and analyze results from all modules")

    # Main tabs
    tabs = st.tabs(["📁 Browse Results", "📜 Job History", "📈 Statistics"])

    # === BROWSE RESULTS TAB ===
    with tabs[0]:
        st.markdown("### Browse Results")

        # Module selector
        module = st.selectbox(
            "Select Module",
            options=[
                "Mono Checker",
                "Image Similarity",
                "Personal Tagger",
                "Color Narrator",
            ],
            key="results_module_select",
        )

        # Map module to default paths
        module_paths = {
            "Mono Checker": {
                "jsonl": OUTPUTS_DIR / "results" / "mono_results.jsonl",
                "markdown": OUTPUTS_DIR / "summaries" / "mono_summary.md",
            },
            "Image Similarity": {
                "jsonl": OUTPUTS_DIR / "results" / "similarity_results.jsonl",
                "markdown": OUTPUTS_DIR / "summaries" / "similarity_summary.md",
            },
            "Personal Tagger": {
                "jsonl": OUTPUTS_DIR / "results" / "tagger_results.jsonl",
                "markdown": OUTPUTS_DIR / "summaries" / "tagger_summary.md",
            },
            "Color Narrator": {
                "jsonl": None,
                "markdown": OUTPUTS_DIR / "summaries" / "narrator_summary.md",
            },
        }

        default_jsonl = (
            str(module_paths[module]["jsonl"]) if module_paths[module]["jsonl"] else ""
        )
        default_markdown = str(module_paths[module]["markdown"])

        # Render unified browser
        render_unified_results_browser(
            key_prefix=f"results_{module.lower().replace(' ', '_')}",
            default_jsonl=default_jsonl,
            default_markdown=default_markdown,
        )

        # Export options
        st.markdown("---")
        st.markdown("### Export Options")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📋 Copy Results Path"):
                if default_jsonl and Path(default_jsonl).exists():
                    st.code(default_jsonl)
                else:
                    st.warning("Results file not found")

        with col2:
            if st.button("📁 Open Results Folder"):
                results_dir = OUTPUTS_DIR / "results"
                st.code(f"# Open in file browser:\n{results_dir}")

        with col3:
            if st.button("🗑️ Clear Old Results"):
                st.warning("Are you sure? This will delete old result files.")
                # Placeholder for cleanup logic

    # === JOB HISTORY TAB ===
    with tabs[1]:
        st.markdown("### Job History")

        # Module filter option
        filter_module = st.checkbox("Filter by current module", value=False)

        if filter_module:
            module_filter = st.selectbox(
                "Module",
                options=["mono", "similarity", "tagger", "narrator"],
                key="job_history_module_filter",
            )
        else:
            module_filter = None

        # Render job history
        selected_job = render_job_history(
            key_prefix="results_job_history", max_jobs=50, module_filter=module_filter
        )

        # Handle job re-run
        if selected_job:
            st.markdown("---")
            st.success(f"✅ Selected job: {selected_job.get('description')}")

            module = selected_job.get("module", "unknown")
            config = selected_job.get("config", {})

            st.write(f"**Module:** {module}")
            st.write(f"**Timestamp:** {selected_job.get('timestamp')}")

            # Navigate to appropriate page
            if st.button("🔄 Re-run in Module Page"):
                # Store config in session state
                st.session_state[f"{module}_config"] = config

                # Navigate to module page
                page_map = {
                    "mono": "pages/3_🖼️_Mono_Checker.py",
                    "similarity": "pages/4_🖼️_Image_Similarity.py",
                    "tagger": "pages/5_🖼️_Personal_Tagger.py",
                    "narrator": "pages/6_🖼️_Color_Narrator.py",
                }

                if module in page_map:
                    st.switch_page(page_map[module])
                else:
                    st.error(f"Unknown module: {module}")

    # === STATISTICS TAB ===
    with tabs[2]:
        st.markdown("### Statistics")

        # Overall stats
        st.subheader("Overall Statistics")

        col1, col2, col3, col4 = st.columns(4)

        # Count result files
        results_dir = OUTPUTS_DIR / "results"
        if results_dir.exists():
            jsonl_files = list(results_dir.glob("*.jsonl"))

            with col1:
                st.metric("Total Result Files", len(jsonl_files))

            # Count total images processed
            total_images = 0
            for jsonl_file in jsonl_files:
                try:
                    results = parse_jsonl(str(jsonl_file))
                    total_images += len(results)
                except Exception:
                    pass

            with col2:
                st.metric("Total Images Processed", total_images)

            # Job count
            job_history = st.session_state.get("job_history", [])
            with col3:
                st.metric("Total Jobs", len(job_history))

            # Success rate
            if job_history:
                success_count = sum(
                    1 for j in job_history if j.get("status") == "success"
                )
                success_rate = (success_count / len(job_history)) * 100
                with col4:
                    st.metric("Success Rate", f"{success_rate:.1f}%")
        else:
            st.info("No results directory found")

        # Per-module stats
        st.markdown("---")
        st.subheader("Per-Module Statistics")

        modules = ["mono", "similarity", "tagger", "narrator"]
        module_stats = []

        for module in modules:
            # Count jobs
            module_jobs = [j for j in job_history if j.get("module") == module]

            # Find result file
            result_files = {
                "mono": "mono_results.jsonl",
                "similarity": "similarity_results.jsonl",
                "tagger": "tagger_results.jsonl",
                "narrator": None,  # No JSONL for narrator
            }

            result_count = 0
            if result_files.get(module) and results_dir.exists():
                result_path = results_dir / result_files[module]
                if result_path.exists():
                    try:
                        results = parse_jsonl(str(result_path))
                        result_count = len(results)
                    except Exception:
                        pass

            module_stats.append(
                {
                    "Module": module.capitalize(),
                    "Jobs Run": len(module_jobs),
                    "Images Processed": result_count,
                    "Last Run": (
                        module_jobs[-1].get("timestamp", "Never")
                        if module_jobs
                        else "Never"
                    ),
                }
            )

        import pandas as pd

        df = pd.DataFrame(module_stats)
        st.dataframe(df, use_container_width=True)

        # Storage statistics
        st.markdown("---")
        st.subheader("Storage")

        if OUTPUTS_DIR.exists():
            total_size = sum(
                f.stat().st_size for f in OUTPUTS_DIR.rglob("*") if f.is_file()
            )
            size_mb = total_size / (1024 * 1024)

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Outputs Directory Size", f"{size_mb:.1f} MB")

            with col2:
                # Count files by type
                jsonl_count = len(list(OUTPUTS_DIR.rglob("*.jsonl")))
                md_count = len(list(OUTPUTS_DIR.rglob("*.md")))
                img_count = len(list(OUTPUTS_DIR.rglob("*.jpg"))) + len(
                    list(OUTPUTS_DIR.rglob("*.png"))
                )

                st.write("**File counts:**")
                st.write(f"- JSONL: {jsonl_count}")
                st.write(f"- Markdown: {md_count}")
                st.write(f"- Images: {img_count}")


if __name__ == "__main__":
    main()
