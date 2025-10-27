"""Results viewer component for JSONL and Markdown files."""

import streamlit as st
import json
from pathlib import Path
from typing import List, Dict, Any, Optional


@st.cache_data(show_spinner="Loading results...")
def parse_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Parse JSONL file (CACHED).

    Args:
        file_path: Path to JSONL file

    Returns:
        List of parsed JSON objects
    """
    results = []

    try:
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        st.error(f"Failed to parse JSONL: {e}")

    return results


@st.cache_data(show_spinner="Loading markdown...")
def load_markdown(file_path: str) -> str:
    """
    Load markdown file (CACHED).

    Args:
        file_path: Path to markdown file

    Returns:
        Markdown content as string
    """
    try:
        with open(file_path, "r") as f:
            return f.read()
    except Exception as e:
        return f"Failed to load markdown: {e}"


def render_jsonl_viewer(
    jsonl_path: str,
    key_prefix: str,
    filterable_fields: Optional[List[str]] = None,
    items_per_page: int = 50,
) -> List[Dict[str, Any]]:
    """
    Render JSONL results viewer with filtering and pagination.

    Args:
        jsonl_path: Path to JSONL file
        key_prefix: Unique prefix for widgets
        filterable_fields: Fields that can be filtered
        items_per_page: Items per page

    Returns:
        List of visible results after filtering
    """

    if not Path(jsonl_path).exists():
        st.error(f"❌ File not found: {jsonl_path}")
        return []

    # Load results (cached)
    results = parse_jsonl(jsonl_path)

    if not results:
        st.info("No results found in file")
        return []

    st.success(f"✅ Loaded {len(results)} results")

    # Filtering
    filtered_results = results

    if filterable_fields:
        st.markdown("### Filters")

        filter_cols = st.columns(len(filterable_fields))

        for i, field in enumerate(filterable_fields):
            with filter_cols[i]:
                # Get unique values for this field
                unique_values = set()
                for result in results:
                    if field in result:
                        value = result[field]
                        if isinstance(value, (str, int, float, bool)):
                            unique_values.add(str(value))

                if unique_values:
                    selected = st.multiselect(
                        field.replace("_", " ").title(),
                        options=sorted(unique_values),
                        key=f"{key_prefix}_filter_{field}",
                    )

                    if selected:
                        filtered_results = [
                            r
                            for r in filtered_results
                            if field in r and str(r[field]) in selected
                        ]

        if len(filtered_results) < len(results):
            st.info(f"Showing {len(filtered_results)} of {len(results)} results")

    # Pagination
    total_pages = (len(filtered_results) + items_per_page - 1) // items_per_page
    start_idx = 0  # Initialize before conditional

    if total_pages > 1:
        page = st.number_input(
            "Page",
            min_value=1,
            max_value=total_pages,
            value=1,
            key=f"{key_prefix}_page",
        )
        start_idx = (page - 1) * items_per_page
        end_idx = start_idx + items_per_page
        page_results = filtered_results[start_idx:end_idx]
    else:
        page_results = filtered_results

    # Display results
    st.markdown("### Results")

    for i, result in enumerate(page_results):
        with st.expander(f"Result {start_idx + i + 1}", expanded=False):
            st.json(result)

    return filtered_results


def render_markdown_viewer(
    markdown_path: str,
    use_columns: bool = False,
) -> None:
    """
    Render markdown file with styling.

    Args:
        markdown_path: Path to markdown file
        use_columns: Whether to use two-column layout
    """

    if not Path(markdown_path).exists():
        st.error(f"❌ File not found: {markdown_path}")
        return

    # Load markdown (cached)
    md_content = load_markdown(markdown_path)

    if use_columns:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### Raw Markdown")
            st.text_area(
                "Markdown",
                value=md_content,
                height=600,
                disabled=True,
                label_visibility="collapsed",
            )

        with col2:
            st.markdown("### Rendered")
            st.markdown(md_content)
    else:
        st.markdown(md_content)


def render_results_summary(
    results: List[Dict[str, Any]],
    verdict_field: str = "verdict",
) -> Dict[str, int]:
    """
    Render summary statistics for results.

    Args:
        results: List of result dicts
        verdict_field: Field name containing verdict

    Returns:
        Dict with verdict counts
    """

    # Count verdicts
    verdict_counts = {}

    for result in results:
        if verdict_field in result:
            verdict = result[verdict_field]
            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1

    # Display metrics
    st.subheader("Summary Statistics")

    cols = st.columns(len(verdict_counts) if verdict_counts else 1)

    if verdict_counts:
        for i, (verdict, count) in enumerate(sorted(verdict_counts.items())):
            with cols[i]:
                # Color-code based on verdict
                if verdict.lower() in ["pass", "ok", "success"]:
                    st.metric(label=verdict.upper(), value=count, delta=None)
                elif verdict.lower() in ["fail", "failed", "error"]:
                    st.metric(label=verdict.upper(), value=count, delta=None)
                else:
                    st.metric(label=verdict.upper(), value=count, delta=None)
    else:
        st.info("No verdict information found")

    return verdict_counts


def render_unified_results_browser(
    key_prefix: str,
    default_jsonl: Optional[str] = None,
    default_markdown: Optional[str] = None,
) -> None:
    """
    Render unified results browser with tabs for different views.

    Args:
        key_prefix: Unique prefix for widgets
        default_jsonl: Default JSONL path
        default_markdown: Default markdown path
    """

    st.subheader("Results Browser")

    # File selection
    col1, col2 = st.columns(2)

    with col1:
        jsonl_path = st.text_input(
            "JSONL Results",
            value=default_jsonl or "",
            key=f"{key_prefix}_jsonl_path",
            help="Path to JSONL results file",
        )

    with col2:
        markdown_path = st.text_input(
            "Markdown Summary",
            value=default_markdown or "",
            key=f"{key_prefix}_markdown_path",
            help="Path to markdown summary file",
        )

    # Tabs for different views
    tabs = st.tabs(["📊 Summary", "📄 JSONL", "📝 Markdown"])

    # Summary tab
    with tabs[0]:
        if jsonl_path and Path(jsonl_path).exists():
            results = parse_jsonl(jsonl_path)
            if results:
                render_results_summary(results)
            else:
                st.info("No results to summarize")
        else:
            st.info("Select a JSONL file to view summary")

    # JSONL tab
    with tabs[1]:
        if jsonl_path and Path(jsonl_path).exists():
            render_jsonl_viewer(
                jsonl_path,
                key_prefix=f"{key_prefix}_jsonl",
                filterable_fields=["verdict", "status"],
            )
        else:
            st.info("Select a JSONL file to view results")

    # Markdown tab
    with tabs[2]:
        if markdown_path and Path(markdown_path).exists():
            render_markdown_viewer(markdown_path)
        else:
            st.info("Select a markdown file to view summary")
