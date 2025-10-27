"""Model registry table component."""

import streamlit as st
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from imageworks.gui.config import MODEL_REGISTRY_PATH


@st.cache_data(ttl=60, show_spinner="Loading registry...")
def load_registry(registry_path: str) -> pd.DataFrame:
    """
    Load model registry as DataFrame (CACHED).

    Args:
        registry_path: Path to registry JSON

    Returns:
        DataFrame with model information
    """
    try:
        with open(registry_path, "r") as f:
            registry = json.load(f)

        models = registry.get("models", [])

        if not models:
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(models)

        # Extract nested fields
        if "metadata" in df.columns:
            # Extract common metadata fields
            for field in ["format", "quantization", "parameter_count"]:
                df[field] = df["metadata"].apply(
                    lambda x: x.get(field, "") if isinstance(x, dict) else ""
                )

        return df

    except Exception as e:
        st.error(f"Failed to load registry: {e}")
        return pd.DataFrame()


def render_registry_table(
    registry_path: Optional[str] = None,
    key_prefix: str = "registry",
    filterable_columns: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Render model registry table with filtering and search.

    Args:
        registry_path: Path to registry JSON (defaults to config path)
        key_prefix: Unique prefix for widgets
        filterable_columns: Columns to enable filtering

    Returns:
        Selected model dict or None
    """

    if registry_path is None:
        registry_path = str(MODEL_REGISTRY_PATH)

    if not Path(registry_path).exists():
        st.error(f"❌ Registry not found: {registry_path}")
        return None

    # Load registry (cached)
    df = load_registry(registry_path)

    if df.empty:
        st.info("No models in registry")
        return None

    st.success(f"✅ Loaded {len(df)} models")

    # Filters
    if filterable_columns:
        st.markdown("### Filters")

        filter_cols = st.columns(len(filterable_columns))

        for i, col_name in enumerate(filterable_columns):
            if col_name in df.columns:
                with filter_cols[i]:
                    unique_values = df[col_name].dropna().unique()

                    if len(unique_values) > 0:
                        selected = st.multiselect(
                            col_name.replace("_", " ").title(),
                            options=sorted([str(v) for v in unique_values]),
                            key=f"{key_prefix}_filter_{col_name}",
                        )

                        if selected:
                            df = df[df[col_name].astype(str).isin(selected)]

    # Search
    search = st.text_input(
        "🔍 Search", key=f"{key_prefix}_search", help="Search in model names and paths"
    )

    if search:
        # Search in name and path columns
        mask = df.apply(lambda row: search.lower() in str(row).lower(), axis=1)
        df = df[mask]
        st.info(f"Found {len(df)} matches")

    # Display table
    st.markdown("### Models")

    # Select columns to display
    display_columns = ["name", "format", "quantization", "parameter_count", "path"]
    available_columns = [col for col in display_columns if col in df.columns]

    if available_columns:
        st.dataframe(
            df[available_columns],
            use_container_width=True,
            height=400,
        )
    else:
        st.dataframe(df, use_container_width=True, height=400)

    # Model selection
    if len(df) > 0:
        model_names = df["name"].tolist() if "name" in df.columns else df.index.tolist()

        selected_name = st.selectbox(
            "Select Model", options=[""] + model_names, key=f"{key_prefix}_select"
        )

        if selected_name:
            # Get full model data
            if "name" in df.columns:
                selected_row = df[df["name"] == selected_name].iloc[0]
            else:
                selected_row = df.loc[selected_name]

            return selected_row.to_dict()

    return None


def render_registry_stats(registry_path: Optional[str] = None) -> None:
    """
    Render registry statistics.

    Args:
        registry_path: Path to registry JSON
    """

    if registry_path is None:
        registry_path = str(MODEL_REGISTRY_PATH)

    if not Path(registry_path).exists():
        st.warning("Registry not found")
        return

    df = load_registry(registry_path)

    if df.empty:
        st.info("No models in registry")
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Models", len(df))

    with col2:
        if "format" in df.columns:
            formats = df["format"].value_counts()
            st.metric("Formats", len(formats))
            for fmt, count in formats.items():
                st.caption(f"{fmt}: {count}")

    with col3:
        if "path" in df.columns:
            total_size = 0
            for path_str in df["path"].dropna():
                path = Path(path_str)
                if path.exists():
                    if path.is_file():
                        total_size += path.stat().st_size
                    elif path.is_dir():
                        total_size += sum(
                            f.stat().st_size for f in path.rglob("*") if f.is_file()
                        )

            st.metric("Total Size", f"{total_size / (1024**3):.1f} GB")
