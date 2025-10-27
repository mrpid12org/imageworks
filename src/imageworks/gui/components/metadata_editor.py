"""Metadata editor component for tag preview and editing."""

import streamlit as st
from typing import List, Dict, Any
from pathlib import Path


def render_metadata_editor(
    images_with_tags: List[Dict[str, Any]],
    key_prefix: str = "metadata_editor",
) -> List[Dict[str, Any]]:
    """
    Render metadata editor for reviewing and editing tags.

    Args:
        images_with_tags: List of dicts with 'path', 'caption', 'keywords', 'description'
        key_prefix: Unique prefix for widgets

    Returns:
        Updated list of images with edited tags
    """

    if not images_with_tags:
        st.info("No tags to edit")
        return []

    st.subheader(f"📝 Edit Tags ({len(images_with_tags)} images)")

    # Bulk operations
    st.markdown("### Bulk Operations")

    col1, col2, col3 = st.columns(3)

    with col1:
        find_text = st.text_input(
            "Find", key=f"{key_prefix}_find", help="Find this text in keywords"
        )

    with col2:
        replace_text = st.text_input(
            "Replace with", key=f"{key_prefix}_replace", help="Replace with this text"
        )

    with col3:
        if st.button("🔄 Apply to All", key=f"{key_prefix}_bulk_replace"):
            if find_text and replace_text:
                for img_data in images_with_tags:
                    if "keywords" in img_data and img_data["keywords"]:
                        keywords = img_data["keywords"]
                        if isinstance(keywords, str):
                            keywords = keywords.replace(find_text, replace_text)
                        elif isinstance(keywords, list):
                            keywords = [
                                kw.replace(find_text, replace_text) for kw in keywords
                            ]
                        img_data["keywords"] = keywords

                st.success(
                    f"Replaced '{find_text}' with '{replace_text}' in all images"
                )
                st.rerun()

    # Approval toggles
    col1, col2 = st.columns(2)

    with col1:
        if st.button("✅ Approve All", key=f"{key_prefix}_approve_all"):
            for img_data in images_with_tags:
                img_data["approved"] = True
            st.success("All images approved")
            st.rerun()

    with col2:
        if st.button("❌ Reject All", key=f"{key_prefix}_reject_all"):
            for img_data in images_with_tags:
                img_data["approved"] = False
            st.warning("All images rejected")
            st.rerun()

    st.markdown("---")

    # Individual image editors
    st.markdown("### Individual Images")

    # Pagination
    items_per_page = 10
    total_pages = (len(images_with_tags) + items_per_page - 1) // items_per_page

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
        page_images = images_with_tags[start_idx:end_idx]
    else:
        page_images = images_with_tags
        start_idx = 0

    # Edit each image
    for i, img_data in enumerate(page_images):
        actual_idx = start_idx + i
        img_path = img_data.get("path", "")

        with st.expander(f"📷 {Path(img_path).name}", expanded=False):
            col1, col2 = st.columns([1, 2])

            with col1:
                # Show image thumbnail
                if Path(img_path).exists():
                    try:
                        from imageworks.gui.components.image_viewer import load_image

                        img_array = load_image(img_path, max_size=300)
                        st.image(img_array, use_container_width=True)
                    except Exception as e:
                        st.error(f"Failed to load image: {e}")

            with col2:
                # Caption editor
                caption = img_data.get("caption", "")
                edited_caption = st.text_area(
                    "Caption",
                    value=caption,
                    height=80,
                    key=f"{key_prefix}_caption_{actual_idx}",
                )
                if edited_caption != caption:
                    images_with_tags[actual_idx]["caption"] = edited_caption

                # Keywords editor
                keywords = img_data.get("keywords", "")
                if isinstance(keywords, list):
                    keywords = ", ".join(keywords)

                edited_keywords = st.text_area(
                    "Keywords (comma-separated)",
                    value=keywords,
                    height=60,
                    key=f"{key_prefix}_keywords_{actual_idx}",
                )
                if edited_keywords != keywords:
                    # Convert back to list
                    kw_list = [
                        k.strip() for k in edited_keywords.split(",") if k.strip()
                    ]
                    images_with_tags[actual_idx]["keywords"] = kw_list

                # Description editor
                description = img_data.get("description", "")
                edited_description = st.text_area(
                    "Description",
                    value=description,
                    height=100,
                    key=f"{key_prefix}_description_{actual_idx}",
                )
                if edited_description != description:
                    images_with_tags[actual_idx]["description"] = edited_description

            # Approval checkbox
            approved = img_data.get("approved", True)
            new_approved = st.checkbox(
                "✅ Approve for writing",
                value=approved,
                key=f"{key_prefix}_approve_{actual_idx}",
            )
            if new_approved != approved:
                images_with_tags[actual_idx]["approved"] = new_approved

    return images_with_tags


def render_compact_tag_list(
    images_with_tags: List[Dict[str, Any]],
    show_approved_only: bool = False,
) -> None:
    """
    Render compact list of tags for final review.

    Args:
        images_with_tags: List of image metadata dicts
        show_approved_only: Whether to show only approved items
    """

    st.subheader("📋 Tags Summary")

    # Filter if needed
    display_images = images_with_tags
    if show_approved_only:
        display_images = [img for img in images_with_tags if img.get("approved", True)]

    if not display_images:
        st.warning("No images to display")
        return

    st.write(f"Showing {len(display_images)} images")

    # Display as table
    rows = []
    for img_data in display_images:
        img_name = Path(img_data.get("path", "")).name
        caption = (
            img_data.get("caption", "")[:50] + "..."
            if len(img_data.get("caption", "")) > 50
            else img_data.get("caption", "")
        )

        keywords = img_data.get("keywords", [])
        if isinstance(keywords, list):
            keywords_str = ", ".join(keywords[:5])
            if len(keywords) > 5:
                keywords_str += f" (+{len(keywords)-5} more)"
        else:
            keywords_str = str(keywords)[:50]

        approved_str = "✅" if img_data.get("approved", True) else "❌"

        rows.append(
            {
                "Image": img_name,
                "Caption": caption,
                "Keywords": keywords_str,
                "Approved": approved_str,
            }
        )

    import pandas as pd

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, height=400)

    # Stats
    approved_count = sum(1 for img in display_images if img.get("approved", True))
    st.write(f"**Approved:** {approved_count} / {len(display_images)}")
