"""In-app help documentation component."""

import streamlit as st
from typing import Optional


# Help content database
HELP_TOPICS = {
    "mono_checker": {
        "title": "🖼️ Mono Checker",
        "description": "Detects monochrome (grayscale/black-and-white) images using Vision-Language Models.",
        "sections": {
            "Overview": """
The Mono Checker analyzes images to determine if they are monochrome (grayscale or black-and-white).
It uses Vision-Language Models (VLMs) to provide intelligent detection with confidence scores.
            """,
            "Presets": """
**Quick**: Fast checks with standard thresholds
- Threshold: 0.8
- Timeout: 10s
- Best for: Quick filtering of obviously monochrome images

**Standard**: Balanced accuracy and speed
- Threshold: 0.7
- Timeout: 30s
- Best for: Most use cases

**Thorough**: Maximum accuracy
- Threshold: 0.5
- Timeout: 60s
- Best for: Critical accuracy needs, mixed content
            """,
            "Outputs": """
**Verdict**: `monochrome` or `color`
**Certainty**: Confidence score (0.0-1.0)
**Reasoning**: Model's explanation (if supported)

Results saved to: `outputs/mono_results.jsonl`
            """,
            "Tips": """
- Start with Standard preset
- Use Quick for large batches
- Enable dry-run to preview without processing
- Check Results tab to verify detections
            """,
        },
    },
    "image_similarity": {
        "title": "🔍 Image Similarity",
        "description": "Find duplicate or similar images using advanced embedding strategies.",
        "sections": {
            "Overview": """
The Image Similarity Checker compares candidate images against a library to find duplicates or similar images.
It uses multiple embedding strategies and can combine results for robust matching.
            """,
            "Presets": """
**Quick**: Single-strategy fast matching
- Strategy: phash only
- Thresholds: relaxed
- Best for: Exact/near-duplicate detection

**Standard**: Balanced multi-strategy
- Strategies: phash + siglip
- Thresholds: moderate
- Best for: Most use cases

**Thorough**: All strategies with strict thresholds
- Strategies: phash + siglip + resnet + metadata
- Thresholds: strict
- Best for: Comprehensive similarity search
            """,
            "Strategies": """
**phash**: Perceptual hashing (fast, good for exact duplicates)
**siglip**: Semantic embeddings (best for similar content)
**resnet**: Deep learning features (balanced)
**open_clip**: Alternative semantic embeddings
**simple**: Basic image metrics
**metadata**: File metadata comparison
            """,
            "Outputs": """
Results include:
- Candidate image path
- List of matches with similarity scores
- Match details (size, dimensions, path)

Results saved to: `outputs/similarity_results.jsonl`
            """,
            "Tips": """
- Use Quick for finding exact duplicates
- Use Standard for general similarity search
- Combine strategies for better results
- Adjust thresholds based on false positive rate
            """,
        },
    },
    "personal_tagger": {
        "title": "🏷️ Personal Tagger",
        "description": "Add AI-generated metadata tags to image files.",
        "sections": {
            "Overview": """
The Personal Tagger uses Vision-Language Models to analyze images and generate descriptive tags.
Tags are written to image metadata (EXIF/XMP) for organization and searchability.
            """,
            "Workflow": """
1. **Configure**: Select images and tagging options
2. **Preview**: Run dry-run to see proposed tags
3. **Edit**: Review and modify tags before committing
4. **Commit**: Write approved tags to image files
            """,
            "Tag Categories": """
**Content**: What's in the image (objects, scenes, people)
**Style**: Artistic style, medium, technique
**Quality**: Technical quality assessments
**Mood**: Emotional tone, atmosphere
**Custom**: User-defined categories
            """,
            "Outputs": """
**Dry-run mode**: Displays proposed tags without writing
**Commit mode**: Writes tags to image metadata

Original files can be backed up automatically.
            """,
            "Tips": """
- ALWAYS run Preview first
- Review tags in Edit tab before committing
- Enable backup originals for safety
- Use bulk find/replace for consistency
- Approve/reject individual tags
            """,
        },
    },
    "color_narrator": {
        "title": "🎨 Color Narrator",
        "description": "Generate natural language color descriptions for images.",
        "sections": {
            "Overview": """
The Color Narrator analyzes image color palettes and generates human-readable descriptions.
It can work standalone or integrate with Mono Checker results.
            """,
            "Presets": """
**Quick**: Basic color descriptions
- Concise output
- Fast processing
- Best for: Batch processing

**Standard**: Detailed color analysis
- Rich descriptions
- Moderate detail
- Best for: Most use cases

**Thorough**: Comprehensive color narratives
- Extensive analysis
- Maximum detail
- Best for: Detailed documentation
            """,
            "Pipeline Mode": """
Enable to automatically import Mono Checker results:
- Filter by verdict (mono/color/all)
- Process only relevant images
- Combine workflows seamlessly
            """,
            "Outputs": """
Results include:
- Dominant colors
- Color palette
- Natural language description
- Technical color metrics

Results saved to: `outputs/narrator_results.jsonl`
            """,
            "Tips": """
- Use pipeline mode with Mono Checker
- Filter by verdict to process only color images
- Standard preset sufficient for most needs
- Review descriptions in Results tab
            """,
        },
    },
    "model_manager": {
        "title": "🎯 Model Manager",
        "description": "Manage AI models, backends, and model profiles.",
        "sections": {
            "Overview": """
The Model Manager provides a central hub for:
- Browsing available models
- Downloading new models
- Monitoring backend services
- Managing model profiles
            """,
            "Registry": """
View all available models with details:
- Model name and ID
- Type (VLM, embedding, etc.)
- Size and format
- Status (available, downloading, etc.)

Search and filter to find models quickly.
            """,
            "Download": """
Download models from Hugging Face:
- Enter model ID (e.g., microsoft/Phi-3.5-vision-instruct)
- Select variant (GGUF quantization)
- Monitor download progress
- Models saved to configured directory
            """,
            "Backends": """
Monitor backend service health:
- **vLLM**: High-performance inference
- **LMDeploy**: Alternative inference engine
- **Ollama**: Local model serving
- **Chat Proxy**: API gateway

View status, models loaded, and test connections.
            """,
            "Tips": """
- Check backend health before running jobs
- Download models during off-peak hours
- Use model profiles for consistent configurations
- Monitor disk space before downloads
            """,
        },
    },
    "results_browser": {
        "title": "📊 Results Browser",
        "description": "View and analyze results from all ImageWorks modules.",
        "sections": {
            "Overview": """
The Results Browser provides unified access to:
- All module outputs
- Job history with re-run capability
- Statistics and metrics
            """,
            "Browse Results": """
View results from any module:
- Select module (Mono, Similarity, etc.)
- Browse output files
- Filter and search results
- View images and metadata
            """,
            "Job History": """
Track all executed jobs:
- View job parameters
- Check execution status
- Re-run previous jobs
- Filter by module or status
            """,
            "Statistics": """
View aggregate statistics:
- Per-module metrics (success rate, count)
- Storage usage
- Processing times
- Error rates
            """,
            "Tips": """
- Use history to re-run successful configurations
- Check statistics to optimize workflows
- Filter results by module for focused analysis
- Export results for external analysis
            """,
        },
    },
    "presets": {
        "title": "🎛️ Presets System",
        "description": "Simplified configuration using predefined presets.",
        "sections": {
            "Overview": """
The Presets System reduces complexity by providing:
- Pre-configured settings for common use cases
- Quick/Standard/Thorough options
- Advanced overrides for experts
            """,
            "Preset Levels": """
**Quick**: Fast processing, lower accuracy
- Best for: Large batches, obvious cases
- Processing: 10-30s per item
- Accuracy: 85-90%

**Standard**: Balanced (RECOMMENDED)
- Best for: Most use cases
- Processing: 30-60s per item
- Accuracy: 90-95%

**Thorough**: Maximum accuracy, slower
- Best for: Critical applications
- Processing: 60-120s per item
- Accuracy: 95-99%
            """,
            "Custom Overrides": """
Expand "Show Advanced Options" to customize:
- All CLI flags available
- Override preset defaults
- Save custom configurations
- Export/import settings
            """,
            "Tips": """
- Start with Standard preset
- Use Quick for initial testing
- Enable advanced options only if needed
- Save custom configurations as presets
            """,
        },
    },
    "caching": {
        "title": "⚡ Caching & Performance",
        "description": "Understanding Streamlit caching for optimal performance.",
        "sections": {
            "Overview": """
The GUI uses aggressive caching to prevent expensive re-computation:
- File scanning cached for 5 minutes
- Model loading cached indefinitely
- Backend health checks cached for 10 seconds
            """,
            "Cache Types": """
**@st.cache_data**: For data/file operations
- Expires after TTL (time-to-live)
- Cleared manually via Settings
- Example: File scanning, JSONL parsing

**@st.cache_resource**: For models/connections
- Never expires automatically
- Cleared manually or on app restart
- Example: Loaded models, database connections
            """,
            "Performance Tips": """
- Clear cache if seeing stale data
- Restart app to reload models
- Large file operations cached automatically
- Backend status refreshes every 10s
            """,
            "Troubleshooting": """
**Stale data**: Clear cache in Settings
**High memory**: Restart app or clear resource cache
**Slow performance**: Check cache TTL settings
**Backend not updating**: Wait 10s or clear cache
            """,
        },
    },
    "troubleshooting": {
        "title": "🔧 Troubleshooting",
        "description": "Common issues and solutions.",
        "sections": {
            "Backend Connection": """
**Issue**: Backend shows "Offline" or "Error"

**Solutions**:
1. Verify backend is running: Check terminal/logs
2. Check URL in Settings → Backends
3. Test connection with curl: `curl <backend_url>/health`
4. Restart backend service
5. Check firewall/network settings
            """,
            "Model Loading": """
**Issue**: Model not found or fails to load

**Solutions**:
1. Check model exists in registry
2. Verify model path in Settings → Paths
3. Download model via Model Manager
4. Check disk space
5. Verify model format compatibility
            """,
            "Processing Failures": """
**Issue**: Job fails or hangs

**Solutions**:
1. Check logs in `logs/chat_proxy.jsonl`
2. Reduce batch size
3. Increase timeout in Advanced Options
4. Enable debug mode in Settings
5. Try dry-run first to verify configuration
            """,
            "Performance Issues": """
**Issue**: GUI is slow or unresponsive

**Solutions**:
1. Clear cache: Settings → General → Clear All Caches
2. Reduce items per page: Settings → Appearance
3. Restart Streamlit app
4. Check system resources (RAM, GPU)
5. Disable unnecessary backends
            """,
            "File Access Errors": """
**Issue**: Cannot read/write files

**Solutions**:
1. Check file permissions
2. Verify paths exist: Settings → Paths
3. Check disk space
4. Try absolute paths instead of relative
5. Run with appropriate user permissions
            """,
        },
    },
}


def render_help_sidebar():
    """Render help icon in sidebar that opens help dialog."""
    if st.sidebar.button("❓ Help", use_container_width=True):
        st.session_state["show_help"] = True


def render_help_dialog():
    """Render help dialog if activated."""
    if st.session_state.get("show_help", False):
        with st.expander("📖 Help Center", expanded=True):
            render_help_browser()

            if st.button("✖ Close Help"):
                st.session_state["show_help"] = False
                st.rerun()


def render_help_browser():
    """Render browsable help content."""
    st.markdown("### 📚 Help Topics")

    # Topic selector
    topic_names = {
        "mono_checker": "🖼️ Mono Checker",
        "image_similarity": "🔍 Image Similarity",
        "personal_tagger": "🏷️ Personal Tagger",
        "color_narrator": "🎨 Color Narrator",
        "model_manager": "🎯 Model Manager",
        "results_browser": "📊 Results Browser",
        "presets": "🎛️ Presets System",
        "caching": "⚡ Caching & Performance",
        "troubleshooting": "🔧 Troubleshooting",
    }

    selected_topic = st.selectbox(
        "Select topic",
        options=list(topic_names.keys()),
        format_func=lambda x: topic_names[x],
        key="help_topic_selector",
    )

    # Display selected topic
    if selected_topic in HELP_TOPICS:
        topic = HELP_TOPICS[selected_topic]

        st.markdown(f"## {topic['title']}")
        st.markdown(topic["description"])
        st.markdown("---")

        # Render sections
        for section_title, section_content in topic["sections"].items():
            with st.expander(f"📄 {section_title}", expanded=False):
                st.markdown(section_content)


def render_inline_help(topic_key: str, section: Optional[str] = None):
    """
    Render inline help for specific topic.

    Args:
        topic_key: Help topic key
        section: Optional specific section to show
    """
    if topic_key not in HELP_TOPICS:
        return

    topic = HELP_TOPICS[topic_key]

    with st.expander(f"ℹ️ Help: {topic['title']}", expanded=False):
        st.markdown(topic["description"])

        if section and section in topic["sections"]:
            st.markdown(f"### {section}")
            st.markdown(topic["sections"][section])
        else:
            for section_title, section_content in topic["sections"].items():
                st.markdown(f"### {section_title}")
                st.markdown(section_content)
                st.markdown("---")


def render_tooltip(text: str, help_text: str):
    """
    Render text with tooltip.

    Args:
        text: Main text to display
        help_text: Tooltip text
    """
    st.markdown(f'<span title="{help_text}">{text} ℹ️</span>', unsafe_allow_html=True)


def quick_help(topic: str, message: str):
    """
    Show quick help info message.

    Args:
        topic: Help topic
        message: Quick help message
    """
    st.info(f"💡 **{topic}**: {message}")


def get_help_link(topic_key: str) -> str:
    """
    Get documentation link for topic.

    Args:
        topic_key: Help topic key

    Returns:
        Documentation link
    """
    doc_links = {
        "mono_checker": "docs/domains/mono/mono-overview.md",
        "image_similarity": "docs/guides/image-similarity-checker.md",
        "personal_tagger": "docs/domains/personal-tagger/overview.md",
        "color_narrator": "docs/domains/color-narrator/reference.md",
    }

    return doc_links.get(topic_key, "docs/index.md")
