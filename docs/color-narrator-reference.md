# Color-Narrator Reference Documentation\n\n> **🤖 AI Models & Prompting**: For comprehensive information about all AI models used across imageworks, model experiments, prompting strategies, and hardware requirements, see the [AI Models and Prompting Guide](./ai-models-and-prompting.md).\n\nThe **Color-Narrator** is a VLM-guided system for generating natural language descriptions of residual color in monochrome competition images. It integrates with mono-checker analysis data and uses the Qwen2-VL-2B vision-language model to create professional metadata descriptions.\n\n## Table of Contents\n\n1. [Overview](#overview)\n2. [Architecture](#architecture)\n3. [Installation & Setup](#installation--setup)\n4. [Usage](#usage)\n5. [Configuration](#configuration)\n6. [API Reference](#api-reference)\n7. [Development](#development)\n8. [Troubleshooting](#troubleshooting)\n\n## Overview\n\n### Purpose\nColor-Narrator analyzes JPEG images that should be monochrome but contain residual color, then generates professional natural language descriptions suitable for XMP metadata embedding. It's designed for photography competition workflows where accurate color contamination documentation is essential.\n\n### Key Features\n- **VLM Integration**: Uses Qwen2-VL-2B for vision-language inference\n- **Mono-Checker Integration**: Leverages existing monochrome analysis data\n- **XMP Metadata**: Embeds structured color descriptions in JPEG files\n- **Batch Processing**: Handles large collections efficiently\n- **Professional Output**: Generates competition-ready metadata descriptions\n- **CUDA Acceleration**: Optimized for RTX 4080/6000 Pro with CUDA 12.9\n\n### Workflow Integration\n```\nMono-Checker Analysis → Color-Narrator VLM Processing → XMP Metadata Embedding\n```### Quick Start
```bash
# Start vLLM server (see vLLM Deployment Guide for details)
./start_vllm_server.py --model Qwen2-VL-2B-Instruct --port 8000

# Basic narration workflow
uv run imageworks-color-narrator narrate \
  --images ./competition_images \
  --overlays ./lab_overlays \
  --mono-jsonl ./mono_results.jsonl

# Validate existing narrations
uv run imageworks-color-narrator validate \
  --images ./competition_images \
  --mono-jsonl ./mono_results.jsonl

# Enhanced mono analysis (hybrid approach)
uv run imageworks-color-narrator enhance-mono \
  --mono-jsonl ./mono_results.jsonl \
  --limit 50 \
  --summary ./outputs/summaries/enhancement_summary.md
```e descriptions of residual color in monochrome competition images. It integrates with mono-checker analysis data and uses the Qwen2-VL-2B vision-language model to create professional metadata descriptions.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Installation & Setup](#installation--setup)
4. [Usage](#usage)
5. [Configuration](#configuration)
6. [API Reference](#api-reference)
7. [Development](#development)
8. [Troubleshooting](#troubleshooting)

## Overview

### Purpose
Color-Narrator analyzes JPEG images that should be monochrome but contain residual color, then generates professional natural language descriptions suitable for XMP metadata embedding. It's designed for photography competition workflows where accurate color contamination documentation is essential.

### Key Features
- **VLM Integration**: Uses Qwen2-VL-2B for vision-language inference
- **Mono-Checker Integration**: Leverages existing monochrome analysis data
- **XMP Metadata**: Embeds structured color descriptions in JPEG files
- **Batch Processing**: Handles large collections efficiently
- **Professional Output**: Generates competition-ready metadata descriptions
- **CUDA Acceleration**: Optimized for RTX 4080/6000 Pro with CUDA 12.9

### Workflow Integration
```
Mono-Checker Analysis → Color-Narrator VLM Processing → XMP Metadata Embedding
```

## Architecture

### Core Components

#### 1. **CLI Interface** (`cli/main.py`)
- **Commands**: `narrate`, `validate`
- **Framework**: Typer with rich help and parameter validation
- **Integration**: Direct integration with mono-checker output formats

#### 2. **VLM Client** (`core/vlm.py`)
- **Model**: Qwen2-VL-2B-Instruct via vLLM server
- **API**: OpenAI-compatible REST interface
- **Features**: Base64 image encoding, structured prompts, confidence estimation

#### 3. **Data Loader** (`core/data_loader.py`)
- **Inputs**: JPEG originals, LAB overlay PNGs, mono-checker JSONL
- **Validation**: File existence, format consistency, contamination filtering
- **Batching**: Configurable batch sizes for efficient processing

#### 4. **Orchestration** (`core/narrator.py`)
- **Pipeline**: Coordinates data loading, VLM inference, metadata writing
- **Error Handling**: Robust error recovery and reporting
- **Progress Tracking**: Detailed progress and statistics reporting

#### 5. **Metadata System** (`core/metadata.py`)
- **XMP Integration**: Structured metadata embedding (via sidecar JSON in development)
- **Versioning**: Metadata schema versioning and migration support
- **Backup**: Automatic file backup before modification

#### 6. **Shared Libraries** (`libs/personal_tagger/`)
- **Color Analysis**: LAB color space analysis, chroma/hue calculations
- **VLM Utils**: Prompt management, batch processing, model lifecycle
- **Image Utils**: Loading, processing, format validation, enhancement

### Data Flow
```
1. Load JPEG originals + LAB overlays + mono JSONL
2. Filter by contamination level and file availability
3. Generate VLM prompts with mono analysis context
4. Process through Qwen2-VL-2B inference
5. Parse responses for color regions and confidence
6. Structure metadata with validation
7. Embed XMP metadata in JPEG files
8. Generate processing reports and statistics
```

## Installation & Setup

### Prerequisites
- **Python**: 3.9+ with CUDA support
- **CUDA**: 12.9 (for RTX 4080/6000 Pro compatibility)
- **PyTorch**: >=2.3.0 with CUDA support
- **vLLM Server**: Running Qwen2-VL-7B-Instruct model

### Installation
```bash
# Install imageworks package with color-narrator dependencies
uv sync

# Verify installation
uv run imageworks-color-narrator --help
```

### VLM Server Setup

> **⚠️ Important**: See the [vLLM Deployment Guide](./vllm-deployment-guide.md) for detailed setup instructions, troubleshooting, and lessons learned from production deployment.

**Quick Start** (for 16GB VRAM):
```bash
# Start vLLM server with Qwen2-VL-2B model (recommended for 16GB VRAM)
nohup uv run vllm serve ./models/Qwen2-VL-2B-Instruct \
  --served-model-name Qwen2-VL-2B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --trust-remote-code \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.8 \
  > vllm_server.log 2>&1 &

# For RTX 6000 Pro (48GB VRAM), you can use the 7B model:
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --port 8000 \
  --gpu-memory-utilization 0.8
```

### Configuration
Color-Narrator reads configuration from `pyproject.toml`:

```toml
[tool.imageworks.color_narrator]
vlm_base_url = "http://localhost:8000/v1"
vlm_model = "Qwen/Qwen2-VL-7B-Instruct"
default_batch_size = 4
min_contamination_level = 0.1
# ... additional settings
```

## Usage

### Basic Commands

#### Generate Color Descriptions
```bash
# Basic narration with auto-discovery
uv run imageworks-color-narrator narrate

# Specify paths explicitly
uv run imageworks-color-narrator narrate \
  --images ./originals \
  --overlays ./overlays \
  --mono-jsonl ./mono_results.jsonl

# Custom batch size and debug mode
uv run imageworks-color-narrator narrate \
  --batch-size 8 \
  --debug
```

#### Validate Existing Descriptions
```bash
# Validate existing XMP metadata
uv run imageworks-color-narrator validate \
  --images ./originals \
  --mono-jsonl ./mono_results.jsonl
```

#### Dry Run Mode
```bash
# Preview processing without making changes
uv run imageworks-color-narrator narrate \
  --images ./originals \
  --overlays ./overlays \
  --mono-jsonl ./mono_results.jsonl \
  --dry-run
```

### Typical Workflow

1. **Run Mono-Checker Analysis**
   ```bash
   uv run imageworks-mono ./images --jsonl-out mono_results.jsonl
   uv run imageworks-mono visualize ./images
   ```

2. **Generate Color Narrations**
   ```bash
   uv run imageworks-color-narrator narrate \
     --images ./images \
     --overlays ./images \
     --mono-jsonl mono_results.jsonl \
     --batch-size 4
   ```

3. **Validate Results**
   ```bash
   uv run imageworks-color-narrator validate \
     --images ./images \
     --mono-jsonl mono_results.jsonl
   ```

### Integration Examples

#### Python API Usage
```python
from imageworks.apps.personal_tagger.color_narrator.core import (
    ColorNarrator, NarrationConfig
)

# Configure processing
config = NarrationConfig(
    images_dir=Path("./images"),
    overlays_dir=Path("./overlays"),
    mono_jsonl=Path("./mono_results.jsonl"),
    batch_size=4,
    dry_run=False
)

# Process all images
narrator = ColorNarrator(config)
results = narrator.process_all()

# Review results
for result in results:
    if result.vlm_response:
        print(f"{result.item.image_path.name}: {result.vlm_response.description}")
```

#### Batch Processing Script
```python
from pathlib import Path
from imageworks.apps.personal_tagger.color_narrator.core import (
    ColorNarratorDataLoader, DataLoaderConfig
)

# Load and validate data
config = DataLoaderConfig(
    images_dir=Path("./competition_images"),
    overlays_dir=Path("./lab_overlays"),
    mono_jsonl=Path("./mono_analysis.jsonl"),
    min_contamination_level=0.15
)

loader = ColorNarratorDataLoader(config)
stats = loader.get_statistics()
print(f"Found {stats['valid_items']} images ready for processing")
```

## Configuration

### Core Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `vlm_base_url` | str | `"http://localhost:8000/v1"` | vLLM server endpoint |
| `vlm_model` | str | `"Qwen/Qwen2-VL-7B-Instruct"` | Model identifier |
| `vlm_timeout` | int | `120` | Request timeout (seconds) |
| `default_batch_size` | int | `4` | Processing batch size |
| `min_contamination_level` | float | `0.1` | Minimum contamination to process |

### Path Configuration

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `default_images_dir` | str | `"outputs/originals"` | Default JPEG images directory |
| `default_overlays_dir` | str | `"outputs/overlays"` | Default LAB overlay directory |
| `default_mono_jsonl` | str | `"outputs/results/mono_results.jsonl"` | Default mono analysis file |

### Processing Options

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `require_overlays` | bool | `true` | Require overlay files for processing |
| `backup_original_files` | bool | `true` | Backup files before modification |
| `overwrite_existing_metadata` | bool | `false` | Overwrite existing XMP metadata |
| `max_concurrent_requests` | int | `4` | Maximum concurrent VLM requests |

### Color Analysis

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `chroma_threshold` | float | `5.0` | Chroma threshold for color detection |
| `min_region_size` | int | `100` | Minimum pixels for color region |
| `high_chroma_threshold` | float | `15.0` | High chroma intensity threshold |

## API Reference

### Core Classes

#### `ColorNarrator`
Main orchestration class for color narration processing.

**Methods:**
- `process_all() -> List[ProcessingResult]`: Process all valid items
- `validate_existing(images_dir: Path) -> Dict[str, Any]`: Validate existing metadata

#### `VLMClient`
Client for VLM inference with Qwen2-VL-7B.

**Methods:**
- `infer_single(request: VLMRequest) -> VLMResponse`: Single image inference
- `infer_batch(requests: List[VLMRequest]) -> List[VLMResponse]`: Batch processing
- `health_check() -> bool`: Check server availability

#### `ColorNarratorDataLoader`
Loads and validates processing data sources.

**Methods:**
- `load() -> None`: Load all data sources
- `get_items(batch_size: Optional[int]) -> Iterator[List[ColorNarratorItem]]`: Get processing batches
- `get_statistics() -> Dict[str, Any]`: Get data statistics

#### `XMPMetadataWriter`
Handles XMP metadata reading and writing.

**Methods:**
- `write_metadata(image_path: Path, metadata: ColorNarrationMetadata) -> bool`: Write metadata
- `read_metadata(image_path: Path) -> Optional[ColorNarrationMetadata]`: Read metadata
- `has_color_narration(image_path: Path) -> bool`: Check metadata presence

### Data Classes

#### `ColorNarrationMetadata`
Structured metadata for color narration results.

**Fields:**
- `description: str`: Natural language color description
- `confidence_score: float`: VLM confidence (0.0-1.0)
- `color_regions: List[str]`: Identified color regions
- `processing_timestamp: str`: ISO timestamp
- `mono_contamination_level: float`: Contamination level from mono analysis
- `vlm_model: str`: VLM model used for inference
- `vlm_processing_time: float`: Inference time (seconds)

#### `VLMResponse`
Response from VLM inference.

**Fields:**
- `description: str`: Generated color description
- `confidence: float`: Estimated confidence
- `color_regions: List[str]`: Extracted color regions
- `processing_time: float`: Processing time
- `error: Optional[str]`: Error message if failed

## Development

### Project Structure
```
src/imageworks/apps/personal_tagger/color_narrator/
├── cli/
│   ├── __init__.py
│   └── main.py          # Typer CLI commands
├── core/
│   ├── __init__.py
│   ├── vlm.py          # VLM client and inference
│   ├── data_loader.py  # Data loading and validation
│   ├── narrator.py     # Main orchestration
│   └── metadata.py     # XMP metadata handling
├── api/                # Future FastAPI endpoints
└── __init__.py

src/imageworks/libs/personal_tagger/
├── __init__.py
├── color_analysis.py   # Color space analysis utilities
├── vlm_utils.py       # VLM management utilities
└── image_utils.py     # Image processing utilities

tests/personal_tagger/color_narrator/
├── __init__.py
├── conftest.py        # Test fixtures
├── test_vlm.py        # VLM client tests
├── test_data_loader.py # Data loading tests
├── test_narrator.py   # Orchestration tests
├── test_metadata.py   # Metadata tests
└── test_cli.py        # CLI tests
```

### Running Tests
```bash
# Run all color-narrator tests
uv run pytest tests/personal_tagger/color_narrator/ -v

# Run specific test module
uv run pytest tests/personal_tagger/color_narrator/test_vlm.py -v

# Run with coverage
uv run pytest tests/personal_tagger/color_narrator/ --cov=imageworks.apps.personal_tagger.color_narrator
```

### Code Style
```bash
# Format code
uv run black src/imageworks/apps/personal_tagger/color_narrator/

# Lint code
uv run ruff check src/imageworks/apps/personal_tagger/color_narrator/

# Type checking
uv run mypy src/imageworks/apps/personal_tagger/color_narrator/
```

### Adding Custom Prompt Templates
```python
from imageworks.libs.personal_tagger.vlm_utils import VLMPromptManager, VLMPromptTemplate

# Create custom template
custom_template = VLMPromptTemplate(
    name="custom_analysis",
    template="Analyze this image for: {focus_area}\n\nContext: {context}",
    required_params=["focus_area"],
    optional_params=["context"],
    description="Custom analysis template"
)

# Register template
prompt_manager = VLMPromptManager()
prompt_manager.register_template(custom_template)
```

## Troubleshooting

### Common Issues

#### VLM Server Connection Failed
**Problem**: `VLM server is not available` error
**Solutions**:
1. Verify vLLM server is running: `curl http://localhost:8000/v1/models`
2. Check server logs for GPU memory issues
3. Adjust `vlm_timeout` in configuration
4. Verify CUDA compatibility with `torch.cuda.is_available()`

#### Out of GPU Memory
**Problem**: CUDA out of memory errors during inference
**Solutions**:
1. Reduce `default_batch_size` in configuration
2. Adjust vLLM `--gpu-memory-utilization` parameter
3. Use `--max-concurrent-requests 1` for sequential processing
4. Monitor GPU memory: `nvidia-smi`

#### Missing Overlay Files
**Problem**: "No overlay found" messages
**Solutions**:
1. Verify overlay naming patterns match mono-checker output
2. Set `require_overlays = false` in configuration if overlays are optional
3. Check overlay file extensions in `DataLoaderConfig`

#### XMP Metadata Issues
**Problem**: Metadata not being written or read correctly
**Solutions**:
1. Verify file write permissions
2. Check backup file creation (indicates write attempt)
3. For development, check sidecar `.cn_metadata.json` files
4. Enable debug mode: `--debug` flag

#### Performance Issues
**Problem**: Slow processing or high memory usage
**Solutions**:
1. Reduce image sizes with `target_size` parameter
2. Implement image caching for repeated processing
3. Use `--dry-run` mode for testing without processing
4. Monitor system resources during processing

### Debug Mode
Enable detailed logging and intermediate file saving:

```bash
uv run imageworks-color-narrator narrate \
  --images ./test_images \
  --debug \
  --dry-run
```

### Log Configuration
Adjust logging level in configuration:
```toml
[tool.imageworks.color_narrator]
log_level = "DEBUG"  # DEBUG, INFO, WARNING, ERROR
debug_save_intermediate = true
debug_output_dir = "outputs/debug"
```

### Health Checks
Verify system components:

```python
from imageworks.apps.personal_tagger.color_narrator.core import VLMClient

# Test VLM server
client = VLMClient()
if client.health_check():
    print("✅ VLM server is healthy")
else:
    print("❌ VLM server is not responding")
```

### Performance Monitoring
```python
from imageworks.apps.personal_tagger.color_narrator.core import ColorNarrator
import time

# Monitor processing performance
start_time = time.time()
results = narrator.process_all()
total_time = time.time() - start_time

successful = sum(1 for r in results if r.vlm_response and not r.error)
avg_time = total_time / len(results) if results else 0

print(f"Processed {successful}/{len(results)} images in {total_time:.1f}s")
print(f"Average time per image: {avg_time:.1f}s")
```

---

**Color-Narrator** integrates seamlessly with the mono-checker workflow to provide professional color contamination documentation for photography competitions. For additional support, see the main project documentation and test examples.
