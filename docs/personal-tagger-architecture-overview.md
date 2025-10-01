# Personal Tagger Architecture Overview

## Introduction

The Personal Tagger is a sophisticated AI-powered system for automatically enriching photo libraries with keywords, captions, and descriptions. It processes various image formats (JPG, JPEG, PNG, TIF, TIFF, ORF, CR2, CR3) and writes metadata directly into JPEG files or as XMP sidecars for RAW/TIFF files, making results visible in Lightroom and other photo management software.

## Operational Flow & Dependencies

The Personal Tagger **does not download or serve AI models on its own**—it orchestrates already-running Vision-Language Model (VLM) endpoints. A typical end-to-end workflow is:

1. **Fetch model weights** with the [Model Downloader](model-downloader.md):
   ```bash
   imageworks-download download "qwen-vl/Qwen2.5-VL-7B-Instruct-AWQ"
   imageworks-download download "llava-hf/llava-v1.6-mistral-7b-hf"
   ```
2. **Launch inference backends** (vLLM, LMDeploy, etc.) that expose OpenAI-compatible APIs using the downloaded paths (for example via `scripts/start_personal_tagger_backends.py --launch`).
3. **Run the Personal Tagger CLI** once the endpoints are healthy; the runner connects to those servers using the configured base URLs and model identifiers.

This separation ensures the tagger can target local GPU servers, remote inference stacks, or hosted APIs interchangeably. The downloader therefore runs *before* the personal tagger when you host models yourself, but the tagger can also point at existing infrastructure where the models are already available.

## Core System Architecture

The Personal Tagger is built as a **modular, pipeline-based system** with clear separation of concerns across multiple layers:

### 1. Core Processing Pipeline

The system implements a **three-stage sequential AI pipeline**:

1. **Caption Generation**: Creates concise 1-2 sentence descriptions
2. **Keyword Extraction**: Generates ~15 relevant searchable tags
3. **Description Generation**: Produces detailed accessibility-friendly text

Each stage can use different AI models and is independently configurable.

## Module Breakdown

### Core Modules (`src/imageworks/apps/personal_tagger/core/`)

#### `models.py` - Data Structures
**Purpose**: Defines the core data structures for the entire pipeline.

**Key Classes**:
- `KeywordPrediction`: Individual keywords with confidence scores
- `GenerationModels`: Tracks which AI models were used for each stage
- `PersonalTaggerRecord`: Complete output record for processed images
- `ensure_unique_keywords()`: Deduplication logic preserving highest confidence

**Data Flow**: All processing stages work with these standardized data structures.

#### `config.py` - Configuration Management
**Purpose**: Hierarchical configuration system supporting multiple sources.

**Key Components**:
- `PersonalTaggerSettings`: Default values loaded from `pyproject.toml`
- `PersonalTaggerConfig`: Runtime configuration with validation
- `load_config()`: Loads settings from project files and environment variables
- `build_runtime_config()`: Merges CLI arguments with defaults

**Configuration Hierarchy**:
```
pyproject.toml defaults → Environment variables → CLI arguments
```

#### `prompts.py` - Prompt Management
**Purpose**: Sophisticated prompt templating system with multiple profiles.

**Key Components**:
- `StagePrompt`: Individual prompt templates with token limits and context variables
- `TaggerPromptProfile`: Contains prompts for all three stages (caption, keywords, description)
- Built-in profiles: "default", "narrative_v2", and "phototools_prompt" with distinct tone/style approaches
- Template rendering with dynamic context substitution

**Prompt Profiles**:
- **Default**: Concise, factual outputs optimized for metadata fields
- **Narrative v2**: More evocative, storyteller tone with broader keyword diversity
- **PhotoTools Prompt**: PhotoTools-inspired phrasing with expert keyword emphasis

#### `inference.py` - AI Orchestration
**Purpose**: Manages the complete AI inference pipeline with multiple backend support.

**Key Components**:
- `OpenAIChatClient`: Thin wrapper around VLM backends with unified interface
- `BaseInferenceEngine`: Abstract interface for different inference approaches
- `OpenAIInferenceEngine`: Sequential 3-stage pipeline implementation
- `FakeInferenceEngine`: Generates deterministic fixtures when `dry_run` mode is enabled
- `create_inference_engine()`: Factory that chooses between real and fake engines
- Base64 image encoding and intelligent JSON response parsing

**Processing Flow**:
1. Image → Base64 encoding
2. Caption stage with error handling
3. Keywords stage with JSON parsing and validation
4. Description stage using previous outputs as context
5. Assembly into complete record with timing and model tracking

#### `post_processing.py` - Text Refinement
**Purpose**: Comprehensive text cleaning and normalization pipeline.

**Keyword Processing Features**:
- **Banned word filtering**: Removes generic photographic terms and subjective adjectives
- **De-pluralization**: Smart singular/plural handling with linguistic exceptions
- **Substring elimination**: Removes redundant keywords that are substrings of others
- **Deduplication**: Preserves order while removing exact duplicates
- **Compound word handling**: Splits "X and Y" constructions appropriately

**Text Processing**:
- **Caption tidying**: Ensures proper punctuation and formatting
- **Description normalization**: Collapses whitespace while preserving meaning

**Quality Control**: Maintains lists of banned words, singular exceptions, and compound word rules.

#### `metadata_writer.py` - ExifTool Integration
**Purpose**: Writes processed metadata to image files using ExifTool CLI.

**Metadata Mapping**:
- **Title**: `XMP-dc:Title` (for captions)
- **Description**: `XMP-dc:Description`, `IPTC:Caption-Abstract`
- **Keywords**: `XMP-dc:Subject`, `IPTC:Keywords`, `XMP-lr:HierarchicalSubject`

**Safety Features**:
- Backup creation before metadata writes
- Existing metadata detection and overwrite protection
- Validation of ExifTool availability
- Comprehensive error handling and logging

#### `runner.py` - Orchestration Engine
**Purpose**: Main coordination layer that ties all components together.

**Responsibilities**:
- **Image discovery**: Recursive directory scanning with extension filtering
- **Processing coordination**: Sequential image handling via the configured inference engine
- **Output generation**: JSONL audit logs and human-readable Markdown summaries
- **Result annotation**: Captures metadata write outcomes and attaches notes per image
- **Dual testing modes**:
  - **Dry-run**: Uses fake test data, no AI inference, no metadata writes
  - **No-meta**: Real AI inference but skips metadata writes

#### `model_registry.py` - Model Catalog
**Purpose**: Structured catalog of supported AI models with metadata.

**Model Information**:
- Display names and HuggingFace model identifiers
- Licensing information (Apache 2.0, etc.)
- Supported quantizations (AWQ, INT4, FP16, etc.)
- Backend compatibility (LMDeploy, vLLM, etc.)
- Usage notes and recommendations

**Supported Models**:
- **Qwen2.5-VL**: Default balanced model for all stages
- **LLaVA-NeXT**: Detailed captions with strong OCR capabilities
- **SigLIP**: Embedding-based deterministic keyword ranking
- **Idefics2**: Narrative descriptions for creative workflows

### 2. Shared Infrastructure (`src/imageworks/libs/`)

#### VLM Backend Abstraction (`vlm/backends.py`)
**Purpose**: Unified interface for multiple Vision-Language Model serving stacks.

**Supported Backends**:
- **vLLM**: High-performance inference with tensor parallelism
- **LMDeploy**: Optimized deployment with eager mode support
- **Triton**: TensorRT-LLM integration (placeholder implementation)

**Features**:
- OpenAI-compatible API standardization
- Health checking with graceful error reporting
- Configurable request timeouts
- Backend-specific extensions (e.g. LMDeploy health endpoints, Triton stub)

#### Prompt Library System (`prompting/`)
**Purpose**: Reusable prompt management infrastructure shared across ImageWorks apps.

**Components**:
- `PromptLibrary`: Generic registry with ID/name-based lookup
- `PromptProfileBase`: Base class for prompt profile implementations
- Version management and profile validation

#### Vision Algorithms (`vision/mono.py`)
**Purpose**: Advanced computer vision algorithms shared between applications.

**Capabilities**:
- LAB color space analysis for monochrome detection
- Split-tone identification and classification
- Color contamination measurement
- ExifTool metadata integration
- Statistical analysis of image color properties

### 3. Backend Infrastructure (`scripts/`)

#### LMDeploy Server (`start_lmdeploy_server.py`)
**Purpose**: Launches LMDeploy OpenAI-compatible servers with vision support.

**Features**:
- PyTorch backend with CUDA optimizations
- Vision-specific batch sizing configuration
- Eager mode for reduced VRAM usage
- Model path resolution from environment variables
- Automatically resolves the default Qwen2.5-VL AWQ checkpoint under
  `$IMAGEWORKS_MODEL_ROOT/weights/qwen-vl/Qwen2.5-VL-7B-Instruct-AWQ`, matching
  the Model Downloader's directory layout.

#### vLLM Server (`start_vllm_server.py`)
**Purpose**: Launches vLLM servers with advanced parallelization support.

**Features**:
- Tensor parallelism across multiple GPUs
- GPU memory utilization controls
- Dynamic batching and sequence management
- Trust-remote-code support for newer models

#### Multi-Backend Orchestrator (`start_personal_tagger_backends.py`)
**Purpose**: Intelligent coordination of multiple backend servers.

**Smart Features**:
- **Model consolidation**: Uses single server when caption/keywords/description models match
- **Port conflict prevention**: Detects and prevents backend port collisions
- **Stage mapping**: Routes different processing stages to appropriate servers
- **Resource optimization**: Minimizes memory usage by sharing model instances

**Configuration Analysis**: Examines personal tagger configuration to determine optimal server topology.

#### Summary Regeneration Utility (`regenerate_summary.py`)
**Purpose**: Fast regeneration of summary reports from existing JSONL data without re-running AI inference.

**Key Features**:
- **JSONL-based regeneration**: Creates summaries from previously processed results
- **Full content display**: Shows complete keywords, captions, and descriptions (no truncation)
- **Flexible output**: Configurable output paths and backend labeling
- **Error resilient**: Handles malformed JSON lines gracefully
- **Format compatibility**: Works with different JSONL schema versions

**Usage Examples**:
```bash
# Basic usage (creates regenerated_summary.md in JSONL directory)
uv run python scripts/regenerate_summary.py outputs/results/personal_tagger.jsonl

# Specify custom output path and backend
uv run python scripts/regenerate_summary.py \
  outputs/results/personal_tagger.jsonl \
  --output outputs/summaries/personal_tagger_summary.md \
  --backend lmdeploy
```

**Use Cases**:
- **Summary format iteration**: Quickly test different summary layouts without expensive re-processing
- **Recovery**: Regenerate lost summaries from existing JSONL audit logs
- **Analysis**: Create multiple summary variants with different filtering or grouping
- **Development**: Rapid prototyping of summary features using real data

## Cross-Application Integration

The Personal Tagger is part of a larger **ImageWorks ecosystem** with three main applications:

### Application Portfolio

1. **Personal Tagger**: AI-powered photo library enrichment (this document)
2. **Color Narrator**: Specialized color analysis and description generation
3. **Mono Checker**: Monochrome image validation for photography competitions

### Shared Infrastructure Benefits

**Common VLM Backends**: All applications use the same OpenAI-compatible serving infrastructure, enabling:
- Resource sharing between applications
- Consistent model management and deployment
- Unified configuration and monitoring

**Metadata Consistency**: Standardized approach to image metadata across applications:
- ExifTool-based writing for reliability and Lightroom compatibility
- XMP/IPTC field mapping following industry standards
- Backup and safety mechanisms

**Configuration Patterns**: Consistent `pyproject.toml`-based configuration system across all apps.

## Metadata Ecosystem

### Multi-Strategy Approach

The ImageWorks project implements **three different metadata strategies** optimized for different use cases:

1. **Personal Tagger**: ExifTool CLI for standard XMP/IPTC fields
   - Optimizes for Lightroom compatibility and industry standards
   - Uses established metadata fields that most software recognizes

2. **Color Narrator**: Python XMP library with custom namespace + JSON sidecar fallback
   - Custom `http://imageworks.ai/color-narrator/1.0/` namespace for specialized data
   - Graceful degradation to JSON sidecars when XMP libraries unavailable

3. **Mono Checker**: ExifTool with custom XMP-MW namespace
   - Photography competition-specific metadata in `XMP-MW` namespace
   - Shell script generation for batch processing

### Lightroom Integration Strategy

**Standard Fields Used**:
- `XMP-dc:Subject` and `IPTC:Keywords` for searchable keywords
- `XMP-dc:Description` and `IPTC:Caption-Abstract` for descriptions
- `XMP-dc:Title` for short captions
- `XMP-lr:HierarchicalSubject` for Lightroom keyword hierarchy support

**File Format Support**:
- **JPEG**: Embedded XMP/IPTC metadata written directly to image files
- **RAW/TIFF**: XMP sidecar files created alongside original files
- **Backup mechanisms**: Original files preserved before metadata modification

## Configuration Architecture

### Hierarchical Configuration System

The Personal Tagger uses a sophisticated configuration system with multiple layers:

```
┌─────────────────────┐
│   CLI Arguments     │ (Highest Priority)
├─────────────────────┤
│ Environment Vars    │
├─────────────────────┤
│  pyproject.toml     │ (Lowest Priority)
└─────────────────────┘
```

### Key Configuration Areas

**Model Selection**: Independent model choice for each processing stage:
- Caption model (e.g., Qwen2.5-VL-7B-AWQ)
- Keyword model (can be same or different)
- Description model (can use larger model for quality)

**Backend Configuration**:
- Endpoint URLs and API keys
- Timeout and retry settings
- Temperature and token limits

**Processing Parameters**:
- Batch sizes for throughput optimization
- Worker thread counts for parallelization
- Recursive directory scanning options

**I/O Configuration**:
- Input directory specifications
- Output paths for JSONL logs and summaries
- Metadata writing behavior and backup options

**Development Settings**:
- Dry-run mode: Uses fake test data, no AI inference, no metadata writes
- No-meta mode: Real AI inference but skips metadata writes
- Debug output levels and intermediate file saving
- Prompt profile selection for experimentation

## Data Flow Architecture

### Complete Processing Pipeline

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   Images    │───▶│  Discovery   │───▶│ Processing  │
│ (Various    │    │ & Filtering  │    │  Pipeline   │
│  Formats)   │    └──────────────┘    └─────────────┘
└─────────────┘                              │
                                            │
┌─────────────┐    ┌──────────────┐         │
│  Metadata   │◀───│ Post-Process │◀────────┘
│  Writing    │    │ & Cleanup    │
└─────────────┘    └──────────────┘
       │
       ▼
┌─────────────┐    ┌──────────────┐
│   Output    │    │   Logging    │
│ Generation  │    │ & Reporting  │
└─────────────┘    └──────────────┘
```

### Detailed Processing Steps

1. **Image Discovery**:
   - Recursive directory scanning with configurable depth
   - Extension-based filtering (supports RAW formats, JPEG, PNG, TIFF)
   - Duplicate detection and path normalization

2. **AI Processing Pipeline**:
   - **Stage 1**: Caption generation with image analysis
   - **Stage 2**: Keyword extraction using image + optional context
   - **Stage 3**: Description generation using image + caption + keywords
   - Error handling and partial result recovery at each stage

3. **Post-Processing**:
   - Keyword deduplication and quality filtering
   - Text normalization and formatting
   - Confidence score assignment and ranking

4. **Metadata Integration**:
   - ExifTool command generation with proper field mapping
   - Backup creation and safety validation
   - Cross-platform path handling and error recovery

5. **Output Generation**:
   - Structured JSONL logs for programmatic processing
   - Human-readable Markdown summaries with statistics
   - Per-directory organization and progress reporting

## Error Handling and Resilience

### Multi-Level Error Recovery

**Per-Stage Recovery**: Each AI processing stage has independent error handling:
- Network timeout recovery with exponential backoff
- JSON parsing fallbacks for malformed responses
- Partial result preservation when individual stages fail

**Metadata Safety**: Multiple layers of protection for original files:
- Automatic backup creation before any modification
- Validation of ExifTool availability before processing
- Rollback capability for failed metadata writes

**Graceful Degradation**: System continues processing even with component failures:
- Processing continues with partial results when individual images fail
- Backend unavailability handled with clear error reporting
- Missing dependencies detected early with helpful error messages

### Comprehensive Logging

**Multi-Level Logging**:
- **DEBUG**: Detailed processing steps and internal state
- **INFO**: Progress updates and successful operations
- **WARNING**: Recoverable errors and fallback usage
- **ERROR**: Processing failures requiring attention

**Structured Output**: JSONL format enables:
- Programmatic analysis of processing results
- Integration with monitoring and alerting systems
- Historical analysis and quality tracking

## Performance Considerations

### Scalability Features

**Parallel Processing**: Configurable worker threads for:
- Image preprocessing and encoding
- Metadata writing operations
- I/O operations separate from AI inference

**Batch Processing**: Intelligent batching strategies:
- Configurable batch sizes based on available VRAM
- Model-specific optimization (different models have different optimal batch sizes)
- Memory usage monitoring and adaptive sizing

**Resource Management**:
- Connection pooling for backend communication
- Model instance sharing across processing stages
- Cleanup and resource release mechanisms

### Hardware Optimization

**GPU Memory Management**:
- Support for quantized models (AWQ, INT4) for memory-constrained systems
- Eager mode options for reduced memory usage
- Tensor parallelism for multi-GPU systems

**Storage Optimization**:
- Efficient image loading with configurable resizing
- Minimal memory footprint during processing
- Streaming I/O for large directory processing

## Testing and Quality Assurance

### Test Coverage Areas

**Unit Testing**:
- Individual component validation
- Configuration loading and merging
- Text processing and keyword filtering
- Metadata field mapping and ExifTool integration

**Integration Testing**:
- End-to-end pipeline execution
- Backend communication and error handling
- File format support and metadata writing

**Mock Infrastructure**:
- Fake inference engines for testing without AI backends
- Tracked metadata writers for validation
- Sample image generation for consistent testing

### Quality Control Measures

**Output Validation**:
- Keyword quality filtering with banned word lists
- Text formatting and length constraints
- Metadata field compliance with industry standards

**Processing Validation**:
- Dry-run modes for safe testing
- Summary generation with processing statistics
- Error aggregation and reporting

## Future Architecture Considerations

### Extensibility Points

**Model Integration**: The model registry and inference engine abstraction support:
- New model architectures and quantization formats
- Alternative inference backends (TensorRT-LLM, etc.)
- Custom fine-tuned models for specific use cases

**Prompt Evolution**: The prompt library system enables:
- A/B testing of different prompt strategies
- Domain-specific prompt profiles (portraits, landscapes, etc.)
- Multi-language prompt support

**Backend Expansion**: The VLM backend abstraction supports:
- Cloud-based inference services (OpenAI, Anthropic, etc.)
- Edge deployment scenarios
- Hybrid cloud/local processing workflows

### Performance Scaling

**Distributed Processing**: Architecture supports extension to:
- Multi-machine processing clusters
- Queue-based job distribution
- Load balancing across backend instances

**Storage Integration**: Modular design enables:
- Cloud storage integration (S3, Google Cloud, etc.)
- Database-backed metadata management
- Content delivery network integration

This architecture provides a solid foundation for production photo library automation while maintaining flexibility for future enhancements and scale requirements.
