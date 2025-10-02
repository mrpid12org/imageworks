# Model Downloader

A comprehensive tool for downloading and managing AI models across multiple formats and directories, with support for quantized models and cross-platform compatibility.

> GGUF: Previous fallback guidance referenced a custom GGUF server script now removed. Use Ollama for GGUF experimentation; production multimodal flows rely on vLLM.

## Features

- 🚀 **Fast Downloads**: Uses aria2c for parallel, resumable downloads
- 🔍 **Format Detection**: Infers GGUF, AWQ, GPTQ, Safetensors, and more from filenames and configs
- 📁 **Smart Routing**: Sends GGUF models to LM Studio paths and other formats to the WSL weights store
- 📋 **Model Registry**: Tracks size, checksum, location, and metadata for every download

- 🔗 **URL Support**: Handles direct HuggingFace URLs and shorthand `owner/repo` identifiers (including `owner/repo@branch`)

- ⚡ **Cross-Platform**: Built for mixed Windows/WSL setups with optional custom overrides
- 🛡️ **Verification**: Validates completed downloads and registry integrity

## Quick Start

### Installation

The model downloader is part of the ImageWorks toolkit. Ensure you have aria2c installed:

```bash
# Ubuntu/Debian
sudo apt install aria2

# macOS
brew install aria2
```

### Basic Usage

```bash
# Download a model by name
imageworks-download download "microsoft/DialoGPT-medium"

# Target a specific branch/tag
imageworks-download download "microsoft/DialoGPT-medium@dev"

# Download from URL
imageworks-download download "https://huggingface.co/microsoft/DialoGPT-medium"

# Download specific format
imageworks-download download "casperhansen/llama-7b-instruct-awq" --format awq

# Force location override (keyword or custom path)
imageworks-download download "TheBloke/Llama-2-7B-Chat-GGUF" --location windows_lmstudio

# List downloaded models
imageworks-download list

# Show statistics
imageworks-download stats
```

### Logging

All downloader commands now stream structured logs to `logs/model_downloader.log` by default so you can audit each transfer. Set the `IMAGEWORKS_LOG_DIR` environment variable to relocate the log directory (for example when running inside a container or packaging the tool). Console output remains unchanged for interactive use, but every status icon and warning is mirrored into the log file for later review.

### Python API

```python
from imageworks.tools.model_downloader import ModelDownloader

# Initialize downloader
downloader = ModelDownloader()

# Download a model
model = downloader.download("microsoft/DialoGPT-medium")

# List models filtered by format
models = downloader.list_models(format_filter="awq")

# Get statistics
stats = downloader.get_stats()
```

## Directory Structure

The downloader manages two separate directories:

### Linux WSL Directory (`~/ai-models/`)
```
~/ai-models/
├── weights/              # Safetensors, AWQ, GPTQ, PyTorch models
│   ├── microsoft/
│   │   └── DialoGPT-medium/
│   └── casperhansen/
├── registry/            # Model registry JSON files
└── cache/               # Temporary files
```

**Compatible with**: vLLM, Transformers, AutoAWQ, AutoGPTQ

### Windows LM Studio Directory
```
/mnt/d/ai stuff/models/llm models/  (Windows: D:\ai stuff\models\llm models\)
├── TheBloke/
│   └── Llama-2-7B-Chat-GGUF/
└── Qwen/
    └── Qwen2.5-1.5B-GGUF/
```

**Compatible with**: LM Studio, llama.cpp, Ollama


Downloads that use `--location windows_lmstudio` (or detect GGUF formats automatically) keep a publisher/`repo` structure. Other formats default to `~/ai-models/weights/<owner>/<repo>`. Supplying a custom path via `--location /path/to/models` stores the model beneath that path. When a non-`main` branch is requested, the repository directory is suffixed with `@branch` (e.g. `DialoGPT-medium@dev`) to avoid collisions with the default branch.

## Format Detection

The downloader aggregates multiple detectors to determine the best storage location:

| Format | Detection Signals | Default Target |
|--------|------------------|----------------|
| **GGUF** | `.gguf` extensions, quantisation suffixes (`Q4_K`) | Windows LM Studio |
| **AWQ** | Repository name patterns, `quantization_config.quant_method == "awq"` | Linux WSL |
| **GPTQ** | Repository name patterns, config quantisation metadata | Linux WSL |
| **Safetensors** | `.safetensors` files | Linux WSL |
| **PyTorch** | `.bin`, `.pth`, `.pt` weights | Linux WSL |

Results are ranked by confidence; provide `--format awq,gguf` to express an explicit preference order when multiple matches are found.


## Validation & Troubleshooting

Every download is verified to ensure files are present and complete before an entry is stored in the registry. The generated directories include all artefacts required by the serving helpers:

- **Core configs**: `config.json` and `tokenizer_config.json`
- **Tokenizer assets**: either `tokenizer.json` or `tokenizer.model`
- **Serving aids**: `generation_config.json`, `chat_template.json`, and quantisation descriptors when the upstream repository ships them
- **Weights**: at least one `.safetensors`, `.bin`, `.pt`, `.awq`, or `.gguf` shard

If any file fails to download, the downloader aborts with a verification error so you can retry without registering a broken model. The [LMDeploy helper](../scripts/start_lmdeploy_server.py) performs an additional sanity check when you launch the server, warning about missing chat templates or generation configs before it starts. This mirrors the most common causes of runtime issues (blank completions, misaligned role handling, or tokenisation failures) and provides actionable remediation guidance.


## Commands

### `download`
Download models from HuggingFace or URLs.

```bash
imageworks-download download MODEL [OPTIONS]

Options:
  --format, -f TEXT          Preferred format(s) (comma separated: gguf, awq, gptq, safetensors)
  --location, -l TEXT        Target location (linux_wsl, windows_lmstudio, or custom path)
  --include-optional, -o     Include optional files (docs, examples)
  --force                    Force re-download even if model exists
  --non-interactive, -y      Non-interactive mode (use defaults)
```

**Examples:**
```bash
# Basic download
imageworks-download download "microsoft/DialoGPT-medium"

# Specific format and location
imageworks-download download "microsoft/DialoGPT-medium" --format safetensors --location linux_wsl

# Branch selection with custom location
imageworks-download download "microsoft/DialoGPT-medium@dev" --location ~/models

# From URL with optional files
imageworks-download download "https://huggingface.co/microsoft/DialoGPT-medium" --include-optional

# Non-interactive
imageworks-download download "microsoft/DialoGPT-medium" --non-interactive
```

### `list`
List downloaded models with filtering.

```bash
imageworks-download list [OPTIONS]

Options:
  --format, -f TEXT      Filter by format
  --location, -l TEXT    Filter by location
  --details, -d          Show detailed information
  --json                 Output in JSON format
```

**Examples:**
```bash
# List all models
imageworks-download list

# Filter by format
imageworks-download list --format awq

# Show details
imageworks-download list --details

# JSON output
imageworks-download list --json
```

### `analyze`
Analyze HuggingFace URLs without downloading.

```bash
imageworks-download analyze URL [OPTIONS]

Options:
  --files    Show detailed file information
```

**Examples:**
```bash
# Analyze repository
imageworks-download analyze "https://huggingface.co/microsoft/DialoGPT-medium"

# Show file details
imageworks-download analyze "https://huggingface.co/microsoft/DialoGPT-medium" --files
```

### `remove`
Remove models from registry and optionally delete files.

```bash
imageworks-download remove MODEL_NAME [OPTIONS]

Options:
  --format, -f TEXT      Specific format to remove
  --location, -l TEXT    Specific location to remove from
  --delete-files         Also delete the model files from disk
  --force               Don't ask for confirmation
```

### `stats`
Show download statistics and registry information.

```bash
imageworks-download stats
```

### `verify`
Verify model integrity and registry consistency.

```bash
imageworks-download verify [MODEL_NAME] [OPTIONS]

Options:
  --fix-missing    Remove registry entries for missing models
```

### `config`
Show current configuration.

```bash
imageworks-download config
```

## Configuration

### Environment Variables

```bash
# Override default paths
export IMAGEWORKS_MODEL_ROOT=~/custom-ai-models/weights
export IMAGEWORKS_LMSTUDIO_ROOT=/mnt/d/custom/lmstudio/models
```

### pyproject.toml Configuration

Add to your `pyproject.toml`:

```toml
[tool.imageworks.model-downloader]
linux_wsl_root = "~/ai-models"
windows_lmstudio_root = "/mnt/d/ai stuff/models/llm models"
max_connections = 16
preferred_formats = ["awq", "safetensors", "gguf"]
```

## Python API Reference

### ModelDownloader

Main class for downloading and managing models.

```python
from imageworks.tools.model_downloader import ModelDownloader

downloader = ModelDownloader()
```

#### Methods

**`download(model_identifier, **kwargs)`**
Download a model from HuggingFace.

```python
model = downloader.download(
    model_identifier="microsoft/DialoGPT-medium",
    format_preference=["awq", "safetensors"],
    location_override="linux_wsl",
    include_optional=False,
    force_redownload=False,
    interactive=True,
)

# Target an alternate branch
experimental = downloader.download("microsoft/DialoGPT-medium@dev", force_redownload=True)
```

**`list_models(**kwargs)`**
List downloaded models with filtering.

```python
models = downloader.list_models(
    format_filter="awq",
    location_filter="linux_wsl"
)
```

**`remove_model(model_name, **kwargs)`**
Remove a model from registry.

```python
success = downloader.remove_model(
    model_name="microsoft/DialoGPT-medium",
    format_type="awq",
    location="linux_wsl",
    delete_files=True,
)
```

**`get_stats()`**
Get download statistics.

```python
stats = downloader.get_stats()
print(f"Total models: {stats['total_models']}")
print(f"Total size: {stats['total_size_bytes']} bytes")
```

### ModelRegistry

Registry for tracking downloaded models.

```python
from imageworks.tools.model_downloader import get_registry

registry = get_registry()
```

#### Methods

**`find_model(model_name, format_type=None, location=None)`**
Find models matching criteria.

```python
models = registry.find_model("microsoft/DialoGPT-medium")
awq_models = registry.find_model("microsoft/DialoGPT-medium", format_type="awq")
```

**`add_model(model_name, format_type, location, path, **kwargs)`**
Add a model to registry.

```python
registry.add_model(
    model_name="microsoft/DialoGPT-medium",
    format_type="awq",
    location="linux_wsl",
    path="/path/to/model",
    size_bytes=1000000
)
```

### FormatDetector

Automatic format detection for models.

```python
from imageworks.tools.model_downloader import FormatDetector

detector = FormatDetector()
formats = detector.detect_comprehensive(
    model_name="microsoft/DialoGPT-medium-awq",
    filenames=["model.safetensors", "config.json"],
    config_content='{"quantization_config": {"quant_method": "awq"}}'
)
```

### URLAnalyzer

Analyze HuggingFace URLs and repositories.

```python
from imageworks.tools.model_downloader import URLAnalyzer

analyzer = URLAnalyzer()
analysis = analyzer.analyze_url("https://huggingface.co/microsoft/DialoGPT-medium")

print(f"Detected formats: {[f.format_type for f in analysis.formats]}")
print(f"Total size: {analysis.total_size} bytes")
```

## Error Handling

The downloader provides detailed error messages for common issues:

### Missing aria2c
```
RuntimeError: aria2c not found. Please install aria2c for optimal download performance.
Ubuntu/Debian: sudo apt install aria2
macOS: brew install aria2
```

### Network Errors
aria2c surfaces HTTP errors directly; failed transfers can be resumed by re-running the command.

### Disk Space
Ensure you have sufficient free space before starting large downloads (the downloader does not pre-validate capacity).

### File Conflicts
If a model already exists, you can choose to:
- Use the existing model
- Re-download and overwrite the registry entry (prompted interactively)
- Re-run with `--location` to place the download somewhere else

## Integration with ImageWorks

The model downloader integrates seamlessly with other ImageWorks tools:

### With vLLM Server
```python
# Download AWQ model
downloader.download("casperhansen/llama-7b-instruct-awq")

# Use with vLLM
from imageworks.apps.vlm_backend import start_vllm_server
start_vllm_server(model_name="llama-7b-instruct-awq")
```

### With Color Narrator
```python
# Download vision model
downloader.download("Qwen/Qwen2.5-VL-7B-Instruct")

# Use with color narrator
from imageworks.apps.color_narrator import ColorNarrator
narrator = ColorNarrator(model_name="Qwen2.5-VL-7B-Instruct")
```

### With Personal Tagger
```bash
# Fetch the caption/keyword/description models you plan to serve
imageworks-download download "qwen-vl/Qwen2.5-VL-7B-Instruct-AWQ"
imageworks-download download "llava-hf/llava-v1.6-mistral-7b-hf"

# Launch backends (vLLM/LMDeploy) with the downloaded paths
python scripts/start_personal_tagger_backends.py --launch \
  --caption-model-path "~/ai-models/weights/qwen-vl/Qwen2.5-VL-7B-Instruct-AWQ" \
  --description-model-path "~/ai-models/weights/llava-hf/llava-v1.6-mistral-7b-hf"

# The LMDeploy helper defaults to `$IMAGEWORKS_MODEL_ROOT/weights/qwen-vl/…`, so
# downloads made with the Model Downloader are discovered automatically when you
# keep the standard directory layout.

# Run the personal tagger once backends are online
python -m imageworks.apps.personal_tagger.cli.main run --input ~/photos --backend http://localhost:8000
```

## Troubleshooting

### Common Issues

**Model not found**
- Verify the model name is correct
- Check if the model exists on HuggingFace
- Try the full URL instead of just the name

**Slow downloads**
- Ensure aria2c is installed and working
- Check your internet connection
- Try reducing max_concurrent_downloads in config

**Permission errors**
- Check directory permissions
- For Windows LM Studio directory, ensure WSL can access the drive

**Registry corruption**
- Run `imageworks-download verify --fix-missing` to clean up
- Delete `~/ai-models/registry/models.json` to reset registry

### Debug Mode

Set environment variable for detailed logging:
```bash
export IMAGEWORKS_DEBUG=1
imageworks-download download "model-name"
```

### File Issues

Check file integrity:
```bash
imageworks-download verify
```

Clean up missing models:
```bash
imageworks-download verify --fix-missing
```

## Contributing

See the main ImageWorks repository for contributing guidelines.

## License

This tool is part of the ImageWorks project and follows the same license terms.
