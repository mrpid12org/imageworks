# Imageworks

Imageworks provides local-first photography tooling for competition compliance, duplicate detection, metadata enrichment, and backend orchestration.

## Features

- **Streamlit Control Center** – Unified GUI for every module with presets, telemetry, and process history. [GUI Reference](docs/reference/gui.md)
- **Mono Checker** – Competition compliance pipeline (metadata ingest, monochrome validation, overlays, Lightroom helpers). [Mono Checker Reference](docs/reference/mono-checker.md)
- **Image Similarity Checker** – Duplicate and near-duplicate detection with embedding, perceptual, and structural strategies plus explanation generation. [Image Similarity Reference](docs/reference/image-similarity-checker.md)
- **Color Narrator** – VLM-guided natural language descriptions of residual colour in monochrome images with backend diagnostics and overlay support. [Color Narrator Reference](docs/domains/color-narrator/reference.md)
- **Personal Tagger** – Multi-stage caption/keyword/critique generator for personal libraries with Lightroom-ready metadata writes. [Personal Tagger Reference](docs/reference/personal-tagger.md)
- **ZIP Extractor** – Batch unzip + metadata normaliser for competition submissions, producing Markdown intake summaries. [ZIP Extractor Reference](docs/reference/zip-extract.md)
- **Model Registry & Loader** – Deterministic registry, hash verification, vLLM single-port activation, and REST API used by all tooling. [Model Loader Reference](docs/reference/model-loader.md)
- **Model Downloader** – Parallel Hugging Face/Ollama retrieval with registry updates, format detection, and GUI parity. [Model Downloader Reference](docs/reference/model-downloader.md)
- **Chat Proxy** – OpenAI-compatible FastAPI layer that routes to the registry-backed backend roster for OpenWebUI and other clients. [Chat Proxy Reference](docs/reference/chat-proxy.md)

All applications share a Python codebase (`src/imageworks`) that targets WSL/Ubuntu and takes advantage of GPU acceleration when available.

See [Mono Workflow](docs/domains/mono/mono-workflow.md) for detailed documentation of the complete competition checking process.

## Quick Start

### GUI Interface (Recommended)

Launch the GUI Control Center:

```bash
./scripts/launch_gui.sh
# or
uv run imageworks-gui
```

The GUI provides a user-friendly interface for all Imageworks tools with preset configurations and visual workflows. See the [GUI Reference](docs/reference/gui.md) for a deep dive into each page.

### CLI Tools

All tools are also available via command-line. Each command is packaged as a console script (run with `uv run <command> ...` to stay inside the project environment):

| Command | Purpose | Reference |
|---------|---------|-----------|
| `imageworks-mono` | Validate monochrome competition entries, write XMP keywords, render overlays. | [Mono Checker Reference](docs/reference/mono-checker.md) |
| `imageworks-image-similarity` | Detect duplicates/near-duplicates against a historical library with optional LLM explanations. | [Image Similarity Reference](docs/reference/image-similarity-checker.md) |
| `imageworks-color-narrator` | Generate residual colour descriptions and diagnostics for monochrome validation. | [Color Narrator Reference](docs/domains/color-narrator/reference.md) |
| `imageworks-personal-tagger` | Produce captions/keywords/critique for personal catalogues with registry-backed model selection. | [Personal Tagger Reference](docs/reference/personal-tagger.md) |
| `imageworks-zip` | Extract competition ZIPs, enforce keywords, and summarise metadata actions. | [ZIP Extractor Reference](docs/reference/zip-extract.md) |
| `imageworks-download` | Download/import model weights, normalise metadata, and maintain the layered registry. | [Model Downloader Reference](docs/reference/model-downloader.md) |
| `imageworks-models` | Inspect, select, lock, or activate models from the registry (CLI). | [Model Loader Reference](docs/reference/model-loader.md) |
| `imageworks-loader` | Sync registry overlays and curated layers (legacy convenience wrapper). | [Model Loader Reference](docs/reference/model-loader.md) |
| `imageworks-chat-proxy` | Launch the OpenAI-compatible FastAPI proxy that fronts the registry-backed models. | [Chat Proxy Reference](docs/reference/chat-proxy.md) |

### Chat Proxy & OpenWebUI
ImageWorks includes a lightweight OpenAI-compatible Chat Proxy that presents your unified model registry (Ollama, vLLM, etc.) to OpenWebUI and other OpenAI clients. The proxy:
- Returns simplified, human-friendly model names (same as the CLI list)
- Supports text, vision, and tools passthrough with optional light normalization
- Exposes `/v1/models` and `/v1/chat/completions` endpoints

A docker-compose file (`docker-compose.openwebui.yml`) runs the proxy and OpenWebUI together. By default, the proxy hides non-installed entries; when containerized, mount your HF weights at the same absolute path so “installed-only” checks pass, or set `CHAT_PROXY_INCLUDE_NON_INSTALLED=1` to relax filtering. Launch locally with `uv run imageworks-chat-proxy` (FastAPI) or `docker-compose -f docker-compose.openwebui.yml up`.

Docs: [Chat Proxy Reference](docs/reference/chat-proxy.md), [OpenWebUI Setup](docs/runbooks/openwebui-setup.md)

## Project Layout

```
imageworks/
├── src/imageworks/         # Main package source code
│   ├── apps/               # CLI + API entry points (mono, similarity, tagger, narrator, zip)
│   ├── gui/                # Streamlit app + pages/components
│   ├── libs/               # Core libraries and utilities shared across apps
│   ├── model_loader/       # Registry, hash verification, backend orchestration
│   ├── chat_proxy/         # OpenAI-compatible proxy service
│   ├── tools/              # Stand-alone utilities (model downloader, zip helpers)
│   └── archive/            # Archived reference code
├── configs/                # Registry snapshots, preset defaults, experiment config
├── downloads/              # Local cache for model weights and artifacts (gitignored)
├── docs/                   # Reference docs, runbooks, domain guides
├── models/                 # Sample/placeholder model assets (gitignored)
├── scripts/                # Helper scripts (GUI launcher, importers)
├── tests/                  # Pytest suite, analysis helpers, test assets
└── uv.lock / pyproject.toml # Dependency and build configuration
```

### Key Directories

- `src/imageworks/` – Core package with shared libraries, services, and entry points
- `configs/` – Layered model registry, presets, and experiment settings
- `docs/` – User guides, runbooks, and architectural references
- `downloads/` – Working directory for model downloads (populated by tooling)
- `tests/` – Test suites, analysis tools, and synthetic image fixtures

## Documentation

See the [Documentation Map](docs/index.md) for the full categorized list. Key entry points:

### Core Guides
- [Mono Workflow](docs/domains/mono/mono-workflow.md) - Complete competition checking process
- [Project Structure](docs/architecture/project-structure.md) - Detailed codebase organization
- [AI Models and Prompting](docs/guides/ai-models-and-prompting.md) - Comprehensive guide to models, experiments, and prompting strategies

### Component Guides
- [Mono Checker Reference](docs/reference/mono-checker.md) - CLI/API/GUI reference for the mono pipeline
- [Image Similarity Reference](docs/reference/image-similarity-checker.md) - Duplicate detection workflows
- [Color-Narrator Reference](docs/domains/color-narrator/reference.md) - VLM-based colour analysis system
- [Personal Tagger Reference](docs/reference/personal-tagger.md) - Caption/keyword workflow
- [Model Loader Reference](docs/reference/model-loader.md) - Registry, hashes, activation flows
- [Model Downloader Reference](docs/reference/model-downloader.md) - Weight acquisition and audits
- [Chat Proxy (OpenAI-compatible)](docs/reference/chat-proxy.md) - Unified model list and chat endpoint
- [ZIP Extractor Reference](docs/reference/zip-extract.md) - Intake automation
- [vLLM Deployment Guide](docs/runbooks/vllm-deployment-guide.md) - Production AI model deployment
- [Deterministic Model Serving](docs/architecture/deterministic-model-serving.md) - Hybrid vLLM + Ollama design (deterministic selection, locking, metrics)

### Developer Environment
- [IDE Setup (WSL/VSCode)](docs/guides/ide-setup-wsl-vscode.md) - Development environment configuration

## Getting Started
1. Install [uv](https://docs.astral.sh/uv/) following the steps in `docs/guides/ide-setup-wsl-vscode.md`.
2. From the repository root run `uv sync` to create the virtual environment and install dependencies.
3. Activate tools with `uv run ...` or `uvx ...` so commands use the project environment.

## Running the Applications
- CLI entry point: `uv run imageworks-mono --help` (Typer app declared in `pyproject.toml`).
- FastAPI (Mono): run `uv run imageworks-api` then call:
  - `GET /healthz`
  - `POST /mono/check` (multipart upload or JSON with `path`)
  - `POST /mono/batch` (JSON: `folder`, optional `exts`, thresholds)
- GUI: `uv run imageworks-gui` or `./scripts/launch_gui.sh` to launch the Streamlit control center.
- Chat Proxy: `uv run imageworks-chat-proxy` to expose OpenAI-compatible endpoints backed by the model registry.
- Model Loader API: `uv run imageworks-models-api` for REST-based registry inspection and selection.
- CLI defaults: `pyproject.toml` `[tool.imageworks.mono]` defines the folder, extensions, and output paths used when you omit options.
- Run with defaults: `uv run imageworks-mono` (produces JSONL and summary without extra flags).
- XMP writing: By default, `uv run imageworks-mono` generates an ExifTool script and writes lightweight keywords derived from results (both XMP-dc:Subject and IPTC:Keywords). Use `--no-write-xmp` to skip writing, or add `--no-xmp-keywords-only` to include custom XMP fields as well.

### Writing diagnostics into images (Lightroom demo)
- Generate JSONL with diagnostics: `uv run imageworks-mono check <folder> --jsonl-out mono_results.jsonl`
- Create an ExifTool script that writes custom XMP fields (and optional keywords):
  - `uv run imageworks-mono-xmp generate mono_results.jsonl --out write_xmp.sh`
  - Include LR-friendly keywords (prefix `mono:`): add `--as-keywords` (the CLI writer uses keywords by default when invoked via `imageworks-mono`)
- Run the script (requires ExifTool): `bash write_xmp.sh`
- In Lightroom: Metadata → Read Metadata from Files to ingest changes.

To remove diagnostics/keywords later, generate a cleanup script:
- `uv run imageworks-mono-xmp clean mono_results.jsonl --out clean_xmp.sh`
- For keywords-only removal: add `--keywords-only`

## Testing
- Execute the full suite with `uv run pytest` from the repository root.
- `tests/test_mono.py` exercises the monochrome detector against synthetic images.
- `tests/test_env.py` validates environment imports and, optionally, CUDA. By default the CUDA test is skipped; set `REQUIRE_CUDA=1` to enforce it: `REQUIRE_CUDA=1 uv run pytest -q`.

### PyTorch / CUDA
- The project pins PyTorch/torchvision/torchaudio to the CUDA 12.8 wheels (2.7.x).
- `uv sync` automatically pulls from the PyTorch CUDA index defined in `pyproject.toml`.
- Ensure the NVIDIA driver on your workstation provides CUDA 12.8 support (required for RTX 6000 PRO / Blackwell).
- CPU-only environments can remove or override these packages if needed; otherwise expect a ~1 GB download the first time the environment is locked.

### Monochrome Checker Logic

The logic for determining if an image is a valid monochrome is sophisticated, taking into account color variation, saturation, split-toning, and various edge cases.

For a detailed, step-by-step explanation of the decision-making process, see [Mono Technical](docs/domains/mono/mono-technical.md).

For architecture or roadmap discussions, start with the specification and follow links into the docs folder.
