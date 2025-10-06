# Imageworks

Imageworks provides local-first photography tooling for competition compliance and metadata enrichment.

- **Mono (Competition Checker)** – Complete workflow for Camera Club entries:
  - Extracts and organises competition images from downloaded ZIP files
  - Embeds title and author metadata from XMP sidecars
  - Integrates with Lightroom for managed ingest
  - Validates true monochrome images
  - Flags near-duplicates and generates overlay visualisations

- **Color Narrator** – VLM-guided natural language descriptions of residual colour in monochrome images using Qwen2-VL-2B inference.

- **Personal Tagger** *(in development)* – Lightroom enrichment utilities that will reuse the shared libraries; current modules provide scaffolding only.

All applications share a Python codebase (`src/imageworks`) that targets WSL/Ubuntu and takes advantage of GPU acceleration when available.

See [Mono Workflow](docs/domains/mono/mono-workflow.md) for detailed documentation of the complete competition checking process.

## Project Layout

```
imageworks/
├── src/imageworks/          # Main package source code
│   ├── apps/               # Application entry points (CLI/Web)
│   ├── libs/               # Core libraries and utilities
│   └── archive/           # Archived reference code
├── tests/                  # Test suite and tools
│   ├── analysis/          # Diagnostic and analysis tools
│   ├── test_images/       # Test image files
│   │   ├── synthetic/     # Generated test images
│   │   └── samples/       # Sample photos for testing
│   └── output/            # Test output files
├── outputs/                # Production outputs
│   ├── summaries/         # Human-readable outputs (MD, CSV)
│   └── results/           # Data files (JSONL)
└── configs/               # Configuration files
```

### Key Directories

- `src/imageworks/` – Core package with shared libraries and entry points
- `tests/` – Test suites, analysis tools, and test data
- `outputs/` – Generated files and analysis results
- `configs/` – Sample configuration and experiment settings

## Documentation

See the [Documentation Map](docs/index.md) for the full categorized list. Key entry points:

### Core Guides
- [Mono Workflow](docs/domains/mono/mono-workflow.md) - Complete competition checking process
- [Project Structure](docs/architecture/project-structure.md) - Detailed codebase organization
- [AI Models and Prompting](docs/guides/ai-models-and-prompting.md) - Comprehensive guide to models, experiments, and prompting strategies

### Component Guides
- [Color-Narrator Reference](docs/domains/color-narrator/reference.md) - VLM-based color analysis system
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
- Web/API: run `uv run imageworks-mono-api` then call:
  - `GET /healthz`
  - `POST /mono/check` (multipart upload or JSON with `path`)
  - `POST /mono/batch` (JSON: `folder`, optional `exts`, thresholds)
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
