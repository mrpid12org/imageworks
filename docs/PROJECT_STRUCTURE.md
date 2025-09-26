# Project Structure

This guide explains the repository layout and where to find things.

## Top Level
- `README.md` – quick start, testing, and pointers into docs.
- `pyproject.toml` – package metadata, dependencies, and console scripts.
- `uv.lock` – pinned dependency lockfile for reproducible installs.
- `configs/` – repo-scoped configuration (e.g., ExifTool namespace).
- `docs/` – developer documentation (environment setup, specs, workflows).
- `src/` – all application and library code (installed via the `src/` layout).
- `tests/` – test suites, analysis tools, and test data.
- `outputs/` – generated files and analysis results.

## `configs/`
- `exiftool/.ExifTool_config` – defines the custom XMP `MW` namespace used by the exporter to write structured monochrome diagnostics.

## `docs/`
- `MONO_WORKFLOW.md` - complete mono checking workflow
- `processing-downloads.md` - detailed file processing documentation
- `monochrome-overview.md` - monochrome detection algorithm overview
- `monochrome-technical.md` - detailed monochrome classification logic
- `dev-env/ide-setup-wsl-vscode.md` - WSL + VS Code environment setup
- `spec/imageworks-specification.md` - product and system specification
- `PROJECT_STRUCTURE.md` - this document

## `src/`
Top-level package: `imageworks`

- `imageworks/apps/`
  - `mono_checker/`
    - `cli/mono.py` – Typer CLI for running checks and generating overlays.
    - `api/main.py` – FastAPI app exposing `/mono/check`, `/mono/batch`, `/healthz`.
    - `__init__.py` – package markers.

- `imageworks/libs/`
  - `vision/mono.py` – core monochrome analysis logic and diagnostics.
  - `__init__.py` – package markers.

- `imageworks/tools/`
  - `write_mono_xmp.py` – generates ExifTool scripts from JSONL to write XMP fields and optional Lightroom-friendly keywords; also includes a cleanup command.

- `imageworks/__init__.py` – package marker.

## `tests/`
- `analysis/` – diagnostic and analysis tools
  - `analyze_hue_lum.py` – hue/luminance relationship analysis
  - `analyze_polyfit.py` – polyfit warning analysis
- `test_images/` – test image files
  - `synthetic/` – generated test images for unit tests
  - `samples/` – real photo samples for testing
- `output/` – test and analysis output files
- Test Files:
  - `test_mono.py` – unit tests for neutral/toned/fail detection
  - `test_mono_split_and_boundaries.py` – split-toning and threshold tests
  - `test_mono_near_cast.py` – near-neutral color cast classification
  - `test_perf_mono.py` – opt-in performance micro-benchmark
  - `test_api.py` – FastAPI route tests
  - `test_env.py` – environment checks

## `outputs/`
- `summaries/` – human-readable outputs (MD, CSV)
  - `mono_summary.md` – detailed analysis results
  - `mono_summary.csv` – tabular results
  - `mono_summary_ir.csv` – intermediate results
- `results/` – data files
  - `mono_results.jsonl` – raw analysis results

## Console Scripts
Declared in `pyproject.toml` and available via `uv run <script>`:
- `imageworks-mono` – CLI (`cli/mono.py`).
  - `imageworks-mono <folder>` – run checks and (optionally) write JSONL.
  - `imageworks-mono visualize <folder>` – write heatmap overlays next to images.
- `imageworks-mono-api` – run the FastAPI app (`api/main.py`).
- `imageworks-mono-xmp` – XMP writer/cleanup tool (`tools/write_mono_xmp.py`).

## Typical Outputs
- `mono_results.jsonl` – produced by the CLI when using `--jsonl-out`, contains per-image verdicts and diagnostics (tone names, failure reasons, etc.).
- Overlays written by `visualize` use the `_<suffix>.jpg` pattern next to originals.

## Where to Start
- CLI usage: see `README.md` (Running the Applications).
- Analysis logic: `src/imageworks/libs/vision/mono.py` (extensively documented).
- XMP export: `src/imageworks/tools/write_mono_xmp.py` and `configs/exiftool`.
- API endpoints: `src/imageworks/apps/mono_checker/api/main.py`.
