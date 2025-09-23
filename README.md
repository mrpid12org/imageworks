# Imageworks

Imageworks provides local-first photography utilities for two workflows:

- **Competition Checker** – validates Cambridge Camera Club entries by confirming true monochrome images and flagging near-duplicates.
- **Personal Tagger** – enriches a personal Lightroom catalogue with concise keywords, optional captions/long descriptions, and similarity search tools.

Both applications share a Python codebase (`src/imageworks`) that targets WSL/Ubuntu and takes advantage of GPU acceleration when available.

## Project Layout
- `src/imageworks` – shared libraries plus CLI/Web entry points.
- `tests/` – pytest suites covering core image analytics and environment checks.
- `configs/` – sample configuration and experiment settings.
- `mono_results.jsonl` – captured monochrome checker runs for analysis/regression.

## Getting Started
1. Install [uv](https://docs.astral.sh/uv/) following the steps in `docs/dev-env/ide-setup-wsl-vscode.md`.
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
- Auto-write XMP: `uv run imageworks-mono --write-xmp` regenerates the ExifTool script and runs it (respecting defaults for script path/keywords/sidecars).

### Writing diagnostics into images (Lightroom demo)
- Generate JSONL with diagnostics: `uv run imageworks-mono check <folder> --jsonl-out mono_results.jsonl`
- Create an ExifTool script that writes custom XMP fields (and optional keywords):
  - `uv run imageworks-mono-xmp generate mono_results.jsonl --out write_xmp.sh`
  - Include LR-friendly keywords (prefix `mono:`): add `--as-keywords`
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

## Documentation
- Developer Environment: `docs/dev-env/ide-setup-wsl-vscode.md`
- Project Specification: `docs/spec/imageworks-specification.md`
- Project Structure: `docs/PROJECT_STRUCTURE.md`

### Monochrome Checker Logic

The logic for determining if an image is a valid monochrome is sophisticated, taking into account color variation, saturation, split-toning, and various edge cases.

For a detailed, step-by-step explanation of the decision-making process, please refer to the [Monochrome Checker Logic Decision Tree](./docs/MONOCHROME_CHECKER_LOGIC.md).

For architecture or roadmap discussions, start with the specification and follow links into the docs folder.
