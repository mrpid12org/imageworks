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
- Web/API components live under `src/imageworks/apps/competition_checker` and can be hosted via FastAPI/Uvicorn.

## Testing
- Execute the full suite with `uv run pytest` from the repository root.
- `tests/test_mono.py` exercises the monochrome detector against synthetic images.
- `tests/test_env.py` validates CUDA availability and imports for FAISS, OpenCLIP, Pillow, and OpenCV; skip or adjust if running on CPU-only hardware.

## Documentation
- Developer Environment: `docs/dev-env/ide-setup-wsl-vscode.md`
- Project Specification: `docs/spec/imageworks-specification.md`

For architecture or roadmap discussions, start with the specification and follow links into the docs folder.
