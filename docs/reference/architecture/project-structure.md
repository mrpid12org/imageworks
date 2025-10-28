# Project Structure

Imageworks currently supports three related workflows:

1. **Mono (Competition Checker)** – monochrome compliance analysis and competition tooling.
2. **Color Narrator** – VLM-powered narration of residual colour for competition feedback.
3. **Personal Tagger** *(in development)* – Lightroom enrichment utilities built on shared libraries; most modules are stubs today.

Mono and Color Narrator are production-ready and share significant infrastructure in `src/imageworks`. Personal Tagger builds on the same libraries but is intentionally isolated while its design stabilises.

## Top Level
- `README.md` – entry point with install instructions and documentation map.
- `pyproject.toml` – project metadata, dependencies, and console scripts.
- `uv.lock` – pinned dependency set for reproducible environments.
- `configs/` – repository-scoped configuration (e.g., ExifTool namespaces).
- `docs/` – architecture notes, specifications, workflow documentation, runbooks, and proposals.
- `scripts/` – developer utilities (model servers, helpers) outside packaged apps.
- `src/` – Python source tree (installed via the `src/` layout).
- `tests/` – unit, integration, and analysis suites plus supporting assets.
- `outputs/` – generated analysis artefacts kept under version control.
- `logs/` – runtime logs (git-ignored).

## `configs/`
- `exiftool/.ExifTool_config` – declares the custom `MW` namespace used when exporting monochrome diagnostics to XMP.

## `scripts/`
- `start_vllm_server.py` – convenience launcher for a local vLLM server.
- `start_lmdeploy_server.py` – launches LMDeploy with tQwen2.5-VL-7B and eager mode defaults.
- Additional helper scripts for development and deployment workflows.

## `docs/`
Key documentation is grouped here:
- `mono-workflow.md` – end-to-end competition workflow.
- `processing-downloads.md` – ingest pipeline for competition ZIP files.
- `mono-overview.md` / `mono-technical.md` – algorithm overviews and deep dives.
- `guides/ide-setup-wsl-vscode.md` – development environment bootstrap.
- `spec/` – product specifications and design artefacts.
- `project-structure.md` – this document.

## `src/`
Top-level package: `imageworks`

- `imageworks/apps/`
  - `mono_checker/`
    - `cli/mono.py` – Typer CLI entry point for competition analysis.
    - `api/main.py` – FastAPI application exposing `/mono/*` endpoints.
  - `color_narrator/`
    - `cli/main.py` – Typer CLI for narrating or validating colour findings.
    - `api/` – Reserved for future FastAPI endpoints (currently a stub).
    - `core/` – orchestration modules:
      - `vlm.py` – backend-agnostic client (default LMDeploy + Qwen2.5-VL-7B-AWQ).
      - `data_loader.py` – coordinates JPEG, overlay, and JSONL inputs.
      - `narrator.py` – main processing pipeline and error handling.
      - `metadata.py` – XMP metadata management helpers.
      - `hybrid_mono_enhancer.py` – optional enhancements combining mono metrics with VLM context.
  - `personal_tagger/` – staging area for the forthcoming Personal Tagger CLI/API. Modules mirror the Color Narrator layout but are pre-production.

- `imageworks/libs/`
  - `vision/mono.py` – core monochrome analysis algorithms and diagnostics.
  - `personal_tagger/`
    - `color_analysis.py` – LAB colour statistics utilities shared by narrators.
    - `vlm_utils.py` – prompt templates, batching help, and VLM orchestration glue.
    - `image_utils.py` – reusable image loading and manipulation helpers.
  - Additional shared libraries live here as features expand.

- `imageworks/tools/`
  - `write_mono_xmp.py` – generates ExifTool scripts to apply or remove diagnostics.

## `tests/`
- `mono/` – unit and integration coverage for the competition checker.
- `color_narrator/` – narration-specific unit, integration, and experimental suites.
- `personal_tagger/` – early tests for the upcoming Personal Tagger workflow.
- `vision/` – lower-level algorithm tests for shared vision utilities.
- `analysis/` – diagnostic notebooks/scripts used during research and validation.
- `shared/` – curated test assets (images, overlays, sample data).
- `test_output/` – scratch space for test artefacts (git-ignored).

## `outputs/`
- `summaries/` – human-readable reports (Markdown, CSV) intended for review.
- `results/` – machine-readable outputs such as JSONL result sets.

## Console Scripts
Declared in `pyproject.toml` and runnable via `uv run <script>`:
- `imageworks-mono` – main CLI for the competition checker.
- `imageworks-mono-api` – FastAPI server hosting the mono endpoints.
- `imageworks-mono-xmp` – export/cleanup of monochrome diagnostics to XMP.
- `imageworks-color-narrator` – narration CLI (`narrate`, `validate`, etc.).

## Typical Outputs
- `mono_results.jsonl` – structured per-image evaluation emitted by the mono CLI.
- Overlay images with `_overlay` suffix written alongside originals by the visualiser.

## Getting Started
- CLI usage: follow the instructions in `README.md` under “Running the Applications”.
- Core algorithm reference: `src/imageworks/libs/vision/mono.py`.
- XMP export workflow: `src/imageworks/tools/write_mono_xmp.py` plus `configs/exiftool`.
- Color Narrator orchestration: `src/imageworks/apps/color_narrator/core/` modules.

## Development and Testing Guidelines

### Test Layout
Organise new tests under `tests/<module>/<unit|integration>/` and keep fixtures in `tests/shared/` when they are reused between modules.

### Output Hygiene
Always direct ad-hoc or test artefacts to `tests/test_output/` to keep the repository clean:
- **Avoid:** `uv run imageworks-mono check folder --jsonl-out results.jsonl`
- **Prefer:** `uv run imageworks-mono check folder --jsonl-out tests/test_output/results.jsonl`

### Reference Material
For detailed testing conventions and tooling, see `tests/README.md`.
