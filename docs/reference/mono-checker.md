# Mono Checker Reference

The mono checker enforces CCC monochrome competition rules by running
LAB-based colour leakage analysis and optional overlay generation.

## Core analysis
- `check_monochrome` loads an image, converts to LAB, and computes chroma
  statistics plus hue concentration to classify frames as `pass`,
  `pass_with_query`, or `fail`. Thresholds default to constants shared with the
  pyproject configuration.【F:src/imageworks/libs/vision/mono.py†L1395-L1454】【F:pyproject.toml†L62-L118】
- The routine exposes grid-region analysis via
  `include_grid_regions=True`, reusing the color narrator grid analyzer for
  spatial context.【F:src/imageworks/libs/vision/mono.py†L1439-L1467】
- `MonoResult` structures diagnostics such as `chroma_ratio_4`, hue drift, split
  tone guesses, and optional 3×3 summaries for later VLM enrichment.【F:src/imageworks/libs/vision/mono.py†L54-L137】

## CLI structure (`imageworks-mono`)
Typer commands live in `apps/mono_checker/cli/mono.py` with defaults sourced from
`[tool.imageworks.mono]` in `pyproject.toml`. Paths fall back to the configured
competition image directory when omitted.【F:src/imageworks/apps/mono_checker/cli/mono.py†L1-L151】【F:src/imageworks/apps/mono_checker/cli/mono.py†L545-L617】

### `check`
Scans a folder tree (filtering files that start with `01_` to match entry naming
conventions) and emits JSONL, CSV, and Markdown summaries. Options cover
threshold overrides, overlay generation, and XMP script execution via
`imageworks.tools.write_mono_xmp`.【F:src/imageworks/apps/mono_checker/cli/mono.py†L167-L363】【F:src/imageworks/apps/mono_checker/cli/mono.py†L624-L707】

### `visualize`
Generates overlays using channel difference, HSV saturation, hue, or LAB heatmaps
for failed entries. Defaults honour pyproject configuration and can run on all
files or failed cases only.【F:src/imageworks/apps/mono_checker/cli/mono.py†L1038-L1164】

## Integration
- Color narrator: consumes JSONL verdicts and overlays produced by the checker.
- Personal tagger: inherits XMP writing helpers for Lightroom metadata.
- Registry/loader: mono defaults share the `[tool.imageworks]` namespace used by
  other applications for consistent configuration.【F:pyproject.toml†L62-L199】
