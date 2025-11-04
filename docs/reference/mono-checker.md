# Mono Checker Operations Guide

The mono checker evaluates monochrome competition images for unintended colour casts, generates overlays, and writes Lightroom-compatible metadata. It powers both the CLI (`imageworks-mono`) and the Streamlit “Mono Checker” page.

---
## 1. Feature Summary

| Capability | Details |
|------------|---------|
| LAB-based tint detection | `libs.vision.mono.check_monochrome` computes hue spread, chroma footprint, split-tone detection, and verdicts (PASS / PASS_WITH_QUERY / FAIL). |
| Metadata export | Optional XMP/keywords via bundled writer scripts; supports in-file EXIF updates or sidecar `.xmp` files. |
| Overlay generation | LAB chroma/residual heatmaps, HSV saturation maps, and channel-difference overlays to visualise colour leaks. |
| Batch processing | Recursive folder traversal with extension filters, JSONL audit logs, Markdown summaries, and CSV exports. |
| Heatmap automation | Auto-generate overlays for failed/query images with configurable modes, suffixes, and quality thresholds. |
| GUI integration | Streamlit page orchestrates CLI invocations, presets thresholds, and visualises results with grids/detail panes. |
| API service | FastAPI microservice (`imageworks-api`) for single-image or batch checks, streaming NDJSON results. |

---
## 2. Workflow Architecture

```
Input folder → CLI/GUI preset → check_monochrome()
   → Verdict + metrics stored per image
   → JSONL + Markdown summary + optional CSV
   → Overlay generation (optional)
   → XMP metadata writer (optional)
```

Key modules:
- `apps.mono_checker.cli.mono`: Typer commands (`check`, `visualize`).
- `libs.vision.mono`: analysis primitives and descriptive helpers.
- `apps.mono_checker.api.main`: `/mono/check` and `/mono/batch` endpoints for automation.
- GUI components: `pages/3_🖼️_Mono_Checker.py` uses preset selector, CLI wrapper, and results viewer components.

---
## 3. CLI Surface (`uv run imageworks-mono ...`)

### 3.1 `check`
- Arguments: `FOLDER` (optional; defaults from `[tool.imageworks.mono]`).
- Key options: `--exts`, `--neutral-tol`, `--lab-neutral-chroma`, `--lab-toned-pass`, `--lab-toned-query`, `--lab-fail-c4-ratio`, `--jsonl-out`, `--csv-out`, `--summary-out`, `--table-only/--no-table-only`, `--auto-heatmap/--no-auto-heatmap`, `--write-xmp/--no-write-xmp`, `--xmp-sidecar`, `--include-grid-regions`.
- Outputs: console summary by subdirectory, JSONL lines per image, Markdown summary grouping PASS/QUERY/FAIL, optional CSV.

### 3.2 `visualize`
- Generates overlays without re-checking metrics.
- Options: `--mode` (`channel_diff`, `saturation`, `hue`, `lab_chroma`, `lab_residual`), `--fails-only`, `--out-suffix`, threshold tunables.
- Can infer failure list from JSONL or run a quick scan when `--fails-only` is set.

### 3.3 Configuration fallbacks
- `[tool.imageworks.mono]` in `pyproject.toml` defines default folder, thresholds, output paths, heatmap settings, and XMP behaviour.
- CLI respects overrides and environment variables (e.g., `IMAGEWORKS_LOG_DIR` for logs).

---
## 4. GUI Experience

- **Preset Selector**: `MONO_CHECKER_PRESETS` offers “Competition Default”, “Strict LAB”, “Debug” etc., mapping directly to CLI options.
- **Custom Overrides Panel**: Allows editing input directory, overlay output, JSONL/Summary paths, LAB thresholds, dry-run toggle.
- **Command Preview**: Renders exact CLI invocation used behind the scenes.
- **Process Runner**: Executes Typer command, streams stdout/stderr, captures return code.
- **Results Browser**: Reads JSONL output, groups by verdict, shows thumbnails with verdict badges, and toggles overlays.
- **Image Detail**: Displays original, overlays, per-image metrics, Lightroom tips (mirrors CLI summary).

GUI persists user overrides in `st.session_state` (per app) and offers quick folder selection from ZIP extraction output.

---
## 5. Outputs & Metadata

| Artifact | Location | Description |
|----------|----------|-------------|
| JSONL log | Default `outputs/results/mono_results.jsonl` | One line per image with full metric dictionary. |
| Markdown summary | Default `outputs/summaries/mono_summary.md` | Human-readable grouped verdict report. |
| CSV (optional) | User-specified | Tabular export for spreadsheet review. |
| Heatmaps | `<image>_mono_vis.png` or per-mode suffix | LAB/HSV overlays saved alongside originals. |
| XMP script | `write_xmp.sh` (if metadata enabled) | Shell script executed to push keywords/metadata via ExifTool. |

---
## 6. API Endpoints (`uv run imageworks-api`)

| Endpoint | Method | Notes |
|----------|--------|-------|
| `/healthz` | GET | Basic health probe. |
| `/mono/check` | POST | Accepts uploaded file (`image`) or `path`, plus threshold overrides; returns JSON verdict. |
| `/mono/batch` | POST | Streams NDJSON for folder scan (`folder`, `exts`, thresholds). |

---
## 7. Troubleshooting

| Issue | Diagnosis | Fix |
|-------|-----------|-----|
| `Folder not provided` exit | Config default missing or path invalid. | Supply folder arg or set `default_folder` in `pyproject.toml`. |
| No files matched | Extensions list wrong. | Adjust `--exts` to include actual formats (e.g., `jpg,jpeg,tif`). |
| Heatmaps missing | `--auto-heatmap` disabled or overlays already exist. | Re-run with `--no-fails-only` or delete existing overlays. |
| XMP script fails | ExifTool missing. | Install `exiftool` or disable `--write-xmp`. |
| GUI run stalls | CLI waiting for ExifTool or disk; check Process Runner logs; disable metadata in GUI. |

---
## 8. Best Practices

1. Use `--table-only` for CI runs to reduce stdout noise.
2. Maintain JSONL history for audit; combine across batches with `cat`.
3. Store overlays separately from originals to avoid Lightroom confusion; configure `default_overlays_dir`.
4. Run `visualize --mode lab_residual --fails-only` post-review to regenerate overlays after metadata adjustments.
5. Use API for automated pipelines when integrating with ingestion workflows.

