# Mono Checker Runbook

Use this runbook to evaluate monochrome competition submissions and prepare
artifacts for downstream narration.

## 1. Configure defaults
- Edit `[tool.imageworks.mono]` in `pyproject.toml` to point at the current
year’s competition directories and desired thresholds.【F:pyproject.toml†L62-L118】
- Override per run via CLI options or environment variables such as
  `IMAGEWORKS_MONO__DEFAULT_FOLDER` if you need ad-hoc paths (same pattern as
  other tools).【F:src/imageworks/apps/mono_checker/cli/mono.py†L69-L152】

## 2. Run the checker
Typical command:
```bash
uv run imageworks-mono check \
  --jsonl-out outputs/results/mono_results.jsonl \
  --csv-out outputs/results/mono_results.csv \
  --summary-out outputs/summaries/mono_summary.md
```
Key flags:
- `--table-only/--no-table-only` toggles per-image console output (disable when
  investigating specific files).【F:src/imageworks/apps/mono_checker/cli/mono.py†L210-L256】
- `--include-grid-regions` enriches JSONL with 3×3 spatial analysis used by the
  color narrator.【F:src/imageworks/apps/mono_checker/cli/mono.py†L302-L333】
- `--no-write-xmp` runs the checker without triggering the metadata exporter.

A bash script is generated and executed when `--write-xmp` remains enabled. The
script calls `imageworks.tools.write_mono_xmp` to apply Lightroom-friendly
keywords and custom tags.【F:src/imageworks/apps/mono_checker/cli/mono.py†L624-L707】

## 3. Generate overlays (optional)
To visualise leak hotspots:
```bash
uv run imageworks-mono visualize --mode lab_chroma --fails-only
```
- `--mode` accepts `channel_diff`, `saturation`, `hue`, `lab_chroma`, or
  `lab_residual` depending on the investigation.【F:src/imageworks/apps/mono_checker/cli/mono.py†L1038-L1121】
- `--out-suffix` controls the exported filename (default `_mono_vis`).

## 4. Review outputs
- JSONL: feed into dashboards or the color narrator pipeline.
- CSV: share verdicts with judges; includes dominant tone analysis and failure
  summaries.
- Markdown summary: quick counts and highlights suitable for weekly updates.

## 5. Troubleshooting
| Symptom | Checks |
| --- | --- |
| `Folder not provided or not found` | Ensure the directory exists or update `default_folder` in pyproject. The CLI validates before scanning.【F:src/imageworks/apps/mono_checker/cli/mono.py†L252-L308】 |
| Heatmaps missing despite `--auto-heatmap` | Generated only when `auto_heatmap` remains enabled and files fail the checker. Confirm `auto_heatmap_modes` in pyproject when troubleshooting.【F:src/imageworks/apps/mono_checker/cli/mono.py†L273-L342】 |
| XMP script fails | Verify ExifTool is installed and accessible; rerun with `--dry-run` to isolate inference vs metadata issues. The CLI logs the generated script path before execution.【F:src/imageworks/apps/mono_checker/cli/mono.py†L644-L707】 |
| Unexpected passes | Lower `lab_fail_c4_ratio`/`lab_fail_c4_cluster` or re-run with `--summary-only` to check aggregate counts before exporting overlays.【F:src/imageworks/apps/mono_checker/cli/mono.py†L230-L288】 |

Log summary artifacts in the shared drive after each batch for future reference.
