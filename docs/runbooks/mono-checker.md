# Mono Checker Runbook

Guidelines for preparing, executing, and reviewing monochrome checks via CLI and GUI.

---
## 1. Preparation

1. Confirm input directory contains JPEG/TIFF competition submissions named `01_XXXX.*` (filter used by `_iter_files`).
2. Ensure overlays directory exists and has sufficient space (defaults from `pyproject.toml`).
3. Install ExifTool if metadata writing is required (`sudo apt install exiftool`).
4. Review `[tool.imageworks.mono]` thresholds before production runs.

---
## 2. Standard Batch Run (CLI)

1. Build command:
   ```bash
   uv run imageworks-mono check \
     /path/to/submissions \
     --jsonl-out outputs/results/run_2024w12.jsonl \
     --summary-out outputs/summaries/run_2024w12.md \
     --auto-heatmap --write-xmp
   ```
2. Monitor progress (Rich progress bar). Expect summary lines per subdirectory.
3. Review Markdown summary for FAIL/QUERY counts.
4. Inspect overlays for flagged images (stored next to originals or configured dir).

---
## 3. Standard Batch Run (GUI)

1. Open Streamlit → “Mono Checker” page.
2. Choose preset (Competition Default).
3. Set **Input Directory**, **Overlay Output**, **JSONL** and **Summary** paths.
4. Toggle **Auto heatmaps** and **Metadata** as required.
5. Click **Run**; monitor real-time logs in the Process Runner.
6. After completion, browse results in the verdict tabs and drill into any FAIL images.

---
## 4. Re-running Overlays Only

- CLI: `uv run imageworks-mono visualize /path/to/submissions --mode lab_residual --fails-only --out-suffix _review`.
- GUI: Use **Regenerate Overlays** preset (custom button) to call the command with saved parameters.

---
## 5. Metadata Export Workflow

1. Run `check` with `--write-xmp` and desired keyword options.
2. Generated script `write_xmp.sh` executes automatically; confirm no errors in CLI log.
3. If re-running metadata only, set `--no-auto-heatmap --table-only` to speed up.
4. Verify Lightroom keywords/collections update accordingly.

---
## 6. Handling Failures

| Symptom | Resolution |
|---------|------------|
| CLI exits 1 with “No files matched” | Verify directory, adjust extensions or ensure naming convention matches `01_`. |
| Process Runner shows ExifTool error | Check `exiftool` installation; rerun with `--no-write-xmp` if necessary. |
| GUI results empty | Confirm JSONL path; use “Open in viewer” button to reload file. |
| Overlays overwritten | Change suffix or output directory; rerun `visualize` with unique `--out-suffix`. |

---
## 7. Post-Run Tasks

1. Archive JSONL and Markdown summary in competition folder.
2. Update tracker with PASS/FAIL/QUERY counts.
3. Notify judges of flagged images with direct links to overlays.
4. If metadata was written, commit keyword changes to DAM system.

---
## 8. Automation Hooks

- API batch endpoint can be invoked from ingestion pipelines: POST `/mono/batch` with folder + thresholds; store NDJSON response.
- Use `imageworks-models` to ensure registry models referenced in prompts remain locked.
- GUI offers “Save preset as default” storing CLI arguments for future runs.

