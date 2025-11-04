# Color Narrator Runbook

Follow this checklist to narrate mono competition finalists and capture the
output in Lightroom-friendly metadata.

## 1. Prepare mono results
1. Run the mono checker to generate the JSONL verdict file referenced by the
   narrator defaults:
   ```bash
   uv run imageworks-mono check --jsonl-out outputs/results/mono_results.jsonl \
     --summary-out outputs/summaries/mono_summary.md
   ```
   The CLI enforces `--write-xmp` support by requiring a JSONL destination before
   invoking the ExifTool script.【F:src/imageworks/apps/mono_checker/cli/mono.py†L624-L707】
2. Ensure overlays exist in the configured directory if `require_overlays` is
   left enabled (default true).【F:src/imageworks/apps/color_narrator/core/narrator.py†L83-L110】

## 2. Configure the VLM backend
- Check `[tool.imageworks.color_narrator]` in `pyproject.toml` for base URLs and
  model names. Override per run with `--backend`, `--model`, or environment
  variables such as `IMAGEWORKS_COLOR_NARRATOR__VLM_BASE_URL`.
- Validate connectivity before long runs:
  ```bash
  uv run imageworks-color-narrator validate
  ```
  This command performs filesystem checks and hits the selected backend’s health
  endpoint.【F:src/imageworks/apps/color_narrator/cli/main.py†L1608-L1754】

## 3. Execute narration
Run batches with pyproject defaults and optional overrides:
```bash
uv run imageworks-color-narrator run --regions --batch-size 2 \
  --summary-path outputs/summaries/color_narrator.md
```
- `--regions` sends grid crops through `RegionBasedVLMAnalyzer` when supported by
  the chosen prompt definition.【F:src/imageworks/apps/color_narrator/core/narrator.py†L51-L60】
- `--no-write-xmp` performs a dry run but retains JSONL and Markdown output for
  review.【F:src/imageworks/apps/color_narrator/cli/main.py†L809-L1208】
- Adjust concurrency with `--max-concurrent-requests` to match GPU VRAM limits.

Progress and per-batch latency are logged through `BatchRunMetrics` in the CLI.
JSONL entries capture both VLM responses and metadata write status for later
inspection.【F:src/imageworks/apps/color_narrator/cli/main.py†L1340-L1435】

## 4. Review deliverables
- Markdown summary (if requested) highlights interesting cases by verdict.
- JSONL log feeds QA dashboards or the `summarise` command:
  ```bash
  uv run imageworks-color-narrator summarise \
    --jsonl inputs/results/mono_results.jsonl --summary-out audits/review.md
  ```
- Lightroom keywords and custom metadata appear after the generated script runs
  `imageworks.tools.write_mono_xmp` unless `--no-write-xmp` was specified.【F:src/imageworks/apps/color_narrator/cli/main.py†L1340-L1412】

## 5. Troubleshooting
| Symptom | Checks |
| --- | --- |
| CLI exits with `VLM backend is not available` | Ensure LMDeploy/vLLM server is running and that `--backend` matches the deployment. Health check uses the configured base URL before processing begins.【F:src/imageworks/apps/color_narrator/core/narrator.py†L124-L146】 |
| `--write-xmp` aborts with missing JSONL path | Provide `--jsonl-out` (or set `default_jsonl`) so the CLI can read verdicts when invoking the metadata writer.【F:src/imageworks/apps/color_narrator/cli/main.py†L1340-L1371】 |
| Overlays skipped despite existing files | Confirm filenames align with mono results and that `require_overlays` remains true. Disabled overlays allow narration without heatmaps.【F:src/imageworks/apps/color_narrator/core/narrator.py†L83-L108】 |
| GPU OOM | Lower `--batch-size`, disable `--regions`, or switch to the lighter vLLM default defined in pyproject.【F:src/imageworks/apps/color_narrator/core/narrator.py†L32-L60】【F:pyproject.toml†L123-L199】 |

Document completion of each run in the project tracker along with summary
artifacts for judging.
