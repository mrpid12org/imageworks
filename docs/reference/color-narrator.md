# Color Narrator Operations Guide

Color Narrator enriches monochrome review with VLM-generated colour descriptions, confidence scores, and Lightroom metadata. It orchestrates batch narration of competition images using overlays from the mono checker and integrates tightly with the GUI.

---
## 1. Capability Matrix

| Area | Details |
|------|---------|
| Multi-backend VLM support | Supports VLLM, LMDeploy, Triton, and Ollama-compatible endpoints via `core.vlm.VLMClient`. |
| Prompt templates | `core.prompts` defines numbered templates (global, validation, region-aware) selectable via CLI/GUI. |
| Region-guided narration | Optional spatial grids using overlay masks with `RegionBasedVLMAnalyzer`. |
| Metadata writing | `core.metadata.XMPMetadataWriter` embeds narration, keywords, and Lightroom section headings. |
| Batch orchestration | `ColorNarrator` processes JSONL from mono checker, pairs with overlays, and manages retries. |
| Audit logging | JSONL export with VLM responses, confidence, mono metrics, and metadata status. |
| Integration hooks | Pulls defaults from `[tool.imageworks.color_narrator]`, shares `BatchRunMetrics` with model loader. |

---
## 2. Core Components

- `core.narrator.ColorNarrator`: high-level runner orchestrating asset loading, prompt building, VLM calls, and metadata writes.
- `core.vlm.VLMClient`: backend abstraction; resolves to HTTP requests depending on backend type.
- `core.region_based_vlm.RegionBasedVLMAnalyzer`: slices overlays into nine regions and packages prompts accordingly.
- `cli.main`: Typer interface exposing narration workflows and utility commands.
- GUI page `6_🖼️_Color_Narrator.py`: wrappers for CLI, preset management, and result browsing.

---
## 3. CLI (`uv run imageworks-color-narrator ...`)

### 3.1 `narrate`
- Key options:
  - `--images`, `--overlays`, `--mono-jsonl`
  - VLM overrides: `--vlm-backend`, `--vlm-base-url`, `--vlm-model`, `--vlm-timeout`, `--vlm-api-key`
  - Registry integration: `--vlm-registry-model`
  - Prompt control: `--prompt`, `--list-prompts`, `--regions`, `--require-overlays/--allow-missing-overlays`
  - Output: `--summary`, `--results-json`, `--no-meta`
- Behaviour:
  1. Loads mono JSONL, ensures overlays exist (optional requirement).
  2. Resolves VLM backend using configuration precedence (CLI > env > pyproject defaults).
  3. Executes narration batches; each result includes description, confidence, processing time, and metadata success flag.
  4. Writes Markdown summary and JSONL audit; optionally writes XMP metadata via `XMPMetadataWriter`.

### 3.2 Auxiliary commands
- `bundle-prompts`: Export prompt definitions for documentation.
- `preview-prompt`: Render template with sample data (supports region placeholders).
- `diagnose-backend`: Connectivity checks for selected backend.

---
## 4. Configuration

From `[tool.imageworks.color_narrator]` in `pyproject.toml`:
- Default backend preference (`vlm_backend`), base URLs for each backend, model IDs, API keys, timeouts.
- Batch settings (`default_batch_size`, `max_concurrent_requests`).
- Overlay requirements (`require_overlays`), contamination thresholds.
- Paths: `default_images_dir`, `default_overlays_dir`, `default_mono_jsonl`.
- Prompt defaults (`default_prompt_id`).

Environment overrides use `IMAGEWORKS_COLOR_NARRATOR__*` (double underscore) to map to config keys.

---
## 5. GUI Integration

- **Preset selector**: chooses prompt template, backend, batch size, metadata mode.
- **Backend status**: shows ping results using `diagnose-backend` logic.
- **Process runner**: executes CLI `narrate` command; logs VLM latency and metadata writes.
- **Results viewer**: lists narrated images with thumbnails, description, confidence, and metadata status. Allows filtering by verdict (Pass/Query/Fail from mono data).
- **Metadata toggle**: UI switch maps to `--no-meta` flag.

GUI persists overrides per user and cross-links to mono checker outputs (auto-suggests overlays directory and JSONL path).

---
## 6. Outputs

| Artifact | Description |
|----------|-------------|
| Markdown summary (`narrate_summary.md`) | Grouped by mono verdict with description excerpts and confidence. |
| JSONL results (`narrate_results.jsonl`) | Each line contains file paths, prompt id, backend info, narration, confidence, metadata flag, and error (if any). |
| Metadata writes | IPTC/XMP keywords, Lightroom sections, custom properties written to image or sidecar depending on config. |
| Logs | Append to `logs/color_narrator.log` if logging configured via `configure_logging`. |

---
## 7. Troubleshooting

| Symptom | Cause | Action |
|---------|-------|--------|
| `❌ Unknown prompt id` | Prompt ID not in registry. | Run `--list-prompts` and choose valid id or update configuration. |
| HTTP 401 from backend | Missing or incorrect API key. | Set `--vlm-api-key` or `IMAGEWORKS_COLOR_NARRATOR__vlm_api_key`. |
| Missing overlays | `require_overlays` true but PNG not found. | Disable requirement or ensure mono checker produced overlays. |
| Metadata write failure | ExifTool or Pillow error. | Check log; rerun with `--no-meta` then troubleshoot writer. |
| GUI stuck on “Waiting for backend” | Backend unreachable. | Use **Diagnose backend** button, verify service port, restart backend. |

---
## 8. Best Practices

1. Run mono checker before narrator; provide JSONL + overlays for richest context.
2. Keep prompts under version control; export with `bundle-prompts` when modifying templates.
3. Use registry models (`--vlm-registry-model`) for consistent backend selection across environments.
4. Capture JSONL artifacts for audit and training future prompt adjustments.
5. Limit batch size when using remote APIs to avoid rate limits; tune via config or CLI flag.

