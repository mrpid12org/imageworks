# Color Narrator Reference

The color narrator application batches monochrome competition entries, enriches
mono-checker output with a vision-language model (VLM), and writes structured
XMP metadata back to the source files.

## Core pipeline
- `NarrationConfig` collects data roots, VLM settings, and post-processing
  toggles, defaulting to LMDeploy with Qwen2.5-VL-7B-AWQ.【F:src/imageworks/apps/color_narrator/core/narrator.py†L19-L61】
- `ColorNarratorDataLoader` filters mono JSONL results by contamination level
  and overlay availability before yielding `ColorNarratorItem` batches for
  inference.【F:src/imageworks/apps/color_narrator/core/narrator.py†L63-L110】
- `VLMClient.infer_single` constructs OpenAI-compatible requests and normalises
  backend-specific quirks; health checks run before every batch.【F:src/imageworks/apps/color_narrator/core/narrator.py†L112-L171】【F:src/imageworks/apps/color_narrator/core/vlm.py†L27-L180】
- Successful calls create `ColorNarrationMetadata` payloads that the
  `XMPMetadataWriter` injects into the image file unless `dry_run` is enabled.【F:src/imageworks/apps/color_narrator/core/narrator.py†L173-L213】

### Region-enhanced prompting
Enable `config.use_regions` to feed 3×3 grid crops plus mono telemetry into
`RegionBasedVLMAnalyzer`. The analyzer aggregates per-region captions and
contamination scores before composing the final prompt template.【F:src/imageworks/apps/color_narrator/core/narrator.py†L51-L60】【F:src/imageworks/apps/color_narrator/core/region_based_vlm.py†L16-L180】

## Configuration sources
- `[tool.imageworks.color_narrator]` in `pyproject.toml` defines default VLM
  endpoints, contamination thresholds, and overlay directories.【F:pyproject.toml†L123-L199】
- Environment overrides follow the `IMAGEWORKS_COLOR_NARRATOR__FOO=bar`
  convention in the CLI loader (case-insensitive key name).【F:src/imageworks/apps/color_narrator/cli/main.py†L69-L106】
- Mono verdict ingestion expects the JSONL path emitted by
  `imageworks-mono check --jsonl-out …`; `require_overlays` skips entries
  lacking heatmap assets.【F:src/imageworks/apps/color_narrator/core/narrator.py†L83-L108】

## CLI entry points (`imageworks-color-narrator`)
Typer commands live in `cli/main.py` and automatically merge pyproject defaults
with command-line overrides.【F:src/imageworks/apps/color_narrator/cli/main.py†L1-L327】

### `run`
Process every qualifying item, log VLM latency, and (optionally) emit a Markdown
summary and BatchRunMetrics JSON. Set `--no-write-xmp` to skip metadata writes
and `--regions` to enable grid-enhanced prompting.【F:src/imageworks/apps/color_narrator/cli/main.py†L809-L1208】【F:src/imageworks/apps/color_narrator/cli/main.py†L1340-L1435】

### `summarise`
Aggregate prior JSONL runs into Markdown or table reports for judges. The command
supports filtering by verdict, contamination level, and manual include/exclude
lists.【F:src/imageworks/apps/color_narrator/cli/main.py†L1175-L1442】

### `overlays`
Generate LAB or mono overlays for ad-hoc inspection without rerunning the full
pipeline. Supports `--mode lab_chroma|lab_residual` and honors the pyproject
heatmap defaults.【F:src/imageworks/apps/color_narrator/cli/main.py†L1468-L1754】

### `validate`
Run sanity checks over pyproject configuration, ensuring overlay directories
exist and the selected backend responds to health probes.【F:src/imageworks/apps/color_narrator/cli/main.py†L1608-L1754】

## Outputs
- JSONL audit trail containing VLM responses, contamination metrics, and XMP
  write status.
- Optional Markdown summary file capturing pass/fail counts and exemplar cases.
- Per-image XMP tags or Lightroom keywords controlled by `--write-xmp` flags.

## Integration points
- Mono checker: consumes JSONL verdicts to prioritise color narration.
- Personal tagger: shares registry role conventions for selecting VLM backends.
- Chat proxy/model loader: VLM endpoints match the single-port orchestrator
  controlled by `imageworks-loader activate-model` when using vLLM.【F:src/imageworks/apps/color_narrator/cli/main.py†L435-L524】【F:src/imageworks/model_loader/cli_sync.py†L202-L270】
