# Personal Tagger Overview

The Personal Tagger automates descriptive captions, keywords, and narrative
summaries for Lightroom libraries. It reuses the shared Imageworks libraries to
run local VLM inference, write structured metadata, and produce per-run
artefacts for downstream review.

## Goals
- Discover candidate images from user-managed folders (optionally recursive).
- Invoke a local OpenAI-compatible VLM backend to generate descriptive text.
- Persist results back into the files (XMP) and as JSONL/Markdown reports.
- Track batch metrics for comparison across models or prompt profiles.
- Provide a path to deterministic model selection via the unified registry.

## Architecture at a Glance
```
┌──────────────────────────┐
│ PersonalTaggerRunner     │  ← coordinates the flow
├────────────┬─────────────┤
│ discovery  │ inference   │
│            ▼             │
│      BaseInferenceEngine │  ← wraps HTTP chat completions
│            │             │
│            ▼             │
│   PersonalMetadataWriter │  ← writes/updates XMP metadata
└────────────┴─────────────┘
          │
          ▼
  outputs/ (JSONL, summary, metrics)
```

Key modules (`src/imageworks/apps/personal_tagger/core/`):

| Module | Responsibility |
| --- | --- |
| `runner.py` | Top-level orchestration; discovers images, handles preflight, aggregates metrics, writes reports. |
| `config.py` | Loads defaults from `pyproject.toml`, environment overrides, and CLI flags into a frozen `PersonalTaggerConfig`. |
| `inference.py` | Defines `BaseInferenceEngine` and concrete HTTP client that streams chat responses from vLLM/LMDeploy. |
| `prompts.py` | Houses prompt templates and profiles for different tagging styles. |
| `metadata_writer.py` | Applies generated content to image metadata (XMP) and gracefully skips on dry runs. |
| `models.py` | Provides `PersonalTaggerRecord` dataclass used for JSONL/summary output. |
| `model_registry.py` | Integrates with the unified registry for role-based model resolution (optional today). |

Shared utilities from `imageworks.model_loader` supply batch metrics, registry
lookups, and role/capability checks once `--use-registry` is enabled.

## Workflow

1. **Configuration** – `PersonalTaggerConfig` is assembled from `pyproject.toml`
   `[tool.imageworks.personal_tagger]`, environment variables prefixed with
   `IMAGEWORKS_PERSONAL_TAGGER__`, and CLI parameters.
2. **Preflight (optional)** – when `preflight=true`, the runner issues a
   `/v1/models` probe and light text/vision completions against the configured
   backend (LMDeploy or vLLM). Failures raise actionable `RuntimeError`s.
3. **Discovery** – `runner.discover_images()` walks the configured input paths
   (recursive by default) and filters by allowed extensions.
4. **Inference** – the inference engine batches or sequentially submits images,
   returning a `PersonalTaggerRecord` containing captions, keywords, and free
   text descriptions together with latency metadata.
5. **Metadata Write** – unless `--dry-run` or `--no-meta` is set, the writer
   updates XMP blocks in-place, taking a backup copy when `backup_originals` is
   true. Failures are logged but do not abort the entire run.
6. **Reporting** – results are persisted to:
   - JSONL (`outputs/results/personal_tagger.jsonl` by default)
   - Markdown summary (`outputs/summaries/personal_tagger_summary.md`)
   - Metrics history (`outputs/metrics/personal_tagger_batch_metrics.json`)

## Model Selection

By default the configuration points at the LMDeploy server exposed on
`http://localhost:24001/v1` serving `Qwen2.5-VL-7B-AWQ`. The CLI flags
`--backend`, `--base-url`, `--description-model`, `--caption-model`, and
`--keyword-model` override these values.

Experimental support for the unified registry is available via `--use-registry`:
- Roles (`caption_role`, `keyword_role`, `description_role`) map to registry
  entries advertising the corresponding capability.
- The runner resolves the first non-deprecated entry that satisfies the role and
  vision requirements, mirroring the Personal Tagger model registry notes.
- This allows central model upgrades by editing `configs/model_registry.json`.

See [Model Registry Notes](model-registry.md) for structured role definitions
and naming conventions.

## Configuration Surface

`PersonalTaggerSettings` in `config.py` captures defaults such as input paths,
backend URLs, token limits, and image extensions. Important toggles:

| Setting | Purpose |
| --- | --- |
| `input-paths` | One or more directories/files to scan. |
| `recursive` | Whether to walk sub-directories. |
| `dry-run` | Skip XMP writes but still emit JSONL/summary. |
| `no-meta` | Never write metadata (even if not a dry run). |
| `backup-originals` | Copy the original file before writing XMP. |
| `overwrite-metadata` | Force replacement instead of merge-if-different. |
| `prompt-profile` | Selects a prompt variant from `prompts.py`. |
| `batch-size` / `max-workers` | Control concurrency during inference. |
| `preflight` | Enable backend readiness checks. |

Environment variables follow the pattern
`IMAGEWORKS_PERSONAL_TAGGER__<SETTING>=value` (e.g.
`IMAGEWORKS_PERSONAL_TAGGER__BASE_URL`).

## CLI Usage

```
uv run imageworks-personal-tagger run \
  --input ~/Photos/2024-portfolio \
  --summary outputs/summaries/portfolio_tags.md \
  --use-registry \
  --description-role description_v2
```

Dry run without metadata:
```
uv run imageworks-personal-tagger run \
  --input ./sample-images \
  --dry-run --no-meta
```

All parameters map to the dataclass fields documented in `config.py`; see the
Typer auto-generated `--help` output for the full flag list.

## Outputs

- **JSONL** – machine-readable per-image records (image path, prompts used,
  responses, timing, metadata status).
- **Markdown summary** – aggregated table of captions/keywords per image.
- **Metrics** – rolling history capturing per-stage latency, tokens generated,
  and throughput to help compare model or prompt experiments.

## Operational Notes

- The runner logs to the standard Imageworks logging pipeline; enable debug
  logging via `IMAGEWORKS_LOG_LEVEL=DEBUG` for verbose tracing.
- If the metadata writer encounters read-only files or unsupported formats, the
  run continues with `metadata_written=False` and a descriptive note.
- The XMP writer currently handles JPEG files; RAW sidecar handling is on the
  backlog.
- Prompt templates are pure Python dictionaries; restart the CLI after editing
  `prompts.py` to pick up changes.

## Future Work

- Full registry integration with automated role resolution defaults.
- Expanded metadata support for Lightroom keyword hierarchies.
- Additional prompt profiles for niche workflows (e.g. competition judging,
  social media variants).
- Batch scheduling improvements (adaptive concurrency based on latency).
