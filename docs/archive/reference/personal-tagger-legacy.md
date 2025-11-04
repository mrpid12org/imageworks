# Personal Tagger Reference

The personal tagger generates captions, descriptions, and keyword metadata for
portfolio images using OpenAI-compatible multimodal backends.

## Configuration model
- `PersonalTaggerSettings` loads defaults from `[tool.imageworks.personal_tagger]`
  (pyproject) and exposes image extension filters, backend defaults, and output
  destinations.【F:src/imageworks/apps/personal_tagger/core/config.py†L55-L128】【F:pyproject.toml†L200-L257】
- `build_runtime_config` merges CLI options, pyproject defaults, and environment
  overrides (`IMAGEWORKS_PERSONAL_TAGGER__*`).【F:src/imageworks/apps/personal_tagger/core/config.py†L1-L54】【F:src/imageworks/apps/personal_tagger/cli/main.py†L19-L120】
- Registry integration toggles (`--use-loader`, `--use-registry`) allow mapping
  functional roles to logical model names via the deterministic loader before
  inference.【F:src/imageworks/apps/personal_tagger/cli/main.py†L121-L214】

## Runtime pipeline
- `PersonalTaggerRunner` discovers files, orchestrates inference, and writes
  JSONL/Markdown summaries plus batch metrics.【F:src/imageworks/apps/personal_tagger/core/runner.py†L1-L104】
- Discovery honours recursive scanning, explicit file inputs, and extension
  filters configured in `PersonalTaggerConfig`.【F:src/imageworks/apps/personal_tagger/core/runner.py†L41-L77】
- `create_inference_engine` picks a backend-specific engine (OpenAI-compatible)
  that issues caption, keyword, and description prompts in sequence per
  image.【F:src/imageworks/apps/personal_tagger/core/inference.py†L29-L220】
- The critique stage can run the `club_judge_json` profile, prompting the model
  for a rubric-guided critique that returns structured JSON (title, category,
  critique body, score) with automatic fallback when parsing fails.【F:src/imageworks/apps/personal_tagger/core/inference.py†L325-L406】【F:src/imageworks/apps/personal_tagger/core/prompts.py†L244-L280】
- Metadata persistence flows through `PersonalMetadataWriter`, supporting
  Lightroom/XMP updates with opt-in backups and overwrite rules.【F:src/imageworks/apps/personal_tagger/core/runner.py†L78-L138】【F:src/imageworks/apps/personal_tagger/core/metadata_writer.py†L23-L170】
- Optional `preflight` mode performs REST probes for model listing, text, and
  vision inference before processing any files.【F:src/imageworks/apps/personal_tagger/core/runner.py†L105-L185】

## CLI (`imageworks-personal-tagger`)
- Root callback mirrors the `run` command for backward compatibility, allowing
  `imageworks-personal-tagger --input-dir …` or explicit subcommands.【F:src/imageworks/apps/personal_tagger/cli/main.py†L19-L120】
- `list-registry` prints registry candidates for caption/keyword/description
  roles when the loader integration is enabled.【F:src/imageworks/apps/personal_tagger/cli/main.py†L215-L232】
- `run` executes the full pipeline, supports dry runs, and records structured
  JSONL plus Markdown summaries for audit trails.【F:src/imageworks/apps/personal_tagger/cli/main.py†L233-L372】
- New options `--critique-title-template`, `--critique-category`, and
  `--critique-notes` feed additional context into the judging prompt, enabling
  reusable competition briefs from both CLI and GUI flows.【F:src/imageworks/apps/personal_tagger/cli/main.py†L100-L164】【F:src/imageworks/apps/personal_tagger/cli/main.py†L462-L515】

## Outputs
- JSONL log of `PersonalTaggerRecord` entries with model identifiers, generated
  text, critique metadata (title/category/score), and metadata write status.
- Markdown summary aggregating keyword frequencies and notable captions.
- Metrics history appended to `outputs/metrics/personal_tagger_batch_metrics.json`.

## Integration
- Model loader: resolves logical names when `--use-loader` is provided, sharing
  registry semantics with the chat proxy and downloader.【F:src/imageworks/apps/personal_tagger/cli/main.py†L121-L214】
- Color narrator: reuses VLM backend configuration for consistent GPU scheduling.
- Zip extract: shares metadata helpers to maintain consistent Lightroom keyword
  handling across ingestion pipelines.【F:src/imageworks/apps/personal_tagger/core/metadata_writer.py†L23-L170】
