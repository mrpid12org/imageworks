# Personal Tagger Runbook

Use this runbook to generate captions, descriptions, and keywords for portfolio
shots while maintaining safe metadata practices.

## 1. Prerequisites
- Confirm `uv run imageworks-loader list --role caption` returns available
  registry entries if you intend to use loader integration.
- Update `[tool.imageworks.personal_tagger]` paths in `pyproject.toml` to point at
  your source directories and default outputs.【F:pyproject.toml†L200-L257】
- Export overrides (optional) using the `IMAGEWORKS_PERSONAL_TAGGER__` prefix.

## 2. Discover candidate images
Run a dry discovery pass to confirm filters:
```bash
uv run imageworks-personal-tagger run --dry-run --no-meta \
  --input-dir ~/photos/portfolio
```
The runner logs skipped paths and honours recursive scanning and extension
filters defined in the config object.【F:src/imageworks/apps/personal_tagger/core/runner.py†L41-L104】

## 3. Execute tagging
```bash
uv run imageworks-personal-tagger run \
  --input-dir ~/photos/portfolio \
  --output-jsonl outputs/results/personal_tagger.jsonl \
  --summary outputs/summaries/personal_tagger.md \
  --batch-size 2 --max-workers 2 --prompt-profile default
```
Key switches:
- `--use-registry` resolves caption/keyword/description models by functional role
  via the deterministic loader.【F:src/imageworks/apps/personal_tagger/cli/main.py†L121-L214】
- `--preflight/--no-preflight` toggles health checks. Leave enabled for remote
  backends to avoid partial runs.【F:src/imageworks/apps/personal_tagger/core/runner.py†L105-L185】
- `--dry-run` retains JSONL/summary output but skips metadata writes and marks
  records accordingly.【F:src/imageworks/apps/personal_tagger/core/runner.py†L78-L138】
- `--critique-title-template`, `--critique-category`, and `--critique-notes` feed
  structured context into the competition-judge critique stage (e.g. with the
  `club_judge_json` prompt profile) for consistent scoring output.【F:src/imageworks/apps/personal_tagger/cli/main.py†L85-L170】

## 4. Review results
- Inspect the Markdown summary for keyword frequency, caption quality, and any
  errors flagged in the `notes` column.
- JSONL entries contain raw text suitable for reformatting or follow-up prompts.
- When a judging profile is active, review the `critique_title`,
  `critique_category`, and `critique_score` fields alongside the narrative
  critique in both JSONL and the Markdown summary.
- Batch metrics append inference duration and throughput history to
  `outputs/metrics/personal_tagger_batch_metrics.json` for regression tracking.【F:src/imageworks/apps/personal_tagger/core/runner.py†L78-L138】

## 5. Troubleshooting
| Symptom | Checks |
| --- | --- |
| `Preflight: failed to connect` | Verify `--base-url` and API key. The preflight performs `/models`, text, and vision probes before running.【F:src/imageworks/apps/personal_tagger/core/runner.py†L105-L185】 |
| Metadata not written | Ensure `--no-meta` is not set and the Exif/XMP sidecar path is writable. The runner records `metadata_written` and any exceptions in `notes`.【F:src/imageworks/apps/personal_tagger/core/runner.py†L78-L138】 |
| Duplicate captions | When running multiple passes, enable `--overwrite-metadata` to replace existing tags or use `--dry-run` for inspection only.【F:src/imageworks/apps/personal_tagger/cli/main.py†L19-L120】 |
| Backend returns wrong model | Use `--caption-model`/`--keyword-model` overrides or update registry aliases. `list-registry` shows available logical names when using loader integration.【F:src/imageworks/apps/personal_tagger/cli/main.py†L215-L372】 |

Archive JSONL outputs with commit hashes for reproducibility after each tagging
cycle.
