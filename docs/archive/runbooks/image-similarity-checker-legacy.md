# Image Similarity Checker Runbook

Use this playbook to detect near-duplicates between candidate submissions and
the historical competition library.

## 1. Prepare configuration
- Update `[tool.imageworks.image_similarity]` in `pyproject.toml` with the current
  library root and default thresholds.【F:pyproject.toml†L258-L325】
- Export overrides with `IMAGEWORKS_IMAGE_SIMILARITY__` variables when running in
  CI or remote environments (e.g. `...__LIBRARY_ROOT`).【F:src/imageworks/apps/image_similarity_checker/core/config.py†L1-L152】

## 2. Warm library manifests
```bash
uv run imageworks-image-similarity check \
  ~/photos/2024-candidates --library-root ~/archive/ccc \
  --dry-run --summary outputs/summaries/similarity_dryrun.md
```
- Dry run populates the library manifest cache and validates discovery without
  executing heavy embeddings.【F:src/imageworks/apps/image_similarity_checker/core/engine.py†L45-L113】
- Inspect the summary for discovered candidate counts and ensure paths are
  resolved correctly.

## 3. Run full similarity analysis
```bash
uv run imageworks-image-similarity check \
  ~/photos/2024-candidates --library-root ~/archive/ccc \
  --output-jsonl outputs/results/similarity.jsonl \
  --summary outputs/summaries/similarity.md \
  --top-matches 5 --fail-threshold 0.92 --query-threshold 0.85
```
- Enable additional strategies using repeated `--strategy` flags (e.g. `embedding`
  plus `perceptual_hash`).【F:src/imageworks/apps/image_similarity_checker/cli/main.py†L13-L140】
- To leverage registry-managed VLM explainers, add `--use-loader --prompt-profile
  default` and configure the desired logical model via the loader CLI.【F:src/imageworks/apps/image_similarity_checker/core/engine.py†L30-L44】
- Set `--write-metadata` to inject verdict keywords into IPTC metadata. Backups
  and overwrite behaviour default to pyproject values but can be toggled with
  CLI flags.【F:src/imageworks/apps/image_similarity_checker/core/metadata.py†L19-L178】

## 4. Review results
- JSONL file contains structured matches for ingestion into dashboards.
- Markdown summary surfaces top matches, scores, and optional VLM explanations.
- Investigate `SimilarityVerdict.FAIL` entries first; `QUERY` results warrant
  manual review but may pass after justification.【F:src/imageworks/apps/image_similarity_checker/core/engine.py†L114-L190】

## 5. Troubleshooting
| Symptom | Checks |
| --- | --- |
| `No candidate images were found` | Ensure paths are correct and include files directly, not just directories; the engine validates before running strategies.【F:src/imageworks/apps/image_similarity_checker/core/engine.py†L45-L68】 |
| `CapabilityError` when using loader | The selected logical model lacks required capabilities (vision/tool). Choose a different registry entry or disable loader integration.【F:src/imageworks/apps/image_similarity_checker/core/engine.py†L30-L44】 |
| VLM explanations time out | Increase `--timeout`, reduce `--top-matches`, or run without explanations by omitting backend parameters.【F:src/imageworks/apps/image_similarity_checker/cli/main.py†L83-L180】 |
| Metadata writes skipped | Confirm `--write-metadata` is enabled and that filesystem paths are writable. Logs note when no matches are found or writes are skipped.【F:src/imageworks/apps/image_similarity_checker/core/metadata.py†L19-L178】 |

Store JSONL and summary artifacts with the competition batch for auditability.
