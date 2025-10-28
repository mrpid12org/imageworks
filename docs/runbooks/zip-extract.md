# ZIP Extractor Runbook

Prepare competition submissions for analysis by extracting ZIP archives and
syncing metadata.

## 1. Configure defaults
- Update `[tool.imageworks.zip-extract]` in `pyproject.toml` with the latest
  download and extract paths.【F:pyproject.toml†L226-L257】
- Confirm ExifTool is installed; the CLI shells out to `exiftool` for keyword and
  metadata updates.【F:src/imageworks/tools/zip_extract.py†L70-L140】

## 2. Dry run
```bash
uv run imageworks-zip run --dry-run \
  --zip-dir /mnt/c/Users/me/Downloads/ccc \
  --extract-root /mnt/d/Competition/images
```
- Dry runs validate directory structure and show planned extractions without
  modifying files.【F:src/imageworks/tools/zip_extract.py†L200-L340】

## 3. Extract with metadata
```bash
uv run imageworks-zip run \
  --include-xmp --update-all-metadata \
  --summary outputs/zip_extract_summary.md
```
- `--include-xmp` loads sidecars for title/author sync; `--update-all-metadata`
  overwrites existing JPEG metadata rather than skipping duplicates.【F:src/imageworks/tools/zip_extract.py†L70-L140】
- Default behaviour ensures the `ccc` keyword exists on every image for Lightroom
  organisation.【F:src/imageworks/tools/zip_extract.py†L70-L140】

## 4. Review summary
- Markdown summary lists extracted files, updated metadata, skipped items, and
  errors per archive.【F:src/imageworks/tools/zip_extract.py†L142-L190】
- Archive outputs with the mono checker results to maintain provenance.

## 5. Troubleshooting
| Symptom | Checks |
| --- | --- |
| Missing target directory | Ensure the ZIP filename contains a `(Folder Name)` block or supply explicit paths; extractor defaults to the ZIP stem otherwise.【F:src/imageworks/tools/zip_extract.py†L31-L70】 |
| Metadata update failures | Verify ExifTool is available on PATH. Inspect the summary for captured exceptions per file.【F:src/imageworks/tools/zip_extract.py†L70-L140】 |
| Files skipped unexpectedly | Existing files are preserved; rerun with `--update-all-metadata` to rewrite metadata while keeping content intact.【F:src/imageworks/tools/zip_extract.py†L52-L120】 |
| Non-image contents extracted | Only allowed extensions are copied. Adjust `IMAGE_EXTS` in the module if new formats arrive.【F:src/imageworks/tools/zip_extract.py†L1-L52】 |

Store extracted folders in the location referenced by the mono checker and color
narrator defaults to maintain end-to-end automation.
