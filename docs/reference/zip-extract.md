# ZIP Extractor Operations Guide

The ZIP extractor unpacks competition submissions, normalises metadata, and ensures Lightroom keywords are applied before downstream processing (mono checker, color narrator, etc.).

---
## 1. Capabilities

| Capability | Details |
|------------|---------|
| Batch extraction | Processes all ZIP files in a directory or a single targeted ZIP. |
| Metadata propagation | Copies XMP sidecars or writes author/title metadata using ExifTool; enforces `ccc` keyword. |
| Selective extraction | Optionally include `.xmp` files; skips files already present unless `--metadata` specified. |
| Colour diagnostics | Can compute colourfulness metrics (via configuration) for reporting (future expansion). |
| Summary reporting | Generates Markdown summary listing extracted files, metadata updates, skips, and errors. |
| CLI-only | Primary interface via `imageworks-zip` Typer CLI; GUI leverages results indirectly via Mono Checker folder picker. |

---
## 2. CLI (`uv run imageworks-zip run ...`)

### Options
- `--zip-dir`: directory containing ZIPs (default from config).
- `--zip-file`: path to single ZIP (overrides `--zip-dir`).
- `--extract-root`: destination directory (created if missing).
- `--output-file`: summary Markdown path (`zip_extract_summary.md` default).
- `--include-xmp`: also extract XMP sidecars.
- `--metadata` (`-m`): force metadata update for existing files.

### Behaviour
1. Determines extract subdirectory from text in parentheses of ZIP filename (fallback to stem).
2. Extracts images (and optional XMP) while skipping existing files unless metadata flag set.
3. Ensures each extracted image contains keyword `ccc`; updates metadata with title/author when XMP available.
4. Records extracted files, skipped files, metadata updates, and errors per ZIP.
5. Writes Markdown summary enumerating actions for audit.

---
## 3. Configuration

`[tool.imageworks.zip-extract]` (pyproject):
- `default_zip_dir`, `default_extract_root`, `default_summary_output`
- Metadata flags: `backup_original_files`, `overwrite_existing_metadata`
- Colour thresholds (used by downstream analytics)
- Prompt templates for integration with color narrator validation

Defaults ensure WSL paths map to Windows downloads and photo storage directories.

---
## 4. Integration Points

- **Mono Checker GUI**: “Select from extracted folders” drop-down enumerates directories under `extract_root`.
- **Color Narrator**: uses same defaults for overlays and JSONL paths; ensures metadata prepared before narration.
- **Personal Tagger**: benefits from keyword baseline (`ccc`) to flag competition entries.

---
## 5. Troubleshooting

| Symptom | Cause | Remedy |
|---------|-------|--------|
| “ZIP file does not exist” | Path typo or missing `.zip` extension. | Verify path or use `--zip-dir` scan. |
| Metadata update errors | ExifTool not installed or missing permissions. | Install ExifTool and ensure output directory writable. |
| Summary empty | No ZIP files found matching glob. | Confirm `--zip-dir` and file extensions. |
| Keywords not added | Extraction skipped because file existed. | Re-run with `-m/--metadata` to force update. |

---
## 6. Best Practices

1. Run ZIP extractor immediately after receiving competition submissions.
2. Keep extracted directories organised by submission batch (naming derived from ZIP filenames).
3. Review Markdown summary to confirm metadata applied; attach to intake ticket.
4. Use version control or backups for metadata scripts if customizing ExifTool behaviour.
5. After extraction, trigger mono checker on the new folder to continue pipeline.

