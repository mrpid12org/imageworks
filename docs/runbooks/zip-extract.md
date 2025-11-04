# ZIP Extractor Runbook

Operational steps for unpacking incoming competition submissions.

---
## 1. Intake Preparation

1. Move received ZIP files into the configured staging directory (see `[tool.imageworks.zip-extract].default_zip_dir`).
2. Ensure extract root has sufficient space and is included in backup plan.
3. Verify ExifTool availability (`exiftool -ver`).

---
## 2. Standard Extraction

1. Execute:
   ```bash
   uv run imageworks-zip run \
     --zip-dir /mnt/c/Users/<user>/Downloads/CCC \
     --extract-root /mnt/d/Proper\ Photos/photos/ccc\ competition\ images \
     --output-file outputs/logs/zip_extract_$(date +%Y%m%d).md \
     --include-xmp -m
   ```
2. Monitor console output for each ZIP (extracted counts, metadata updates, errors).
3. Review generated Markdown summary; confirm each ZIP lists extracted images and metadata updates.
4. File summary in intake ticket or shared folder.

---
## 3. Single ZIP Rerun

- Use `--zip-file path/to/file.zip` to reprocess a specific submission. Combine with `-m` to update metadata even if files exist.

---
## 4. Post-Extraction Tasks

1. Spot-check a few images in Lightroom to confirm `ccc` keyword present and metadata populated.
2. Notify downstream operators that folder is ready for mono checker.
3. If overlays or additional diagnostics required, trigger relevant pipelines.

---
## 5. Troubleshooting

| Issue | Action |
|-------|--------|
| Command exits with permission error | Ensure extract root writable; run as user with correct permissions. |
| ExifTool missing | Install via package manager or disable metadata updates (omit `-m`). |
| Files already exist, metadata skipped | Rerun with `-m` or delete stale files before extraction. |
| Summary missing entries | Confirm ZIP contained supported image extensions; check logs for errors. |

---
## 6. Maintenance

1. Periodically archive processed ZIPs to long-term storage to keep staging directory manageable.
2. Update configuration defaults if directory structure changes (e.g., new Windows username or drive letter).
3. Review Markdown summaries quarterly to refine process or identify recurring issues.

