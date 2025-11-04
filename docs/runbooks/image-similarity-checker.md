# Image Similarity Checker Runbook

Step-by-step procedures for detecting duplicates between new entries and the competition library.

---
## 1. Before You Begin

1. Confirm historical library is mounted and accessible.
2. Verify embedding models are downloaded (OpenCLIP weights) or remote endpoints reachable.
3. Ensure `[tool.imageworks.image_similarity_checker]` defaults (thresholds, library root) are correct.
4. Decide whether explanations and metadata writes are required for this run.

---
## 2. CLI Workflow

1. Build command:
   ```bash
   uv run imageworks-image-similarity check \
     submissions/2024-week12 \
     --library-root /mnt/archive/library \
     --output-jsonl outputs/results/similarity_week12.jsonl \
     --summary outputs/summaries/similarity_week12.md \
     --strategy embedding --strategy perceptual_hash \
     --embedding-backend open_clip --embedding-model ViT-B-32::laion2b_s34b_b79k \
     --fail-threshold 0.92 --query-threshold 0.85 \
     --explain --backend vllm --model qwen2-7b-instruct \
     --write-metadata
   ```
2. Monitor CLI output; each candidate prints verdict, score, and top matches.
3. Review Markdown summary for FAIL/QUERY sections and share with judging team.
4. Store JSONL output for audit and replays.

---
## 3. GUI Workflow

1. Open Streamlit → “Image Similarity” page.
2. Drop candidate files or select directories.
3. Confirm library root and optional cache refresh.
4. Toggle strategies, augmentations, explanations, and metadata to match run plan.
5. Click **Run Check**; track progress and view per-image cards.
6. Use filters to review FAILs first, then QUERIES; open side-by-side viewer for inspection.
7. Download JSONL/Markdown via provided buttons.

---
## 4. Handling Explanations

- Enable `--explain` (CLI) or GUI toggle.
- Provide backend/model overrides or rely on defaults.
- If using model loader: add `--use-loader --registry-model similarity/explainer`.
- Watch for API rate limits; reduce `--top-matches` if necessary.

---
## 5. Metadata Workflow

1. Set `--write-metadata` and optionally `--backup-originals`.
2. Metadata writer applies similarity keywords (`similarity:fail`, etc.).
3. For dry runs, use `--dry-run --summary <path>` to review without writing.
4. GUI exposes metadata toggle and warns before overwriting existing keywords.

---
## 6. Troubleshooting

| Issue | Resolution |
|-------|------------|
| CLI crash referencing cache | Delete `.cache/imageworks/similarity_manifest.json` and rerun with `--refresh-library-cache`. |
| GPU OOM during embedding | Switch to CPU-friendly backend or reduce batch size via config; disable augmentation pooling. |
| Explanations time out | Increase `--timeout`, ensure backend healthy, or disable explanations temporarily. |
| Metadata conflicts | Use `--no-overwrite-metadata` to append only when empty, or disable metadata and handle manually. |

---
## 7. Post-Run Actions

1. Share Markdown summary with review panel.
2. Update competition tracker with flagged duplicates.
3. Archive JSONL and explanation text in evidence folder.
4. Reset GUI presets if thresholds changed (Settings → Save preset).
5. Plan next manifest refresh (set reminder before TTL expiry).

