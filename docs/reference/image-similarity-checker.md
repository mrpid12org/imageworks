# Image Similarity Checker Operations Guide

The image similarity checker finds duplicates and near-duplicates between new competition entries and the historical library. It uses configurable embedding strategies, augmentation pools, and explanation generation, surfacing results via CLI, API (planned), and GUI.

---
## 1. Capabilities Overview

| Capability | Details |
|------------|---------|
| Multi-strategy matching | Combines embedding search (OpenCLIP, remote OpenAI-compatible), perceptual hashing, and structural similarity. |
| Threshold-based verdicts | Configurable `fail_threshold` and `query_threshold` produce FAIL/QUERY/PASS verdicts with rationale. |
| Augmentation pooling | Optional grayscale conversion and five-crop averaging increase robustness to slight edits. |
| Explanation generation | Integrates with LLM/VLM backends to describe why images are similar (`--explain`). |
| Metadata tagging | Writes IPTC/XMP metadata to candidates (keywords, descriptions) when enabled. |
| JSONL/Markdown output | Machine-readable logs plus human-friendly summary grouped by verdict. |
| GUI integration | Streamlit page for uploading candidates, running checks, and browsing results with filters and galleries. |

---
## 2. Architecture

- `core.engine.SimilarityEngine`: orchestrates indexing library manifests, computing embeddings, scoring, and producing `SimilarityVerdict` objects.
- `core.config`: merges defaults from `[tool.imageworks.image_similarity_checker]` (if present) with CLI overrides.
- `core.reporting`: writes JSONL and Markdown outputs.
- `cli.main`: Typer CLI entrypoint (`check`).
- GUI page `4_🖼️_Image_Similarity.py`: wraps CLI command via process runner and interactive result viewer.

Data flow:
```
Candidate images → manifest builder → embedding/feature computation → score ranking
  → apply thresholds → optional explanation generation → metadata writer → outputs
```

---
## 3. CLI (`uv run imageworks-image-similarity check ...`)

### Required arguments
- `CANDIDATE...`: one or more files/directories.
- Optional `--library-root` to point at historical archive (defaults from config).

### Key options
- Thresholds: `--fail-threshold`, `--query-threshold`, `--top-matches`.
- Strategies: repeatable `--strategy embedding`, `--strategy perceptual_hash`, etc.
- Embeddings: `--embedding-backend`, `--embedding-model`, `--augment-pooling`, `--augment-grayscale`, `--augment-five-crop`, `--augment-five-crop-ratio`.
- Explanations: `--explain`, `--backend`, `--base-url`, `--model`, `--api-key`, `--prompt-profile`.
- Metadata: `--write-metadata`, `--backup-originals`, `--overwrite-metadata`, `--dry-run`.
- Registry integration: `--use-loader`, `--registry-model`, `--registry-capability`.
- Performance: `--perf-metrics`, `--refresh-library-cache`, `--manifest-ttl-seconds`.

### Outputs
- JSONL (`--output-jsonl`): contains candidate path, top matches with scores, verdict, explanation, metadata status.
- Markdown summary (`--summary`): grouped by FAIL/QUERY/PASS with bullet list of matches and reasons.
- Console output prints aggregate counts and table-style verdict lines.

---
## 4. GUI Features

- **Candidate selector**: drag-and-drop or directory selection.
- **Library configuration**: choose root, refresh manifest cache, inspect match counts.
- **Strategy toggles**: checkboxes for embedding/hash/structural analysis, augmentation options.
- **Explanation controls**: toggle explanation generation, select backend/model, view live prompt previews.
- **Result explorer**: thumbnail gallery sorted by score; clicking reveals side-by-side comparison, explanation text, metadata status, and direct open-in-viewer links.
- **Export actions**: buttons to download JSONL/Markdown artifacts and copy explanation text for judges.

---
## 5. Configuration Defaults

Configure `[tool.imageworks.image_similarity_checker]` (if absent, CLI uses hardcoded fallbacks):
- `library_root`
- `output_jsonl`, `summary_path`
- Default thresholds, strategies, embedding backend/model
- Metadata behaviour flags (write/overwrite/backups)
- Explanation backend defaults (base URL, model, prompt profile)

---
## 6. Troubleshooting

| Symptom | Diagnosis | Remedy |
|---------|-----------|--------|
| Library manifest stale | Manifest TTL expired; new library images not detected. | Run with `--refresh-library-cache` or delete cached manifest. |
| Embedding backend errors | Missing model weights or endpoint unreachable. | Verify `--embedding-model` path, install dependencies, or update base URL/API key. |
| No matches found | Thresholds too strict or strategies disabled. | Lower `--fail-threshold` / `--query-threshold`, enable additional strategies. |
| Metadata writes skipped | `--write-metadata` false or `dry-run` true. | Remove dry-run or enable metadata flag. |
| Explanation blank | Backend disabled or prompt profile missing. | Ensure `--explain` set and backend configured; check prompt templates. |

---
## 7. Best Practices

1. Refresh manifest weekly to capture new historical images.
2. Use augmentation pooling for competitions prone to cropping or grayscale tweaks.
3. Keep JSONL logs for auditing duplicate rejections.
4. Feed explanation summaries back into judging notes to justify decisions.
5. When using remote APIs, throttle requests via smaller `--top-matches` and `--batch-size` (config dependent).

