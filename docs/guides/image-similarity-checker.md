# Image Similarity Checker

The image similarity checker flags competition entries that match or closely resemble
prior submissions. It combines deterministic perceptual hashes with configurable
embedding models, produces JSONL/Markdown reports, and can annotate image metadata
for downstream workflows.

## Key Concepts

- **Strategies** – Pluggable similarity algorithms. The default stack combines a
  simple embedding descriptor with a perceptual hash for fast duplicate detection.
  Strategies are configured via the `default_strategies` setting.
- **Thresholds** – Scores above `fail_threshold` trigger a fail verdict, while scores
  between `query_threshold` and `fail_threshold` yield a query for manual review.
- **Prompt Profiles** – When explanations are enabled, the checker uses templated
  prompts (stored in code) to request rationales from an OpenAI-compatible backend.
- **Outputs** – Every run can emit JSONL (`outputs/results/similarity_results.jsonl`) and
  Markdown (`outputs/summaries/similarity_summary.md`) summaries. Metadata updates are
  optional and require ExifTool.

## CLI Usage

```bash
uv run imageworks-image-similarity check \
  /path/to/candidate/folder \
  --library-root "/mnt/d/Proper Photos/photos/ccc competition images" \
  --strategy perceptual_hash --strategy embedding \
  --fail-threshold 0.92 --query-threshold 0.82 \
  --output-jsonl outputs/results/similarity.jsonl \
  --summary outputs/summaries/similarity_summary.md
```

### Common Flags

| Option | Purpose |
| --- | --- |
| `--strategy` | Enable specific similarity strategies (`embedding`, `perceptual_hash`). |
| `--embedding-backend` | Choose the embedding implementation (`simple`, `open_clip`, `remote`). |
| `--library-root` | Override the default archive location. |
| `--fail-threshold`, `--query-threshold` | Adjust sensitivity bands for fail/query verdicts. |
| `--explain/--no-explain` | Toggle natural-language rationales using prompt profiles. |
| `--write-metadata` | Append keywords with similarity verdicts to image files via ExifTool. |
| `--use-loader/--no-use-loader` | Resolve base URLs/models through the deterministic model loader. |
| `--registry-model` | Provide the logical model name from `configs/model_registry.json`. |
| `--registry-capability` | Require additional registry capabilities (defaults to `vision`). |

## Reports

The JSONL file records one object per candidate image with raw scores and metadata. The
Markdown report summarises verdicts in a table, then expands each candidate with notes,
score breakdowns, and optional explanations. These artefacts align with the existing
ImageWorks mono checker and personal tagger outputs.

## Integration Tips

1. **Pipeline Ordering** – Run the similarity checker immediately after ingesting entries.
   Its JSONL output includes a `verdict` field that upstream automation can inspect before
   running heavier modules (mono checker, colour narrator, personal tagger).
2. **Caching** – Embedding strategies cache library vectors in `outputs/cache/similarity`.
   Delete this cache to force regeneration after large library updates.
3. **Metadata Writing** – Ensure ExifTool is installed when enabling metadata annotations.
   Keywords follow the `similarity:*` namespace for easy filtering in Lightroom.
4. **Model Loader Integration** – Enable `--use-loader` to reuse the deterministic model
   registry. Combine with `--registry-capability` (repeatable) to ensure resolved models
   advertise capabilities such as `vision` or future `embedding` support.
5. **Explanations** – Configure an OpenAI-compatible endpoint in `pyproject.toml` when
   enabling `--explain`. Prompt profiles live in code and can be versioned to tune tone.

## Adjusting Thresholds

The default thresholds (`0.92` fail, `0.82` query) were chosen to bias toward catching
obvious duplicates. After reviewing real-world runs, update the `default_fail_threshold`
and `default_query_threshold` values in `pyproject.toml`. Lower thresholds increase
sensitivity (more queries/fails), while higher values reduce false positives.

## Troubleshooting

- **Missing embeddings** – Use `--strategy perceptual_hash` while debugging embedding
  backends. The checker logs cache builds and errors per strategy.
- **Long runtimes** – Build an embedding cache once, then keep `outputs/cache/similarity`
  alongside the library. For large archives consider distributing strategy execution by
  splitting the candidate list.
- **Explanation failures** – Errors from the VLM backend are logged; the run still
  completes without explanations.
