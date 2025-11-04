# Image Similarity Checker Reference

Detect duplicate and near-duplicate competition images by combining perceptual
hashing, embedding comparisons, and optional VLM explanations.

## Engine architecture
- `SimilarityEngine` orchestrates discovery, strategy priming, candidate
  evaluation, and verdict classification. Dry-run mode generates placeholder
  passes for smoke tests.【F:src/imageworks/apps/image_similarity_checker/core/engine.py†L20-L113】
- Discovery pulls candidates and historical library images with caching and
  manifest TTLs to avoid repeated directory scans.【F:src/imageworks/apps/image_similarity_checker/core/engine.py†L28-L68】【F:src/imageworks/apps/image_similarity_checker/core/discovery.py†L17-L140】
- Strategies implement `prime()` and `match()`; defaults include perceptual hash
  and CLIP embeddings. `build_strategies` assembles them based on CLI flags and
  config.【F:src/imageworks/apps/image_similarity_checker/core/strategies.py†L19-L230】
- Matches are aggregated into `CandidateSimilarity` objects with score-ordered
  lists and stored thresholds for audit trails.【F:src/imageworks/apps/image_similarity_checker/core/engine.py†L114-L190】
- Optional `SimilarityExplainer` uses the configured OpenAI-compatible backend to
  generate textual justifications when matches trigger query/fail verdicts.【F:src/imageworks/apps/image_similarity_checker/core/engine.py†L191-L265】【F:src/imageworks/apps/image_similarity_checker/core/explainer.py†L18-L188】

## Configuration
- `[tool.imageworks.image_similarity]` (pyproject) sets defaults for image
  extensions, thresholds, library roots, and enabled strategies.【F:pyproject.toml†L258-L325】
- `load_config` merges pyproject defaults, environment overrides, and CLI values
  using the `IMAGEWORKS_IMAGE_SIMILARITY__` prefix.【F:src/imageworks/apps/image_similarity_checker/core/config.py†L1-L152】
- Loader integration resolves explainer models via `select_model`, ensuring only
  backends with required capabilities are used.【F:src/imageworks/apps/image_similarity_checker/core/engine.py†L30-L44】

## CLI (`imageworks-image-similarity`)
- `check` accepts candidate paths, resolves configuration, runs the engine, and
  writes JSONL plus Markdown summaries when requested.【F:src/imageworks/apps/image_similarity_checker/cli/main.py†L13-L180】
- Default strategy stack performs perceptual hash (fast reject) followed by
  embedding similarity. Additional strategies can be enabled with repeated
  `--strategy` flags.【F:src/imageworks/apps/image_similarity_checker/core/strategies.py†L19-L230】
- Metadata export writes IPTC keywords and JSON sidecars through
  `SimilarityMetadataWriter` when `--write-metadata` is enabled.【F:src/imageworks/apps/image_similarity_checker/core/metadata.py†L19-L178】

## Outputs
- JSONL: list of candidate results with match arrays, verdicts, and thresholds.
- Markdown summary: per-candidate top matches with thumbnails and explanation
  text (when generated).
- Optional metadata keywords tagging duplicates for Lightroom review.

## Integration
- Personal tagger & color narrator share backend configuration patterns and can
  reuse the same registry roles for VLM explanations.
- Chat proxy/model loader supply OpenAI-compatible endpoints for explainer
  prompts, keeping inference orchestration centralised.【F:src/imageworks/apps/image_similarity_checker/cli/main.py†L83-L180】
