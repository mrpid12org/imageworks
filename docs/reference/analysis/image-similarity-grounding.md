# Grounding Duplicate Detection Decisions

The similarity checker is deliberately hybrid: deterministic perceptual hashes provide
fast, reproducible flags for obvious duplicates, while learned embeddings capture tonal,
crop, and minor edit variants. This combination keeps the system explainable and tunable.

## Why Perceptual Hashes?

Perceptual hashes (pHash/dHash) reduce an image to a binary signature resilient to minor
changes such as scale, rotation, and exposure tweaks. They are:

- **Fast** – simple downsampling and comparisons require no GPU.
- **Deterministic** – identical inputs yield identical hashes, which helps during appeals.
- **Grounded** – high similarity scores map directly to Hamming distance, enabling rule-based
  explanations without AI involvement.

However, hashes struggle with more aggressive edits (heavy colour grading, textures) or
recomposed images. They form the "hard evidence" baseline and guard against drift.

## Why Embeddings?

Embedding models capture semantic and stylistic information beyond raw pixels. Using a
configurable backend (`simple`, `open_clip`, `remote`) lets teams trade accuracy for
performance or integrate with a model registry. Cosine similarity across normalised vectors
works well for near-duplicate frames, alternate crops, and B&W conversions.

Embedding outputs are averaged with deterministic evidence by picking the maximum score per
candidate reference. This avoids double-counting while still surfacing the most convincing
match across strategies.

## Adjustable Verdict Bands

Humans ultimately decide the boundary between "too similar" and "acceptably different". The
checker exposes two thresholds:

- **Fail threshold** – values above this are treated as automatic fails. Start high (0.9+) to
  catch exact or near-exact duplicates and lower gradually if missed duplicates appear.
- **Query threshold** – the caution band. Scores between `query_threshold` and `fail_threshold`
  are flagged for manual review, keeping false positives manageable.

During rollout, collect a set of known duplicates and borderline cases. Plot the scores and
adjust thresholds until the pass/fail split matches curator expectations. Because strategies
return normalised scores in `[0, 1]`, thresholds remain intuitive.

## Alternative Signals Considered

- **Classical feature matching (SIFT/SURF)** – accurate but slower at scale and patent-laden.
- **Learned perceptual image patch similarity (LPIPS)** – powerful but requires GPU inference
  and is less interpretable.
- **FAISS vector databases** – ideal for very large archives, but adds operational overhead.
  The current cache-based approach keeps complexity low. FAISS integration can be layered on
  if latency becomes an issue.

The chosen design prioritises reliability and incremental deployment. Strategies are modular:
add a new implementation (e.g., SigLIP embeddings, LPIPS) and list it in `default_strategies`
without rewriting orchestration code.

## Tuning Guidance

1. **Start with deterministic-only runs** (`--strategy perceptual_hash`) to establish baseline
   behaviour and confirm metadata/report pipelines.
2. **Introduce embeddings** with conservative fail thresholds and review cases above 0.85.
3. **Log borderline scores** and gather curator feedback. Adjust thresholds accordingly.
4. **Enable explanations** once thresholds stabilise. Prompt profiles reside in
   `core/prompts.py` and can be versioned when the communication style evolves.

Document agreed thresholds in the repository so future competitions understand the rationale.
