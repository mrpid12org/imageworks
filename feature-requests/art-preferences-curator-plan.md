# Implementation Plan: Art Preferences Curator & Inspiration Assistant

## Goal
Aggregate my personal art references into a structured knowledge base that captures stylistic preferences, recommends galleries/artists to explore, and provides creative prompts for producing similar work.

## Why it matters
- Removes the guesswork when buying art: identifies galleries/ artists aligned with my taste and gives context on price/availability.
- Distils my scattered notes/photos into clear stylistic clusters and narratives.
- Serves as an inspiration engine for my own photography projects.

## Scope
- Data ingestion (images, label photos, documents, notes).
- OCR/text extraction and metadata enrichment.
- Clustering and preference profiling (historical + contemporary parallels).
- Gallery/artist recommendation report with source references.
- Inspiration prompts and technique suggestions tied to clusters.
- Optional Lightroom metadata export and JSON knowledge base.

## Requirements & Acceptance
- **R1:** Ingest and normalise references from Google Photos export, documents, label snapshots.
  **AC1:** Sample ingestion run produces a structured dataset with image references, extracted text, and provenance pointers.
- **R2:** Cluster artworks into stylistic families with descriptive summaries.
  **AC2:** Report lists ≥3 clusters with example images/artists and historical contemporary pairings.
- **R3:** Identify galleries/dealers aligned with clusters, including availability or price notes.
  **AC3:** Gallery report includes actionable info (location/contact/link) for top recommendations.
- **R4:** Generate creative prompts/technique suggestions for personal photography tied to each cluster.
  **AC4:** Prompt library references specific techniques (e.g., ICM, multiple exposure) and artists.
- **R5:** Export data for Lightroom (keywords, styles) and JSON knowledge base.
  **AC5:** Lightroom sample catalogue or metadata file loads with clustered tags; JSON lists galleries/artists/prompts.

## Inputs & Outputs
- **Inputs:** Local folders (Google Photos download, documents), path to label snapshots, optional manual annotations, existing Lightroom catalogue (if available).
- **Outputs:**
  - Markdown + JSON reports summarising clusters, galleries, prompts.
  - Optional Lightroom metadata (keywords, collections) via sidecar or plugin script.
  - Local knowledge base (JSON/SQLite) storing structured entries with provenance.

## Design Sketch
1. **Ingestion layer:** Batch load images/docs; apply OCR (Tesseract/vision API) and metadata extraction; store references with provenance.
2. **Embedding & clustering:** Use image/text embeddings (e.g., CLIP) to group works; overlay manual tags/historical references.
3. **Knowledge synthesis:** For each cluster, summarise style descriptors, link to historical inspirations, map to contemporary artists/galleries (via web lookup/manual sources).
4. **Recommendation engine:** Generate gallery list with context (style, price tier, availability) + maintain update cadence (quarterly).
5. **Inspiration prompts:** For each cluster, craft technique prompts referencing known artists and photographic approaches.
6. **Reporting/export:** Output Markdown/JSON, optionally write Lightroom metadata (keywords/lists).

## Tasks & Sizes
1. **T1 (M):** Define data schema & storage (JSON/SQLite), build ingestion pipeline for photos/docs with OCR.
2. **T2 (M):** Generate embeddings and initial clustering (experiment with CLIP/text clustering); manual review tools for messy data.
3. **T3 (M):** Implement cluster summarisation (historical + contemporary mapping); integrate manual curation options.
4. **T4 (M):** Build gallery recommendation module (lookup, structured output, price/context notes).
5. **T5 (S):** Craft inspiration prompt generator tied to clusters (Leverage VLM/LLM for prompt synthesis).
6. **T6 (S):** Produce Markdown + JSON reports; optional Lightroom metadata export script.
7. **T7 (M):** Testing on pilot dataset; refine thresholds; document workflow & quarterly update process.
8. **T8 (S):** Prepare onboarding docs: how to add new references, review clusters, update gallery list.

## Risks & Mitigations
- **Messy inputs/poor OCR:** Provide manual annotation hook and track items needing review; allow user edits in knowledge base.
- **Clustering accuracy:** Combine automated embeddings with manual tagging; allow reclassification/override.
- **Gallery info freshness:** Include data source timestamps; plan quarterly refresh workflow (possibly assisted by search prompts).

## Savepoint
- **Current status:** Requirements & plan drafted; no implementation yet.
- **Next action:** Create branch `feature/art-preferences-curator`; prototype ingestion/OCR pipeline (T1).
- **Branch:** _feature/art-preferences-curator_ (to be created).
- **Files in flight:** None yet.
- **Open questions:** Decide on storage format (SQLite vs. JSON); determine depth of price research; identify initial dataset paths.
