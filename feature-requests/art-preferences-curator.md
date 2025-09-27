# Feature Request: Art Preferences Curator & Inspiration Assistant

## 0) TL;DR (1–2 sentences)
Build a module that consolidates my scattered art references (photos, notes, documents) into a searchable taste profile, highlights galleries/artists aligned with those preferences, and surfaces creative prompts for my own photography.

---

## 1) Why / Outcome
- **Why now:** I need a reliable way to remember artists I like, discover similar purchasable art, and generate inspiration for my own work without manually trawling years of rough notes and photos.
- **Definition of success:** A curated knowledge base describing my stylistic preferences, practical lists of galleries/artists to explore (with price/context hints), and a set of prompts/ideas I can query before visiting fairs or creating art myself.

---

## 2) Scope
- **In scope:**
  - Aggregating personal references (Google Photos, documents, snapped labels) into structured metadata.
  - Identifying recurring themes/styles/artist clusters and generating textual summaries.
  - Producing actionable gallery/artist lists (especially contemporary/available work) with basic research notes.
  - Providing inspirational prompts/technique reminders tailored to my tastes.
  - Optional Lightroom catalogue export for browsing by style/artist.
- **Out of scope:**
  - Full commercial art valuation, live price feeds, automated buying tools.
  - Museum-collection scraping beyond personal references.
  - Polished GUI (initially CLI/reports).

---

## 3) Requirements
### 3.1 Functional
- [ ] F1 — Import references from local Google Photos export, documents, and note dumps; capture images + associated text (OCR where available).
- [ ] F2 — Cluster artworks/styles into labelled groups describing my preferences, referencing both historical inspiration and contemporary parallels.
- [ ] F3 — Generate a report listing recommended galleries, artists, and key works to explore/buy (with provenance and affordability cues where possible).
- [ ] F4 — Create inspirational prompts/technique suggestions for producing similar styles in photography (multiple exposure, ICM, abstract experiments, etc.).
- [ ] F5 — Produce optional Lightroom-compatible catalogue/metadata (keywords, genres, periods) for quick search.
- [ ] F6 — Output consolidated Markdown/JSONL reports with summary, clusters, gallery list, and prompt library.

### 3.2 Non-functional (brief)
- **Performance:** Batch processing acceptable (one-off + quarterly updates); able to run on existing ImageWorks environment (WSL, RTX 4080/6000 Pro, CUDA ≥ 12.8).
- **Reliability / failure handling:** Tolerate messy inputs (blurry photos, partial labels) and flag items needing manual review.
- **Compatibility:** Leverage existing AI/VLM infrastructure where helpful; allow use of third-party/open-source tools (OCR, CLIP, curated data).
- **Persistence / metadata:** Store structured knowledge base locally (e.g., JSON, SQLite) with clear provenance and ability to append new finds.

**Acceptance Criteria (testable)**
- [ ] AC1 — Running the module on a sample subset produces a Markdown/JSON summary with at least three preference clusters and associated artists.
- [ ] AC2 — Recommended gallery list includes source references (image/label) and context (style, price bracket if known).
- [ ] AC3 — Lightroom export (or metadata package) allows browsing by style tags compiled from the knowledge base.
- [ ] AC4 — Inspirational prompts reference specific techniques/styles tied to identified clusters.
- [ ] AC5 — Reports clearly list items needing manual review (missing OCR, ambiguous matches).

---

## 4) Effort & Priority (quick gut check)
- **Effort guess:** Extra Large (research-heavy, iterative ingestion + curation pipeline).
- **Priority / sequencing notes:** Long-term groundwork for personal art curation; initial MVP can focus on ingestion + basic clustering.

---

## 5) Open Questions / Notes
- What taxonomy best expresses style clusters (historical movements, visual motifs, colour palettes)?
- How to balance automated clustering vs. manual curation for messy inputs?
- Preferred storage format for the knowledge base (Lightroom catalogue vs. JSON/SQLite)?
- How deep should price research go (manual integration, API lookups, references to partner galleries)?

---

## 6) Agent Helper Prompt
"Read this feature request. Ask me concise questions about data onboarding (formats, volumes), desired clustering granularity, and reporting cadence before proposing designs. Then highlight missing requirements or risks."

---

## 7) Links (optional)
- **References:** Personal Google Photos export, gallery label snapshots, art notes/docs (paths TBD)
- **Related tools:** Open-source OCR (Tesseract), CLIP/embedding search, art market research APIs (if available)
