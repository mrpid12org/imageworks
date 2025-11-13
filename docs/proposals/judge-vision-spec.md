# Judge Vision Specification

## 1. Purpose

Design and implement a “Judge Vision” capability that delivers consistent, transparent, UK camera-club style photography judging. The system must:

- Provide human-readable critiques aligned with regional guidelines (PAGB, RPS, PSA/FIAP).
- Emit structured JSON suitable for downstream automation (awards, leaderboards, metadata).
- Support bias-controlled ranking and award assignment.
- Offer tooling for both automated VLM judges and human judges (briefing aids, calibration).


## 2. Background & Practitioner Guidance

### 2.1 UK/PAGB Judging Context
- PAGB Awards for Photographic Merit use six adjudicators voting YES/NO with special “near miss (3)” and “good enough (5)” buttons. Totals are reported numerically but originate from binary votes. [1]
- Club competitions vary: some allocate 1st/2nd/3rd + HC/C, others mark on 14–20 or 5–9 scales. Understanding the local scheme is essential. [3]
- Category rules (Open, Nature, Creative, Themed) and manipulation allowances differ per event. [7][8]

### 2.2 Core Evaluation Dimensions
All guidance sources converge on three primary axes (impact, composition/design, technical quality) plus category compliance:
1. **Impact & Communication** – subject clarity, story, mood, originality. Often weighted highest (~50%). [4]
2. **Composition & Design** – light, focal balance, geometry, colour/mono intent, background control, timing. [5]
3. **Technical Quality & Presentation** – exposure/tonal range, sharpness where needed, local contrast, artefacts (halos, banding, CA), crop/borders, presentation for prints/PDIs. [4][6]
4. **Category Fit** (optional) – adherence to Nature/Creative/Themed briefs. [8]

### 2.3 Human Judging Flow (Club Night Playbook)
1. Collect brief, rules, file specs, category definitions in advance. [7][9]
2. Calibrate expectations for each category; identify time limits. [8]
3. On the night:
   - First pass to gauge range/outliers.
   - Establish anchor images to prevent drift.
   - Critique per image (~20–30 s):
     - Impact (intent/story).
     - Composition (strengths and distractions).
     - Technical observations.
     - Actionable improvement. [8]
4. Assign awards/marks per local scheme.

### 2.4 Common Pitfalls & Mitigations
- Over-focus on technical minutiae vs message (avoid). [4]
- Scale drift across evening—use anchors.
- Genre bias (portraits judged as nature, etc.)—apply correct rubric.
- Title/sequence bias—prefer anonymity and random order. [10][20]

### 2.5 Judge Development Resources
Federation judge lists/training (SCPF/SPA). [11]
RPS distinction criteria for benchmark panels. [12]
Essays (Photomills, David Travis) for judgement craft. [13]


## 3. System Overview

### 3.1 High-Level Architecture
```
Input Images ──► Stage0 Compliance Checks
                  │
                  ├──► Stage1 Technical Signals (MUSIQ, NIMA/LAION)
                  │
                  └──► Stage2 VLM Critique (rubric prompt) ──► JSON output
                                         │
                                         ├──► Stage3 Pairwise Tournament & Ranking
                                         │
                                         └──► Stage4 Bias Controls, Calibration & Analytics
```

### 3.2 Components
1. **Compliance Engine** (Stage 0)
   - Validate resolution, aspect ratio, borders, watermark presence, manipulation limits (where rules are explicit).
   - Configurable per competition via rules registry.

2. **Technical Signal Collectors** (Stage 1)
   - MUSIQ: detect exposure/contrast/compression issues. [14]
   - NIMA / LAION aesthetic predictor: supply weak aesthetic priors and outlier detection. [15][21]
   - Optional: CLIP-based detectors for content moderation.

3. **VLM Critique Engine** (Stage 2)
   - Vision-Language model (Qwen-VL or equivalent) prompted with fixed rubric and JSON schema:
     ```json
     {
       "title": "<string|null>",
       "style": "Open|Nature|Portrait|Creative|Documentary|Abstract|Street|Landscape|Other|null",
       "critique": "<80-120 words>",
       "total": 0-20,
       "award_suggestion": "Gold|Silver|Bronze|HC|C|None",
       "compliance_flag": "<optional string>"
     }
     ```
   - Critique must reference rubric dimensions, mention technical faults only when material, and include one actionable improvement. Category suitability is now described narratively instead of via a numeric “category fit” subscore.

4. **Pairwise Judging/Tournament Module** (Stage 3)
   - Pairwise comparisons using Swiss or Bradley–Terry/Thurstone aggregation.
   - Randomize order and left/right placement per comparison.
   - Support conversion to:
     - Club placements (1/2/3/HC/C or similar).
     - PAGB-like thresholds (≥15 = YES, ≥19 = “good enough”). [1]

5. **Bias Control & Analytics** (Stage 4)
   - Multi-judge ensemble (different seeds, prompt paraphrases).
   - Anchor image injections with expected score bands.
   - Record Kendall’s W / Spearman ρ against anchors or human panels.
   - Drift monitoring for technical signal priors.

6. **Human Judge Toolkit**
   - One-page crib with rubric summary and critique cadence.
   - Web UI timer and note capture referencing the same rubric.
   - Option to export human notes into the same JSON schema.


## 4. Detailed Prompting Strategy

### 4.1 System Prompt (VLM Critique)
```
You are an experienced UK camera-club competition judge.
Apply this rubric: Impact & Communication, Composition & Design, Technical Quality & Presentation, Category Fit.
Prioritise impact; note technical faults only if they materially affect the image.
Be constructive and specific; include one actionable improvement.
Respect the category rules. Avoid personal taste statements.
Return only the JSON schema supplied. Length target for critique: 80–120 words.
```

### 4.2 User Prompt Template
```
Judge the following entry using the rubric.
Title: {title}
Category: {category}
Competition notes: {notes}
Caption context: {caption}
Keyword highlights: {keyword_preview}
Compliance signals: {compliance_findings}
Technical priors: {technical_signals}

Return only the JSON object defined previously.
```

### 4.3 Pairwise Comparison Prompt
```
Two images (A left, B right) are presented from the same competition and category.
Select the stronger entry for the rubric and brief. Ignore any prior decisions.
Return JSON { "winner": "A|B", "reason": "<≤40 words referencing rubric>" }.
```

### 4.4 Prompt Variants
- Maintain paraphrased system prompts (3–5 variants) to reduce ensemble correlated errors.
- Temperature ≤0.3 for deterministic critique; pairwise comparisons may use 0.4–0.6 while still controlling randomness.


## 5. Data Models & Storage

### 5.1 Competition Config
```toml
[competition.my_club_open_2025]
categories = ["Open", "Nature"]
rules = { borders = "disallowed", manipulation = "allow minor cloning", max_width = 1920, max_height = 1200 }
awards = ["Gold", "Silver", "Bronze", "HC", "C"]
score_bands = { Gold = [19, 20], Silver = [18], Bronze = [17], HC = [16], C = [15] }
pairwise_rounds = 4
anchors = ["anchors/open_gold", "anchors/open_reject"]
```

### 5.2 Judging Output JSON
Extends VLM schema with metadata:
```json
{
  "image": "...",
  "profile": "club_judge_json",
  "judge_id": "vlm_qwen_fp8_seed42",
  "rubric_scores": {...},
  "total": 19,
  "award_suggestion": "Gold",
  "technical_signals": {...},
  "compliance": {...},
  "pairwise_stats": {...},
  "notes": "...",
  "timestamp": "..."
}
```

### 5.3 Tournament Output
```json
{
  "competition": "...",
  "rounds": [...],
  "final_rankings": [
    { "image": "...", "score": 19.2, "award": "Gold" },
    ...
  ],
  "anchors_report": {...},
  "stability_metrics": { "kendall_w": 0.78, "seed_variance": 0.12 }
}
```


## 6. Implementation Plan

### 6.1 Phase 1 – Prompt & Critique Foundation
1. Update personal tagger prompt system:
   - Add rubric subscores and award to schema.
   - Extend JSON parsing and validation (pydantic or dataclass schema).
   - Surface new fields in GUI (editor, summary, JSONL viewer).
2. Integrate technical signal collectors (MUSIQ, NIMA/LAION).
   - Batch inference wrappers with caching.
   - Map signals into textual hints for prompts and structured data.
3. Barrel tests:
   - Unit tests for JSON parsing, score ranges, error recovery.
   - Baseline manual review on sample set.

### 6.2 Phase 2 – Tournament & Bias Controls
1. Build pairwise comparison API wrapper.
2. Implement tournament orchestrator:
   - Swiss rounds with randomised sides.
   - Bradley–Terry or Thurstone estimation for final scores.
   - Mapping to award bands.
3. Add multi-judge ensemble logic:
   - Configurable seeds, prompt variants.
   - Aggregation by median/majority vote for awards.
4. Analytics dashboard:
   - Visualise rankings, comparison graphs, anchors, drift metrics.
5. Calibration anchors workflow:
   - Store labelled anchor images per category.
   - Enforce calibration bounds (e.g., anchor must appear top quartile).

### 6.3 Phase 3 – Human Judge Toolkit & QA
1. Human UI:
   - Timing/notes interface reflecting same rubric.
   - Export to JSON schema for comparability.
2. Evaluation harness:
   - Collect human judge outputs, compute Kendall’s W / Spearman ρ vs VLM.
   - Document rotation of anchors, stability under reorder/seed variation.
3. Compliance rule registry:
   - Structured config per club/competition.
   - CLI validation tool for entrants.

### 6.4 Phase 4 – Deployment & Ops
1. Package as CLI + Streamlit module.
2. Add scheduling/integration with club competitions.
3. Logging & monitoring: track prompt changes, model versions, performance metrics.
4. Ethics/bias documentation: note limitations, require organiser review for final awards.


## 7. Testing & Validation

### 7.1 Automated Tests
- JSON schema validation for critique outputs.
- Score clamping and subscores sanity checks.
- Tournament ranking invariants (no cycles, top-n award mapping).
- Bias control tests (random seed reproducibility).

### 7.2 Manual/Pilot Evaluations
- Run on archived competition sets with known placements.
- Compare to human rankings (Kendall’s W, Spearman ρ).
- Evaluate outliers: high technical quality but low impact, etc.
- Monitor compliance flag accuracy (detect rule breaches).


## 8. Risks & Mitigations

| Risk | Mitigation |
| --- | --- |
| VLM hallucinations / schema violations | Strict JSON parsing, retry logic, streaming validator. |
| Position bias in pairwise judging | Randomised order, multi-round pairings, multi-judge ensemble. [17][19] |
| Category misinterpretation | Explicit category instructions, compliance hints, anchor checks. |
| Drift over time (model updates) | Versioned prompts, anchor regression suite, logging. |
| Human pushback (lack of trust) | Transparent rubric scoring, actionable feedback, human override, audit log. |


## 9. Deliverables Checklist

- [x] Prompt library update with new rubric & schema.
- [x] Technical signal collectors (MUSIQ, NIMA/LAION) integration.
- [x] Personal tagger/GUI updates for subscores, awards, compliance flags.
- [x] Tournament module with Swiss/Bradley–Terry aggregation.
- [ ] Bias control features (randomisation, multi-judge ensemble, anchors).
- [ ] Analytics dashboard & reports (JSON + Markdown).
- [ ] Human judge toolkit (crib, timer UI, export).
- [ ] Evaluation report comparing VLM vs human panels.
- [x] Documentation (README, runbook, architecture diagram).

## 11. Implementation Notes (2024-)

- Added Stage 0 compliance checks and Stage 1 technical heuristics within the personal tagger pipeline, surfacing concise summaries to prompts and GUI reviewers.
- Expanded the Judge Vision prompt profile to emit rubric subscores, 20-point totals, award suggestions, and compliance flags in strict JSON.
- Captured the new schema via `PersonalTaggerRecord`, including technical signals, compliance reports, and optional pairwise tournament results.
- Introduced Swiss-style pairwise ranking utilities and automatic award band mapping driven by competition configuration files.
- Updated the Streamlit metadata editor to expose subscores, total scores, compliance findings, and technical priors for human calibration.


## 10. References
1. PAGB APM Leaflet – https://thepagb.org.uk/wp-content/uploads/apm_leaflet_1.pdf
2. PAGB APM Overview – https://thepagb.org.uk/awards/apmstill/
3. Holmfirth Camera Club Competition Rules – https://holmfirthcameraclub.co.uk/section141973.html
4. Club Judging Guidelines (N&EMPF) – https://www.nempf.org/uploads/3/7/5/2/37524881/club_judging_guidelines_v4_by_david_gibbins_1.pdf
5. RPS Licentiate Criteria – https://rps.org/qualifications/licentiate/licentiate-criteria/
6. RPS Checkpoints to Appraise Photographs – https://rps.org/news/groups/landscape/news/newsletter-articles/2024-articles/july-2024/a-series-of-checkpoints-to-appraise-photographs/
7. Example Competition Terms (Royal Society) – https://royalsociety.org/journals/publishing-activities/photo-competition/terms-conditions/
8. Guidelines for Judges – https://gatewaycameraclub.visualpursuits.com/Downloads/0eed478e-4daf-4ba6-9745-170f81dd7a03?o=y
9. WPF PAGB info – https://thewpf.co.uk/information/pagb-distinctions-2
10. Judging Bias in Photography Competitions – https://www.mattpaynephotography.com/gallery/judging-bias-photography-competitions/
11. SCPF Judge Pathway – https://southerncountiespf.org.uk/judging/judging-becoming-a-judge/
12. RPS Distinctions Guidance – https://rps.org/media/jlro4i1r/rps-distinctions-licentiate-criteria-guidance-rev6a-upload.pdf
13. Photomills “Judging Photography” – https://www.photomills.co.uk/uploads/images/Gallery/Judging%20Photography.pdf
14. MUSIQ Paper – https://openaccess.thecvf.com/content/ICCV2021/papers/Ke_MUSIQ_Multi-Scale_Image_Quality_Transformer_ICCV_2021_paper.pdf
15. NIMA: Neural Image Assessment – https://arxiv.org/pdf/1709.05424
16. Paired Comparison Methods (Bramley) – https://assets.publishing.service.gov.uk/media/5a80d75940f0b62305b8d734/2007-comparability-exam-standards-i-chapter7.pdf
17. Survey on LLM-as-a-Judge – https://arxiv.org/abs/2411.15594
18. WPF Judge Guidelines – https://mywpf.org/judge-guidelines
19. Position Bias in Pairwise Evaluation – https://arxiv.org/html/2406.07791v5
20. Nikonians: Competition Judges – https://www.nikonians.org/reviews/photography_competition_judges
21. LAION Aesthetic Predictor – https://github.com/LAION-AI/aesthetic-predictor
22. HPS-v2 Benchmark – https://arxiv.org/abs/2306.09341
23. PSA Club Judging Service – https://psaphotoworldwide.org/page/co-club-judging-service
