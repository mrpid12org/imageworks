# Judge Vision Runbook

## Overview

The Judge Vision workflow extends the personal tagger pipeline with UK camera-club
judging capabilities. The pipeline now performs four coordinated stages:

1. **Stage 0 – Compliance Checks**
   * Validates image dimensions and policy hints (borders, manipulation, watermarks).
   * Competition-specific rules are loaded from a TOML registry file.

2. **Stage 1 – Technical Signals**
   * Lightweight heuristics compute mean luma, contrast, edge density, and saturation.
   * Summaries surface as prompt context and appear in the GUI for human review.

3. **Stage 2 – VLM Critique**
   * The `club_judge_json` prompt profile now returns 80–120 word critiques plus rubric
     subscores, 20-point totals, award suggestions, and optional compliance flags.

4. **Stage 3 – Pairwise Tournament**
   * Optional Swiss-style rounds derive rankings and expose final placements and
     stability metrics. Award bands are mapped automatically from competition configs.

## Quick Start

1. Create a competition registry (see example below) and point the CLI at it:

   ```bash
   imageworks-personal-tagger run \
     --input-dir ~/photos/competition \
     --competition-config configs/competitions.toml \
     --competition club_open_2025 \
     --pairwise-rounds 3
   ```

2. Review results in Streamlit: subscores, totals, compliance findings, and technical
   signals are editable in-line. Pairwise tournament summaries appear in the generated
   Markdown report and JSONL audit log.

3. Exported JSON records now include:
   * `critique_subscores` (impact, composition, technical, category_fit)
   * `critique_total`, `critique_award`, and `critique_compliance_flag`
   * `technical_signals.metrics` and compliance `issues`/`warnings`
   * Optional `pairwise` rounds/final rankings/stability metrics

## Competition Registry Example

```toml
[competition.club_open_2025]
categories = ["Open", "Nature"]
rules = { max_width = 1920, max_height = 1200, borders = "disallowed", watermark_allowed = false }
awards = ["Gold", "Silver", "Bronze", "HC", "C"]
score_bands = { Gold = [19, 20], Silver = [18], Bronze = [17], HC = [16], C = [15] }
pairwise_rounds = 4
```

## GUI Enhancements

* Subscores (0–5) and total scores (0–20) are editable via number inputs.
* Award suggestions and compliance flags can be edited or cleared.
* Compliance summaries and technical priors render as captions for quick scanning.
* Compact summaries include total score, award, and compliance status per image.

## Outputs

* **JSONL** – Each record follows schema version `1.1` and embeds compliance,
  technical, and pairwise reports for downstream analytics.
* **Summary Markdown** – Adds a “Pairwise Tournament” section with match logs,
  final rankings, and stability metadata.

## Next Steps

* Integrate multi-judge ensembles and anchor analysis for bias control.
* Extend analytics dashboards to visualise rankings and drift over time.
* Build the human judge toolkit (timers, crib sheets, JSON export) atop the new schema.
