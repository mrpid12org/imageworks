# Monochrome Checker Decision Tree (Technical Reference, v2)

*For a user-friendly summary, see: [Monochrome Checker Logic](MONOCHROME_CHECKER_LOGIC.md)*

---

## Purpose and Audience

This document is a detailed, technical mapping of the monochrome checker’s logic, written in plain English but closely following the code. It is intended for developers, advanced users, and anyone debugging or extending the checker. Variable names and thresholds are included to help link the explanation to the Python codebase.

---

## Table of Key Variables and Thresholds

| Variable                        | Default Value | Description                                                                 |
|----------------------------------|--------------|-----------------------------------------------------------------------------|
| LAB_TONED_PASS_DEFAULT           | 10.0         | Max hue std-dev (deg) for clear toned pass                                  |
| LAB_TONED_QUERY_DEFAULT          | 14.0         | Max hue std-dev (deg) for toned query                                       |
| LAB_STRONG_TONE_HUE_STD          | 14.0         | Max hue std-dev for strong tone override                                    |
| LAB_STRONG_TONE_CONCENTRATION    | 0.85         | Min hue concentration for strong tone override                              |
| LAB_STRONG_TONE_PRIMARY_SHARE    | 0.97         | Min primary hue share for strong tone override                              |
| LAB_HARD_FAIL_C4_RATIO_DEFAULT   | 0.10         | Min strong color fraction for hard fail                                     |
| LAB_HARD_FAIL_C4_CLUSTER_DEFAULT | 0.08         | Min single cluster fraction for hard fail                                   |
| LAB_SHADOW_QUERY_SHARE           | 0.55         | Min neutral shadow share for stage-lit override                             |
| LAB_SHADOW_QUERY_HUE_STD         | 24.0         | Max hue std-dev for stage-lit override                                      |
| LAB_SHADOW_QUERY_PRIMARY_SHARE   | 0.95         | Min primary hue share for stage-lit override                                |

---

## Decision Flow (with Variable Names)

(See the mermaid diagram in v1 for a visual overview.)

1. **Neutral Check**
   - `chroma_p99 <= neutral_chroma` (default 2.0)
   - Outcome: PASS (Neutral)
2. **Split-Tone Collapse**
   - `fail_two_peak` and `delta_h_highs_shadows_deg < 45.0`
   - If `hue_std > LAB_TONED_PASS_DEFAULT`: PASS WITH QUERY (Toned)
   - Else: PASS (Toned)
3. **Stage-Lit Override**
   - `force_fail` and `single_hue_stage_lit`
   - Outcome: PASS WITH QUERY (Toned)
4. **Uniform Strong Tone Override**
   - `uniform_strong_tone` and `hue_std > LAB_TONED_PASS_DEFAULT`
   - Outcome: PASS (Toned)
5. **Refined Pass**
   - `not force_fail` and `hue_std <= LAB_TONED_PASS_DEFAULT` and `merge_ok`
   - Outcome: PASS (Toned)
6. **Refined Query**
   - `not force_fail` and (`hue_std <= LAB_TONED_QUERY_DEFAULT` or (peak_delta_deg in 12-18 and second_mass < 0.15))
   - Outcome: PASS WITH QUERY (Toned)
7. **Default Fail Conditions**
   - If `fail_two_peak` or `hilo_split` or (`R < 0.4 and R2 > 0.6`): split_toning_suspected
   - If `cf >= 25.0` or `chroma_p95 > neutral_chroma + 8.0`: multi_color
   - If `chroma_med < neutral_chroma * 0.75 and hue_std < 30.0`: near_neutral_color_cast
   - Else: color_present
8. **Degrade to Pass/Query**
   - If `not force_fail` and (`small_footprint` or (`soft_large_footprint` and `chroma_ratio4 < 0.12`)) and (`large_drift` or `hue_std < 45.0`): PASS (Toned)
   - If `not force_fail` and (`moderate_footprint` or `subtle_cast` or `soft_large_footprint` or (`large_drift` and `chroma_ratio4 < 0.05`)): PASS WITH QUERY (Toned)
   - Else: FAIL (Not Monochrome)

---

## Mapping to Code

- Main logic: `check_monochrome` and `_check_monochrome_lab` in `src/imageworks/libs/vision/mono.py`
- CLI and config: `src/imageworks/apps/competition_checker/cli/mono.py`
- Constants: Defined at the top of `mono.py`
- Output structure: `MonoResult` dataclass

---

## Example Output Mapping

A typical result dictionary (see `_result_to_json` in `cli/mono.py`):

```json
{
  "verdict": "pass_with_query",
  "mode": "toned",
  "hue_std_deg": 13.2,
  "dominant_color": "sepia",
  "failure_reason": "split_toning_suspected",
  ...
}
```
- `verdict` and `mode` map to the main decision outcome.
- `hue_std_deg`, `dominant_color`, and `failure_reason` are used in the logic steps above.

---

## Change Log

- **v2 (2025-09-23):** Added explicit variable mapping, code references, and a table of thresholds. Improved clarity for debugging and code review.
- **v1:** Initial version.

---

*For a user-friendly summary, see: [Monochrome Checker Logic](MONOCHROME_CHECKER_LOGIC.md)*
