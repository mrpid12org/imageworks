# Monochrome Checker Decision Tree (Comprehensive Reference, v3)

*For a user-friendly summary, see: [Monochrome Checker Logic](MONOCHROME_CHECKER_LOGIC.md)*

---

## Purpose and Audience

This document is a comprehensive reference for the Imageworks Monochrome Checker. It combines a high-level summary, philosophy, and practical guidance for photographers and judges, with a detailed, code-mapped technical explanation for developers and advanced users. Variable names and thresholds are included to help link the explanation to the Python codebase.

---

## Quick Summary

The Monochrome Checker is designed to help photographers and judges determine if an image meets monochrome competition rules. It analyzes the final rendered pixels, not the editing process, and is consistent with FIAP/PSA definitions: neutral black-and-white and single-tone images pass, while split-toned or multi-color images fail. Borderline cases are flagged for human review.

---

## Glossary

- **Chroma**: The intensity or purity of color. Low chroma means nearly neutral (gray), high chroma means strong color.
- **Hue**: The attribute of a color that lets us classify it as red, yellow, green, etc. Measured in degrees around a color wheel.
- **Split-tone**: An image with two distinct color tones, often in highlights and shadows.
- **Toned**: An image with a single, consistent color tint (e.g., sepia).
- **Neutral**: An image with no discernible color tint—pure black, white, and gray.
- **Query**: A result where the checker is unsure and flags the image for human review.
- **Override**: A special rule that allows an image to pass or be flagged for review even if it would otherwise fail, based on specific characteristics (e.g., strong but uniform tone, stage lighting).
- **Degrade**: The process of downgrading a fail to a pass or query if the color presence is minor or subtle.

---

## Philosophy and Purpose

The checker aims for fairness, consistency, and transparency in monochrome competitions. It is designed to:
- Align with FIAP/PSA rules (final image, not editing method)
- Be lenient on borderline or subtle cases (flag for review, not auto-fail)
- Use perceptual color metrics (LAB color space, chroma-weighted)
- Provide clear, actionable results for both photographers and judges

### Interpreting the Results: Judging the Output, Not the Method

A critical distinction for judges is that the checker analyzes the final rendered pixels of an image, not the photographer's editing process.

From pixels alone, we can reliably tell whether the resulting image reads as (i) neutral, (ii) single-toned, or (iii) split-toned. What we cannot do with certainty is prove whether the photographer applied a "split-toning" tool in their software.

#### Many-to-One Mapping
Different editing pipelines can lead to the same final result. A split-toning tool with a strong balance pushed to one side can create a single-toned output. Conversely, a single tint combined with complex curve adjustments can mimic a weak split-tone.

**Competition Rules:** Salon rules (FIAP/PSA) are concerned with the final image. If the output exhibits a single, uniform tone, it should pass as monochrome, regardless of the tools used to create it. If the output shows two or more distinct tones, it should fail.

The checker is aligned with this principle: it judges the result, not the artist’s intent or method.

---

## Key Variables and Thresholds

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

## Detailed Explanation of Each Step

### 1. Neutral Monochrome Check
*   **Purpose**: To quickly identify images that are truly grayscale with negligible color.
*   **Criteria**: `chroma_p99 <= neutral_chroma`
    *   `chroma_p99`: The 99th percentile of chroma values across the image. This means 99% of the pixels have a chroma value less than or equal to this number.
    *   `neutral_chroma`: A configurable threshold (default 2.0). If almost all pixels have very low chroma, the image is considered neutral.
*   **Outcome**: `PASS (Neutral Monochrome)`.

### 2. Toning Collapse Check
*   **Purpose**: To reclassify images that might initially appear split-toned (due to two distinct hue peaks) but where the highlights and shadows actually fall within the same general hue family. This prevents false positives for split-toning.
*   **Criteria**: `fail_two_peak` is true AND `delta_h_highs_shadows_deg < 45.0`
    *   `fail_two_peak`: Indicates that the two most dominant hue peaks are significantly separated (`>= 15.0°`) and both have substantial mass (`>= 10%`).
    *   `delta_h_highs_shadows_deg`: The circular hue difference between the mean hue of the brightest 25% of pixels (highlights) and the darkest 25% of pixels (shadows).
    *   `45.0°`: A threshold. If the hue difference between highlights and shadows is less than 45 degrees, it suggests they are part of the same broad hue family, even if two distinct peaks exist.
*   **Outcome**:
    *   If `hue_std > toned_pass_deg`: `PASS WITH QUERY (Toned)`. The toning collapsed, but the overall hue variation is still wider than a clear pass.
    *   Otherwise: `PASS (Toned)`. The toning collapsed to a single hue family.

### 3. Stage-Lit Override
*   **Purpose**: To correctly classify images that have a strong color element in the subject but a largely neutral background, common in stage photography.
*   **Criteria**: `force_fail` is true AND `single_hue_stage_lit` is true
    *   `force_fail`: A flag set if the image has strong, widespread color that would normally cause a failure.
    *   `single_hue_stage_lit`: Indicates a large neutral-shadow region (e.g., >55% of pixels with low lightness and chroma) with a subject that has a single, dominant hue.
*   **Outcome**: `PASS WITH QUERY (Toned)`.

### 4. Uniform Strong Tone Override
*   **Purpose**: To allow images with a very strong but consistent single tone (e.g., a deeply sepia-toned image) to pass, even if their overall hue spread is slightly wider than the standard pass threshold.
*   **Criteria**: `uniform_strong_tone` is true AND `hue_std > toned_pass_deg`
    *   `uniform_strong_tone`: Indicates a narrow hue spread (`<= 14.0°`), high hue concentration (`R >= 0.85`), a very dominant primary hue (`primary_share >= 0.97`), and a significant presence of strong color (`chroma_ratio4 >= 0.05`).
    *   `hue_std > toned_pass_deg`: The overall hue spread is wider than the standard pass threshold.
*   **Outcome**: `PASS (Toned)`.

### 5. Refined Pass Condition
*   **Purpose**: The primary condition for an image to be considered a clear "pass" as a toned monochrome.
*   **Criteria**: NOT `force_fail` AND `hue_std <= toned_pass_deg` AND `merge_ok`
    *   `force_fail`: Must not be triggered.
    *   `hue_std <= toned_pass_deg`: The circular hue standard deviation is within the tight "pass" limit (default 10.0°).
    *   `merge_ok`: No significant split-toning is detected (either no second peak, or it's too close or too weak).
*   **Outcome**: `PASS (Toned)`.

### 6. Refined Query Condition
*   **Purpose**: To flag images for review that are borderline toned monochromes.
*   **Criteria**: NOT `force_fail` AND (`hue_std <= toned_query_deg` OR (`peak_delta_deg` is not None AND `12.0 < peak_delta_deg <= 18.0` AND `second_mass < 0.15`))
    *   `force_fail`: Must not be triggered.
    *   `hue_std <= toned_query_deg`: The circular hue standard deviation is within the "query" limit (default 14.0°).
    *   OR: There is a detected second hue peak that is moderately separated (`12.0° < peak_delta_deg <= 18.0°`) but has a relatively small mass (`second_mass < 0.15`). This catches subtle secondary tones that might warrant a human review.
*   **Outcome**: `PASS WITH QUERY (Toned)`.

### 7. Default Fail Conditions
*   **Purpose**: If none of the above conditions result in a pass or query, the image is considered a failure. This section determines the specific reason for failure.
*   **Conditions (evaluated in order):**
    *   **Split-Toning Suspected**: `fail_two_peak` is true OR `hilo_split` is true OR (`R < 0.4` AND `R2 > 0.6`).
        *   *Explanation*: Indicates clear evidence of two distinct hue families, either from peak analysis, highlight/shadow comparison, or a legacy bimodality metric.
        *   *Failure Reason*: `split_toning_suspected`.
    *   **Multi-Color**: `cf >= 25.0` OR `chroma_p95 > neutral_chroma + 8.0`.
        *   *Explanation*: The image is generally too colorful or has strong, widespread color.
        *   *Failure Reason*: `multi_color`.
    *   **Near-Neutral Color Cast**: `chroma_med < neutral_chroma * 0.75` AND `hue_std < 30.0`.
        *   *Explanation*: A very subtle color cast is present, but it's not strong enough to be considered intentional toning.
        *   *Failure Reason*: `near_neutral_color_cast`.
    *   **Color Present (General)**: Any other case where color is detected but doesn't fit other categories.
        *   *Failure Reason*: `color_present`.

### 8. Degrade to Pass/Query (from an initial Fail)
*   **Purpose**: To re-evaluate certain "fail" conditions and potentially downgrade them to a "pass" or "pass with query" if the color presence is minor or subtle, especially if `force_fail` was not initially triggered.
*   **Criteria (only if NOT `force_fail`):**
    *   **Degrade to PASS**: If (`small_footprint` OR (`soft_large_footprint` AND `chroma_ratio4 < 0.12`)) AND (`large_drift` OR `hue_std < 45.0`).
        *   *Explanation*: The color footprint is very small or soft, and either there's significant hue drift (which might be an artistic choice) or the hue spread is still relatively tight.
        *   *Outcome*: `PASS (Toned)`.
    *   **Degrade to PASS WITH QUERY**: If `moderate_footprint` OR `subtle_cast` OR `soft_large_footprint` OR (`large_drift` AND `chroma_ratio4 < 0.05`).
        *   *Explanation*: The color footprint is moderate, or there's a subtle cast, or a soft large footprint, or large drift with a small strong color footprint. These are borderline cases that warrant review.
        *   *Outcome*: `PASS WITH QUERY (Toned)`.

### 9. Final Fail
*   **Purpose**: If an image still remains a "fail" after all degradation checks, it is definitively classified as not monochrome.
*   **Outcome**: `FAIL (Not Monochrome)`.

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

- **v3 (2025-09-23):** Combined summary, philosophy, glossary, and detailed technical mapping into a single comprehensive reference.
- **v2:** Added explicit variable mapping, code references, and a table of thresholds. Improved clarity for debugging and code review.
- **v1:** Initial version.

---

*For a user-friendly summary, see: [Monochrome Checker Logic](MONOCHROME_CHECKER_LOGIC.md)*
