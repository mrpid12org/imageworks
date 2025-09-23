# Monochrome Checker Decision Tree

This document outlines the decision-making logic used by the monochrome checker to classify images as "pass", "pass with query", or "fail" for monochrome compliance. The process involves a series of checks, starting with neutrality, then evaluating toning, and finally identifying various failure modes.

## Key Concepts and Criteria Explained

*   **`neutral_chroma`**: A threshold (default 2.0 C*) below which pixels are considered essentially neutral.
*   **`toned_pass_deg`**: The maximum circular hue standard deviation (σ) in degrees (default 10.0°) for an image to be considered a clear "pass" as a toned monochrome.
*   **`toned_query_deg`**: The maximum circular hue standard deviation (σ) in degrees (default 14.0°) for an image to be considered "pass with query" as a toned monochrome.
*   **`force_fail`**: A flag indicating if the image has strong, widespread color that should generally lead to a failure, unless overridden by specific conditions.
*   **`uniform_strong_tone`**: An override for images with exceptionally strong but uniform tones, allowing them to pass if the hue spread remains narrow and the image stays single-hued.
    *   *Criteria:* `hue_std <= LAB_STRONG_TONE_HUE_STD` (14.0°), `hue_concentration (R) >= LAB_STRONG_TONE_CONCENTRATION` (0.85), `primary_share >= LAB_STRONG_TONE_PRIMARY_SHARE` (0.97), and `chroma_ratio4 >= 0.05`.
*   **`single_hue_stage_lit`**: An override for images with a large neutral shadow region but a single-hue subject, often seen in stage lighting.
    *   *Criteria:* `shadow_share >= LAB_SHADOW_QUERY_SHARE` (0.55), `subject_share >= 0.05`, `hue_std <= LAB_SHADOW_QUERY_HUE_STD` (24.0°), `primary_share >= LAB_SHADOW_QUERY_PRIMARY_SHARE` (0.95), and `chroma_ratio4 >= 0.05`.
*   **`merge_ok`**: A condition indicating that any detected secondary hue peaks are either non-existent, very close to the primary peak (`<= MERGE_DEG` = 12.0°), or have insignificant mass (`< MINOR_MASS` = 0.10).
*   **`fail_two_peak`**: A condition indicating that two distinct hue peaks were found, separated by a significant angular distance (`>= FAIL_DEG` = 15.0°) and with a substantial secondary mass (`>= MINOR_MASS` = 0.10).
*   **`hilo_split`**: A condition indicating that the circular hue difference between highlights and shadows (`delta_h_highs_shadows_deg`) is significant (`>= HILO_SPLIT_DEG` = 45.0°).
*   **`R` (Hue Concentration)**: Resultant length of the circular mean of hues. A value closer to 1.0 indicates a tighter concentration of hues.
*   **`R2` (Hue Bimodality)**: Resultant length of the circular mean of doubled hues. A high value (e.g., > 0.6) can indicate bimodality, suggesting two distinct hue clusters.
*   **`cf` (Colorfulness)**: Hasler–Süsstrunk colorfulness metric. A higher value indicates a more colorful image.
*   **`chroma_p95`**: 95th percentile of chroma values.
*   **`chroma_med`**: Median chroma value.
*   **`small_footprint`, `moderate_footprint`, `soft_large_footprint`, `subtle_cast`**: These describe the extent and intensity of color presence in the image, used for degrading a potential "fail" to a "pass" or "query".
*   **`large_drift`**: Indicates a significant shift in hue across the tonal range (`abs(hue_drift_deg_per_l) > 120.0`).

---

## Decision Flow

The checker evaluates an image through the following sequence of conditions. The first condition met determines the verdict.

```mermaid
graph TD
    A[Start: Check Monochrome] --> B{Is chroma_p99 <= neutral_chroma?};
    B -- Yes --> C[Verdict: PASS - Neutral Monochrome];
    B -- No --> D{Is fail_two_peak AND delta_h_highs_shadows_deg < 45.0?};
    D -- Yes --> E{Is hue_std > toned_pass_deg?};
    E -- Yes --> F[Verdict: PASS WITH QUERY - Toned - Toning collapsed, but wider hue variation];
    E -- No --> G[Verdict: PASS - Toned - Toning collapsed to single hue family];
    D -- No --> H{Is force_fail AND single_hue_stage_lit?};
    H -- Yes --> I[Verdict: PASS WITH QUERY - Toned - Stage-lit override];
    H -- No --> J{Is uniform_strong_tone AND hue_std > toned_pass_deg?};
    J -- Yes --> K[Verdict: PASS - Toned - Uniform strong tone override];
    J -- No --> L{Is NOT force_fail AND hue_std <= toned_pass_deg AND merge_ok?};
    L -- Yes --> M[Verdict: PASS - Toned - Refined Pass Condition];
    L -- No --> N{Is NOT force_fail AND (hue_std <= toned_query_deg OR (peak_delta_deg > 12.0 AND peak_delta_deg <= 18.0 AND second_mass < 0.15))?};
    N -- Yes --> O[Verdict: PASS WITH QUERY - Toned - Refined Query Condition];
    N -- No --> P{Default Fail Conditions};

    P --> Q{Is fail_two_peak OR hilo_split OR (R < 0.4 AND R2 > 0.6)?};
    Q -- Yes --> R[Failure Reason: split_toning_suspected];
    Q -- No --> S{Is cf >= 25.0 OR chroma_p95 > neutral_chroma + 8.0?};
    S -- Yes --> T[Failure Reason: multi_color];
    S -- No --> U{Is chroma_med < neutral_chroma * 0.75 AND hue_std < 30.0?};
    U -- Yes --> V[Failure Reason: near_neutral_color_cast];
    U -- No --> W[Failure Reason: color_present];

    R --> X{Is NOT force_fail?};
    T --> X;
    V --> X;
    W --> X;

    X -- Yes --> Y{Can degrade to PASS? (small_footprint OR soft_large_footprint AND chroma_ratio4 < 0.12) AND (large_drift OR hue_std < 45.0)};
    Y -- Yes --> Z[Verdict: PASS - Toned - Degraded from Fail];
    Y -- No --> AA{Can degrade to PASS WITH QUERY? (moderate_footprint OR subtle_cast OR soft_large_footprint OR (large_drift AND chroma_ratio4 < 0.05))};
    AA -- Yes --> BB[Verdict: PASS WITH QUERY - Toned - Degraded from Fail];
    AA -- No --> CC[Verdict: FAIL - Not Monochrome - Final Fail];
    X -- No --> CC;

    CC --> End;
    Z --> End;
    BB --> End;
    C --> End;
    F --> End;
    G --> End;
    I --> End;
    K --> End;
    M --> End;
    O --> End;
```

---

## Detailed Explanation of Each Step:

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
