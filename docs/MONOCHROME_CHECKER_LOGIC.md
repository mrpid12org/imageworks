# Monochrome Checker Logic: A Decision Tree

```mermaid
graph TD
    subgraph "Step 1: Neutral Check"
        A{"Is image neutral?<br>(Chroma-99th <= 2.0)"};
    end

    subgraph "Step 2: Hard-Fail Check"
        B{"Obvious widespread color?<br>(Large footprint AND high hue spread)"};
    end

    subgraph "Step 3: Toned Pass Check"
        C{"Valid single tone?<br>(hue_std <= 10° AND passes two-peak check)"};
    end

    subgraph "Step 4: Query/Borderline Check"
        D{"Borderline case?<br>(hue_std 10-14° OR borderline split-tone)"};
    end

    subgraph "Step 5: Final Failure Analysis"
        E{"Clear split-tone?<br>(Distant peaks AND significant mass)"};
    end

    subgraph "Verdicts"
        direction LR
        P1[PASS - Neutral];
        P2[PASS - Toned];
        Q[PASS - Query];
        F1[FAIL - Color Present];
        F2[FAIL - Split-Toning Suspected];
    end

    A -- No --> B;
    A -- Yes --> P1;

    B -- Yes --> F1;
    B -- No --> C;

    C -- Yes --> P2;
    C -- No --> D;

    D -- Yes --> Q;
    D -- No --> E;

    E -- Yes --> F2;
    E -- No --> F1;

    classDef check fill:#fff,stroke:#333,stroke-width:2px,color:#333;
    classDef verdict_pass fill:#dfd,stroke:#3a3,stroke-width:2px;
    classDef verdict_query fill:#fdf,stroke:#a3a,stroke-width:2px;
    classDef verdict_fail fill:#fdd,stroke:#a33,stroke-width:2px;

    class A,B,C,D,E check;
    class P1,P2 verdict_pass;
    class Q verdict_query;
    class F1,F2 verdict_fail;
```

This document outlines the step-by-step logic the Imageworks Competition Checker uses to determine if an image is a valid monochrome. The logic is designed to be consistent with FIAP/PSA definitions, which allow for neutral black-and-white images as well as images toned with a single, consistent hue.

### Guiding Principles

- **Perceptual Accuracy:** All color and hue measurements are performed in the LAB color space and are weighted by chroma (color intensity) to better align with human perception.
- **Lenience on Borderline Cases:** The logic aims to flag ambiguous images for human review (`Query`) rather than failing them outright, especially when a color cast is very subtle or covers a microscopic area.
- **Robust Split-Tone Detection:** A two-peak hue analysis is used to differentiate between an acceptable color wobble within a single tone and a true, multi-toned image.

---

## The Decision Tree

The logic is applied after the image has been loaded and analyzed to gather key metrics. It follows this sequence:

**Step 1: Check for True Neutral**

-   **Question:** Is the image almost perfectly black and white, with no discernible tint?
-   **Test:** The tool checks if the chroma (color intensity) of even the most colorful pixels (ignoring the top 1% of outliers) is below a tiny threshold (`C*99 ≤ 2.0`).
-   **Result:**
    -   **YES:** ➡️ **VERDICT: PASS (Neutral)**. The image is considered a true neutral monochrome.
    -   **NO:** ➡️ Proceed to the next step.

---

**Step 2: Check for Obvious, Widespread Color**

-   **Question:** Does the image have a large area of obvious color and significant color variation, making it an almost certain failure?
-   **Test:** This is a "hard fail" check. It fails if **ALL** of the following are true:
    1.  The most colorful pixels are clearly visible (`C*99 ≥ 6.0`).
    2.  A large portion of the image is colored (over 10% of pixels have visible color, or a single colored patch covers over 8%).
    3.  The color varies significantly (`hue_std > 10°`).
    4.  The image doesn't qualify for the "Uniform Strong Tone" exception (see below).
-   **Result:**
    -   **YES:** ➡️ **VERDICT: FAIL (Color Present)**. The image has too much color variation over too large an area.
    -   **NO:** ➡️ Proceed to the next step. The color is either not widespread or not varied enough for an automatic failure.

---

**Step 3: Check for Valid Single-Toned Images**

-   **Question:** Is the image a legitimate, consistently toned monochrome (like a classic sepia or selenium print)?
-   **Test:** The image passes if it has a tight, consistent tint. This is true if **BOTH** of the following conditions are met:
    1.  **Low Hue Variation:** The color that *is* present is very consistent across the image (`hue_std ≤ 10°`).
    2.  **Passes Two-Peak Check:** If two main "hues" are detected, they are either so close they are perceived as one (e.g., yellow and orange, `delta < 12°`), OR the second hue is insignificant (its "mass" is less than 10% of the main one). This prevents images with a minor, acceptable color wobble from failing.
-   **Result:**
    -   **YES:** ➡️ **VERDICT: PASS (Toned)**. The image is a valid toned monochrome.
    -   **NO:** ➡️ Proceed to the next step.

---

**Step 4: Check for Borderline Cases (Query)**

-   **Question:** Is the image not a clear pass, but also not a definite fail?
-   **Test:** The image is flagged for human review if it's in a "caution" zone. This happens if **EITHER** of these is true:
    1.  **Moderate Hue Variation:** The color variation is in a middle range—not tight enough to pass, but not wide enough to fail outright (`10° < hue_std ≤ 14°`).
    2.  **Borderline Split-Tone:** The two-peak analysis finds two distinct hues, but the case is borderline (e.g., peak delta between 12-18° and the second peak has less than 15% mass).
-   **Result:**
    -   **YES:** ➡️ **VERDICT: PASS (Query)**. The image is flagged for a judge to make the final call.
    -   **NO:** ➡️ Proceed to the final step.

---

**Step 5: Final Verdict (Fail)**

-   **Question:** If the image has reached this point, what is the reason for failure?
-   **Test:** An image that isn't a Neutral Pass, Toned Pass, or Query is a Fail. The logic assigns a specific reason:
    1.  **Clear Split-Tone:** The two-peak analysis finds two distinct hues that are far apart (`delta ≥ 15°`) and the second hue is significant (`mass ≥ 10%`).
    2.  **General Color:** If it doesn't meet the split-tone criteria, it fails simply because its color variation (`hue_std`) was too high.
-   **Result:**
    -   ➡️ **VERDICT: FAIL (Split-Toning Suspected or Color Present)**.

---

### Important Exceptions & Overrides

-   **Uniform Strong Tone:** An image with a very strong, saturated color (like a deep blue cyanotype) can still **PASS** if that color is extremely consistent across the entire frame. The logic has a specific override for this case.
-   **Tiny Leak Downgrade:** If an image would otherwise fail, but the actual colored pixels cover a microscopic portion of the image (e.g., less than 1%), the verdict is often downgraded from `FAIL` to `PASS (Query)` to be lenient on tiny, insignificant color halos.
-   **Stage Lighting:** A special override exists to handle cases where a mostly-neutral scene has a single, strong colored light on a subject. These are typically flagged as a `Query` instead of a `Fail`.
