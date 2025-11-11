# ImageWorks -- Tonal & Technical Analysis Integration Guide

*For use with Qwen-3-VL / VLM-based photography judges*

------------------------------------------------------------------------

## 1. Purpose

This document describes how to: 1. Implement lightweight **tonal
analysis** for photographic images using OpenCV and NumPy.\
2. Combine **MUSIQ**, **NIMA**, and tonal metrics with **Qwen-3-VL**
through structured prompting, so the model produces human-like
competition critiques and scores.

------------------------------------------------------------------------

## 2. Why add tonal analysis

Vision--language models are strong at composition and subject
interpretation but weak at: - evaluating **contrast, brightness,
clipping, or local sharpness**, and\
- differentiating intentional low contrast from under-exposure.

A tonal-analysis module provides objective numeric cues about exposure
and contrast that the model can then interpret linguistically.

------------------------------------------------------------------------

## 3. What tonal analysis measures

  -----------------------------------------------------------------------
  Aspect                              Meaning
  ----------------------------------- -----------------------------------
  **Global contrast**                 Spread of lightest vs. darkest
                                      tones (histogram range).

  **Mid-tone balance**                Whether exposure favours shadows or
                                      highlights.

  **Highlight control**               Degree of clipping near white.

  **Shadow detail**                   Degree of crushing near black.

  **Local contrast**                  Micro-contrast giving "snap" or
                                      clarity.

  **Colour balance**                  Warm/cool bias or channel clipping
                                      (optional).
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 4. Core metrics to compute

  ------------------------------------------------------------------------
  Feature          Formula / method              Interpretation
  ---------------- ----------------------------- -------------------------
  **Dynamic range  `(P99 – P1)` of luminance     Narrow → flat; wide →
  ratio**          histogram                     punchy

  **Mean           Average of Y channel (0--1)   Exposure level
  luminance**

  **Histogram      Weighted mean of luminance    Skew to dark/bright
  centre**         histogram

  **Clipping       \% of pixels \< 2 % or \> 98  \> 1--2 % → clipping risk
  percentage**     % luminance

  **Mid-tone       Slope of cumulative histogram Flat → low mid-tone
  slope**          25--75 %                      contrast

  **Local contrast Std-dev of Laplacian filter   Edge "snap" /
  (Laplacian σ)**  response                      micro-clarity
  ------------------------------------------------------------------------

------------------------------------------------------------------------

## 5. Python implementation example

``` python
import cv2, numpy as np

def tonal_analysis(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.0

    # Histogram
    hist = cv2.calcHist([gray],[0],None,[256],[0,1]).ravel()
    hist /= hist.sum()
    cdf = np.cumsum(hist)

    p1, p99 = np.searchsorted(cdf,[0.01,0.99])/255
    dyn_range = p99 - p1
    mean_lum = gray.mean()
    clip_lo = (gray < 0.02).mean()
    clip_hi = (gray > 0.98).mean()

    # Local contrast
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    local_contrast = lap.std()

    result = {
        "mean_luminance": float(mean_lum),
        "dynamic_range": float(dyn_range),
        "clip_low_percent": 100*float(clip_lo),
        "clip_high_percent": 100*float(clip_hi),
        "local_contrast": float(local_contrast)
    }
    return result
```

### Typical interpretation ranges

  -----------------------------------------------------------------------
  Metric                  Good range              Comment
  ----------------------- ----------------------- -----------------------
  Mean luminance          0.40 -- 0.60            Balanced exposure

  Dynamic range           \> 0.70                 Punchy tonal spread

  Clip high / low         \< 1 % each             Controlled highlights &
                                                  shadows

  Local contrast          0.03 -- 0.07            Too low = flat, too
  (Laplacian σ)                                   high = over-sharpened
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 6. Generating a tonal summary

After computing metrics, auto-generate a one-sentence description:

``` python
def tonal_summary(metrics):
    m = metrics
    summary = []
    if m["dynamic_range"] < 0.5:
        summary.append("Overall contrast appears low and mid-tones are somewhat flat.")
    if m["clip_high_percent"] > 1:
        summary.append("Highlights show minor clipping.")
    if m["clip_low_percent"] > 1:
        summary.append("Shadows verge on blockage.")
    if not summary:
        summary.append("Tonal balance looks well controlled with moderate contrast.")
    return " ".join(summary)
```

Output example:\
\> "Exposure is slightly dark with moderate contrast; highlights and
shadows are well controlled but local contrast is a bit low."

------------------------------------------------------------------------

## 7. Feeding data into the VLM judge

### Inputs per image

1.  **Image** file.\
2.  **TECHNICAL ANALYSIS block** containing:
    -   MUSIQ MOS (0--100, higher = better)\
    -   NIMA Aesthetic (1--10)\
    -   NIMA Technical (1--10)\
    -   Tonal metrics from §5\
    -   Tonal summary sentence from §6

------------------------------------------------------------------------

## 8. System prompt for Qwen-3-VL

``` text
You are an experienced UK camera-club competition judge.

You will receive:
1) An image.
2) A TECHNICAL ANALYSIS block with objective measurements
   (MUSIQ, NIMA, tonal metrics). These are reliable and should be
   treated as ground truth for technical quality and tonality.

Your task:
- Combine what you see with the TECHNICAL ANALYSIS to produce a
  concise, constructive critique and a score out of 20 using:
  • Impact & Communication
  • Composition & Design
  • Technical Quality & Presentation (use the analysis data)

Guidelines:
- Refer to the TECHNICAL ANALYSIS when commenting on exposure,
  contrast, tonal range, and sharpness. Do not contradict it.
- Translate numbers into natural phrases (e.g. “slightly flat mid-tones”).
- Mention at least one technical or tonal aspect explicitly.
- Keep critique 100–130 words.
- Use the entire 14–20 band: in a ~90 image field award several 18s and at least one 19–20 when deserved; justify the higher marks.
- Score = integer 0–20 (14–20 typical club range).

Return valid JSON only:
{
  "title": "<string>",
  "category": "Open|Nature|Creative|Themed|null",
  "style": "Open|Nature|Creative|Documentary|Abstract|Record|Other|null",
  "critique": "<100–130 words>",
  "score": <integer>,
  "subscores": {
    "impact": 0-5,
    "composition": 0-5,
    "technical": 0-5,
    "category_fit": 0-5
  }
}
```

------------------------------------------------------------------------

## 9. User prompt template

``` text
Image: <attach or link>

Title: "Rory goes for it"
Category: "Creative"

TECHNICAL ANALYSIS:
- MUSIQ MOS: 42.98 / 100 (higher = better)
- NIMA Aesthetic: 5.27 / 10
- NIMA Technical: 5.24 / 10
- Tonal metrics:
    • mean_luminance: 0.46
    • dynamic_range: 0.62
    • clip_low_percent: 0.4 %
    • clip_high_percent: 0.1 %
    • local_contrast: 0.031
- Tonal summary:
    "Exposure slightly dark, moderate contrast, highlights/shadows
     controlled, local contrast a bit low."

Instruction:
Evaluate this image for a club competition using the rubric and
the TECHNICAL ANALYSIS above. Return only the JSON object defined
in the system prompt.
```

------------------------------------------------------------------------

## 10. Why this works

-   **Context plus authority** --- clearly signals that numerical data
    are factual.\
-   **Human-like phrasing** --- model translates numbers into natural
    commentary.\
-   **Balanced weighting** --- the model still judges impact/composition
    but grounds technical remarks in real metrics.\
-   **Consistency** --- identical data produce stable tone evaluations
    across sessions.

------------------------------------------------------------------------

## 11. Example expected output

``` json
{
  "title": "Rory goes for it",
  "category": "Creative",
  "critique": "A dynamic panning shot conveying impressive speed and commitment as the rider leans hard into the bend. The blurred background and matching reds strengthen the motion effect. Technical analysis shows moderate perceptual quality and slightly flat mid-tone contrast, consistent with the subdued look; a modest mid-tone lift would add punch. Sharpness on the rider is adequate and both highlights and shadows are well controlled. A compelling image of action and risk, handled with assurance.",
  "score": 17
}
```

------------------------------------------------------------------------

## 12. Integration summary

    [Preprocess resize 448 px] ─► [MUSIQ/NIMA models]
                                  ├── objective scores
                                  ├── tonal analysis (OpenCV)
                                  ├── textual summary
                                  └── TECHNICAL ANALYSIS block
                                            ↓
                                  [Qwen-3-VL judge prompt]
                                            ↓
                             → JSON critique + score (human-like)

**Suggested weighting for composite technical merit:** - MUSIQ 40 % -
NIMA Technical 20 % - Tonal module 20 % - VLM Technical sub-score 20 %

------------------------------------------------------------------------

## 13. Notes & extensions

-   Compute optional **colour clipping** and **white-balance bias** via
    Lab a*/b* channels.\
-   Add an **"auto-curves simulation"** that re-computes MUSIQ/NIMA
    after an S-curve adjustment to estimate potential tonal gain.\
-   Maintain club-night statistics to calibrate average dynamic-range
    levels per genre.

------------------------------------------------------------------------

### **Summary**

> **Implementation:** use OpenCV metrics (dynamic range, clipping,
> contrast).\
> **Integration:** feed results to Qwen-3-VL as a TECHNICAL ANALYSIS
> block.\
> **Prompting:** state clearly that these numbers are ground truth and
> must inform the Technical Quality section.\
> **Outcome:** consistent, human-like photographic critiques that
> explicitly discuss tone and technical quality.
