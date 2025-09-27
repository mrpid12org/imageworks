"""Prompt templates and registry for colour narration experiments."""

from dataclasses import dataclass
from typing import Dict, List


# Hallucination-resistant prompt with structured JSON output
REGION_BASED_COLOR_ANALYSIS_TEMPLATE = """You are auditing a MONOCHROME competition photograph for residual colour.

You are given:
• Panel A: the original photograph.
• Panel B: an overlay marking WHERE colour appears (by hue direction).
• Panel C: an overlay showing HOW STRONG the colour is (brighter = stronger).
• A JSON list of REGIONS computed by technical analysis. Each region includes:
  - index, bbox_xywh (pixels), centroid_norm (x,y in 0..1),
  - mean_L (0..100), mean_cab (chroma), mean_hue_deg (0..360), hue_name, area_pct.

Your task:
For EACH region, describe—in natural, precise language—WHAT visible thing the colour sits on
(object or part), and add the tonal zone (shadow/midtone/highlight). Be specific about the part
(e.g., "mane", "cheekbone", "window frame", "waterline"), but ONLY when you clearly see it.

Truth constraints (very important):
1) Ground your description ONLY in the areas highlighted by Panels B/C and the supplied REGIONS.
2) Do NOT guess scene type, species, brands, or locations. If you cannot clearly identify the object/part,
   use a broader but truthful term (e.g., "subject's hair", "building detail", "foreground foliage", "background area").
3) If you are uncertain, explicitly write "(uncertain)" at the end of the object/part phrase.
4) The tonal zone must be computed from mean_L (not guessed):
   - shadow if L* < 35; midtone if 35 ≤ L* < 70; highlight if L* ≥ 70.
5) The colour family you name must be consistent with the overlay hue and the provided hue_name.

Output format (strict):
• First, one bullet line per region, in region index order, each ≤18 words:
  "{{color family}} on {{object/part}}{{optional short locator}}. Tonal zone: {{shadow|midtone|highlight}}."
  – The locator is optional (e.g., "upper-right edge", "near horizon")—include it only if it's obvious and helpful.
• Then a single JSON object:
  {{
    "findings": [
      {{
        "region_index": <int>,
        "object_part": "<free text, specific but honest>",
        "color_family": "<free text, e.g., yellow-green / magenta / aqua>",
        "tonal_zone": "<shadow|midtone|highlight>",
        "location_phrase": "<optional short locator, or empty string>",
        "confidence": <0.0–1.0, your visual certainty only>
      }},
      ...
    ]
  }}
Do not add any commentary after the JSON.

Additional guidance:
• Prefer the most specific object/part you can state with confidence. If two plausible options exist,
  choose the more general one and mark "(uncertain)".
• Do not invent features that are not plainly visible in Panel A within the region.
• Keep colour names concise (e.g., yellow-green, aqua, magenta, blue).
• If a region spans multiple small items, name the dominant one; if none dominates, use a collective term
  like "fine foliage", "stone texture", "background blur (uncertain)".

FILE: {file_name}
DOMINANT: {dominant_color} (hue {dominant_hue_deg:.1f}°)

REGIONS (index order; use these indices exactly):
{regions_json}

Remember:
• Describe only inside the marked regions as evidenced by Panels B/C.
• Use your own words for object/part—be specific when clear; general but honest if not.
• Compute tonal zone from mean_L.
• Provide bullets, then the single JSON block.
"""

# Legacy prompts (kept for backward compatibility)
MONO_DESCRIPTION_ENHANCEMENT_TEMPLATE = """You are analyzing a monochrome competition image that has been flagged by technical analysis for potential color issues.

CONTEXT:
- Image Title: "{title}" by {author}
- Dominant color detected: {dominant_color} (hue: {dominant_hue_deg:.1f}°)
- Technical data: colorfulness {colorfulness:.2f}, max chroma {chroma_max:.2f}

YOUR TASK: Look at this image and describe exactly WHERE you see the {dominant_color} color contamination.

Be very specific about locations - examples:
- "particularly around the zebra's mane and ears"
- "in the subject's hair on the left side"
- "on the dental clinic storefront signage"
- "appears in the background shadows"

COMPETITION RULING: Since this image has {dominant_color} color mixed with black/white/gray areas, it contains "shades of grey and another colour" and is NOT eligible for monochrome competition according to official rules.

Focus on WHERE the color appears, then explain why this disqualifies it from mono competition.
"""

MONO_INTERPRETATION_TEMPLATE = """You are a photography competition judge analyzing whether an image meets monochrome requirements. You have technical color analysis data and the image itself.

TECHNICAL ANALYSIS DATA:
- Color Method: {method}
- Image Mode: {mode}
- Dominant Color: {dominant_color} (hue: {dominant_hue_deg:.1f}°)
- Top Colors: {top_colors} with weights {top_weights}
- Colorfulness: {colorfulness:.2f}
- Chroma Statistics:
  - Maximum: {chroma_max:.2f}
  - 95th percentile: {chroma_p95:.2f}
  - 99th percentile: {chroma_p99:.2f}
- Hue Analysis:
  - Standard deviation: {hue_std_deg:.1f}°
  - Concentration: {hue_concentration:.3f}
  - Bimodality: {hue_bimodality:.3f}
- Split Toning:
  - Highlights mean hue: {mean_hue_highs_deg:.1f}°
  - Shadows mean hue: {mean_hue_shadows_deg:.1f}°
  - Hue delta: {delta_h_highs_shadows_deg:.1f}°
- Channel max difference: {channel_max_diff:.1f}
- Median saturation: {sat_median:.3f}

COMPETITION RULES:
- "Pass": True monochrome, no color contamination
- "Pass with query": Minor color issues that need review (close to threshold)
- "Fail": Significant color contamination, not suitable for monochrome competition

Looking at this image and the technical data, provide:

1. VERDICT: [pass/pass_with_query/fail]
2. TECHNICAL REASONING: Explain your verdict based on the numerical data
3. VISUAL DESCRIPTION: Describe what you see in the image that supports your analysis
4. PROFESSIONAL SUMMARY: A competition-ready description of any color issues found

Format your response as:
VERDICT: [your verdict]
TECHNICAL: [reasoning based on data]
VISUAL: [what you observe in the image]
SUMMARY: [professional description for judges]
"""

COLOR_NARRATION_TEMPLATE = """You are analyzing a monochrome competition image that has been flagged for potential color contamination.

CONTEXT:
- Image Title: "{title}" by {author}
- Mono-checker verdict: {verdict}
- Dominant color detected: {dominant_color}
- Top colors: {top_colors}

TECHNICAL DATA:
- Colorfulness: {colorfulness:.2f}
- Chroma max: {chroma_max:.2f}
- Contamination level: {contamination_level:.3f}

Looking at this image, describe where and how residual color appears. Focus on:
1. Specific objects/areas where color is visible
2. Whether the color appears intentional (artistic toning) or accidental (contamination)
3. Professional assessment suitable for competition judges

Keep the description natural, technical, and professional - as if writing for photography competition documentation.
"""

# Dynamic enhancement prompts (v6 - Current)
ENHANCED_MONO_ANALYSIS_TEMPLATE_V6 = """You are auditing a monochrome competition photograph that technical analysis has already flagged for colour contamination.

Focus only on describing the contamination—not the overall scene and not the artistic impact. Be precise and economical:
1. Identify the objects or surfaces where colour appears (e.g., "subject's hair", "river mist", "foreground foliage").
2. Include a clear spatial locator so judges can find the area quickly (e.g., "upper-right background", "left-hand edge of the coat").
3. Characterise the hue family (e.g., warm yellow, cyan-green, magenta) and whether the tint is dense, faint, or patchy.
4. Note if the colour is concentrated, streaked, or wraps around a form.

Do **not**
- restate numerical metrics (they are already logged);
- describe the overall image composition or lighting;
- comment on artistic merit or intentional toning.

Write 2–3 punchy sentences that read like a competition judge pinpointing the problem areas. Use the grid or hotspot hints below when helpful:
{mono_context}{region_section}
"""

# Template variations for quick A/B testing
ENHANCED_MONO_ANALYSIS_TEMPLATE_V5 = """You are analyzing a monochrome competition image that has been flagged by technical analysis for potential color issues.

CONTEXT:
- Image Title: "{title}" by {author}
{mono_context}{region_section}

YOUR TASK: Look at this image and describe exactly WHERE you see color contamination.

Be very specific about locations - examples:
- "particularly around the zebra's mane and ears"
- "in the subject's hair on the left side"
- "on the dental clinic storefront signage"
- "appears in the background shadows"

Focus on WHERE the color appears, then explain how it affects the monochrome competition eligibility."""

ENHANCED_MONO_ANALYSIS_TEMPLATE_V4 = """You are a photography competition judge analyzing whether an image meets monochrome requirements.

IMAGE: "{title}" by {author}

TECHNICAL ANALYSIS DATA:
{mono_context}{region_section}

COMPETITION RULES:
- "Pass": True monochrome, no color contamination
- "Pass with query": Minor color issues that need review
- "Fail": Significant color contamination

Looking at this image and the technical data, describe what you see and where color contamination appears."""


TRIPTYCH_HUE_ANCHORED_TEMPLATE = """You are auditing a MONOCHROME competition image for unwanted colour.

Panel A = original photo. Panel B = LAB residual (where/which hue). Panel C = LAB chroma (how strong).
Text facts below come from a deterministic analyser; trust them.

Only describe colour that is visible in Panels B/C and consistent with the facts.
Do NOT comment on composition or rules. Do NOT guess scene types, species or brands.
If the object/part is unclear, use a plain phrase like "background area", "sky", "hair", "fabric", "stone" or "(uncertain)".

Write 2–3 short sentences that a camera-club judge can read aloud:
• WHAT visible thing the colour sits on (object/part), and WHERE (natural words like "left edge", "along the waterline", "on the subject’s hair").
• WHICH colour family you see (simple families: red, orange, yellow, yellow-green, green, aqua, blue, violet, purple, magenta, pink).
• Optionally say if it’s concentrated or scattered.

If you cannot see any colour consistent with the overlays and the facts, respond exactly with: NONE

FILE: {file_name}
Verdict: {verdict}    Mode: {mode}
Dominant: {dominant_color} (hue {dominant_hue_deg:.1f}°)
Top hues (deg, weight): {top_hues_deg} / {top_weights}
Chroma: max {chroma_max:.2f}, p95 {chroma_p95:.2f}, ratio_2/4 {chroma_ratio_2_4:.2f}
Reason: {reason_summary}

Remember: base statements ONLY on areas indicated by Panels B/C and consistent with the above hues.
Output 2–3 plain sentences. If none seen, output NONE."""


REGION_FIRST_TEMPLATE = """Task: describe WHERE unwanted colour appears in this monochrome image.

Use ONLY the areas shown by the overlays and the region/grid hints.
Name the visible object/part if clear; otherwise use a simple, honest phrase ("background area", "foreground foliage", "window frame", "(uncertain)").
Do not infer scene type, species or brand names.
Do not mention rules or scoring. Keep to facts.

Write 2–3 short sentences, each of the form:
• "[Colour family] on [object/part][, optional brief locator]."
Optionally add "concentrated" or "patchy" if clearly visible in the chroma map.

If no colour consistent with the hints is visible, reply exactly: NONE

FILE: {file_name}
Verdict: {verdict}  Mode: {mode}
Dominant: {dominant_color} (hue {dominant_hue_deg:.1f}°)
Regions/Hints: {region_guidance}
Metrics: chroma_max {chroma_max:.2f}, p95 {chroma_p95:.2f}, ratio_2/4 {chroma_ratio_2_4:.2f}

Panels: A=original, B=LAB residual (where/which hue), C=LAB chroma (strength).
Base your sentences ONLY on those marked areas and the hints above.
Output 2–3 sentences in the requested style. If none seen, output NONE."""


TECHNICAL_ANALYST_TEMPLATE = """You are a precise technical image analyst for a photography competition. Your role is to describe colour faults in monochrome images factually and concisely.

Analyse the provided image, its overlays, and the deterministic metrics.
Provide a single paragraph (≤3 sentences) describing the location and hue of any colour contamination.

Deterministic Analysis:
{analysis_json}

Instructions:
- Reference the visual evidence in the original image and overlays to locate the colour.
- Use the hue names from the analysis as your source of truth.
- State where the colour appears (object + position). If widespread, describe the distribution.
- Do NOT comment on artistic merit or the pass/fail decision.

Report:"""


FACTUAL_REPORTER_TEMPLATE = """ROLE: You are an automated image analysis tool that identifies and locates colour faults. Respond only in the format requested below.

CONTEXT: A monochrome image has been flagged for colour contamination. The analysis has produced the following metrics:
{analysis_json}

INSTRUCTIONS:
- Examine the original image and LAB overlays.
- Identify up to two primary locations of colour contamination.
- For each location, provide the information in the exact bullet format below.

If the colour is a general cast not tied to a specific object, use the "General cast" format.

OUTPUT FORMAT:
- Location: [Describe the specific object and its position in the frame]
- Hue: [Select from the provided colour list]
- Distribution: [Describe if the colour is solid, soft glow, or patchy]

General cast format:
- Location: General cast
- Hue: [{dominant_color}]
- Distribution: [Describe if the cast is in the highlights, mid-tones, or shadows]
"""


TRIPTYCH_ANALYST_BRIEF_TEMPLATE = """You are a neutral analyst. Describe WHERE residual colour appears in a monochrome competition image.

Inputs:
• Panel A = original JPEG.
• Panel B = lab_residual overlay (hue direction; brighter/saturated = stronger tint).
• Panel C = lab_chroma overlay (colour intensity; Inferno colormap; bright ≈ stronger).
• Deterministic metrics from the analyser.

Rules:
• Describe only colour consistent with Panels B/C and visible in Panel A.
• Mention object/part + simple locator (e.g., "left edge", "near horizon").
• Name simple colour family: red, orange, yellow, green, aqua, blue, purple, magenta, pink.
• Optionally state "concentrated" or "patchy" when Panel C clearly shows it.
• No rules commentary, brands, or subject guesses. If unclear, use neutral wording ("background area", "fabric", "stone", "(uncertain)").

If no colour matches the overlays and metrics, reply exactly: NONE.

Write 2–3 short sentences (≤ 18 words each). Otherwise output NONE.

FILE: {file_name}    Title: {title}    Author: {author}
Verdict: {verdict}    Mode: {mode}
Dominant: {dominant_color} (hue {dominant_hue_deg:.1f}°)
Top hues (deg → weight): {top_hues_deg} → {top_weights}
Chroma: max {chroma_max:.2f}, p95 {chroma_p95:.2f}, ratio_2/4 {chroma_ratio_2_4:.2f}
ΔHue highs↔shadows: {delta_h_highs_shadows_deg:.1f}°
Reason: {reason_summary}

Panels: A=original, B=hue direction, C=intensity (Inferno; bright=stronger).
Describe only colour consistent with B/C and these metrics. 2–3 sentences. If none: NONE."""


REGION_HINT_BULLETS_TEMPLATE = """Task: State WHERE residual colour appears. Be factual and minimal.

Use:
• Panel A (original), Panel B (hue direction), Panel C (intensity).
• Deterministic metrics and optional region/grid hints.

Requirements:
• Mention the visible object/part and a brief locator only if obvious (e.g., "near left edge", "upper-right sky").
• Name the colour family (red/orange/yellow/green/aqua/blue/purple/magenta/pink).
• Optionally describe distribution ("concentrated" / "patchy") when Panel C clearly shows it.
• No discussion of rules, brands, or imaginative guesses. If unclear, use neutral phrases ("background area", "fabric", "stone", "(uncertain)").

If nothing consistent with overlays + metrics is visible, respond exactly: NONE.

Return 2–4 bullet points in the format:
• colour on object/part[, locator][. optional distribution]
Otherwise output NONE.

FILE: {file_name}    Verdict: {verdict}    Mode: {mode}
Dominant: {dominant_color} (hue {dominant_hue_deg:.1f}°)
Top hues: {top_colors} at {top_hues_deg} (weights {top_weights})
Chroma stats: max {chroma_max:.2f}, p95 {chroma_p95:.2f}, ratio_2/4 {chroma_ratio_2_4:.2f}
Hints (optional): {region_guidance}
Panels: B shows hue; C shows intensity (Inferno; bright=stronger).
Write 2–4 bullets. If none: NONE."""


FORENSIC_SPECIALIST_TEMPLATE = """ROLE: You are a forensic image analyst reporting to a photography judge. Describe residual colour faults factually and precisely.

You have:
• Panel A (original image)
• Panel B (hue direction overlay)
• Panel C (intensity map)
• Deterministic metrics (see below)

DETERMINISTIC DATA:
{analysis_json}

TASK & INSTRUCTIONS:
- Synthesize the evidence from all panels with the deterministic data.
- Only report hues listed in "top_colors".
- Use the original image to identify the object/texture; use overlays to confirm location/distribution.
- For each colour fault, describe object + frame position, hue family, and distribution (e.g., concentrated in highlights, patchy).
- If you cannot visually confirm any listed hue in the suggested areas, respond with the single word NONE.
- Do not comment on rules, composition, brands, or artistic merit. Keep the tone neutral and factual.

Provide a concise paragraph (≤3 sentences) or the word NONE."""


STRUCTURED_OBSERVER_TEMPLATE = """ROLE: You are an automated colour fault reporting system. Populate a structured report by correlating deterministic data with visual evidence.

REASONING STEPS:
1. Identify primary/secondary colour faults from `top_colors`.
2. Use lab_residual and lab_chroma overlays to find the brightest areas (likely contamination zones).
3. Inspect the original image at those locations to name the object/area.
4. Output results in the strict JSON-like format below.

If you cannot confirm any colour faults visually, respond with the single word NONE.

DETERMINISTIC METRICS:
{analysis_json}

OUTPUT SPECIFICATION:
[
  {
    "location": "[Object/Area and Position in Frame]",
    "hue": "[Colour name from top_colors]",
    "distribution": "[Concentrated/Patchy/Diffuse Cast]"
  }
]

Only list entries you can confirm. Return NONE if nothing is confirmed."""


@dataclass(frozen=True)
class PromptDefinition:
    """Describes a prompt template available to the CLI."""

    id: int
    name: str
    template: str
    supports_regions: bool
    description: str


DEFAULT_ENHANCED_TEMPLATE = ENHANCED_MONO_ANALYSIS_TEMPLATE_V6


PROMPT_LIBRARY: Dict[int, PromptDefinition] = {
    1: PromptDefinition(
        id=1,
        name="baseline_narration",
        template=COLOR_NARRATION_TEMPLATE,
        supports_regions=False,
        description="Concise competition summary referencing dominant colour and contamination metrics.",
    ),
    2: PromptDefinition(
        id=2,
        name="enhanced_v6",
        template=ENHANCED_MONO_ANALYSIS_TEMPLATE_V6,
        supports_regions=True,
        description="Current enhanced template with mono context and optional spatial guidance.",
    ),
    3: PromptDefinition(
        id=3,
        name="enhanced_v5",
        template=ENHANCED_MONO_ANALYSIS_TEMPLATE_V5,
        supports_regions=True,
        description="Location-focused narration emphasising where colour appears.",
    ),
    4: PromptDefinition(
        id=4,
        name="judge_v4",
        template=ENHANCED_MONO_ANALYSIS_TEMPLATE_V4,
        supports_regions=True,
        description="Competition-judge style reasoning with tonal discussion.",
    ),
    5: PromptDefinition(
        id=5,
        name="legacy_enhancement",
        template=MONO_DESCRIPTION_ENHANCEMENT_TEMPLATE,
        supports_regions=False,
        description="Legacy enhancement wording retained for comparison.",
    ),
    6: PromptDefinition(
        id=6,
        name="region_json",
        template=REGION_BASED_COLOR_ANALYSIS_TEMPLATE,
        supports_regions=True,
        description="Structured bullet + JSON response; expects explicit region lists.",
    ),
    7: PromptDefinition(
        id=7,
        name="triptych_hue_anchored",
        template=TRIPTYCH_HUE_ANCHORED_TEMPLATE,
        supports_regions=False,
        description="Overlay-grounded prompt tying narration to deterministic hue metrics.",
    ),
    8: PromptDefinition(
        id=8,
        name="region_minimal_locator",
        template=REGION_FIRST_TEMPLATE,
        supports_regions=True,
        description="Region-guided sentences with strict colour/location phrasing.",
    ),
    9: PromptDefinition(
        id=9,
        name="technical_analyst",
        template=TECHNICAL_ANALYST_TEMPLATE,
        supports_regions=False,
        description="Single-paragraph technical summary grounded in analyser metrics.",
    ),
    10: PromptDefinition(
        id=10,
        name="factual_reporter",
        template=FACTUAL_REPORTER_TEMPLATE,
        supports_regions=False,
        description="Structured bullet report with hue selected from analyser metrics.",
    ),
    11: PromptDefinition(
        id=11,
        name="triptych_analyst_brief",
        template=TRIPTYCH_ANALYST_BRIEF_TEMPLATE,
        supports_regions=False,
        description="Short sentence triptych analyst grounded in overlays and metrics.",
    ),
    12: PromptDefinition(
        id=12,
        name="region_hint_bullets",
        template=REGION_HINT_BULLETS_TEMPLATE,
        supports_regions=True,
        description="2–4 bullet summary leveraging optional region hints.",
    ),
    13: PromptDefinition(
        id=13,
        name="forensic_specialist",
        template=FORENSIC_SPECIALIST_TEMPLATE,
        supports_regions=True,
        description="Narrative forensic analyst synthesising panels and top colours.",
    ),
    14: PromptDefinition(
        id=14,
        name="structured_observer",
        template=STRUCTURED_OBSERVER_TEMPLATE,
        supports_regions=True,
        description="JSON-style structured observer with strict format.",
    ),
}


# Prompt ranking / observations:
#  - 9 (technical_analyst) — current default; best factual balance so far.
#  - 3 (enhanced_v5) — reliable fallback, location-first.
#  - 10 (factual_reporter) — structured bullets; good for parsing.
#  - 11/12/13/14 — new contenders; compare outputs to promote/demote.
#  - 7/8 — previous experiments with repetition issues; avoid unless improved.


CURRENT_PROMPT_ID = 9


def get_prompt_definition(prompt_id: int) -> PromptDefinition:
    """Return prompt definition for the given id (fallback to default)."""

    return PROMPT_LIBRARY.get(prompt_id, PROMPT_LIBRARY[CURRENT_PROMPT_ID])


def list_prompt_definitions() -> List[PromptDefinition]:
    """Return all registered prompt definitions."""

    return sorted(PROMPT_LIBRARY.values(), key=lambda p: p.id)


__all__ = [
    "PromptDefinition",
    "PROMPT_LIBRARY",
    "CURRENT_PROMPT_ID",
    "get_prompt_definition",
    "list_prompt_definitions",
    "REGION_BASED_COLOR_ANALYSIS_TEMPLATE",
    "MONO_DESCRIPTION_ENHANCEMENT_TEMPLATE",
    "MONO_INTERPRETATION_TEMPLATE",
    "COLOR_NARRATION_TEMPLATE",
    "ENHANCED_MONO_ANALYSIS_TEMPLATE_V6",
    "ENHANCED_MONO_ANALYSIS_TEMPLATE_V5",
    "ENHANCED_MONO_ANALYSIS_TEMPLATE_V4",
    "DEFAULT_ENHANCED_TEMPLATE",
    "TRIPTYCH_HUE_ANCHORED_TEMPLATE",
    "REGION_FIRST_TEMPLATE",
    "TECHNICAL_ANALYST_TEMPLATE",
    "FACTUAL_REPORTER_TEMPLATE",
    "TRIPTYCH_ANALYST_BRIEF_TEMPLATE",
    "REGION_HINT_BULLETS_TEMPLATE",
    "FORENSIC_SPECIALIST_TEMPLATE",
    "STRUCTURED_OBSERVER_TEMPLATE",
]

# Default template (can be changed for quick experimentation)
DEFAULT_ENHANCED_TEMPLATE = ENHANCED_MONO_ANALYSIS_TEMPLATE_V6
