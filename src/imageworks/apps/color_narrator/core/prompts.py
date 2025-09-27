"""
Prompt templates for VLM-based mono analysis and color narration.
"""

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
ENHANCED_MONO_ANALYSIS_TEMPLATE_V6 = """Analyze this monochrome competition photograph for color contamination.

IMAGE: "{title}" by {author}

TECHNICAL CONTEXT:
{mono_context}{region_section}

INSTRUCTIONS:
Describe WHERE you see color contamination in this image. Be specific about:
1. Objects or parts that show color (e.g., "subject's hair", "background shadows")
2. How the color relates to the overall image
3. Whether it appears intentional (toning) or accidental (contamination)

Provide a natural, professional description suitable for competition documentation."""

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

# Default template (can be changed for quick experimentation)
DEFAULT_ENHANCED_TEMPLATE = ENHANCED_MONO_ANALYSIS_TEMPLATE_V6
