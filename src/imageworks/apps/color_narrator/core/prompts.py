"""
Prompt templates for VLM-based mono analysis and color narration.
"""

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
