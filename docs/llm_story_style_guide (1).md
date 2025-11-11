# 🧭 Directing an LLM to Write Stories in a Particular Style
*A unified guide combining best practices from narrative craft and prompt design.*

---

## 1. Core Principles

1. **Clarity beats quantity** – An LLM performs best when guided by *a few well-defined, non-contradictory* constraints.
2. **Iterate, don’t overload** – Build the story in rounds: concept → tone → structure → polish.
3. **Anchor, don’t dictate** – Stylistic imitation (“in the style of X”) works better when you supply *a short sample* or *a few descriptive adjectives* than by naming many authors at once.
4. **Be diagnostic between drafts** – After each version, tell the model *what worked and what didn’t* (“the tone is right, but the dialogue feels flat; make it more subtext-driven”).
5. **Preserve consistency** – Keep POV, tense, and style stable unless deliberately changed.

---

## 2. The Story-Style Control Menu

Use this list as a *planning and iterative reference*, **not** as a single prompt dump.

| # | Parameter | Purpose | Examples |
|---|------------|----------|-----------|
|1|**Genre / Sub-genre**|Defines tropes, pacing, and reader expectations.|Cyber-punk noir; Gothic romance; pastoral fantasy.|
|2|**Target Audience / Age Group**|Shapes diction, complexity, and permitted themes.|Middle-grade; adult literary; YA thriller.|
|3|**Tone / Mood**|Establishes emotional color.|Melancholic, wry, elegiac, absurdist.|
|4|**Narrative Voice & POV**|Selects grammatical person and psychic distance.|First-person unreliable; third-person limited; omniscient narrator.|
|5|**Length / Scope**|Controls depth and focus.|≈1,000 words; novella opening; 4-scene vignette.|
|6|**Structure / Temporal Design**|Dictates rhythm and chronology.|Three-act arc; in-media-res with flashbacks; epistolary log.|
|7|**Characters**|Anchors motivation and conflict.|“Maya, a teenage hacker, and Echo, a rogue AI.”|
|8|**Setting**|Supplies sensory and sociopolitical context.|Neo-Tokyo 2145; post-industrial marshland.|
|9|**Central Conflict / Stakes**|Drives plot momentum.|Expose the AI vs protect her father.|
|10|**Theme / Moral Question**|Adds intellectual unity.|The cost of progress vs human connection.|
|11|**Stylistic Cues**|Micro-level texture: rhythm, imagery, syntax.|Short punchy lines; tactile metaphors; clipped dialogue.|
|12|**Reference Author / Period**|Guides voice mimicry.|“In the style of Octavia E. Butler’s clarity and restraint.”|
|13|**Constraints / Prohibitions**|Eliminates clichés or tone breaks.|No explicit gore; avoid “it was all a dream.”|
|14|**Desired Ending / Payoff**|Sets closure expectations.|Ambiguous; tragic irony; redemptive.|
|15|**Optional Devices**|Adds structural flourishes.|Foreshadowing, parallelism, unreliable narration.|

*(Sixteen compressed to fifteen — merged “voice” and “POV” for clarity.)*

---

## 3. Recommended Workflow (Iterative Coaching)

### **Stage 1 – Core Frame (“The Big 5–7”)**
Start small: genre, tone, voice/POV, protagonist, setting, conflict, length.

> **Prompt 1:**
> “Write a ~1,200-word cyber-punk noir.
>  **Voice:** first-person, cynical but lyrical.
>  **Protagonist:** Maya, 17-year-old hacker confronting her estranged father and a rogue AI called *Echo*.
>  **Setting:** Neo-Tokyo 2145, neon rain.
>  **Conflict:** expose Echo or save her father.
>  **Tone:** melancholic.
>  End ambiguously.”

Request this first draft as an **outline or synopsis**, not full prose:
> “Before writing, outline the three-act structure and emotional beats.”

---

### **Stage 2 – Stylistic Refinement**
Once the skeleton works, tune the language and theme.

> **Prompt 2:**
> “Revise the story to:
>  1. Deepen the melancholic tone; show sensory detail (smell of ozone, hum of servers).
>  2. Keep sentences under ~15 words.
>  3. Reinforce the theme *‘the cost of progress vs human connection’.*”

You can paste a short exemplar paragraph (“Here’s the rhythm I want…”) for stronger anchoring.

---

### **Stage 3 – Structural Experimentation**
Manipulate chronology, tension, and reveal ordering.

> **Prompt 3:**
> “Rewrite so it opens *after* her decision, then reveal how she got there through flashbacks. Keep the ending ambiguous.”

---

### **Stage 4 – Polishing and Consistency Checks**
Ask for targeted line-edits.

> “Tighten dialogue; remove redundancies; ensure consistent first-person POV; vary sentence rhythm.”

Optionally request *authorial emulation*:
> “Infuse it with Bradbury-like sensory lyricism, but maintain noir brevity—short description bursts between clipped dialogue.”

---

## 4. Advanced Prompting Techniques

| Technique | Purpose | Example |
|------------|----------|----------|
|**Style Priming**|Feed 1–2 paragraphs of reference prose before writing.|“Here’s a paragraph from Chandler. Match its cadence.”|
|**Few-Shot Tone Anchors**|Provide 2–3 sample sentences of your desired tone.|“Write future sentences like these.”|
|**Iterative Memory Prompting**|Paste back the previous version, then give numbered revision goals.|“Using the text below, revise according to items 1-3.”|
|**Critique-and-Rewrite Loop**|Ask the model to first critique its own draft, then apply those notes.|“List three issues with pacing, then fix them.”|
|**Hybrid Temperature Control**|If your interface allows, run low-temperature (≤ 0.5) passes for structure, then high-temperature (0.8–1.0) passes for stylistic color.|
|**External Style Tokenizing**|Tag exemplar adjectives for reuse: `[tone:grim][syntax:staccato][imagery:chromatic]` — models often retain these tags internally.|

---

## 5. Common Pitfalls

- **Conflicting directives** (e.g., “first-person” + “third-person limited”).
- **Over-specification** – too many style or author references muddy the signal.
- **Neglecting iteration** – expecting a single prompt to achieve literary nuance.
- **Inconsistent tense or register** – always restate desired tense if it drifts.
- **Losing focus on theme** – remind the model of the thematic axis in each round.

---

## 6. Example Full Workflow Summary

1. **Define core concept** → genre, tone, POV, protagonist, conflict.
2. **Ask for outline** → confirm story logic.
3. **Generate first draft.**
4. **Critique & refine** → tone, style, imagery.
5. **Experiment with structure / ending.**
6. **Polish and proof** → pacing, dialogue, sensory coherence.
7. **Optionally request title, blurb, or logline** in the same style for consistency.

---

## 7. Optional Integration Appendix

### Manual or Chat-based Use
Use the checklist interactively in any chat interface. Begin with the **core frame**, then progressively refine tone, pacing, and structure. Keep prompts short and explicit.

### Programmatic Workflow (Scripted or GUI)
If integrated into code or a GUI (e.g., Streamlit, Gradio, or CLI):
1. **Core configuration inputs:** dropdowns or text fields for *genre, tone, POV, theme, style cues, and target length.*
2. **Generate outline first.** Capture the returned structure in memory/state.
3. **Stage refinement passes:** send successive instructions (Stage 2–4 above) automatically or interactively.
4. **Output display:** side-by-side comparison between drafts for iterative selection.

### Suggested Module Pattern (Pseudocode)
```python
story = llm.generate(prompt=core_prompt)
for refinement in refinements:
    story = llm.generate(prompt=f"Revise this story according to: {refinement}\n\n{story}")
```

This architecture stays model-agnostic and works with OpenAI, Ollama, or vLLM endpoints.

---

## 8. Extracting Style from Example Stories

### 1) Curate Inputs
- Provide 1–3 short stories or excerpts that exemplify your desired voice.
- Choose passages that show tone, rhythm, and character work.

### 2) Extract a Style Card (Analysis Pass)
Ask an LLM to analyze—**not imitate**—the stories and output a structured style description.

**Prompt (Style Card Extractor)**
> You are a literary analyst. Read the passages below and produce a **Style Card** capturing the *transferable* stylistic features without naming or copying.
> Return JSON with keys for voice_register, tense, sentence_rhythm, diction, imagery_devices, dialogue_style, pacing_profile, structure_moves, characterization_methods, theme_axes, constraints, positive_checklist, negative_checklist, and style_examples.

### 3) Audit & Refine the Style Card
Ask the same or another model to check for contradictions or derivative phrases and revise the card for operational clarity.

### 4) Build a Portable Generation Brief
Convert the Style Card into a *Generation Brief* containing:
- Core parameters (genre, audience, POV, tense, length)
- Tone & diction rules
- Structural expectations
- Do/Avoid checklists
- Motifs & imagery anchors
- Outline-first instruction
- Revision protocol

### 5) Use Two LLM Roles (Optional)
- **LLM-A:** Analysis and brief building
- **LLM-B:** Story generation and revision

This separation minimizes stylistic contamination, but the same model can handle both roles sequentially.

### 6) Add Measurable Constraints
Embed objective metrics such as:
- Average sentence length (12–16 words)
- One tactile or auditory image per scene
- Scene break every 400–600 words
- Limit two dialogue tags per exchange

### 7) Quality Gates
Before finalizing a draft:
- Check POV/tense consistency.
- Self-score against the do/avoid checklist.
- Ask the model to paraphrase any suspiciously derivative lines.

### 8) Prompt Bundle Summary
- **A)** Style Card Extractor (JSON)
- **B)** Consistency & Originality Audit
- **C)** Generation Brief Builder
- **D)** Drafting Loop (Outline → Draft → Revise)

### 9) When to Use Alternatives
- For high fidelity across many stories → fine-tune or LoRA on style corpus.
- For world consistency → add RAG priming with style card + glossary.
- For copyright caution → use style cards only, never raw text.

### 10) Pitfalls & Mitigations
- **Conflicting specs:** run audit passes to resolve POV/tense.
- **Over-imitation:** enforce author-agnostic phrasing.
- **Drift:** insert mid-draft checkpoints for corrections.
- **Energy loss:** re-inject tone anchors before final pass.

---

### 📚 References & Influences

- ChatGPT / Claude prompt-engineering papers on iterative refinement.
- James Wood, *How Fiction Works* (2008).
- Ursula K. Le Guin, *Steering the Craft* (1998).
- Robert McKee, *Story* (1997).
- Blake Snyder, *Save the Cat!* (2005).

---

### ✅ Summary Takeaway

> **Use the 16-point list as a *planning map*, not a single mega-prompt.**
> Build a reusable *Style Card* from sample stories, audit it for clarity, and feed it into a clean *Generation Brief*.
> Whether manual or automated, this layered process yields a stable, distinctive storytelling voice without style drift or imitation risk.
