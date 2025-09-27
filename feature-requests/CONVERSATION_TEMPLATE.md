# Feature Request Conversation Template

Use this flow whenever capturing a new idea. The aim is to gather enough structure to collaborate with an AI agent and resume later without rework.

---

## Step 1 – Inputs from requester
Checklist to collect up front:
- [ ] Short headline (1–2 sentences summarising the idea).
- [ ] Free-form notes describing desired change or outcome.

---

## Step 2 – Draft initial plan
Produce a lightweight, resumable plan containing:
1. **Goal (one line).**
2. **Why it matters.**
3. **Scope:** bullets for in-scope / out-of-scope.
4. **Must-have requirements:** 3–5 bullets.
5. **Acceptance checks:** objective tests to confirm success.
6. **Inputs / outputs:** files, fields, APIs touched.
7. **Design sketch:** 3–5 bullets covering the approach.
8. **Task breakdown:** small steps with rough sizes (XS/S/M/L).
9. **Savepoint block:** current status, next action, branch, files in flight.

---

## Step 3 – Clarifying questions
Ask the requester the following (adapt wording as needed):
1. What’s the specific intent or outcome you want to achieve?
2. Which existing ImageWorks components does this touch (eg CLI, narrator core, personal tagger, etc.)?
3. Any non-negotiables (eg APIs, file formats, performance, tooling)?
4. Does the change need to coexist with other modules or future expansions?
5. Environment constraints (eg GPU/VRAM, CUDA versions, offline requirements)?

adjust questions as needed given the initial inputs from the requestor and gaps in the initial plan

Wait for answers before refining the plan.

---

## Step 4 – Refine
After clarifications:
- Update requirements/acceptance criteria with new details.
- Expand the design sketch if needed.
- Adjust task sizes and add risks/mitigations.
- Capture effort estimate (t-shirt size or dev-days).

---

## Step 5 – Store artefacts
1. Save the final feature brief under `feature-requests/<slug>.md`.
2. Save the implementation plan alongside it (`feature-requests/<slug>-plan.md`).
3. Note any follow-up questions or dependencies in the plan’s Savepoint block.

---

## Step 6 – Interaction prompt (for AI agent)
When you want deeper assistance, share this standard prompt:
> “Read the feature brief and plan. Ask me concise questions about intent, dependencies, and non-negotiables before proposing solutions. Once clarified, suggest missing requirements, design options, test coverage, and task breakdowns—scaffolding only (no logic) until I confirm the plan.”

---

Keep this template handy so every feature request follows a predictable, collaborative path.
