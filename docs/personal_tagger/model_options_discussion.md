# Personal Tagger Model Options Discussion

## Purpose
- Track potential model/back-end combinations for the personal tagger pipeline (caption → keywords → description).
- Capture requirements, trade-offs, and reference notes while the deployment strategy is still fluid.
- Provide seed material that can later be formalised into the public docs.

## Open Questions
- Which serving backends (LMDeploy, vLLM, Triton, etc.) do we plan to support in production versus experiments?
- Do we expect to mix model families per stage (e.g., smaller keyword model, larger description model)?
- What latency/throughput targets should each stage meet for on-device or workstation usage?
- How much VRAM headroom do we need for parallel batches and future prompt variants? (Quantised AWQ/INT4 checkpoints on Hugging Face keep 7B models within 8–10 GB VRAM.)

## Candidate Matrix
| Stage       | Models of Interest                | Backend Options         | Early Thoughts |
|-------------|-----------------------------------|-------------------------|----------------|
| Caption     | Qwen2.5-VL 7B AWQ, LLaVA-NeXT 7B  | LMDeploy, vLLM          | Needs crisp subject/action naming; short outputs; both have community AWQ builds. |
| Keywords    | VLM JSON prompt, SigLIP embeddings| LMDeploy, embedding svc | JSON compliance and ranking stability matter most; SigLIP-384 and EVA-CLIP checkpoints ship quantised FP16/INT8 variants. |
| Description | Qwen2.5-VL 7B/32B, Idefics2 8B    | LMDeploy, vLLM          | Longer-form prose; track token budgets vs. latency; 32B fits on 96 GB in BF16. |

## Evaluation Criteria
- Output quality: accuracy, avoidance of hallucinations, consistency across datasets.
- Runtime: warm start latency, throughput under 1–N concurrent images.
- Resource profile: VRAM usage with current prompt lengths; ability to quantise or load AWQ checkpoints from Hugging Face.
- Ecosystem: availability of open weights, licensing, community support.
- Tooling: compatibility with our prompt registry + model launcher scripts.

## Reference Notes
- Current defaults set in `pyproject.toml` (`[tool.imageworks.personal_tagger]`).
- Prompt registry located at `src/imageworks/apps/personal_tagger/core/prompts.py`.
- Shared VLM client utilities live under `src/imageworks/libs/vlm/`.
- Backend launcher script: `scripts/start_personal_tagger_backends.py` (see how stages map to servers).

## Next Steps
- Populate the candidate matrix with benchmark observations (latency, token cost, win/loss notes).
- Prototype SigLIP-based keyword ranking and capture results.
- Decide whether to formalise a shared model registry (narrator + tagger) once the picture stabilises.
- Gather public AWQ/INT4 checkpoints for preferred models to simplify deployment experiments.
