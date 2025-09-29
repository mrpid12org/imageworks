# Personal Tagger Model Options Discussion

## Purpose
- Track potential model/back-end combinations for the personal tagger pipeline (caption → keywords → description).
- Capture requirements, trade-offs, and reference notes while the deployment strategy is still fluid.
- Provide seed material that can later be formalised into the public docs.

## Open Questions
- Which serving backends (LMDeploy, vLLM, Triton, etc.) do we plan to support in production versus experiments?
- Do we expect to mix model families per stage (e.g., smaller keyword model, larger description model)?
- What latency/throughput targets should each stage meet for on-device or workstation usage?
- How much VRAM headroom do we need for parallel batches and future prompt variants?

## Candidate Matrix
| Stage       | Models of Interest                | Backend Options         | Early Thoughts |
|-------------|-----------------------------------|-------------------------|----------------|
| Caption     | Qwen2.5-VL 7B, LLaVA-NeXT 7B      | LMDeploy, vLLM          | Needs crisp subject/action naming; short outputs. |
| Keywords    | VLM JSON prompt, SigLIP embeddings| LMDeploy, embedding svc | JSON compliance and ranking stability matter most. |
| Description | Qwen2.5-VL 7B/32B, Idefics2 8B    | LMDeploy, vLLM          | Longer-form prose; evaluate token budgets vs. latency. |

## Evaluation Criteria
- Output quality: accuracy, avoidance of hallucinations, consistency across datasets.
- Runtime: warm start latency, throughput under 1–N concurrent images.
- Resource profile: VRAM usage with current prompt lengths; ability to quantise.
- Ecosystem: availability of open weights, licensing, community support.
- Tooling: compatibility with our prompt registry + model launcher scripts.

## Reference Notes
- Current defaults set in `pyproject.toml` (`[tool.imageworks.personal_tagger]`).
- Prompt registry located at `src/imageworks/apps/personal_tagger/core/prompts.py`.
- Shared VLM client utilities live under `src/imageworks/libs/vlm/`.
- Backend launcher script: `scripts/start_personal_tagger_backends.py` (see how stages map to servers).

## Next Steps
- Populate the candidate matrix with benchmark observations (latency, token cost, win/loss notes).
- Document any per-stage prompt tweaks or guardrails that emerge from experiments.
- Decide whether to formalise a shared model registry (narrator + tagger) once the picture stabilises.
