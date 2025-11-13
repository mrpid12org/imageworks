# Judge Vision Runbook

## Overview

Judge Vision now runs as its own backend module (`imageworks-judge-vision`) rather
than reusing the Personal Tagger CLI. The dedicated runner keeps metadata writes
disabled, emits a progress JSON file for the Streamlit UI, and reuses the shared
compliance/technical/tournament helpers. The pipeline performs four coordinated stages:

1. **Stage 0 – Compliance Checks**
   * Validates image dimensions and policy hints (borders, manipulation, watermarks).
   * Competition-specific rules are loaded from a TOML registry file.

2. **Stage 1 – Technical Signals**
   * Lightweight heuristics compute mean luma, contrast, edge density, and saturation.
   * Summaries surface as prompt context and appear in the GUI for human review.

3. **Stage 2 – VLM Critique**
   * The `club_judge_json` prompt profile now returns 100–150 word critiques plus a direct
     0–20 score from the model. We record the raw VLM score (before rounding/clamping) and
     the rounded/clamped integer so the GUI can display both the judge narrative and the
     numeric total without re-deriving it from subscores.

4. **Stage 3 – Pairwise Playoff**
   * Optional post-processing for the ≥17 cohort. Colour and Mono entries each play ~5 head-to-head
     comparisons via the same VLM; we compute win ratios, promote the top ~15 % to 20, the next ~20 %
     to 19, leave the middle ~40 % unchanged, and demote the bottom ~20 % (of the ≥17 band) by one point.
     The GUI displays win ratios, wins/comparisons, and the adjusted scores.

Implementation note: shared compliance, technical-signal, pairwise, and rubric models now
live under `src/imageworks/apps/judge_vision/` so the CLI and GUI entry points call the same
logic.

## Quick Start

### Deterministic IQA prerequisites

Judge Vision expects both the MobileNet-based NIMA checkpoints *and* the MUSIQ TF-Hub
module to be present under the shared model store (the same tree used for HuggingFace/Qwen
weights). Run the helper once before judging so Stage 1 can load the deterministic metrics
without hitting the network:

```bash
uv run python scripts/download_judge_iqa_models.py
```

The script places the assets under `$IMAGEWORKS_MODEL_ROOT/weights/judge-iqa/` (NIMA in
`nima/`, MUSIQ inside `musiq/tfhub-cache/`).

### TensorFlow IQA Service (containerised inference only)

Stage 1 no longer tries to run the entire pipeline inside Docker. The host performs all orchestration,
while the TensorFlow-heavy MUSIQ/NIMA scoring runs inside a tiny NVIDIA container that exposes an HTTP API.
`docker-compose.chat-proxy.yml` now builds this helper as `imageworks-tf-iqa` (via `Dockerfile.tf-iqa`), so the service can stay warm alongside the chat proxy and vLLM executor.

1. **Prerequisites**
   * Docker with NVIDIA Container Toolkit so `docker run --gpus all …` works.
   * Access to `nvcr.io/nvidia/tensorflow:24.02-tf2-py3` (default image) or a compatible tag.
2. **Start/stop helpers**

   ```bash
   # Recommended: compose-managed service
   docker compose -f docker-compose.chat-proxy.yml up -d tf-iqa-service

   # Legacy helpers (still available for ad-hoc hosts)
   scripts/start_tf_iqa_service.sh   # pulls image if needed and starts judge-tf-iqa
   scripts/stop_tf_iqa_service.sh    # removes the container
   ```

   The service binds to `http://127.0.0.1:${JUDGE_VISION_TF_PORT:-5105}` and mounts the repo plus
   `${IMAGEWORKS_MODEL_ROOT}` so it can read cached weights. It installs only the missing dependencies
   (`numpy<2`, `tensorflow-hub`, `tomli`) and then serves `/health`, `/infer`, and `/shutdown` endpoints.

3. **How Stage 1 uses it**
   * When `--iqa-device gpu` is selected, `aesthetic_models.score_*` sends JPEG bytes to the service via
     `tf_container_wrapper`. If the HTTP call fails, the host automatically falls back to a one-off `docker run`.
   * The service encodes MUSIQ/NIMA outputs as JSON and never loads the rest of Judge Vision, so cv2/tonal
     dependencies stay on the host.
   * After Stage 1 finishes, the CLI posts to `/shutdown` (unless `JUDGE_VISION_TF_AUTO_SHUTDOWN=0`) so the
     container exits and releases VRAM before Stage 2 reloads vLLM.

4. **Operational notes**
   * GPU leases still flow through the chat proxy before Stage 1 starts; the lease is released whether IQA succeeds
     or not, and the TensorFlow service is shut down automatically so the Qwen/VLLM model can restart for Stage 2.
   * You can keep the service running for repeated IQA batches by setting `JUDGE_VISION_TF_AUTO_SHUTDOWN=0`.
   * To force a different image/registry/port, export `JUDGE_VISION_TF_IMAGE`, `JUDGE_VISION_TF_PORT`, or
     `JUDGE_VISION_TF_SERVICE_URL` (for remote hosts) before launching the GUI/CLI.
   * `scripts/start_tf_iqa_service.sh` automatically injects `PYTHONPATH`, `TFHUB_CACHE`, and `IMAGEWORKS_MODEL_ROOT
     so the container always finds the same weights as the host.

5. **CPU fallback**
   * Passing `--iqa-device cpu` (or losing the GPU lease) skips the service entirely and runs TensorFlow with the
     local wheel. This is handy on laptops or WSL installations without Docker but is slower than the GPU path.
   * When Compose is running you can keep `tf-iqa-service` up; the host simply bypasses it if CPU mode is selected.

Documenting both options here so we can revisit once all workstations have reliable GPU-enabled Docker access.

### Two-pass (IQA + Critique) workflow

You can now let the CLI/GUI chain both stages automatically:

```bash
uv run imageworks-judge-vision run \
  --input-dir ~/photos/competition \
  --stage two-pass \
  --iqa-device gpu \
  --iqa-cache outputs/cache/judge_vision_iqa.jsonl \
  --competition-config configs/competitions.toml \
  --competition club_open_2025 \
  --output-jsonl outputs/results/judge_vision.jsonl \
  --summary outputs/summaries/judge_vision_summary.md
```

Stage 1 writes MUSIQ/NIMA/tonal metrics for every image (see `outputs/cache/judge_vision_iqa.jsonl`)
and streams status updates into `outputs/metrics/judge_vision_progress.json`. Once IQA completes, the CLI
verifies the cache exists and immediately starts Stage 2 (critique-only) using the same cache path.
Set `--iqa-device gpu` to keep TensorFlow on the GPU; switch to `cpu` if VRAM is limited.
When GPU mode is enabled, the CLI now asks the chat proxy for a temporary **GPU lease**: the proxy pauses
any running vLLM model, issues a keep-alive unload to Ollama (since `/api/stop` never existed), and only then hands the lease to TensorFlow.
As soon as Stage 1 finishes, the lease release triggers `/shutdown` on `tf-iqa-service` and tells the vLLM admin service to reload the previous model.
This works for both the CLI and GUI so you never need to manually stop/start Qwen just to run deterministic IQA.

#### Watching the hand-off
- `logs/judge_vision.log` – look for `GPU lease granted` / `GPU lease released`.
- `docker logs imageworks-chat-proxy` – confirms `[ollama-manager] Requesting unload…` and `[vllm-manager] Starting model …` lines.
- `docker logs imageworks-vllm` – shows `/admin/activate` handling plus per-model stdout/stderr.
- `nvidia-smi` – VRAM usage should drop when Stage 1 acquires the lease and spike again when Stage 2 starts.

> Prefer the GUI? Pick **Execution stage → Two-pass (auto IQA → Critique)** and the Streamlit runner will launch
> a single command that performs both passes sequentially. The progress indicator resets automatically between
> stages so you can monitor the full run without manually reloading anything.

You can still run the stages manually when needed, or interrogate the new admin endpoints:

1. **IQA stage** (CPU or GPU):

   ```bash
   uv run imageworks-judge-vision run \
     --input-dir ~/photos/competition \
     --stage iqa \
     --iqa-device gpu \
     --iqa-cache outputs/cache/judge_vision_iqa.jsonl
   ```

2. **Critique stage** (reuses cache):

   ```bash
   uv run imageworks-judge-vision run \
     --stage critique \
     --iqa-cache outputs/cache/judge_vision_iqa.jsonl \
     --competition-config configs/competitions.toml \
     --competition club_open_2025 \
     --output-jsonl outputs/results/judge_vision.jsonl \
     --summary outputs/summaries/judge_vision_summary.md
   ```

1. Create a competition registry (see example below) and point the CLI at it:

   ```bash
   uv run imageworks-judge-vision run \
     --input-dir ~/photos/competition \
     --competition-config configs/competitions.toml \
     --competition club_open_2025 \
     --output-jsonl outputs/results/judge_vision.jsonl \
     --summary outputs/summaries/judge_vision_summary.md \
     --pairwise-rounds 3
   ```

   Pairwise rounds run a Swiss-style head-to-head comparison using the cached Stage 2 critiques to produce a ranked shortlist (plus stability metrics). Set `--pairwise-rounds` to a positive integer (we recommend `4`) to enable the playoff; leave it at `0` (the default) to skip Pass 3 entirely. Use the new `--stage pairwise` mode when you want to rerun only the playoff against an existing JSONL without recomputing IQA or critiques.

> ℹ️ The CLI writes incremental status to
> `outputs/metrics/judge_vision_progress.json`; the ⚖️ Judge Vision Streamlit page
> now runs the CLI asynchronously so the progress bar refreshes while the run is in-flight (and resets between
> the IQA and critique phases when two-pass mode is enabled).
> The GUI also validates that `--competition-config` points to an existing TOML file. Create it under
> `configs/competitions.toml` (or update the path in the UI) before running.
>
> 🆕 Stage 1 now also appends deterministic IQA metrics to
> `outputs/cache/judge_vision_iqa.jsonl`, making it easy to monitor NIMA/MUSIQ
> scores even before the critiques finish. Re-running the VLM stage can reuse this
> cache in upcoming multi-pass workflows.

2. Review results in Streamlit: the ⚖️ page shows the progress indicator while the
   run executes, and the 📊 Results browser (Judge Vision module) picks up the default
   JSONL/summary artifacts automatically once the run finishes.

3. Exported JSON records now include:
  * `critique_subscores` (impact, composition, technical)
   * `critique_total`, `critique_award`, and `critique_compliance_flag`
   * `technical_signals.metrics` and compliance `issues`/`warnings`
* Optional Pass 3 `pairwise` rounds/final rankings/stability metrics

## Competition Registry Example

```toml
[competition.club_open_2025]
categories = ["Open", "Nature"]
rules = { max_width = 1920, max_height = 1200, borders = "disallowed", watermark_allowed = false }
awards = ["Gold", "Silver", "Bronze", "HC", "C"]
score_bands = { Gold = [19, 20], Silver = [18], Bronze = [17], HC = [16], C = [15] }
pairwise_rounds = 4
```

## GUI Enhancements

* Final scores (0–20) are editable via number inputs.
* Award suggestions and compliance flags can be edited or cleared.
* Compliance summaries and technical priors render as captions for quick scanning.
* Compact summaries include total score, award, and compliance status per image.

## GUI Workflow

1. **Stage from Personal Tagger** – The Personal Tagger page now shows a
   “Send This Batch to Judge Vision” button inside the Configure tab. It copies
   the current selection, output paths, and competition hints into
   `st.session_state["judge_prefill"]`.
2. **⚖️ Judge Vision Page** – The Brief & Rules tab now reads the TOML
   competition registry and turns each `[competition.*]` entry into a preset.
   Selecting a competition populates pairwise rounds, default category,
   critique template, and judge notes automatically. Only one selector is shown—the
   registry replaces ad-hoc presets. A Pass 3 control, located directly under the IQA device drop-down,
   lets you keep the playoff disabled (`0`) or dial it up (e.g., `4`) whenever you want the tournament overlay.
3. **Run Pipeline** – Executes `imageworks-judge-vision run` in dry-run/no-meta
   mode so metadata never changes during judging. The live progress tracker feeds
   `st.progress` and the current filename.
4. **Results** – Streams the JSONL + Markdown artifacts to show rubric tables,
   compliance flags, awards, and tournament stability metrics. The output paths
   match what the 📊 Results page expects, so Judge Vision exports appear there
   without additional configuration.
5. **Review & Iterate** – The JSONL viewer highlights totals/awards per image while the
   pairwise panel surfaces stability metrics, making it obvious when another run is needed.

## Outputs

* **JSONL** – `outputs/results/judge_vision.jsonl`; each record follows schema
  version `1.1` and embeds compliance, technical, and pairwise reports for downstream analytics.
* **Summary Markdown** – `outputs/summaries/judge_vision_summary.md` adds a “Pairwise Tournament”
  section with match logs, final rankings, and stability metadata.
* **Progress JSON** – `outputs/metrics/judge_vision_progress.json` updates after every image,
  enabling the GUI progress indicator and providing a hook for future automation. The Streamlit
  runner now keeps the CLI process detached so this file refreshes live instead of only at completion.
* **IQA Cache** – `outputs/cache/judge_vision_iqa.jsonl` stores deterministic per-image IQA metrics,
  tonal summaries, and timestamps. Stage 2 consumes this file automatically; advanced users can
  inspect MUSIQ/NIMA outputs mid-run or feed them into downstream analytics.

## Next Steps

* Integrate multi-judge ensembles and anchor analysis for bias control.
* Extend analytics dashboards to visualise rankings and drift over time.
* Build the human judge toolkit (timers, crib sheets, JSON export) atop the new schema.
