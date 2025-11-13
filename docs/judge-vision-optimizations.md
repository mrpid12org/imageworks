# Judge Vision Runtime Optimization Notes

This document captures potential performance improvements for Judge Vision and
the instrumentation required to measure them. Treat this as a working plan when
we revisit runtime tuning.

## 1. Optimization Opportunities

### Stage 1 – Deterministic IQA
- **Parallel MUSIQ/NIMA calls**: drive the TensorFlow service with a small
  thread pool (or asyncio) so multiple images are in-flight while the GPU waits.
- **Cache fidelity**: hash image path + size + mtime so renamed-but-unchanged
  files hit the IQA cache. Skip recompute for existing entries.
- **Optional fast mode**: allow users to disable MUSIQ or tonal metrics when
  they only need a quick pass.

### Stage 2 – VLM Critique
- **Lower `max_new_tokens`**: measure actual output length; likely we can cut
  from 512 to ~320 tokens without hurting 110–150 word critiques.
- **Compact prompt**: trim redundant text in the TECHNICAL ANALYSIS block to
  shorten context ingestion.
- **Prefix caching**: move more static instructions into the `system` message so
  vLLM reuses cached KV blocks.
- **Pipelined requests**: overlap network round-trips by submitting the next
  request as soon as the previous one enters decoding. Keeps the CPU busy even
  though we won’t batch on the GPU.

### Stage 3 – Pairwise Playoff
- **Adaptive comparisons**: reduce `comparisons_per_image` for large cohorts or
  switch to Swiss-style pairings (e.g., only round robin when ≤ 8 eligible
  entries).
- **Image encoding cache**: avoid re-encoding the same JPEG twice by caching
  Base64 strings for the duration of Stage 3.

### General Pipeline
- **Buffered JSONL writes**: flush every N records instead of per image to cut
  disk I/O overhead.
- **Resume support**: detect partial Stage 2 results and skip straight to
  unfinished images when rerunning after a crash.
- **Concurrency toggle**: expose `--concurrency` (and GUI switch) so advanced
  users can experiment with overlapping Stage 2 requests when resources allow.

## 2. Metrics & Telemetry Plan

To make data-driven decisions we need consistent, per-run telemetry.

### 2.1 Run Manifest
Write `judge_vision.metadata.json` next to each JSONL with:
- Git commit hash + dirty flag
- Prompt profile ID and SHA256 hash of the prompt text
- CLI/GUI parameters (backend, `max_new_tokens`, temperature, concurrency, etc.)
- Start/end timestamps per stage and total wall-clock duration
- GPU snapshot (device name, VRAM total) at run start

### 2.2 Per-Image Timing Fields
Augment each JSONL record with:
- `stage1_duration_seconds`
- `stage2_duration_seconds`
- Optional `stage3_duration_seconds` for pairwise matches
- `timestamp_stage2_started` / `timestamp_stage2_finished`

### 2.3 Stage-Level Aggregates
Track during the run:
- Count, min/max/average duration per stage
- Pairwise metrics (comparisons scheduled, completed, average time per category)
- Number of cache hits/misses for IQA

Dump the summary into the manifest so we can read it without parsing the whole
JSONL.

### 2.4 GPU Utilization Sampling
Sample GPU stats (VRAM used, utilization %) every N images during Stage 2 and
store the readings in the manifest. Lightweight `nvidia-smi --query` runs are
enough.

### 2.5 Logging Enhancements
Emit structured log lines per stage event, e.g.:
```json
{ "event": "stage2_request", "image": "foo.jpg", "latency": 16.2, "tokens": 380 }
```
This keeps `logs/judge_vision.log` useful for postmortems.

### 2.6 GUI Performance Tab
Add a fourth tab that reads the manifest + JSONL and shows:
- Run header (duration, images, prompt hash, commit)
- Stage timeline bar chart
- Distribution of per-image Stage 2 durations
- Pairwise stats and GPU utilization sparkline
- Download buttons for manifest and raw metrics

### 2.7 Prompt/Version Tracking
Until we integrate the full prompt manager, hashing the prompt text is enough to
tie runs to a specific wording. Longer term we can adopt the Personal Tagger
prompt registry so runs reference a named prompt version.

---

With the metrics above in place we can benchmark the baseline, implement the
optimizations incrementally, and validate whether each change meaningfully
reduces Stage 2 latency or overall wall-clock time.***
