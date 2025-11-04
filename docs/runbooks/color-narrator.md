# Color Narrator Runbook

Operational guide for running the colour narration pipeline with both CLI and GUI entry points.

---
## 1. Preconditions

1. Mono checker run completed with JSONL + overlays ready.
2. Target VLM backend reachable (vLLM/LMDeploy/Triton/Ollama). Test with `uv run imageworks-color-narrator diagnose-backend`.
3. ExifTool installed if metadata writing enabled.
4. `[tool.imageworks.color_narrator]` defaults reviewed for backend URLs and prompt ids.

---
## 2. CLI Batch Execution

1. Command template:
   ```bash
   uv run imageworks-color-narrator narrate \
     --images /path/to/jpegs \
     --overlays /path/to/overlays \
     --mono-jsonl outputs/results/mono_results.jsonl \
     --summary outputs/summaries/narrate_summary.md \
     --results-json outputs/results/narrate_results.jsonl \
     --prompt 3 --vlm-backend lmdeploy --vlm-model "Qwen2.5-VL-7B-AWQ" \
     --batch-size 4
   ```
2. Monitor CLI output for per-image narration and metadata status lines.
3. Inspect summary Markdown for outliers (low confidence or missing metadata).
4. Archive JSONL results alongside mono checker outputs.

---
## 3. GUI Workflow

1. Launch Streamlit GUI and open **Color Narrator** page.
2. Select preset (e.g., “LMDeploy Narration”).
3. Confirm directories auto-populated from last mono run.
4. Toggle metadata/regions/backends as required.
5. Click **Run Narration**; observe progress and backend latency charts.
6. Review results table, filter by **Needs review** (metadata errors) if any.

---
## 4. Prompt & Backend Validation

- List prompts: `uv run imageworks-color-narrator narrate --list-prompts` (exits immediately).
- Preview prompt: `uv run imageworks-color-narrator preview-prompt --prompt 2 --regions`.
- Backend health: `uv run imageworks-color-narrator diagnose-backend --vlm-backend vllm --vlm-base-url http://localhost:8000/v1`.

---
## 5. Metadata Handling

1. To disable metadata temporarily: add `--no-meta` (CLI) or disable toggle in GUI.
2. To write sidecars: use `--no-meta` false plus `--metadata-sidecar` (config-driven) or adjust config.
3. Verify `narrate_results.jsonl` includes `metadata_written: true` for successful rows.
4. If errors occur, rerun with `--no-meta` to capture narratives while investigating ExifTool issues.

---
## 6. Error Handling

| Symptom | Mitigation |
|---------|-----------|
| Backend timeout | Increase `--vlm-timeout` or scale `batch-size` down; check backend logs. |
| HTTP 401/403 | Provide `--vlm-api-key` or fix API credentials. |
| Missing overlays | Use `--allow-missing-overlays` if overlays optional, or regenerate from mono checker. |
| Region prompts misaligned | Ensure overlays match image dimensions; rerun mono checker to regenerate heatmaps. |
| Metadata collisions | Set `--no-meta`, manually inspect existing keywords before retrying with metadata on. |

---
## 7. Post-run Checklist

- [ ] Markdown summary archived with competition batch.
- [ ] JSONL results stored in repository or blob storage.
- [ ] Backend usage metrics reviewed (BatchRunMetrics output at end of CLI run).
- [ ] GUI presets updated if configuration changed.
- [ ] Any low-confidence narrations flagged to judges.

