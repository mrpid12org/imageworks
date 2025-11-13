# Personal Tagger Runbook

Operational checklist for running the personal tagging pipeline.

---
## 1. Preparation

1. Organise input directories (e.g., per shoot) and ensure backups exist.
2. Decide on prompt profile and keyword limits for the batch.
3. Confirm chosen backend/registry models are online (use `imageworks-models select` for registry names).
4. Review metadata overwrite policy to avoid clobbering existing captions.

---
## 2. CLI Execution

1. Sample command:
   ```bash
   uv run imageworks-personal-tagger run \
     --input-dir /photos/to_tag \
     --recursive \
     --output-jsonl outputs/results/tagger_week12.jsonl \
     --summary outputs/summaries/tagger_week12.md \
     --backend vllm \
     --base-url http://localhost:8100/v1 \
     --use-registry \
     --caption-role caption --keyword-role keywords --description-role description \
     --prompt-profile competition \
     --max-keywords 30 --batch-size 4 --max-workers 8
   ```
2. Monitor output; ensure preflight passes and each stage logs success.
3. If running in dry-run mode (`--dry-run` or `--no-meta`), inspect JSONL before committing metadata.
4. For metadata application after dry-run, rerun with identical options but without `--no-meta`.
5. During preflight the runner now hits `/v1/models`, a text-only chat, and a 1×1 vision chat through the chat proxy. If Stage 2 (Judge Vision) is holding the GPU lease you may see transient 429s—rerun after the lease is released or point Personal Tagger at a different registry model.

---
## 3. GUI Workflow

1. Open Streamlit → “Personal Tagger”.
2. Select directories and recursion preference.
3. Choose backend/registry models (GUI will display resolved served IDs from the chat proxy).
4. Configure prompt profile, keyword limit, metadata toggle.
5. Execute run and watch progress.
6. Use results table to preview generated metadata; apply metadata if run was dry-run.

---
## 4. Handling Metadata

- Backups: enable `--backup-originals` (CLI) or GUI toggle to copy originals before writing.
- Overwrite rules: use `--overwrite-metadata` only when existing keywords should be replaced.
- Sidecars: configure in `pyproject.toml` if writing `.xmp` sidecars preferred.
- Post-run validation: spot-check in Lightroom or DAM; confirm keywords and captions present.

---
## 5. Troubleshooting

| Issue | Mitigation |
|-------|-----------|
| Preflight fails due to capability mismatch | Adjust roles or use explicit registry models that advertise the required capabilities (vision + gqa). |
| Preflight can’t reach `/v1/models` | Confirm `imageworks-chat-proxy` is running (or use `CHAT_PROXY_BASE_URL` env to point at remote). |
| API timeout | Increase `--timeout`, reduce `--batch-size`, or check backend load (Stage 2 GPU leases will pause Personal Tagger requests). |
| Prompt profile missing | Run `imageworks-personal-tagger --help` to confirm profile names; update configuration if custom profile removed. |
| Excessive runtime | Increase `--max-workers`, reduce image count per run, or narrow the directory scope. |
| Metadata not written | Ensure run not in dry-run mode and no permission issues on filesystem. |

---
## 6. Post-Run Activities

1. Archive JSONL/summary outputs with shoot deliverables.
2. Record prompt profile, registry logical name, and backend versions in changelog (the batch metrics file `outputs/metrics/personal_tagger_batch_metrics.json` now logs this metadata automatically).
3. Notify stakeholders (photographers, DAM admins) when new metadata ready for review.
4. Review randomly sampled images to ensure tone and keywords meet expectations; adjust prompts for next run if needed.
5. Update GUI preset if configuration changes should persist.
