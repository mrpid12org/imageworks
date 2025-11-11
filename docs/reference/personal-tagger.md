# Personal Tagger Operations Guide

Personal Tagger generates captions, keywords, and descriptions for personal photo libraries. It supports multi-model workflows, deterministic registry integration, and metadata writes compatible with Lightroom and DAM tooling.

---
## 1. Feature Summary

| Capability | Details |
|------------|---------|
| Multi-stage prompting | Separate caption, keyword, and description prompts tuned for metadata quality. |
| Backend flexibility | Works with LMDeploy, vLLM, Ollama, or remote OpenAI-compatible APIs; supports per-stage model overrides. |
| Registry integration | `--use-loader` plus `--caption-role`, `--keyword-role`, etc. resolve models from deterministic registry. |
| Metadata writing | Writes IPTC/XMP captions, keywords, and custom namespaces. |
| Batch orchestration | Processes recursive directories with concurrency control, skipping previously-tagged files if desired. |
| Preset profiles | `prompt_profile` bundles instructions for the caption/keyword/description trio. |
| Judge Vision hand-off | One-click staging from the Personal Tagger GUI to the dedicated Judge Vision page for critiques. |
| Audit outputs | JSONL log per image, Markdown summary, and CLI console table. |
| GUI integration | Streamlit page for selecting inputs, toggling registry usage, and reviewing generated metadata before commit. |

---
## 2. Architecture

- `core.config`: loads defaults from `[tool.imageworks.personal_tagger]`, merges CLI overrides, handles environment variables.
- `core.runner.PersonalTaggerRunner`: orchestrates pipeline (preflight, batching, prompts, metadata writer).
- `core.prompts`: defines prompt profiles and templates for each stage.
- CLI entrypoint `apps.personal_tagger.cli.main`: Typer app with root callback (legacy compatibility), `run`, and `list-registry` commands.
- GUI page `5_🖼️_Personal_Tagger.py`: process runner, preview tables, and metadata diff viewer.

---
## 3. CLI Usage (`uv run imageworks-personal-tagger ...`)

### 3.1 Root invocation
- Running without subcommand mirrors `run` (legacy compatibility) while still accepting options.

### 3.2 `run`
- Key options include:
  - Input selection: repeatable `--input-dir`, `--recursive`, `--image-exts`.
  - Backend: `--backend`, `--base-url`, `--model`, `--api-key`, `--timeout`, `--max-new-tokens`, `--temperature`, `--top-p`.
  - Stage overrides: `--caption-model`, `--keyword-model`, `--description-model`, `--caption-role`, `--keyword-role`, `--description-role`.
  - Registry: `--use-loader`, per-stage registry model flags (`--caption-registry-model`, etc.).
  - Prompting: `--prompt-profile`.
  - Output: `--output-jsonl`, `--summary`, `--max-keywords`, `--batch-size`, `--max-workers`.
  - Metadata: `--dry-run`, `--no-meta`, `--backup-originals`, `--overwrite-metadata`.
  - Preflight: `--skip-preflight`, `--use-loader`, `--use-registry`.

### 3.3 Outputs
- JSONL log containing captions, keywords, descriptions, and metadata status.
- Markdown summary with per-folder counts and highlight excerpts.
- Console progress with Rich tables summarising each stage.

---
## 4. GUI Highlights

- **Input selection**: browse directories, toggle recursion, preview image counts.
- **Model selection**: choose between explicit backend/model combos or registry roles; displays resolved served id.
- **Prompt controls**: pick prompt profile and keyword limits.
- **Process runner**: executes CLI `run` command, streaming logs.
- **Result review**: table of generated metadata; supports “apply metadata” action when run in dry-run mode.
- **Preset management**: save frequently used configurations per photographer.

---
## 5. Configuration Defaults

Define in `[tool.imageworks.personal_tagger]`:
- Default directories, summary/JSONL paths.
- Backend defaults (backend/base-url/model for each stage).
- Prompt profile defaults and keyword limits.
- Metadata behaviour (overwrite, backups, dry-run default).

Environment overrides follow `IMAGEWORKS_PERSONAL_TAGGER__KEY=value` format.

---
## 6. Troubleshooting

| Symptom | Likely Cause | Resolution |
|---------|--------------|------------|
| Preflight failure | Backend unreachable or capabilities mismatch. | Use `--skip-preflight` during triage; verify backend endpoints and registry roles. |
| Metadata skipped | `--no-meta` true or file already tagged. | Remove flag or allow overwrite via `--overwrite-metadata`. |
| Keyword overflow | `max_keywords` too low. | Increase via CLI or config. |
| Registry lookup error | Role name missing in registry. | Run `imageworks-personal-tagger list-registry` to inspect available entries. |
| GUI preview empty | Run executed with `--no-meta`; enable metadata or load JSONL file via viewer. |

---
## 7. Best Practices

1. Maintain separate prompt profiles for different catalog styles (e.g. travel vs studio) to tailor tone.
2. Use registry roles for consistent backend selection across environments.
3. Keep JSONL logs for training future prompts and auditing metadata changes.
4. Use dry-run mode during prompt tuning; apply metadata only after reviewer approval.
5. Schedule periodic reviews of generated keywords to refine prompt templates and synonym lists.
