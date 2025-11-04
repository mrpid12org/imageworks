# Model Downloader Runbook

Use this runbook for day-to-day ImageWorks downloader operations. Each section highlights both CLI commands (`uv run imageworks-download ...`) and the equivalent GUI workflow (`2_🎯_Models.py`). Refer to [Model Downloader Operations Guide](../reference/model-downloader.md) for deep context.

---
## 1. Before You Start
- ✅ Install `aria2c` and ensure the ImageWorks environment is active.
- ✅ Confirm the layered registry files (`configs/model_registry.curated.json`, `.discovered.json`) are under version control.
- ✅ Set environment overrides (model roots, LM Studio paths, logging) when deviating from defaults.
- ✅ Keep `logs/model_downloader.log` for audit trails; GUI actions emit the same structured log.

---
## 2. Quick Reference Matrix

| Task | CLI | GUI |
|------|-----|-----|
| Download HF model | `download` | 📥 **Download & Import → 🌐 Download** |
| Scan local weights | `scan` | 📥 **Download & Import → 📁 Scan** |
| Import Ollama store | `python scripts/import_ollama_models.py` | 📥 **Download & Import → 🦙 Import Ollama** |
| Normalise metadata | `normalize-formats` | 🔧 **Registry Maintenance → 🔄 Normalize** |
| Purge / cleanup | `purge-*`, `prune-duplicates`, `reset-discovered`, `restore-ollama`, `backfill-ollama-paths` | 🔧 **Registry Maintenance → 🗑️ Purge / 🔨 Cleanup** |
| Verify installs | `verify [--fix-missing]` | ⚙️ **Advanced Operations → ✅ Verify** |
| Remove / purge | `remove [--delete-files] [--purge] --force` | ⚙️ **Advanced Operations → 🗑️ Remove** |
| Inspect registry | `list`, `list-roles`, `stats`, `config` | 📚 **Browse & Manage** (table + stats) |

---
## 3. Standard Operating Procedures

### 3.1 Acquire a model (HF or direct URL)
1. **CLI**
   ```bash
   uv run imageworks-download download "owner/repo[@branch]" \
     --format awq --location linux_wsl \
     --weights "<file1,file2>" --support-repo other/repo \
     --include-optional --non-interactive
   ```
   - Review success banner for path, location label, format/quant, file count, chat template detection.
2. **GUI**
   - Open **Download & Import → 🌐 Download**.
   - Fill model ID, branch (if needed), format preferences, optional weights, support repo, target location, optional files, and force toggle.
   - Verify command preview, then click **Download**.

### 3.2 Ingest existing repositories
1. **CLI**
   ```bash
   uv run imageworks-download scan --base ~/ai-models/weights --dry-run
   uv run imageworks-download scan --base ~/ai-models/weights --update-existing --include-testing
   ```
   - Use `--format` only as a fallback when detection fails; placeholders are skipped unless `--include-testing` is set.
2. **GUI**
   - **Download & Import → 📁 Scan**, choose base directory, update/dry-run flags, include-testing toggle, fallback format.
   - Review preview output before applying.

### 3.3 Import Ollama variants
1. **CLI**
   ```bash
   uv run python scripts/import_ollama_models.py --dry-run
   uv run python scripts/import_ollama_models.py --backend ollama --location linux_wsl --deprecate-placeholders
   ```
   - Strategy A naming: `<name>:<tag>` → `<family>-ollama-gguf[-<quant>]`; `served_model_id` preserves the original tag.
2. **GUI**
   - **Download & Import → 🦙 Import Ollama** (dry-run defaults to ON).
   - Supply backend/location overrides, optional placeholder deprecation.

### 3.4 Maintain registry fidelity
1. **Normalize**
   ```bash
   uv run imageworks-download normalize-formats --dry-run
   uv run imageworks-download normalize-formats --apply
   uv run imageworks-download normalize-formats --rebuild --prune-missing --apply
   ```
   - Diff columns: format, quant, size, directory checksum, file count. Use `--no-backup` only when running inside ephemeral pipelines.
2. **Cleanup**
   ```bash
   uv run imageworks-download prune-duplicates --dry-run
   uv run imageworks-download prune-deprecated --placeholders-only --dry-run
   uv run imageworks-download purge-logical-only --dry-run
   uv run imageworks-download reset-discovered --backend ollama --dry-run
   uv run imageworks-download restore-ollama --backup <file> --dry-run
   uv run imageworks-download backfill-ollama-paths --dry-run
   ```
   - Re-run without `--dry-run` to apply; GUI tabs mirror each command and show the exact invocation.

### 3.5 Verify installations and roles
1. **CLI**
   ```bash
   uv run imageworks-download verify
   uv run imageworks-download verify llama-3-8b-vllm-awq --fix-missing
   uv run imageworks-download list --details
   uv run imageworks-download list-roles --show-capabilities
   uv run imageworks-download stats
   ```
2. **GUI**
   - **Advanced Operations → ✅ Verify** (auto-fix toggle) and **Browse & Manage** (table filters, stats widget).

### 3.6 Removal workflow
1. Inventory candidates with `list --details` (or GUI selection); copy served IDs for Ollama.
2. **CLI**
   ```bash
   uv run imageworks-download remove <variant> --delete-files --force
   uv run imageworks-download remove <variant> --purge --delete-files --force
   ```
   - Ollama store cleanup: `ollama rm <served_model_id>`.
   - Reconcile metadata: `uv run imageworks-download verify --fix-missing` followed by `normalize-formats --rebuild --prune-missing --apply`.
   - Optional HF cache hygiene: `huggingface-cli scan-cache` / `delete-cache`.
3. **GUI**
   - **Advanced Operations → 🗑️ Remove**, choose mode (metadata only, files only, purge), confirm destructive actions, execute.

### 3.7 Rediscovery from scratch
1. **CLI (loader commands)**
   ```bash
   uv run imageworks-loader purge-imported --apply --providers all
   uv run imageworks-loader ingest-local-hf --root ~/ai-models/weights
   uv run imageworks-loader rebuild-ollama --location linux_wsl
   uv run imageworks-loader discover-all --hf-root ~/ai-models/weights
   ```
2. **GUI**
   - Use maintenance purge/reset tabs followed by relevant download/import tabs to repopulate entries.

---
## 4. Monitoring & Audit
- `imageworks-download list --json` (plain JSON) → pipe to `jq` for reporting.
- GUI statistics widget surfaces totals and disk consumption.
- Structured logs in `logs/model_downloader.log` mirror CLI icons/status codes.
- Enable `IMAGEWORKS_DEBUG=1` during incident response to capture verbose traces.

---
## 5. Troubleshooting Cheatsheet

| Symptom | Action |
|---------|--------|
| Download stalls | Confirm `aria2c`, retry with `--no-aria2`, inspect network policies. |
| Variant flagged `✗` | `uv run imageworks-download verify --fix-missing`. |
| Registry noise after manual edits | `uv run imageworks-download normalize-formats --dry-run` then `--apply`. |
| Ollama entry missing path | `uv run imageworks-download backfill-ollama-paths`. |
| Legacy `ModelRegistry` import errors | Switch to `imageworks.model_loader.download_adapter` helpers. |
| Need to compare quant families | `uv run imageworks-download list --json | jq -r '.[] | select(.quantization == "q4_k_m") | .name'`. |

---
## 6. Change Management Tips
- Capture the CLI preview displayed in the GUI or the actual command run in shells within change tickets.
- Keep timestamped backups generated by downloader commands; restore with `uv run imageworks-download restore-ollama --backup <file>` when needed.
- After large batch modifications, re-run `stats` and `list --details` to confirm backend, format, and quantisation alignment before promoting changes to production.

---
## 7. Related Reference
- [Model Downloader Operations Guide](../reference/model-downloader.md)
- [GUI Models Page Reference](../reference/gui-models-page.md)
- [Deterministic Model Serving](../reference/architecture/deterministic-model-serving.md)

This runbook should be updated alongside any downloader CLI or GUI changes to preserve parity between modalities.
