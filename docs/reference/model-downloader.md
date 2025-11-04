# Model Downloader Operations Guide

The ImageWorks model downloader acquires, organises, and audits model weights across Hugging Face, Ollama, and local stores. This guide merges the previous reference and runbook into a single definitive source covering both the command-line interface (CLI) and the Streamlit-based GUI.

---
## 1. Capabilities at a Glance

| Capability | Details |
|------------|---------|
| Fast transfers | Parallel, resumable downloads via `aria2c` (fall back to single-threaded when `--no-aria2` is specified). |
| Format + quant detection | Heuristics recognise containers (GGUF, safetensors, PyTorch, AWQ, GPTQ) and quant tokens (`q4_k_m`, `w4g128`, `fp16`, `bf16`, `fp8`, `int4`, `int8`, `squeezellm`, `bnb`, etc.). |
| Smart routing | GGUF defaults to the Windows/LM Studio store, while tensor formats land under the Linux/WSL weights root unless overridden. |
| README & license capture | README, LICENSE, SHA-256 (for small READMEs), and curated signals (quant hints, backend claims, licence hints) are persisted for audits. |
| Registry integration | All operations write to the deterministic layered registry (`configs/model_registry.*.json`) using the shared download adapter. |
| Normalisation & rebuild | `normalize-formats` re-detects formats and quantisation, with `--rebuild` recomputing size, checksum, and file inventory for drift detection. |
| Logging | Structured logs land in `logs/model_downloader.log`; set `IMAGEWORKS_LOG_DIR` to redirect. Console output remains interactive. |
| Python API | Programmatic access through `ModelDownloader`, `FormatDetector`, `URLAnalyzer`, and the download adapter helpers (`record_download`, `list_downloads`, `remove_download`). |

---
## 2. Operating Modes

### 2.1 CLI (baseline)
- Entry point: `uv run imageworks-download <command>` (packaged console script `imageworks-download`).
- Subcommands mirror every lifecycle task: acquisition (`download`, `scan`, `import` helpers), maintenance (`normalize-formats`, `prune-*`, `reset-discovered`), audit (`verify`, `stats`, `list`, `list-roles`), troubleshooting (`analyze`, `config`).
- JSON-first automation: `list --json` (now plain stdout JSON), dry-run flags, and `--no-backup` toggles give precise control for scripts and CI.

### 2.2 GUI (Streamlit Models page)
- File: `src/imageworks/gui/pages/2_🎯_Models.py`.
- Tabs group CLI-equivalent actions:
  1. **📚 Browse & Manage** – renders `list --details`, allows role editing, backend config review.
  2. **📥 Download & Import** – wrappers for `download`, `scan`, and `scripts/import_ollama_models.py` with command previews and progress capture.
  3. **🔧 Registry Maintenance** – normalisation and purge helpers (all dry-run by default).
  4. **🔌 Backends** – monitoring, manual launch notes.
  5. **⚙️ Advanced Operations** – destructive flows (`remove`, `verify`) with confirmation gates.
- All GUI forms display the exact CLI command they execute so you can replicate runs or record change-management entries.

---
## 3. Environment & Installation

### 3.1 Prerequisites
1. Install `aria2c` (parallel downloads):
   ```bash
   # Ubuntu / Debian
   sudo apt install aria2

   # macOS
   brew install aria2
   ```
2. Ensure the ImageWorks virtual environment is active (`uv` handles this automatically when using `uv run`).
3. Confirm write access to the registry and target directories.

### 3.2 Default directories
- **Linux / WSL weights root (`~/ai-models`)**
  ```
  ~/ai-models/
  ├── weights/              # safetensors, AWQ, GPTQ, PyTorch
  │   └── <owner>/<repo>
  ├── registry/             # registry fragments (curated, discovered)
  └── cache/                # temporary downloads
  ```
- **Windows LM Studio root (`/mnt/d/ai stuff/models/llm models`)**
  ```
  /mnt/d/ai stuff/models/llm models/
  └── <owner>/<repo>
  ```
- Custom locations use the supplied path verbatim; branch-qualified downloads append `@branch` to avoid collisions (`repo@dev`).

### 3.3 Configuration switches
- Environment variables:
  ```bash
  export IMAGEWORKS_MODEL_ROOT=~/custom-ai-models/weights
  export IMAGEWORKS_LMSTUDIO_ROOT=/mnt/d/custom/lmstudio/models
  export IMAGEWORKS_LOG_DIR=/var/log/imageworks
  export IMAGEWORKS_DEBUG=1                     # verbose tracing
  export IMAGEWORKS_TEST_MODEL_PATTERNS="testzzz,model-awq,model-fp16"
  export IMAGEWORKS_OLLAMA_HOST=imageworks-ollama
  export OLLAMA_MODELS=/srv/ollama/models
  export OLLAMA_BASE_URL=http://127.0.0.1:11434
  export IMAGEWORKS_REGISTRY_DIR=/tmp/test-registry  # for integration tests
  ```
- `pyproject.toml` (`[tool.imageworks.model-downloader]`):
  ```toml
  linux_wsl_root = "~/ai-models"
  windows_lmstudio_root = "/mnt/d/ai stuff/models/llm models"
  max_connections = 16
  preferred_formats = ["awq", "safetensors", "gguf"]
  ```

---
## 4. Registry & Naming Essentials

### 4.1 Layered registry
- `model_registry.curated.json` – manually maintained baseline (under VCS).
- `model_registry.discovered.json` – tooling overlay populated by downloader/importers.
- `model_registry.json` – merged snapshot (auto-generated; do not edit directly).

### 4.2 Variant naming
```
<family>-<backend>-<format>-<quant>
```
- `family`: repo tail (and branch) normalised to lowercase with `/`, `_`, spaces, `@` → `-`; quant tokens preserve underscores (`q6_k`).
- `backend`: target stack (`vllm`, `ollama`, `lmdeploy`, `gguf`, ...).
- `format`: container (`awq`, `gguf`, `fp16`, `bf16`, `safetensors`, ...); omitted if singular/unambiguous.
- `quant`: precision string (`q4_k_m`, `int4`, `awq`, ...); omitted if redundant with format.
- Collisions on identical tuples update the existing entry; vary format/quant to retain multiple variants.

Human-readable `display_name`/slug variants are stored alongside the canonical key; CLI and GUI favour the simplified name (e.g. `llama 12.2b (Q4 K M)`).

### 4.3 Metadata captured per download
- `download_path`, `download_location`, `download_format`, `download_size_bytes`.
- `download_directory_checksum`, `download_files_count`, `download_files[]` (relative paths).
- README annotations: `metadata.readme_file`, `metadata.readme_sha256` (≤10 MiB), `metadata.readme_signals`.
- Provenance: `source_provider`, `hf_id`, `served_model_id` (Ollama tag), timestamps.
- Capabilities, roles, aliases, version locks remain in the registry for serving alignment.

### 4.4 README signal vocabulary
- Quant schemes (AWQ, GPTQ, `w4g128` etc.), backend claims (vLLM, LM Studio, Ollama), reasoning hints, first detected licence reference.

---
## 5. Format & Location Detection

1. Container detection from file extensions (`.gguf`, `.safetensors`, `.bin`, `.pth`, `.pt`).
2. Quant inference from `quantization_config.json` / `quant_config.json`, AWQ descriptors (`w<bit>g<group>`), filenames (`q4_k_m`, `int8`, `mxfp4`, `bnb`, etc.).
3. Second-pass verification after download writes; `--format` acts only as a fallback when detection fails.
4. Location routing defaults:
   - GGUF → `windows_lmstudio` (LM Studio / llama.cpp / Ollama usage).
   - Tensor formats → `linux_wsl`.
   - Overrides: `--location <label|path>` or global config/ENV.

---
## 6. Core Workflows (Runbook)

Each workflow lists CLI steps first, followed by GUI parity.

### 6.1 Acquire a new Hugging Face model
1. **CLI**
   ```bash
   uv run imageworks-download download "owner/repo[@branch]" \
     --format awq --location linux_wsl \
     --include-optional --weights "file1,file2" \
     --support-repo other/repo --non-interactive
   ```
   - Output highlights display the resolved path, location, detected format/quant, file count, chat template status, size.
2. **GUI** – *Download & Import → Download from HuggingFace*
   - Fill model identifier (and branch if needed).
   - Choose preferred formats, optional weights, support repo, location, include optional files, force re-download toggle.
   - Review the generated CLI preview; click **Download**.

### 6.2 Scan existing weights
1. **CLI**
   ```bash
   uv run imageworks-download scan --base ~/ai-models/weights --dry-run
   uv run imageworks-download scan --base ~/ai-models/weights --update-existing --include-testing --format awq
   ```
   - Skips testing/demo placeholders unless `--include-testing` is supplied.
   - Migrates `backend=unassigned` entries to the detected backend to avoid duplicates.
2. **GUI** – *Download & Import → Scan Existing*
   - Select base directory, toggle dry-run/update existing/include testing.
   - Optional fallback format; review preview before applying.

### 6.3 Import Ollama models
1. **CLI**
   ```bash
   uv run python scripts/import_ollama_models.py --dry-run
   uv run python scripts/import_ollama_models.py --backend ollama --location linux_wsl --deprecate-placeholders
   ```
   - Strategy A naming: `<name>:<tag>` becomes `<family>-ollama-gguf[-<quant>]` with `served_model_id` preserved.
   - Name normalisation replaces `/` and spaces with `-`, collapses repeated dashes, and preserves underscores inside quant tokens (e.g. `Q6_K` → `q6_k`).
   - Detects Ollama store via `$OLLAMA_MODELS` or `~/.ollama/models`; falls back to `ollama://<id>` when path unavailable.
2. **GUI** – *Download & Import → Import Ollama*
   - Choose dry-run, backend, location, placeholder deprecation; execute with preview.

### 6.4 Normalise / rebuild registry metadata
1. **CLI**
   ```bash
   uv run imageworks-download normalize-formats --dry-run
   uv run imageworks-download normalize-formats --apply
   uv run imageworks-download normalize-formats --rebuild --prune-missing --apply
   ```
   - Diff columns: `download_format`, `quantization`, `download_size_bytes`, `download_directory_checksum`, `download_files_count`.
   - `--no-backup` skips automatic merged snapshot backup (use sparingly).
2. **GUI** – *Registry Maintenance → Normalize*
   - Default dry-run; optionally rebuild dynamic fields, prune missing entries, toggle backup.

### 6.5 Verify installations
1. **CLI**
   ```bash
   uv run imageworks-download verify
   uv run imageworks-download verify mistral-7b-instruct-vllm-awq --fix-missing
   ```
2. **GUI** – *Advanced Operations → Verify*
   - Choose verify-all or single variant, enable auto-fix missing as needed; run with progress feedback.

### 6.6 Remove or purge models (safe workflow)
1. Identify target variants (`list --details`, JSON filtering) and served IDs for Ollama.
2. **CLI**
   ```bash
   uv run imageworks-download remove <variant> --delete-files --force
   uv run imageworks-download remove <variant> --purge --delete-files --force
   ```
   - For Ollama storage cleanup: `ollama rm <served_model_id>`.
   - For HF path drift: `imageworks-download verify --fix-missing` then optional HF cache pruning (`huggingface-cli scan-cache`).
   - Final consistency: `normalize-formats --rebuild --prune-missing --apply` or `purge-deprecated`.
3. **GUI** – *Advanced Operations → Remove*
   - Select variant; choose metadata-only, files-only, or purge entirely; confirm destructive actions via checkbox; command preview shown.

### 6.7 Refresh discovered registry layers
1. **CLI (`imageworks-loader`)**
   ```bash
   uv run imageworks-loader purge-imported --apply --providers all
   uv run imageworks-loader ingest-local-hf --root ~/ai-models/weights
   uv run imageworks-loader rebuild-ollama --location linux_wsl
   uv run imageworks-loader discover-all --hf-root ~/ai-models/weights
   ```
   - Ensures HF + Ollama imports flow through the unified adapter path.
2. **GUI** – Initiate via maintenance tabs (purge/reset operations) followed by corresponding download/import steps.

### 6.8 Audit & reporting
- **CLI**: `list`, `list --json`, `list --details`, `list --backend`, `list --location`, `list --include-logical`, `list-roles`, `stats`.
- **GUI**: Browse & Manage tab (table filters, statistics widget) mirrors these outputs.

---
## 7. CLI Command Reference

### 7.1 Acquisition & inspection
- `download MODEL [--format ...] [--location ...] [--include-optional] [--weights files] [--support-repo repo] [--force] [--non-interactive]`
  - Supports Hugging Face shorthand (`owner/repo`, `owner/repo@branch`) and direct URLs.
  - Weight selection downloads only chosen shards plus required configs.
  - Success banner summarises path, location label, format/quant, size, file count, chat template status.
- `analyze URL [--files]`
  - Fetches repository metadata without downloading; `--files` lists remote artefacts.
- `scan --base PATH [--dry-run] [--update-existing] [--include-testing] [--format fallback]`
  - Imports directory structures laid out as `<owner>/<repo>`; fallback format only fills blanks.
  - Clean-before-write filters skip placeholders (names containing `testzzz`, `model-awq`, `demo-model`, `r` etc.).

### 7.2 Normalisation & cleanup
- `normalize-formats [--dry-run|--apply] [--rebuild] [--prune-missing] [--backup/--no-backup]`
- `prune-duplicates [--backend NAME] [--dry-run] [--verbose]`
- `restore-ollama [--backup PATH] [--include-deprecated] [--dry-run]`
- `reset-discovered [--backend NAME] [--dry-run] [--backup/--no-backup]`
- `purge-hf [--weights-root PATH] [--backend NAME] [--dry-run]`
- `purge-logical-only [--include-curated|--discovered-only] [--dry-run|--apply] [--backup/--no-backup]`
- `purge-deprecated [--placeholders-only] [--dry-run]`
- `preview-simple-slugs [--include-ollama] [--include-hf] [--json]`
- `apply-simple-slugs [--include-ollama] [--include-hf] [--disambiguate STRATEGY] [--rename-slugs/--display-only] [--allow-skip-on-collision/--no-skip-on-collision] [--dry-run|--apply] [--backup/--no-backup] [--json] [--tests-only]`
- `backfill-ollama-paths [--dry-run] [--no-set-format]`
  - Detects store (via `$OLLAMA_MODELS` or default) and populates `download_path`, `download_format`, and location labels for logical-only entries.

### 7.3 Verification & removal
- `verify [VARIANT] [--fix-missing]`
- `remove VARIANT [--delete-files] [--purge] --force`
- `stats`
- `list [--format] [--location] [--backend] [--show-deprecated] [--details] [--json] [--include-logical] [--dedupe] [--show-internal-names] [--show-backend] [--show-installed] [--include-testing]`
  - `producer` field in JSON simplifies filtering (`jq '.[].producer'`).
  - Example JSON filters:
    ```bash
    uv run imageworks-download list --json       | jq -r '.[] | select(.producer == "thebloke") | .name'

    uv run imageworks-download list --json       | jq -r '.[] | select((.quantization // "") | test("awq"; "i")) | .name'
    ```
- `list-roles [--show-capabilities] [--include-testing] [--json]`
- `config` – displays merged configuration from env, pyproject, defaults.

### 7.4 Ollama helpers
- `scripts/import_ollama_models.py [--dry-run] [--backend NAME] [--location LABEL] [--deprecate-placeholders]`
- Registry cleanup for legacy placeholders: `purge-deprecated --placeholders-only`.

### 7.5 Exit codes
- `0` success, `1` failure (not found, verification error, unexpected exception).

---
## 8. GUI Reference Matrix

| GUI tab & sub-action | Primary CLI command(s) |
|----------------------|------------------------|
| 📚 Browse & Manage – table, roles, backend config | `imageworks-download list --details`, registry save utilities |
| 📥 Download & Import → 🌐 Download | `imageworks-download download` |
| 📥 Download & Import → 📁 Scan | `imageworks-download scan` |
| 📥 Download & Import → 🦙 Import Ollama | `python scripts/import_ollama_models.py` |
| 🔧 Registry Maintenance → 🔄 Normalize | `imageworks-download normalize-formats` |
| 🔧 Registry Maintenance → 🗑️ Purge | `purge-deprecated`, `purge-logical-only`, `purge-hf`, `reset-discovered` |
| 🔧 Registry Maintenance → 🔨 Cleanup | `prune-duplicates`, `restore-ollama`, `backfill-ollama-paths` |
| ⚙️ Advanced Operations → 🗑️ Remove | `imageworks-download remove` |
| ⚙️ Advanced Operations → ✅ Verify | `imageworks-download verify` |
| ⚙️ Advanced Operations → 📊 Profiles | `imageworks-download config` + pyproject viewer |

Safety net: all destructive GUI flows require confirmation, default to dry-run, and emit the CLI command used for audit trails.

---
## 9. Programmatic APIs

### 9.1 ModelDownloader class
```python
from imageworks.tools.model_downloader import ModelDownloader

downloader = ModelDownloader()
model = downloader.download("microsoft/DialoGPT-medium")
models = downloader.list_models(format_filter="awq")
downloader.remove_model("my-model", delete_files=True)
is_valid = downloader.verify_model("my-model")
stats = downloader.get_stats()
```
- Download pipeline steps: `_build_repository_metadata`, `_resolve_target_dir`, `_partition_files`, `_download_selected_files`, `_inspect_chat_templates`, `_register_download`.

### 9.2 Download adapter utilities
```python
from imageworks.model_loader.download_adapter import record_download, list_downloads, remove_download

entry = record_download(
    hf_id="liuhaotian/llava-v1.5-13b",
    backend="vllm",
    format_type="awq",
    quantization=None,
    path="/abs/path/to/weights/liuhaotian/llava-v1.5-13b",
    location="linux_wsl",
)
for d in list_downloads(only_installed=False):
    print(d.name, d.download_size_bytes)
remove_download(entry.name, keep_entry=True)
```
- `record_download` persists immediately; no explicit `save_registry()` required.

### 9.3 Auxiliary analyzers
```python
from imageworks.tools.model_downloader import FormatDetector, URLAnalyzer

detector = FormatDetector()
formats = detector.detect_comprehensive(
    model_name="microsoft/DialoGPT-medium-awq",
    filenames=["model.safetensors", "config.json"],
    config_content='{"quantization_config": {"quant_method": "awq"}}',
)

analyzer = URLAnalyzer()
analysis = analyzer.analyze_url("https://huggingface.co/microsoft/DialoGPT-medium")
print(f"Detected formats: {[f.format_type for f in analysis.formats]}")
print(f"Total size: {analysis.total_size} bytes")
```

---
## 10. Integration Highlights

- **Chat Proxy** – `/v1/models` mirrors downloader naming; hide non-installed entries by default. Override with `CHAT_PROXY_INCLUDE_NON_INSTALLED=1` if container paths differ.
- **vLLM server helper** – `scripts/start_vllm_server.py` consumes registry paths.
- **Color Narrator & Personal Tagger** – rely on registry roles/capabilities; download required multimodal variants first, then run `imageworks-model-registry verify <name> --lock`.
- **LMDeploy helper** – warns about missing chat templates or generation configs before launching.

---
## 11. Logging & Auditing

- Structured logs: `logs/model_downloader.log` (override via `IMAGEWORKS_LOG_DIR`).
- Each CLI invocation mirrors the Rich console output into the log file.
- GUI runs pipe their CLI command output into the Streamlit log and the same structured log.
- Enable debug traces with `IMAGEWORKS_DEBUG=1`.

---
## 12. Troubleshooting & FAQ

| Symptom | Likely cause | Resolution |
|---------|--------------|------------|
| `imageworks-download list` returns no rows | No downloads yet or registry fragments missing | Confirm `configs/model_registry.*.json` exist; download a model or re-run discovery. |
| Variant shows `✗` in `Installed` column | Path missing or files removed | Re-download or run `verify --fix-missing`. |
| `aria2c not found` | Dependency missing | `sudo apt install aria2` / `brew install aria2`. |
| Download stalls | Network blocking multi-connection transfers | Retry with `--no-aria2`; verify firewall rules. |
| Removal command fails (`not found`) | Variant name mismatch | Use `list --json` / GUI search to copy the canonical variant key. |
| Checksum drift warning | Files modified after download | Re-download or rebuild checksums (`normalize-formats --rebuild --apply`). |
| Registry diff noisy after manual edits | Stale format/quant metadata | `normalize-formats --dry-run` to preview, then `--apply`. |
| Legacy registry import errors | Deprecated `imageworks.tools.model_downloader.registry` import | Switch to adapter helpers (`record_download`, `list_downloads`). |
| Ollama GGUF variants missing from list | Logical-only entries without paths | Run `backfill-ollama-paths` or re-import via Ollama helper. |

**Debug tips**
- `imageworks-download config` to inspect merged configuration.
- `imageworks-download analyze <url>` to confirm repository availability before downloading.
- `IMAGEWORKS_TEST_MODEL_PATTERNS` extends testing filters for local experiments.

---
## 13. Safe Removal Checklist (Detailed)

1. **Inventory** – `imageworks-download list --details` (copy variant names, served IDs for Ollama).
2. **CLI removal** – `remove <variant> --delete-files --force` (keep entry) or `--purge --delete-files --force` (remove entry). Confirm location-specific cleanup (Ollama `ollama rm`, HF directories removed).
3. **Post-clean audit** – `verify --fix-missing`, `normalize-formats --rebuild --prune-missing --apply`, optionally `purge-deprecated`.
4. **HF cache hygiene** – `huggingface-cli scan-cache` / `huggingface-cli delete-cache` if global cache needs trimming.
5. **Change log** – record CLI command + resulting diff (GUI shows the same command for copy/paste into tickets).

---
## 14. Appendix: Registry Field Glossary

| Field | Description |
|-------|-------------|
| `name` | Canonical variant key. |
| `display_name`, `slug` | Human-friendly labels. |
| `backend` | Serving stack (vLLM, LMDeploy, Ollama, etc.). |
| `served_model_id` | Runtime identifier (e.g., Ollama `name:tag`). |
| `backend_config.model_path` | Local path consumed by serving helpers. |
| `download_path`, `download_location`, `download_format`, `download_size_bytes` | Primary download metadata. |
| `download_directory_checksum`, `download_files_count`, `download_files[]` | Inventory for drift detection. |
| `downloaded_at`, `last_accessed` | ISO timestamps. |
| `source_provider`, `hf_id`, `metadata.readme_*`, `metadata.readme_signals` | Provenance + README capture. |
| `roles[]`, `role_priorities`, `capabilities` | Serving intent and modality flags. |
| `model_aliases[]` | Alternate lookup strings. |
| `artifacts.aggregate_sha256`, `version_lock.locked` | Deterministic serving guardrails. |
| `deprecated` | Hidden by default; purge via maintenance tools. |

---
## 15. Testing Conventions & Multi-backend Notes

- Use `testzzz` or configure `IMAGEWORKS_TEST_MODEL_PATTERNS` for temporary entries. CLI and chat proxy hide testing models unless `--include-testing` or `CHAT_PROXY_INCLUDE_TEST_MODELS=1` is set.
- Imports default to sensible backends (`scan` → `vllm`, Ollama importer → `ollama`). Clean legacy `unassigned` entries with `purge-hf --backend unassigned` and a subsequent normalisation run.
- Coexistence of multiple backends per family is supported; variant keys encode backend so vLLM and Ollama builds can live side by side.

---
## 16. Related Tools

- `docs/reference/gui-models-page.md` – UI wiring details.
- `docs/reference/architecture/deterministic-model-serving.md` – naming rationale and serving pipeline.
- `docs/reference/architecture/model-loader-overview.md` – registry lifecycle and hashing strategy.
- `scripts/start_vllm_server.py`, `scripts/import_ollama_models.py` – operational helpers invoked throughout this guide.

This consolidated guide should be treated as the source of truth for downloader operations. Update both CLI scripts and GUI bindings in tandem to keep parity intact.
