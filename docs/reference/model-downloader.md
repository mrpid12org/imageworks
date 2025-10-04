# Model Downloader

A comprehensive tool for downloading and managing AI models across multiple formats and directories, with support for quantized models and cross-platform compatibility.

> GGUF: Previous fallback guidance referenced a custom GGUF server script now removed. Use Ollama for GGUF experimentation; production multimodal flows rely on vLLM.

## Features

- 🚀 **Fast Downloads**: Uses aria2c for parallel, resumable downloads
- 🔍 **Format Detection**: Infers GGUF, AWQ, GPTQ, Safetensors, and more from filenames and configs
- 📁 **Smart Routing**: Sends GGUF models to LM Studio paths and other formats to the WSL weights store
- 📋 **Model Registry**: Tracks size, checksum, location, and metadata for every download
- ♻️ **Normalization & Rebuild**: `normalize-formats` command re-detects formats/quantization and optionally rebuilds dynamic fields safely

- 🔗 **URL Support**: Handles direct HuggingFace URLs and shorthand `owner/repo` identifiers (including `owner/repo@branch`)

- ⚡ **Cross-Platform**: Built for mixed Windows/WSL setups with optional custom overrides
- 🛡️ **Verification**: Validates completed downloads and registry integrity

## Quick Start

### Installation

The model downloader is part of the ImageWorks toolkit. Ensure you have aria2c installed:

```bash
# Ubuntu/Debian
sudo apt install aria2

# macOS
brew install aria2
```

### Basic Usage

```bash
# Download a model by name
imageworks-download download "microsoft/DialoGPT-medium"

# Target a specific branch/tag
imageworks-download download "microsoft/DialoGPT-medium@dev"

# Download from URL
imageworks-download download "https://huggingface.co/microsoft/DialoGPT-medium"

# Download specific format
imageworks-download download "casperhansen/llama-7b-instruct-awq" --format awq

# Force location override (keyword or custom path)
imageworks-download download "TheBloke/Llama-2-7B-Chat-GGUF" --location windows_lmstudio

# List downloaded models
imageworks-download list

# Show statistics
imageworks-download stats
```

### Unified Registry & Variant Names
All downloads are recorded in a single deterministic registry: `configs/model_registry.json`.
Variant names follow the pattern `<family>-<backend>-<format>-<quant>` (see Variant Naming Convention section below). These names are what you pass to commands like `remove`, `verify`, or future serving selectors.

### Listing Variants
Human-readable table (installed + metadata):
```bash
imageworks-download list
```

Add roles column & quant info (already included by default):
```bash
imageworks-download list --details
```

Filter by format (e.g. AWQ only):
Filter by backend (e.g. show only Ollama-managed logical/synthetic GGUF entries):
```bash
imageworks-download list --backend ollama
```

Show `served_model_id` and roles (details adds these columns; served_model_id now always shown when `--details` supplied):
```bash
imageworks-download list --details
```

Include logical (non-downloaded) entries that lack `download_path` (mainly historical Ollama logical-only variants) alongside real downloads:
```bash
imageworks-download list --include-logical
```

```bash
imageworks-download list --format awq
```

Filter by location:
```bash
imageworks-download list --location linux_wsl
```

JSON output for scripting:
```bash
imageworks-download list --json > downloads.json
```

`installed` in JSON is computed (path exists) so stale entries show `false`.


### Scanning Existing Downloads
Import previously downloaded HuggingFace repositories laid out as `~/ai-models/weights/<owner>/<repo>`:

Dry run (no changes):
```bash
imageworks-download scan --base ~/ai-models/weights --dry-run
```
Import:
```bash
imageworks-download scan --base ~/ai-models/weights
```
Include testing/demo placeholders explicitly (otherwise they are skipped at import):
```bash
imageworks-download scan --base ~/ai-models/weights --include-testing
```
Update existing entries and (optionally) supply a fallback format used ONLY when auto-detection fails:
```bash
imageworks-download scan --base ~/ai-models/weights --update-existing --format awq
```
Enhanced heuristics detect (in priority order):
- gguf (`*.gguf`)
- awq (directory name contains `awq`, weight shards `*.awq`, or `quantization_config.json` with `quant_method: awq`)
- gptq (`*.gptq` or name contains `gptq`)
- safetensors → fp16 fallback

Quant hints extracted from filenames or config: `q4_k_m`, `q5_k_m`, `q6_k`, `int4`, `int8`, and for AWQ configs a compact `w<bit>g<group>` label (e.g. `w4g128`).

After scanning:
```bash
imageworks-download list --details | grep awq
```

Clean-before-write semantics:
- The importer skips testing/demo placeholder variants by default using the central testing filters (names containing `testzzz`, or placeholders like `model-awq`, `model-fp16`, `demo-model`, `r` with optional suffixes). Use `--include-testing` to override.
- When a previously scanned entry exists with backend `unassigned` for the same family/format/quant, a new scan with a concrete backend (e.g. `vllm`) will reconcile it: the unassigned entry is migrated to the requested backend instead of creating a duplicate.
- If `hf_id` is missing and no `family_override` is provided, the adapter derives the family from the path tail. Overly generic tails like `model`, `demo`, `demo-model`, or `r` are treated as testing/demo and skipped by default to avoid polluting the registry.

If a variant was misclassified earlier (e.g. AWQ marked fp16), re-run with `--update-existing`. Supplying `--format` will NOT overwrite a correctly detected format; it only fills in missing ones.

### Normalizing / Rebuilding Registry Entries
Keep the registry consistent as heuristics improve or files change on disk:

Preview proposed updates (format/quant changes only):
```bash
imageworks-download normalize-formats --dry-run
```
Apply detected format/quantization updates:
```bash
imageworks-download normalize-formats --apply
```
Rebuild dynamic fields (size, file list, checksum) while preserving curated metadata (backend, roles, aliases):
```bash
imageworks-download normalize-formats --rebuild --apply
```
Prune entries whose `download_path` no longer exists:
```bash
imageworks-download normalize-formats --rebuild --prune-missing --apply
```
Mark (rather than remove) missing entries as deprecated (default if not pruning). A timestamped backup of `configs/model_registry.json` is written unless `--no-backup` is supplied.

Diff display columns:
- `download_format:old→new`
- `quantization:old→new`
- `download_size_bytes:old→new` (rebuild mode)
- `download_directory_checksum:old→new` (rebuild mode)
- `download_files_count:old→new` (rebuild mode)

Use this before commits to ensure deterministic metadata following manual edits or external modifications.


### Importing Ollama Models
Locally pulled Ollama models (e.g. via `ollama pull qwen2.5vl:7b`) are stored in Ollama's internal model store (typically `~/.ollama/models`). To ingest them into the unified registry use the helper script. The importer now implements **Strategy A naming**:

> Strategy A Naming: `<base>:<tag>` becomes a variant where:
>  * If `<tag>` is a quant token (e.g. `q6_k`, `q4_k_m`, `int4`, `int8`, `fp16`) → `family = base`, `quant = normalized tag`
>  * Otherwise `<tag>` is treated as part of the family → `family = base-tag`, `quant = None`
>  * Variant name pattern: `<family>-ollama-gguf[-<quant>]`
>  * `served_model_id` stores the original `base:tag` so runtime tooling can reference the canonical Ollama identifier.

Examples:
```
qwen2.5vl:7b                -> qwen2.5vl-7b-ollama-gguf
llava:7b                    -> llava-7b-ollama-gguf
hf.co/...-GGUF:Q6_K         -> hf.co/...-gguf-ollama-gguf-q6_k  (quant detected)
pixtral-local:latest        -> pixtral-local-latest-ollama-gguf
```

Dry run (no writes):
```bash
uv run python scripts/import_ollama_models.py --dry-run
```

Import for real:
```bash
uv run python scripts/import_ollama_models.py
```

Options:
```bash
uv run python scripts/import_ollama_models.py --backend ollama --location linux_wsl
# Mark legacy placeholder entries (model-ollama-gguf*) deprecated while importing
uv run python scripts/import_ollama_models.py --deprecate-placeholders
```

The importer also migrates existing Strategy A entries that were previously stored
without a quant suffix. When a quantization tag is detected, any matching
`<family>-ollama-gguf` record is renamed to `<family>-ollama-gguf-<quant>` so the
registry keys align with the documented naming convention.

What gets populated:
* `download_format = gguf`
* `family` derived per Strategy A
* `quantization` detected (regex: `q\d(_k(_m)?)?|int4|int8|fp16|f16` case-insensitive)
* `served_model_id = original_name_with_tag`
* `source_provider = ollama`
* `download_path` points to the real store directory if discoverable, else a synthetic `ollama://<name>` URI

Normalization update:
The importer now normalizes names by replacing `/` and spaces with `-`, collapsing repeated dashes, while preserving underscores inside quant tokens (e.g. `Q6_K` -> `q6_k`).

### Backfilling Legacy Ollama Entries (Option A)
### Undeprecating / Normalizing Ollama Entries

If earlier placeholder or partially imported Ollama entries are marked deprecated (and thus hidden) you can clear their deprecated flags and normalize synthetic paths:

```bash
imageworks-download undeprecate-ollama --dry-run -v   # preview
imageworks-download undeprecate-ollama -v            # apply
```

Behaviour:
* Clears `deprecated` where `backend=ollama`
* Normalizes `download_path` to `ollama://<served_model_id|name>` for consistency
* Idempotent (re-running when clean produces no changes)


If you previously created logical Ollama entries (e.g. via manual edits or earlier imports) they may lack `download_path` and thus not appear in `imageworks-download list` (which enumerates entries with download metadata). Populate synthetic paths so they show up:

```bash
uv run imageworks-download backfill-ollama-paths --dry-run -v
uv run imageworks-download backfill-ollama-paths
```

Behaviour:
* Detects store via `$OLLAMA_MODELS` or `~/.ollama/models`
* Sets `download_path` → `<store>/<served_model_id>` (or `ollama://<id>` if store missing)
* Sets `download_format=gguf` when missing (disable with `--no-set-format`)
* Sets `download_location` if empty (default `linux_wsl`)
* Leaves size/checksum blank (can rebuild later with `normalize-formats --rebuild` once per-entry export is desired)

Dry-run first to review; then re-run the standard list:
```bash
uv run imageworks-download list
```
```

Deprecated placeholder imports:
Previously imported placeholder variants named `model-ollama-gguf*` can be deprecated automatically with `--deprecate-placeholders`. Deprecated entries are hidden by default in listings (see below) but remain in the registry for auditability until purged.

You can still assign roles later by editing the registry JSON or a future role-assignment command.

### Deprecated Entries & Purging

Entries may be marked `deprecated: true` (e.g. when original files were removed, or placeholder naming was superseded). Tools now support:

Hide deprecated by default when listing (default behaviour):
```bash
imageworks-download list
```
Show deprecated explicitly:
```bash
imageworks-download list --show-deprecated
```

Purge all deprecated entries:
```bash
imageworks-download purge-deprecated
```
Only purge legacy placeholder Ollama entries:
```bash
imageworks-download purge-deprecated --placeholders-only
```
Preview (no write):
```bash
imageworks-download purge-deprecated --dry-run
```

### JSON Output Stability
`imageworks-download list --json` now emits plain stdout JSON (no Rich wrapping) to avoid inserted newlines in long names; safe for piping into `jq`:
```bash
imageworks-download list --json | jq '.[].name'
```
### Roles Overview
List every role-capable model (one row per role):
```bash
imageworks-download list-roles
```
Include capability flags:
```bash
imageworks-download list-roles --show-capabilities
```

### Verifying Downloads
Check that directories still exist and (if previously hashed) checksum unchanged:
```bash
imageworks-download verify
```
Specific variant:
```bash
imageworks-download verify llava-v1.5-13b-vllm-awq
```
Auto-clear missing/broken download metadata (keeps entry for re-download):
```bash
imageworks-download verify --fix-missing
```

### Removing vs Purging
Remove just the download (retain logical entry with roles/capabilities):
```bash
imageworks-download remove llava-v1.5-13b-vllm-awq --force
```
Remove and delete files:
```bash
imageworks-download remove llava-v1.5-13b-vllm-awq --delete-files --force
```
Purge (delete entry + optionally files):
```bash
imageworks-download remove llava-v1.5-13b-vllm-awq --purge --delete-files --force
```

### Statistics
Summarize counts & total size:
```bash
imageworks-download stats
```

### Programmatic Adapter Usage
Instead of the legacy `ModelRegistry`, use the unified adapter:
```python
from imageworks.model_loader.download_adapter import record_download, list_downloads, remove_download

entry = record_download(
  hf_id="liuhaotian/llava-v1.5-13b",
  backend="unassigned",  # fill with concrete serving backend later
  format_type="awq",
  quantization=None,
  path="/abs/path/to/weights/liuhaotian/llava-v1.5-13b",
  location="linux_wsl",
)

for d in list_downloads():
  print(d.name, d.download_size_bytes)

remove_download(entry.name, keep_entry=True)   # clears download fields only
```

### Troubleshooting
| Symptom | Likely Cause | Resolution |
|---------|--------------|-----------|
| `imageworks-download list` empty | No downloads yet or removed metadata | Download a model or verify registry path (`configs/model_registry.json`). |
| Variant shows `✗` in Inst column | Path missing (deleted/moved) | Re-download or run `verify --fix-missing` to clear download fields. |
| `aria2c not found` | aria2c not installed | Install via `sudo apt install aria2` or `brew install aria2`. |
| Removal failed (not found) | Wrong variant name | Use `list` / `--json` to confirm exact name. |
| Checksum changed warning in verify | Files altered after download | Re-download if integrity matters; lock logic (future) can enforce stability. |
| ImportError referencing legacy registry | Using deprecated `imageworks.tools.model_downloader.registry` | Switch to adapter functions (`record_download`, `list_downloads`). |

Exit Codes:
- `0` success
- `1` generic failure (not found, verification error, exception)

### Cross References
- Variant naming rationale: `../architecture/deterministic-model-serving.md` (section 4.1)
- Naming quick reference below.

### Logging

All downloader commands now stream structured logs to `logs/model_downloader.log` by default so you can audit each transfer. Set the `IMAGEWORKS_LOG_DIR` environment variable to relocate the log directory (for example when running inside a container or packaging the tool). Console output remains unchanged for interactive use, but every status icon and warning is mirrored into the log file for later review.

### Python API

```python
from imageworks.tools.model_downloader import ModelDownloader

# Initialize downloader
downloader = ModelDownloader()

# Download a model
model = downloader.download("microsoft/DialoGPT-medium")

# List models filtered by format
models = downloader.list_models(format_filter="awq")

# Get statistics
stats = downloader.get_stats()
```

## Directory Structure

The downloader manages two separate directories:

### Linux WSL Directory (`~/ai-models/`)
```
~/ai-models/
├── weights/              # Safetensors, AWQ, GPTQ, PyTorch models
│   ├── microsoft/
│   │   └── DialoGPT-medium/
│   └── casperhansen/
├── registry/            # Model registry JSON files
└── cache/               # Temporary files
```

**Compatible with**: vLLM, Transformers, AutoAWQ, AutoGPTQ

### Windows LM Studio Directory
```
/mnt/d/ai stuff/models/llm models/  (Windows: D:\ai stuff\models\llm models\)
├── TheBloke/
│   └── Llama-2-7B-Chat-GGUF/
└── Qwen/
    └── Qwen2.5-1.5B-GGUF/
```

**Compatible with**: LM Studio, llama.cpp, Ollama


Downloads that use `--location windows_lmstudio` (or detect GGUF formats automatically) keep a publisher/`repo` structure. Other formats default to `~/ai-models/weights/<owner>/<repo>`. Supplying a custom path via `--location /path/to/models` stores the model beneath that path. When a non-`main` branch is requested, the repository directory is suffixed with `@branch` (e.g. `DialoGPT-medium@dev`) to avoid collisions with the default branch.

## Format Detection

The downloader aggregates multiple detectors to determine the best storage location:

| Format | Detection Signals | Default Target |
|--------|------------------|----------------|
| **GGUF** | `.gguf` extensions, quantisation suffixes (`Q4_K`) | Windows LM Studio |
| **AWQ** | Repository name patterns, `quantization_config.quant_method == "awq"` | Linux WSL |
| **GPTQ** | Repository name patterns, config quantisation metadata | Linux WSL |
| **Safetensors** | `.safetensors` files | Linux WSL |
| **PyTorch** | `.bin`, `.pth`, `.pt` weights | Linux WSL |

Results are ranked by confidence; provide `--format awq,gguf` to express an explicit preference order when multiple matches are found.

Unified Detection Logic:
Both `scan` and post-download registration now share the same utility (`detect_format_and_quant`) which:
1. Prioritises: gguf → awq → gptq → fp16
2. Parses `quantization_config.json` / `quant_config.json` for AWQ and derives `w<bit>g<group>` labels
3. Extracts filename-based quant hints (`q4_k_m`, `q5_k_m`, `q6_k`, `int4`, `int8`)
4. Leaves format unset when insufficient evidence (allowing future improvements without overwriting existing correct entries)

The downloader performs a second pass after files are written to refine format/quantization (e.g., if initial network analysis was ambiguous). The `--format` option during `scan` acts only as a fallback when auto-detection fails (never overwrites determined values).


## Validation & Troubleshooting

Every download is verified to ensure files are present and complete before an entry is stored in the registry. The generated directories include all artefacts required by the serving helpers:

- **Core configs**: `config.json` and `tokenizer_config.json`
- **Tokenizer assets**: either `tokenizer.json` or `tokenizer.model`
- **Serving aids**: `generation_config.json`, `chat_template.json`, and quantisation descriptors when the upstream repository ships them
- **Weights**: at least one `.safetensors`, `.bin`, `.pt`, `.awq`, or `.gguf` shard

If any file fails to download, the downloader aborts with a verification error so you can retry without registering a broken model. The [LMDeploy helper](../scripts/start_lmdeploy_server.py) performs an additional sanity check when you launch the server, warning about missing chat templates or generation configs before it starts. This mirrors the most common causes of runtime issues (blank completions, misaligned role handling, or tokenisation failures) and provides actionable remediation guidance.


## Commands

### `download`
Download models from HuggingFace or URLs.

```bash
imageworks-download download MODEL [OPTIONS]

Options:
  --format, -f TEXT          Preferred format(s) (comma separated: gguf, awq, gptq, safetensors)
  --location, -l TEXT        Target location (linux_wsl, windows_lmstudio, or custom path)
  --include-optional, -o     Include optional files (docs, examples)
  --force                    Force re-download even if model exists
  --non-interactive, -y      Non-interactive mode (use defaults)
```

**What to expect:**

Successful runs echo a concise summary so you can immediately confirm where the
assets landed and how they were classified. The CLI highlights:

- ✅ success banner with the resolved display name
- 📁 download directory and 🗂️ logical location label (when available)
- 🔧 detected format/quantisation pair and 💾 aggregate size
- 📄 file-count metadata, 🧩 model-type/library hints, and 💬 chat template status

Example output:

```text
✅ Successfully downloaded: DialoGPT-medium
   📁 Files stored at: /home/user/ai-models/weights/microsoft/DialoGPT-medium
   🗂️  Location label: linux_wsl
   🔧 Format: safetensors
   💾 Size: 1.47 GiB
   📄 Files downloaded: 17
   💬 Chat template detected: external file (chat_template.json)
```

**Examples:**
```bash
# Basic download
imageworks-download download "microsoft/DialoGPT-medium"

# Specific format and location
imageworks-download download "microsoft/DialoGPT-medium" --format safetensors --location linux_wsl

# Branch selection with custom location
imageworks-download download "microsoft/DialoGPT-medium@dev" --location ~/models

# From URL with optional files
imageworks-download download "https://huggingface.co/microsoft/DialoGPT-medium" --include-optional

# Non-interactive
imageworks-download download "microsoft/DialoGPT-medium" --non-interactive
```

### `list`
List downloaded models with filtering.

```bash
imageworks-download list [OPTIONS]

Options:
  --format, -f TEXT      Filter by format
  --location, -l TEXT    Filter by location
  --details, -d          Show detailed information
  --json                 Output in JSON format
```

**Examples:**
```bash
# List all models
imageworks-download list

# Filter by format
imageworks-download list --format awq

# Show details
imageworks-download list --details

# JSON output
imageworks-download list --json
```

### `analyze`
Analyze HuggingFace URLs without downloading.

```bash
imageworks-download analyze URL [OPTIONS]

Options:
  --files    Show detailed file information
```

**Examples:**
```bash
# Analyze repository
imageworks-download analyze "https://huggingface.co/microsoft/DialoGPT-medium"

# Show file details
imageworks-download analyze "https://huggingface.co/microsoft/DialoGPT-medium" --files
```

### `remove`
Remove models from registry and optionally delete files.

```bash
imageworks-download remove MODEL_NAME [OPTIONS]

Options:
  --format, -f TEXT      Specific format to remove
  --location, -l TEXT    Specific location to remove from
  --delete-files         Also delete the model files from disk
  --force               Don't ask for confirmation
```

### `stats`
Show download statistics and registry information.

```bash
imageworks-download stats
```

### `verify`
Verify model integrity and registry consistency.

```bash
imageworks-download verify [MODEL_NAME] [OPTIONS]

Options:
  --fix-missing    Remove registry entries for missing models
```

### `config`
Show current configuration.

```bash
imageworks-download config
```

## Configuration

### Environment Variables

```bash
# Override default paths
export IMAGEWORKS_MODEL_ROOT=~/custom-ai-models/weights
export IMAGEWORKS_LMSTUDIO_ROOT=/mnt/d/custom/lmstudio/models
```

### pyproject.toml Configuration

Add to your `pyproject.toml`:

```toml
[tool.imageworks.model-downloader]
linux_wsl_root = "~/ai-models"
windows_lmstudio_root = "/mnt/d/ai stuff/models/llm models"
max_connections = 16
preferred_formats = ["awq", "safetensors", "gguf"]
```

## Python API Reference

### ModelDownloader

Main class for downloading and managing models.

```python
from imageworks.tools.model_downloader import ModelDownloader

downloader = ModelDownloader()
```

#### Methods

**`download(model_identifier, **kwargs)`**
Download a model from HuggingFace.

```python
model = downloader.download(
    model_identifier="microsoft/DialoGPT-medium",
    format_preference=["awq", "safetensors"],
    location_override="linux_wsl",
    include_optional=False,
    force_redownload=False,
    interactive=True,
)

# Target an alternate branch
experimental = downloader.download("microsoft/DialoGPT-medium@dev", force_redownload=True)
```

#### Download pipeline internals

`ModelDownloader.download` now coordinates a series of focused helpers so the
workflow remains understandable despite new heuristics:

1. `_build_repository_metadata` normalises owner/branch information for reuse.
2. `_resolve_target_dir` maps formats and overrides onto a concrete folder.
3. `_partition_files` separates required artefacts from optional extras.
4. `_download_selected_files` orchestrates aria2c and verifies completion.
5. `_inspect_chat_templates` captures tokenizer metadata and sidecar templates.
6. `_register_download` writes the unified registry entry (via `record_download`).

This structure keeps the CLI prompts, verification, and registry bookkeeping the
same while making it easier to audit individual stages when behaviour changes.

**`list_models(**kwargs)`**
List downloaded models with filtering.

```python
models = downloader.list_models(
    format_filter="awq",
    location_filter="linux_wsl"
)
```

**Unified Removal / Purge (CLI)**
The legacy `ModelRegistry` API has been deprecated in favor of the unified deterministic registry plus the download adapter.

Remove a downloaded variant (keeps entry metadata):
```bash
imageworks-download remove mistral-7b-instruct-awq --force
```

Purge (delete entry entirely) and delete files:
```bash
imageworks-download remove mistral-7b-instruct-awq --purge --delete-files --force
```

Verify integrity (checks existence + directory checksum where available):
```bash
imageworks-download verify
```

List roles (from unified registry):
```bash
imageworks-download list-roles
```

Stats:
```bash
imageworks-download stats
```

Programmatic recording now flows through:
```python
from imageworks.model_loader.download_adapter import record_download
entry = record_download(
  hf_id="microsoft/DialoGPT-medium",
  backend="unassigned",
  format_type="awq",
  quantization=None,
  path="/abs/path/to/model",
  location="linux_wsl",
)
```

`record_download` persists updates internally via `update_entries(..., save=True)`,
so additional `save_registry()` calls are unnecessary.

Listing programmatically:
```python
from imageworks.model_loader.download_adapter import list_downloads
downloads = list_downloads(only_installed=False)
```

Removal programmatically (retain entry):
```python
from imageworks.model_loader.download_adapter import remove_download
remove_download("mistral-7b-instruct-awq", keep_entry=True)
```

Attempting to import `imageworks.tools.model_downloader.registry` now raises an ImportError with guidance; update any legacy code to use the adapter functions above.

## Variant Naming Convention

All downloaded variants are registered using a deterministic hyphenated pattern to ensure stable selection and scriptability:

```
<family>-<backend>-<format>-<quant>
```

Where:
- `family`: Normalized HuggingFace repo tail (and branch if present). Lowercase; `/`, `_`, spaces, and `@` → `-`.
- `backend`: Serving backend target (`vllm`, `ollama`, `lmdeploy`, `gguf`, ...).
- `format` (optional): Artifact/weight packaging (`awq`, `gguf`, `fp16`, `bf16`, `safetensors`, ...). Omitted if singular/unambiguous.
- `quant` (optional): Quantization spec (`q4_k_m`, `int4`, `awq`, `gptq`, etc.). Not repeated if identical to `format`.

Examples:
| Source HF ID | Backend | Format | Quant | Variant Name |
|--------------|---------|--------|-------|--------------|
| liuhaotian/llava-v1.5-13b | vllm | awq | (implied) | llava-v1.5-13b-vllm-awq |
| liuhaotian/llava-v1.5-13b | vllm | fp16 | - | llava-v1.5-13b-vllm-fp16 |
| TheBloke/Mistral-7B-Instruct-v0.2-GGUF | ollama | gguf | q4_k_m | mistral-7b-instruct-v0.2-ollama-gguf-q4_k_m |
| google/siglip-base-patch16-256 | vllm | (single) | - | siglip-base-patch16-256-vllm |

Rules Recap:
1. Skip empty components; collapse multiple `-`.
2. Length target <= 80 chars (future abbreviation may apply for extreme cases).
3. Name collision (same tuple) updates existing entry; vary `format` or `quant` to keep both.

Why this matters: All CLI subcommands (`remove`, `verify`, `list`, `list-roles`) reference the variant name. Scripts can reliably grep/sort by suffixes (e.g. `-awq`, `-gguf-q4_k_m`).

See also: Detailed rationale and collision policy in `../architecture/deterministic-model-serving.md` (section 4.1).

### FormatDetector

Automatic format detection for models.

```python
from imageworks.tools.model_downloader import FormatDetector

detector = FormatDetector()
formats = detector.detect_comprehensive(
    model_name="microsoft/DialoGPT-medium-awq",
    filenames=["model.safetensors", "config.json"],
    config_content='{"quantization_config": {"quant_method": "awq"}}'
)
```

### URLAnalyzer

Analyze HuggingFace URLs and repositories.

```python
from imageworks.tools.model_downloader import URLAnalyzer

analyzer = URLAnalyzer()
analysis = analyzer.analyze_url("https://huggingface.co/microsoft/DialoGPT-medium")

print(f"Detected formats: {[f.format_type for f in analysis.formats]}")
print(f"Total size: {analysis.total_size} bytes")
```

## Error Handling

The downloader provides detailed error messages for common issues:

### Missing aria2c
```
RuntimeError: aria2c not found. Please install aria2c for optimal download performance.
Ubuntu/Debian: sudo apt install aria2
macOS: brew install aria2
```

### Network Errors
aria2c surfaces HTTP errors directly; failed transfers can be resumed by re-running the command.

### Disk Space
Ensure you have sufficient free space before starting large downloads (the downloader does not pre-validate capacity).

### File Conflicts
If a model already exists, you can choose to:
- Use the existing model
- Re-download and overwrite the registry entry (prompted interactively)
- Re-run with `--location` to place the download somewhere else

## Integration with ImageWorks

The model downloader integrates seamlessly with other ImageWorks tools:

### With vLLM Server
```python
# Download AWQ model
downloader.download("casperhansen/llama-7b-instruct-awq")

# Use with vLLM
from imageworks.apps.vlm_backend import start_vllm_server
start_vllm_server(model_name="llama-7b-instruct-awq")
```

### With Color Narrator
```python
# Download vision model
downloader.download("Qwen/Qwen2.5-VL-7B-Instruct")

# Use with color narrator
from imageworks.apps.color_narrator import ColorNarrator
narrator = ColorNarrator(model_name="Qwen2.5-VL-7B-Instruct")
```

### With Personal Tagger (Role-Based Recommended)
```bash
# Fetch at least one multimodal model (will satisfy caption/keywords/description roles)
imageworks-download download "qwen-vl/Qwen2.5-VL-7B-Instruct-AWQ"

# (Optional) Fetch an alternative model to compare for description role
imageworks-download download "llava-hf/llava-v1.6-mistral-7b-hf"

# Update / add entries in configs/model_registry.json assigning roles, then verify & (optionally) lock
uv run imageworks-model-registry verify qwen2.5-vl-7b-awq --lock

# Start serving backend(s) (example vLLM helper referencing registry paths)
python scripts/start_vllm_server.py --model "~/ai-models/weights/existing/Qwen2.5-VL-7B-Instruct-AWQ"

# Run the tagger using role-based resolution
uv run imageworks-personal-tagger run \
  -i ~/photos \
  --use-registry \
  --caption-role caption \
  --keyword-role keywords \
  --description-role description \
  --output-jsonl outputs/results/role_mode.jsonl \
  --summary outputs/summaries/role_mode.md
```

Legacy explicit flags (`--caption-model-path`, `--description-model-path`) are deprecated. Use them only for temporary experiments; long-term reproducibility should flow through the unified registry.

## Appendix: Unified Registry Glossary (Condensed)
Field | Meaning
----- | -------
`name` | Canonical key (edit cautiously—affects role resolution caches).
`backend` | Serving stack (`vllm`, `lmdeploy`, `ollama`, etc.).
`served_model_id` | External identity exposed at runtime (used in OpenAI API requests).
`backend_config.model_path` | Local weight path (download target).
`roles[]` | Functional assignments (e.g. `caption`, `keywords`, `description`).
`capabilities` | Modalities and feature switches (`vision`, `text`, etc.).
`artifacts.aggregate_sha256` | Deterministic hash representing tracked files.
`version_lock.locked` | If true, drift from expected hash blocks verification.
`model_aliases[]` | Alternate names for preflight/discovery.
`deprecated` | Kept for rollback; excluded from default role auto-selection.
`download_format` | Raw downloaded format (awq, gguf, safetensors, etc.) if acquired via downloader.
`download_location` | Logical storage location label (`linux_wsl`, `windows_lmstudio`, or `custom`).
`download_path` | Absolute path to downloaded weights directory.
`download_files[]` | Relative file paths captured at download time (basis for artifact hashing when artifact list empty).
`download_directory_checksum` | Lightweight directory hash (names + sizes) for quick invalidation.
`downloaded_at` / `last_accessed` | ISO timestamps for provenance & scheduling heuristics.

Use `uv run imageworks-model-registry verify <name> --lock` after adding new files to ensure reproducibility across environments.

## Troubleshooting

### Common Issues

**Model not found**
- Verify the model name is correct
- Check if the model exists on HuggingFace
- Try the full URL instead of just the name

**Slow downloads**
- Ensure aria2c is installed and working
- Check your internet connection
- Try reducing max_concurrent_downloads in config

**Permission errors**
- Check directory permissions
- For Windows LM Studio directory, ensure WSL can access the drive

**Registry corruption**
- Run `imageworks-download verify --fix-missing` to clean up missing download paths
- Hard reset: edit `configs/model_registry.json` directly (single unified registry) then re-run `imageworks-model-registry verify <name>`

### Debug Mode

Set environment variable for detailed logging:
```bash
export IMAGEWORKS_DEBUG=1
imageworks-download download "model-name"
```

### File Issues

File integrity (reproducible hash drift): use `imageworks-model-registry verify <name>` which now derives artifacts from `download_files` when present.

## Contributing

See the main ImageWorks repository for contributing guidelines.

## License

This tool is part of the ImageWorks project and follows the same license terms.

## Addendum: Testing Convention and Clean Imports

### Testing Convention

To prevent test or demo entries from polluting the main registry and UI:

- Use the token `testzzz` in names or display names for all test-created entries.
- Optionally set `metadata.testing=true` when creating test entries programmatically.
- The downloader list and chat proxy hide test entries by default.
  - CLI: `imageworks-download list --include-testing` to show them.
  - Proxy: set `CHAT_PROXY_INCLUDE_TEST_MODELS=1` to show them in `/v1/models`.
- You can extend patterns via `IMAGEWORKS_TEST_MODEL_PATTERNS` (comma-separated regex).
  Defaults include `testzzz`, and legacy placeholders like `model-awq*`, `model-fp16*`, `demo-model*`, and minimalist names like `r*`.

Recommended for tests:
- Point tests at a temp registry using `IMAGEWORKS_REGISTRY_DIR`.
- Use cleanup helpers or in-memory registries where possible.

### Import Cleanliness & Default Backends

- Scans of HF-style weights now default to `backend=vllm` so imports align with typical serving and avoid `unassigned` duplicates.
- The Ollama importer uses `backend=ollama` and preserves the canonical `served_model_id`.
- Name normalization trims noisy prefixes, consolidates quant tokens, and fills `download_*` metadata.
- If you ever imported unassigned entries in older versions, you can clean them with:
  - `uv run imageworks-download purge-hf --backend unassigned`
  - then `uv run imageworks-download normalize-formats --rebuild --apply`
  - optionally `uv run imageworks-download prune-duplicates`

### Multi-backend Variants

It is valid to have the same family available on multiple backends (e.g., one vLLM safetensors variant and one Ollama GGUF variant). The registry treats these as distinct variants because the variant name includes the backend token. The proxy exposes them separately so you can select explicitly in OpenWebUI.
