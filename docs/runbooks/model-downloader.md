# Model Downloader Runbook

Download, scan, and normalise model weights using the `imageworks-download` CLI.

## 1. Prepare environment
- Install `aria2c` for multi-connection downloads (see package manager commands
  in the reference guide).【F:docs/reference/model-downloader.md†L1-L60】
- Ensure `configs/model_registry.curated.json` is under version control before
  importing new entries.【F:src/imageworks/model_loader/registry.py†L1-L200】

## 2. Download models
```bash
uv run imageworks-download download thebloke/llama-3-8b-instruct-awq \
  --format awq --location linux_wsl
```
- Shortcuts accept `owner/repo`, `owner/repo@branch`, or direct URLs. Formats and
  quantisation are auto-detected unless overridden.【F:src/imageworks/tools/model_downloader/cli.py†L40-L200】
- Use `--location windows_lmstudio` (or custom paths) to route GGUF models to
  host directories when working across WSL and Windows.【F:src/imageworks/tools/model_downloader/cli.py†L201-L330】

## 3. Scan existing weights
```bash
uv run imageworks-download scan --base ~/ai-models/weights --include-testing
```
- Imports repositories laid out as `<owner>/<repo>`, skipping demo placeholders
  unless `--include-testing` is set. Existing entries merge using shared naming
  rules to avoid duplicates.【F:src/imageworks/tools/model_downloader/download_adapter.py†L40-L210】
- Add `--update-existing` to refresh metadata and quantisation heuristics.

## 4. Normalise metadata
```bash
uv run imageworks-download normalize-formats --apply
uv run imageworks-download normalize-formats --rebuild --apply
```
- `--apply` updates format/quant fields; `--rebuild` recomputes size and checksum
  data for drift detection.【F:src/imageworks/tools/model_downloader/cli.py†L331-L480】
- `--prune-missing` removes entries whose `download_path` no longer exists. Use
  `--no-backup` sparingly to skip automatic JSON snapshot backups.【F:docs/reference/model-downloader.md†L61-L160】

## 5. Verify installations
```bash
uv run imageworks-loader verify qwen2-vl-2b
```
- Confirms checksums recorded by the downloader match files on disk. Run after
  large downloads or manual edits.【F:src/imageworks/model_loader/hashing.py†L1-L200】

## 6. Troubleshooting
| Symptom | Checks |
| --- | --- |
| Download stalls | Confirm `aria2c` is installed and network allows parallel connections. Retry with `--no-aria2` for single-threaded downloads.【F:src/imageworks/tools/model_downloader/cli.py†L40-L200】 |
| Entries missing from registry | Ensure `scan` or `download` commands completed successfully; the layered registry writes to `.discovered` before merging.【F:src/imageworks/tools/model_downloader/download_adapter.py†L40-L210】 |
| GGUF targets not served | Use Ollama exports or LM Studio for GGUF variants; the downloader now records served-model hints but the chat proxy focuses on vLLM/Ollama endpoints.【F:docs/reference/model-downloader.md†L1-L160】 |
| Registry diff noisy | Run `normalize-formats --dry-run` first to preview changes and capture diffs in PRs before applying.【F:src/imageworks/tools/model_downloader/cli.py†L331-L480】 |

Log download commands and resulting JSON diffs in change management records.
