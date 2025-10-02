# Ollama Integration: Summary & Action Log

This document consolidates all work, design decisions, fixes, and next steps related to integrating Ollama models into the unified registry from the point where the Ollama import path was first untested through subsequent debugging and enhancements.

---
## 1. Objective
Integrate locally pulled Ollama models into the deterministic unified registry (`configs/model_registry.json`) so that they:
- Use consistent Strategy A naming.
- Represent quantized vs non-quantized variants deterministically.
- Are visible via `imageworks-download list` and downstream tooling.
- Support lifecycle operations (deprecation, purge, backfill, normalization).

## 2. Initial Gap
Early placeholder entries (e.g. `model-ollama-gguf*`) existed. Real imports unverified; naming + quant parsing + visibility rules incomplete. User confirmed they “hadn’t tested the Ollama import yet.”

## 3. Strategy A Naming (Chosen)
Given an Ollama model identifier `base[:tag]`:
- If `tag` matches quant regex → `family = base`, `quant = tag`.
- Else `family = base-tag`, `quant = None`.
- Variant name pattern: `<family>-ollama-gguf[-<quant>]`.
- Original identifier stored in `served_model_id`.

Quant token regex (case-insensitive):
```
^(q\d(?:_k(?:_m)?)?|int4|int8|fp16|f16)$
```
Rationale: Prevents falsely classifying non-quant tags (like `7b`, `latest`) as quant; stable, readable variant names.

## 4. Importer Implementation (`scripts/import_ollama_models.py`)
Features:
- Attempts `ollama list --format json` first; falls back to plaintext parsing with regex.
- Parses size units (KB/MB/GB/TB) into bytes.
- Derives `(family, quant)` via Strategy A.
- Calls `record_download` with:
  - `backend=ollama`
  - `format_type=gguf`
  - `quantization` (if quant token)
  - `family_override`
  - `served_model_id` = original `base[:tag]`
  - `source_provider=ollama`
  - `path` = resolved store path or synthetic (`ollama://<name>`) if store unknown.
- Dry-run mode prints planned imports without writing.
- Option: `--deprecate-placeholders` marks legacy `model-ollama-gguf*` entries deprecated.

Plaintext fallback regex:
```
^(?P<name>\S+)\s+(?P<id>[0-9a-f]{6,})\s+(?P<size_val>\d+(?:\.\d+)?)\s+(?P<size_unit>[KMGTP]B)\s+(?P<modified>.+)$
```

## 5. Registry Effects
Each imported variant sets or updates:
- `name`, `backend`, `family`, `quantization`, `download_format=gguf`, `download_path`, `download_size_bytes` (approx), `served_model_id`, `source_provider=ollama`.
- Roles initially empty; capabilities inferred (basic heuristic for vision models containing `llava`, `vl`).

## 6. Deprecation Lifecycle
Enhancements added:
- `--show-deprecated` flag for `imageworks-download list`.
- Deprecated entries hidden by default.
- `purge-deprecated` command with `--placeholders-only` & `--dry-run`.
- Importer flag `--deprecate-placeholders` to retire legacy placeholder entries while preserving audit trail until purged.

## 7. Visibility Issue & Root Cause
Problem: User saw only one (placeholder) Ollama model in `imageworks-download list`.
Cause: Strategy A entries existed but lacked `download_path`; underlying listing uses only entries with non-null `download_path` (via `list_downloads`). Not a deprecation filter issue.

## 8. Resolution Option A (Chosen): Backfill Command
Implemented `imageworks-download backfill-ollama-paths` to populate metadata for existing logical Ollama entries missing `download_path`.

Command behavior:
- Detect store using `$OLLAMA_MODELS` or `~/.ollama/models`.
- For each `backend=ollama` and `download_path is null`:
  - Set `download_path` to `<store>/<served_model_id or name>` or `ollama://<id>` fallback.
  - Optionally (`--set-format` default on) set `download_format=gguf` if missing.
  - Set `download_location` if empty (default: `linux_wsl`).
  - Leave size/checksum blank (can rebuild later with normalization).
- Supports `--dry-run` and `-v/--verbose`.
- Does not require the `ollama` binary (offline safe).

Alternative options considered (not selected):
- Add `--include-logical` flag to `list` (would require extended semantics).
- Force re-run of importer to populate paths (requires working Ollama CLI).
- Auto-populate during normalization (mixes concerns; rejected).

## 9. Supporting CLI Enhancements
- JSON list output changed to plain `print` (avoid Rich wrapping artifacts).
- Installed state override: Ollama entries treated as installed even if synthetic path.
- Added `purge-deprecated` command.
- Added `backfill-ollama-paths` command.

## 10. Tests & Quality Measures
Added / existing relevant tests:
- `test_deprecated_filtering.py`: Checks show/hide deprecated.
- `test_family_override.py`: Validates `family_override` & `served_model_id` persistence.
- `test_normalize_formats.py`: Ensures dry-run restoration (no mutation).
Pending (not yet implemented):
- Test for `backfill-ollama-paths` dry-run + apply.

## 11. Bugs Fixed
| Issue | Fix |
|-------|-----|
| Normalization dry-run mutated registry | File snapshot + restore logic added |
| JSON output contained mid-token wraps | Switched to plain stdout JSON |
| Missing Ollama visibility | Implemented backfill command |
| Placeholder clutter | Deprecation + purge workflow |
| Potential quant misclassification via underscore normalization | Preserved underscores during quant token regex pass |

## 12. Commands Reference
| Purpose | Command |
|---------|---------|
| Dry-run import | `uv run python scripts/import_ollama_models.py --dry-run` |
| Import + deprecate placeholders | `uv run python scripts/import_ollama_models.py --deprecate-placeholders` |
| Backfill missing paths | `uv run imageworks-download backfill-ollama-paths --dry-run -v` then rerun without `--dry-run` |
| Show all (incl deprecated) | `uv run imageworks-download list --show-deprecated` |
| Purge deprecated placeholders | `uv run imageworks-download purge-deprecated --placeholders-only` |
| Normalize (preview) | `uv run imageworks-download normalize-formats --dry-run` |
| Normalize + rebuild | `uv run imageworks-download normalize-formats --rebuild --apply` |

## 13. Known Limitations
1. Size/checksum for Ollama entries may remain unset unless a future explicit export occurs.
2. Backfill does not compute file list or hashes (intentional for performance).
3. No current `--include-logical` listing flag.
4. Capabilities heuristic may over-match vision.
5. No test coverage yet for backfill command.

## 14. Future Improvements (Proposed)
| Priority | Enhancement | Description |
|----------|------------|-------------|
| High | Backfill test | Add pytest covering dry-run + apply behavior. |
| High | Optional `--include-logical` | List logical entries without download metadata. |
| Medium | Export + hash Ollama models | Deterministic artifact hashing & size. |
| Medium | Show `served_model_id` in `list --details` | Improves traceability. |
| Medium | Role assignment command | Batch annotate roles for Ollama variants. |
| Low | Quant detection centralization | Share regex logic across modules. |
| Low | Capability refinement | More precise vision detection. |
| Low | JSON output for backfill | Machine-friendly diff consumption. |

## 15. Quick Diagnostics Table
| Symptom | Check | Resolution |
|---------|-------|-----------|
| Ollama model missing in list | `download_path` null? | Run backfill or re-import |
| Wrong family (quant folded) | Tag not matched by regex? | Extend regex / adjust logic |
| Placeholder still visible | `deprecated=false` | Re-run import w/ `--deprecate-placeholders` or manual mark |
| JSON parsing issues | Using older CLI version | Update to latest after switch to plain print |

## 16. Current Status
- Strategy A implemented and stable.
- Importer resilient to different Ollama CLI versions (JSON + plaintext).
- Backfill command executed (at least one entry was populated in recent run).
- Deprecation lifecycle operational.
- Documentation updated (model-downloader.md + this summary).

## 17. Immediate Actionable Next Steps
1. Implement backfill command test.
2. Decide on `--include-logical` vs always requiring backfill.
3. Optionally add `served_model_id` column for `list --details`.
4. Plan an export/hashing flow if reproducible artifact integrity becomes a requirement.

---
**End of consolidated log.** This document is intended for handoff or new discussion threads without loss of decision history or rationale.
