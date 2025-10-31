# Model naming, display IDs, and lingering Ollama entries

This document explains how models are named across backends, how display names differ from logical names, why non-existent ("lingering") Ollama models can appear in OpenWebUI, and options to fix or improve the situation.

## Key terms

- Logical name (registry key)
  - The stable, unique identifier for a variant in the unified registry.
  - Used internally by tools and code paths; does not need to be pretty.
  - Derived from: family + backend + (format) + (quant). Exact shape is backend dependent.

- Display name
  - Human-friendly label for UIs (OpenWebUI, etc.).
  - Comes from curated metadata or derived heuristics; may not be unique.
  - We often disambiguate by appending the quantization to create a “display id.”

- Display id (published by chat proxy)
  - The id field returned by `/v1/models` is intended for UI and is usually `display_name[-quant]`.
  - The logical name is also returned in `extensions.logical_id`.

## Current behavior by backend

### vLLM / HF (filesystem-backed)
- Registry entry has a `download_path` to a local directory.
- Chat proxy filters out entries where the path does not exist (hides ghosts).
- Logical name derives from family/backend/format/quant (e.g., `qwen2-vl-vllm-awq-q4_0`).
- Display name is curated or normalized for readability.

### Ollama (tag-backed)
- Models referenced by Ollama tag, stored in `served_model_id` and/or `name`.
- There is no filesystem path we can check. The proxy currently does not verify the tag exists in the Ollama store before listing.
- Logical variant name is still derived for consistency, but existence should map to the local Ollama tag list.

## Why “lingering” Ollama models appear

Even if you haven’t deleted any Ollama models manually, lingering entries can appear because of these factors:

1) Importer is additive unless purged
- The Ollama importer (`scripts/import_ollama_models.py`) creates/updates entries via the Ollama HTTP API (`/api/tags`, `/api/show`).
- It does not remove older Ollama entries unless you pass `--purge-existing`. Over time, renamed tags or format/quant changes leave old entries behind in the registry.

2) Proxy does not validate Ollama tags (current behavior)
- `/v1/models` hides non-existent paths for vLLM/HF, but Ollama entries aren’t checked against the local tag list (`/api/tags`).
- Result: stale registry entries for Ollama can still be listed by the proxy in OpenWebUI.

3) Early placeholders and renames
- Early discovery runs and evolving naming rules (e.g., adding quant tokens) can leave placeholder or legacy variants that were never actively removed.

## Evidence in code

- Importer logic:
  - File: `scripts/import_ollama_models.py`
  - Key paths:
    - `_collect_model_data()` derives family + quant and constructs `OllamaModelData` (stores `name` = tag, `display_name`, and pseudo path like `ollama://<tag>`).
    - `_persist_model()` creates or updates entries. It does not remove unrelated Ollama entries unless `--purge-existing` is used. There is an internal helper `_purge_existing_ollama_entries()` but it’s only used when explicitly requested.

- Chat proxy listing:
  - File: `src/imageworks/chat_proxy/app.py`
  - `/v1/models` filters by `download_path` only for non-Ollama backends. Ollama variants aren’t validated against current tags.

## Display id and duplicate quant suffix

- The proxy historically constructed `id` for UI as `display_name[-quant]`, leading to strings like `pixtral-local-latest-Q4_K_M-q4_k_m`.
- The new `ModelIdentity` pipeline emits a single, clean display label (e.g., `Pixtral 12B Captioner Relaxed (GGUF Q4 K M, Ollama)`).
- `/v1/models` now surfaces that label directly so the OpenWebUI picker and CLI tables show consistent human-readable names while still exposing the logical slug in `extensions.logical_id`.

## Recommended fixes and options

Pick any subset of these; they’re independent.

1) Importer reconciliation (recommended)
- During each import, compute the set of current Ollama tags from `/api/tags`
  (the same data surfaced by `ollama list`).
- Remove (or deprecate) any registry entries where `backend == 'ollama'` and `served_model_id` is not present in the tags set.
- Benefit: The registry reflects the current local Ollama state after each import, without needing `--purge-existing`.

2) Proxy-side Ollama tag verification (fast UX win)
- At `/v1/models` generation time, call `GET http://127.0.0.1:11434/api/tags` (short timeout, small cache) and include only Ollama entries whose tags are present.
- Optional opt-in env to show all anyway (e.g., `CHAT_PROXY_INCLUDE_MISSING_OLLAMA=1`).
- Benefit: Hides ghosts in OpenWebUI immediately without altering the registry.

3) One-time cleanup utility
- Use the download/remove CLI to prune Ollama entries that are not present in `ollama list`, and optionally run `ollama rm` (if applicable) to remove local tags.
- Benefit: Cleans the current registry state. This is complementary to (1), which keeps it clean going forward.

4) Suppress duplicate quant in display ids (UX polish)
- ✅ Implemented by replacing the ad-hoc concatenation with the `ModelIdentity` label; the proxy falls back to the logical slug only when duplicates occur.

5) Normalize before persistence (new)
- ✅ `record_download` and all importers call `ModelIdentity` prior to writing registry entries, ensuring slugs + labels are clean at source instead of relying on downstream hiding/deduping.

## CLI naming guidance

- Chat proxy (OpenAI-compatible `/v1/chat/completions`):
  - Accepts both the logical id (registry key) and the display id.
  - If given a display id, the proxy resolves it to the logical id internally.

- Internal CLIs (downloader, registry tools):
  - Prefer the logical name (registry key). It’s unique and stable and unambiguous for scripting.
  - Display names are user-friendly but not guaranteed unique.

## Environment flags (current vs proposed)

- Current flags:
  - `CHAT_PROXY_INCLUDE_TEST_MODELS` — Include testing/demo models in `/v1/models` when set to a truthy value.
  - `IMAGEWORKS_IMPORT_INCLUDE_TESTING` — Allow importer to persist testing/demo entries (otherwise they’re skipped).

- Proposed (for proxy-side Ollama filtering):
  - `CHAT_PROXY_INCLUDE_MISSING_OLLAMA` — If truthy, include Ollama entries even if their tags aren’t present locally. Default would be to hide missing.

## Practical workflows

- Keep registry aligned with local Ollama:
  - Run `uv run imageworks-loader reset-discovered --backend ollama` to clear additive records before rerunning the importer.
  - Optionally pass `--purge-existing` to the importer for a full rebuild in one shot.
  - Or rely on the proxy-side tag check to hide ghosts from UI while retaining historical entries in the registry (less preferred).

- Ensure clean UI naming:
  - Implement duplicate-quant suppression in the proxy, so display ids don’t repeat the quant suffix.

## Pointers

- Importer: `scripts/import_ollama_models.py`
- Chat Proxy app (models list + endpoints): `src/imageworks/chat_proxy/app.py`
- Registry accessors and schema: `src/imageworks/model_loader/registry.py` and `src/imageworks/model_loader/models.py`
- Download adapter (unified record/update): `src/imageworks/model_loader/download_adapter.py`

## Notes

- The fixes under “Recommended” are safe and low-impact:
  - (1) keeps the registry true to local Ollama state.
  - (2) immediately removes ghost entries from the UI without modifying data.
  - (4) resolves a cosmetic but confusing duplication in names.
- We can stage these one at a time; nothing requires a breaking change.
