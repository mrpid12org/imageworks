# ADR 0001: Unified Model Identity and Registry Hygiene

- **Status:** Accepted
- **Date:** 2025-10-04
- **Owners:** Imageworks platform team

## Context

Historically the registry mixed logical model names, human-readable display
labels, and backend-specific identifiers. Individual importers inferred names
with ad-hoc string concatenation, leading to:

- Duplicate or inconsistent slugs (`pixtral-local-latest-Q4_K_M-q4_k_m`).
- Stale Ollama entries lingering after tag renames.
- Multiple code paths guessing display names in slightly different ways.
- Chat proxy logic that had to dedupe IDs at runtime and append quant suffixes.

The new download flows (`record_download`) needed a single source of truth for
both machine IDs and UI labels, and documents such as the Ollama runbook called
out the need for pre-write cleaning rather than post-hoc filtering.

## Decision

1. **Introduce `ModelIdentity`** (`src/imageworks/model_loader/naming.py`) as the
   canonical normalisation pipeline. All importers and downloaders call it to
   derive:
   - `slug`: `family-backend-[format]-[quant]`
   - `display_name`: concise human label (`Family Size Variant (Format Quant, Backend)`).

2. **Update persistence paths** to require `ModelIdentity` before writing
   registry entries (`record_download`, Ollama importer, CLI tooling). This
   enforces clean data at write time and keeps the slugs consistent.

3. **Expose display names as-is** through the chat proxy. `/v1/models` now
   publishes the friendly `display_name` directly and falls back to logical IDs
   only when duplicates appear.

4. **Add maintenance tooling** (`imageworks-loader reset-discovered`) to clear
   discoverable overlays and allow importer reruns to rebuild the registry from
   scratch without manual JSON edits.

## Consequences

- Registry files now contain deterministic slugs and labels, eliminating
  duplicate quant suffixes.
- Display IDs remain human-friendly across CLI tables, OpenWebUI, and the proxy.
- Importers must populate the identity fields; callers that circumvent
  `ModelIdentity` risk raising tests or lint checks.
- Operators can reconcile state by running `reset-discovered` and rerunning the
  importers, or invoke Ollama imports with `--purge-existing` for a full wipe.
- Future ADRs can build on this foundation for additional naming policies or
  backend-specific decorators.

## Links

- Implementation: `src/imageworks/model_loader/naming.py`
- Runbook: [`docs/runbooks/model-naming-and-ollama-lingering.md`](../runbooks/model-naming-and-ollama-lingering.md)
- Registry architecture: [`docs/architecture/model-loader-overview.md`](../architecture/model-loader-overview.md)
