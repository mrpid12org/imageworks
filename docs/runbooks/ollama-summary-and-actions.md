## Summary of Ollama Integration Changes & Actions

This document summarizes the evolution of Ollama model integration into the unified
`imageworks` registry and provides a runbook for managing related artifacts.

### Key Integration Points

1.  **Unified Registry**: All models, regardless of source (HuggingFace, Ollama),
    are tracked in the layered registry (`configs/model_registry.curated.json`
    + `model_registry.discovered.json`, merged into `model_registry.json` for
    compatibility).
2.  **Ollama Importer**: The script `scripts/import_ollama_models.py` is the
    canonical way to ingest locally-pulled Ollama models.
3.  **Logical Entries**: Imported Ollama models are treated as "logical" entries.
    - Are visible via `imageworks-download list` and downstream tooling.
    - `download_path` points to the Ollama store (`~/.ollama/models/...`).
    - `served_model_id` stores the canonical Ollama tag (e.g., `llava:7b`).

### Evolution of Naming and Discovery

-   **Initial Strategy (Placeholder)**: Created generic entries like
    `model-ollama-gguf`.
    -   **Problem**: Did not distinguish between different Ollama models.
    -   **Status**: Superseded.

-   **Strategy A (Tag-based Naming)**: `ollama pull my-model:7b` becomes a variant
    named `my-model-7b-ollama-gguf`.
    -   **Problem**: Still potential for ambiguity if tags are not unique.
    -   **Status**: Implemented in the current importer.

-   **Strategy B (Content-based Hashing)**: Future goal to use Ollama's manifest
    digest for a truly unique ID.
    -   **Status**: Not yet implemented.

### Common Issues and Resolutions

#### 1. Stale or Duplicate Entries

**Symptom**: `imageworks-download list` shows old, generic, or duplicate Ollama
variants.

**Cause**: Remnants of earlier import strategies or manual registry edits.

**Resolution**:

1.  **Reset**: Clear the discovered layer for Ollama models:
    ```bash
    uv run imageworks-loader reset-discovered --backend ollama
    ```
2.  **Re-import**: Run the canonical importer to rebuild the entries cleanly:
    ```bash
    uv run python scripts/import_ollama_models.py
    ```

#### 2. Logical Entries Not Appearing

**Symptom**: `ollama list` shows models, but they are missing from
`imageworks-download list`.

**Cause**: The registry entry may lack the `download_path` metadata required by
the `list` command to consider it "installed".

**Resolution**:

-   **Backfill Metadata**: A dedicated command was added to fill in the necessary
    `download_path` and other fields for existing logical entries.
    ```bash
    uv run imageworks-loader rebuild-ollama --location linux_wsl
    ```

#### 3. Deprecated Placeholders

**Symptom**: The registry contains old `model-ollama-gguf*` entries that are no
longer in use.

**Cause**: Legacy imports before per-model naming was established.

**Resolution**:

-   The importer can mark these as deprecated during a run:
    ```bash
    uv run python scripts/import_ollama_models.py --deprecate-placeholders
    ```
-   Deprecated entries are hidden by default. Use the `--show-deprecated` flag for
    `imageworks-download list`.
-   They can be permanently removed with the `purge-deprecated` command.

### Runbook: Common Ollama Management Tasks

| Task                        | Command                                                              |
| --------------------------- | -------------------------------------------------------------------- |
| Import all local models     | `uv run python scripts/import_ollama_models.py`                      |
| Reset & re-import all       | `uv run imageworks-loader reset-discovered --backend ollama` then re-import |
| Backfill missing paths      | `uv run imageworks-loader rebuild-ollama --location linux_wsl`       |
| Show all (incl deprecated)  | `uv run imageworks-download list --show-deprecated`                  |
| Purge deprecated placeholders | `uv run imageworks-download purge-deprecated --placeholders-only`    |
| Normalize (preview)         | `uv run imageworks-download normalize-formats --dry-run`             |
| Normalize + rebuild         | `uv run imageworks-download normalize-formats --rebuild --apply`     |
