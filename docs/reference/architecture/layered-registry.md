# Layered Model Registry Design

This document describes the layered registry architecture introduced on 2025-10-02.

## Files

* `configs/model_registry.curated.json` – Hand-maintained, stable metadata (display names, roles, licensing, stable backend assignments, preferred alternatives, version locks, etc.). Avoid writing dynamic operational state here.
* `configs/model_registry.discovered.json` – Auto-generated / mutable overlay produced by tooling (downloads, importers, scans, normalization). Contains dynamic runtime fields and new entries not yet promoted to curated.
* `configs/model_registry.json` – Materialized merged snapshot (curated overlaid by discovered). Generated on each load/save for backward compatibility with legacy code paths still expecting a single file. Treat as read-only.

## Merge Semantics

1. Load curated list first.
2. Overlay discovered entries by `name` (discovered replaces curated if duplicate).
3. In-memory registry only contains final merged entries.
4. Save operation writes only the discovered overlay (plus regenerated snapshot). Curated file remains untouched unless manually edited.

## Classification Heuristic (Legacy Migration)

When migrating from a pre-layer single `model_registry.json`:
* Entries with `metadata.created_from_download` = true → discovered
* Entries whose `backend` in {`ollama`, `unassigned`} → discovered (logical or transient sources)
* All others → curated

The original unified file is renamed: `model_registry.json.backup.pre_split.json`.

## Dynamic Field Rules

Any of these fields mark an entry as dynamic (and thus belonging in discovered or requiring an overlay if originally curated):
* `download_path`, `download_format`, `download_location`
* `download_size_bytes`, `download_files`, `download_directory_checksum`
* `downloaded_at`, `last_accessed`
* Non-empty `performance` metrics, `probes`
* `metadata.created_from_download`

When an existing curated entry gains a dynamic attribute, a full entry copy is written into the discovered overlay on save.

## Duplicate Handling

* Curated + discovered merge should yield unique names. If the **same name** appears in both, discovered replaces curated (intentional override).
* Explicit (test) loads that pass a custom path use single-file mode and enforce strict duplicate errors (unless `IMAGEWORKS_ALLOW_REGISTRY_DUPES=1`).

## Environment Variables

* `IMAGEWORKS_REGISTRY_DIR` – Override base directory containing registry files.
* `IMAGEWORKS_REGISTRY_NO_LAYERING=1` – Force single-file mode **only when an explicit path is also provided**.
* `IMAGEWORKS_ALLOW_REGISTRY_DUPES=1` – Enable tolerant dedupe (mainly for transitional tooling; avoid in production curated sets).

## Rationale

Separating curated from discovered:
* Prevents volatile operational writes (e.g., last access timestamps) from creating noisy diffs in human-reviewed curated content.
* Simplifies regeneration flows: discovered layer can be safely discarded and rebuilt from actual artifacts and local runtime state.
* Enables future promotion workflow (curated review of newly discovered variants).

## Typical Workflows

| Task | File(s) touched |
|------|-----------------|
| Add new first-class model with canonical metadata | Edit curated JSON |
| Download / import a model (CLI) | Updates discovered overlay |
| Recompute sizes / rebuild checksums | Updates discovered overlay (and snapshot) |
| Promote a discovered entry to curated | Manually move entry from discovered to curated then remove from discovered |
| Reset dynamic state | Delete `model_registry.discovered.json` and re-run importers/scans |

## Promotion Guide

1. Identify candidate entry in discovered overlay.
2. Copy entry object into curated file (optionally prune transient fields like `downloaded_at`, `last_accessed`).
3. Remove that entry from discovered overlay.
4. Run any normalization command and confirm merged snapshot stable.

## Future Enhancements (Planned)

* Schema version tagging (`schema_version`) per file to enable automated upgrades.
* Sparse overlay (only storing changed dynamic fields instead of full entry copy) to reduce duplication.
* Promotion CLI command to automate moving entries from discovered → curated.
* Integrity audit command verifying curated file has no dynamic-only fields.

## Troubleshooting

| Symptom | Cause | Resolution |
|---------|-------|------------|
| Entry appears twice in git history with different dynamic timestamps | Dynamic fields were merged into curated unintentionally | Remove dynamic fields from curated or move entry to discovered only |
| New download not visible | Discovered layer not saved or snapshot stale | Trigger any CLI that saves (download/scan) or manually call a `save_registry()` path |
| Tests unexpectedly load production models | Explicit path not passed; layering pulled global files | In tests, always pass `--registry-path` pointing to a temp file |
| Duplicate name error in tests | Two entries in single-file registry share name | Rename one or enable `IMAGEWORKS_ALLOW_REGISTRY_DUPES=1` (not recommended) |

## Reference

Primary implementation logic in `imageworks/model_loader/registry.py`.
