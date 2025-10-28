# Unified Model Registry & Migration (Personal Tagger + Cross‑App)

This document now describes the **unified deterministic model registry** and how we are migrating away from the legacy in‑code Personal Tagger registry.

## 1. Current State

The legacy in‑code registry (`model_registry.py`) was removed in favor of the unified deterministic registry at `configs/model_registry.json`.

Removal details:
- Removal date: 2025-10-02
- Replacement: role-based selection + hash/version locking via unified registry
- Migration status: Complete (tests now validate roles directly against `configs/model_registry.json`).

The unified registry is **backend + artifact + capability + roles** centric and supports all current & future applications (Personal Tagger, Color Narrator, etc.).

### 1.1 Quickstart (Role-Based Personal Tagger)
Run the Personal Tagger end‑to‑end using dynamic role resolution (no hard‑coded model names):

```bash
uv run imageworks-personal-tagger run \
	-i ./sample_images \
	--use-registry \
	--caption-role caption \
	--keyword-role keywords \
	--description-role description \
	--output-jsonl outputs/results/tagger_role_demo.jsonl \
	--summary outputs/summaries/tagger_role_demo.md \
	--dry-run
```

What happens:
- Each role is mapped to the first non‑deprecated registry entry advertising that role *and* `vision` capability.
- The resolved model names are logged (`event_type=role_resolution`).
- Because `--dry-run` is used, no metadata is written—remove it for production.

Upgrade workflow: edit `configs/model_registry.json`, adjust roles or backend settings, re‑lock hash (`verify <name> --lock`), and re‑deploy—no script changes required.

Skip `--caption-role` etc. to accept default role names; supply them only when experimenting with alternative role labels.

## 2. Unified Schema (Dataclass `RegistryEntry`)

Field (grouped) | Description
----------------|------------
`name` | Logical registry key (unique, kebab/lowercase preferred).
`display_name` | Human friendly label; defaults to `name` if absent.
`backend` | Serving backend: `vllm | ollama | gguf | lmdeploy`.
`backend_config.port` | Listening port for the backend process.
`backend_config.model_path` | Local filesystem root of model weights / HF snapshot.
`backend_config.extra_args[]` | Extra launch flags (e.g. tensor parallel options).
`capabilities{}` | Boolean feature flags: examples `text, vision, audio, embedding`.
`artifacts.aggregate_sha256` | Deterministic aggregate of per‑file hashes (version identity).
`artifacts.files[]` | `{path, sha256}` for tracked subset (tokenizer, projector, etc.).
`chat_template.{source,path,sha256}` | Prompt formatting source (embedded code vs external file).
`version_lock.locked` | If true, drift from `expected_aggregate_sha256` is an error.
`version_lock.expected_aggregate_sha256` | The hash we expect when locked.
`version_lock.last_verified` | ISO timestamp of last successful verification.
`performance.*` | Lightweight rolling summary (TTFT avg, throughput, last sample). (Advisory only — not authoritative for scheduling yet.)
`probes.vision` | Latest vision probe result (health + latency snapshot).
`served_model_id` | Identifier required by backend (e.g. Ollama tag `qwen2.5vl:7b`).
`model_aliases[]` | Alternative names we’ll attempt during endpoint selection / preflight.
`roles[]` | Functional roles satisfied (e.g. `caption`, `keywords`, `description`, `narration`, `embedding`). Enables future stage resolution without legacy map.
`license` | SPDX or short license label (informational / compliance filtering).
`source` | Raw downloader import block: `{huggingface_id, format, path, size_bytes, directory_checksum, files:[{path,size}]}`.
`metadata{}` | Freeform extension bag (notes, tuning lineage, etc.).
`deprecated` | If true, still loadable but hidden from defaults / emits warning.
`profiles_placeholder` | Reserved for future performance profiles (currently unused).

Minimal required for a usable entry: `name`, `backend`, `backend_config`, `capabilities`.

## 3. Hashing & Version Locking

We maintain reproducibility via artifact hashing:

Workflow | Steps
---------|------
Initial hash | `compute_artifact_hashes` (invoked by sync or manual tool) collects file hashes and sets `artifacts.aggregate_sha256`.
Lock version | Run `verify` with `--lock` to set `version_lock.locked=true` and store the current aggregate in `expected_aggregate_sha256`.
Continuous verify | Re-run `verify` later; if locked and hash changed → failure (drift detected) before deployment.

Aggregate hash definition (current phase): SHA256 of sorted lines `"<file_sha256>  <relative_path>"` for each tracked file (tokenizer / projector / config subset). Future enhancement: full directory traversal for stronger integrity.

## 4. CLI Maintenance Commands

Command | Purpose | Example
--------|---------|--------
`sync-downloader <models.json>` | Import / enrich unified registry from a downloader manifest (adds `source` and skeleton entries). | `uv run imageworks-model-registry sync-downloader downloads/models.json` (exact entrypoint name depends on packaging; currently module `imageworks.model_loader.cli_sync` app)
`verify <name> [--lock]` | Recompute hashes, optionally lock to current aggregate. | `uv run imageworks-model-registry verify llava-1.5-7b-awq --lock`

Dry run support: `sync-downloader --dry-run` to view planned create/update counts without modifying the file.

## 5. Migration Plan (Detailed)

Phase | Action | Exit Criteria
------|--------|--------------
A (active) | Keep legacy registry; mark deprecated; introduce roles in unified entries. | All new code reads unified registry for listing / selection.
B | Add role-based resolution (e.g. `--caption-role caption --prefer vision`). | Personal Tagger can run with zero legacy references in normal path.
C | Remove legacy `MODEL_REGISTRY`; tests pivot to unified roles fixture. | No imports of legacy module remain (except migration test).
D (optional) | Expand hashing scope to entire model snapshots; enforce on CI before deploy. | CI gate denies drift.

## 6. Adding / Updating a Model
1. Place / download model into a stable path (e.g. `home/<user>/ai-models/...`).
2. Run `sync-downloader` with the downloader manifest (or add entry manually if external).
3. Edit backend specifics (`backend_config.port`, add `served_model_id`, adjust `capabilities`, add any `roles`).
4. Run `verify <name>` to compute hashes.
5. If satisfied, run `verify <name> --lock` to freeze version.
6. Commit updated `configs/model_registry.json`.

## 7. Roles vs Capabilities
Roles answer: “For which pipeline function can I pick this model?”; capabilities answer: “What modalities / features does this model support?”

Example: A multimodal chat model used for captioning and description might have:
```
roles: ["caption", "description"]
capabilities: {"text": true, "vision": true, "embedding": false}
```

## 8. Deprecation Strategy
Legacy registry import triggers a `DeprecationWarning`. New code should avoid referencing it—use role filters over `configs/model_registry.json` instead. During Phase C we will remove the legacy file after tests migrate.

## 9. FAQ
Q: Did we lose older curated stage model lists?
A: No. They still exist in the legacy module until removal; their semantics migrate into `roles[]` plus `model_aliases`.

Q: How do alias fallbacks work?
A: Resolution order (planned): explicit selection → `served_model_id` → first matching `model_aliases` reachable on backend → error.

Q: Where should performance metrics go long‑term?
A: Only summary snapshots stay in the registry; detailed time‑series metrics belong in application / observability logs.

Q: How broad is hashing coverage now?
A: Targeted important config/projector/tokenizer files. Future phases may expand to all files for stronger guarantees.

## 10. Next Enhancements (Tracked)
- Populate per‑file sizes & (optionally) hashes inside `source.files` during sync.
- Consolidate duplicate hashing helper code sections (two implementations currently present in `hashing.py`).
- Add tests: hash determinism, sync idempotency, lock enforcement, legacy warning.
- Role-based model auto‑selection (e.g. `select_models(role="caption", vision=True)`).

---
*Last updated after unified registry implementation & initial hashing CLI integration.*

## 11. Role-Based Selection in Personal Tagger (Implemented)

The Personal Tagger can now resolve models dynamically by functional role instead of hard-coding per‑stage model names.

### 11.1 Enabling Role Mode
Pass `--use-registry` to the tagger CLI. The stage model flags become *preferences* instead of mandatory identifiers.

Example:
```bash
uv run imageworks-personal-tagger run \
	-i /data/images \
	--use-registry \
	--caption-role caption \
	--keyword-role keywords \
	--description-role description \
	--output-jsonl outputs/results/role_mode.jsonl
```

### 11.2 Resolution Order
For each role (caption / keywords / description):
1. Preferred model name (if a legacy `--caption-model` etc. was provided) and it advertises the role + required capabilities.
2. First non‑deprecated registry entry whose `roles` includes the role and satisfies required capabilities (`vision` for these roles).
3. Failure with an actionable error if nothing matches.

### 11.3 Capability Enforcement
Roles for visual stages automatically require `capabilities.vision == true`. This prevents accidental selection of text‑only models.

### 11.4 Logging
When role resolution occurs, a log line is emitted with `event_type=role_resolution` and the mapping:
```
caption_role → resolved caption model
keyword_role → resolved keyword model
description_role → resolved description model
```

### 11.5 Interaction with Version Locking
Because resolved models always originate from the unified registry, version locking (hash drift detection) applies uniformly. Upgrading a role simply means editing `configs/model_registry.json` (and re‑verifying / locking), with no CLI changes for operators.

### 11.6 When to Use
Use role mode in production or shared environments where:
- You want centralized upgrades without editing deployment scripts.
- You rely on hash locking for reproducibility.
- You plan to add fallback / profile semantics later.

Use legacy explicit model flags only for rapid experiments or temporary local overrides.

### 11.7 Troubleshooting
Symptom | Likely Cause | Remedy
--------|--------------|-------
`Failed to resolve role 'caption'` | No registry entry has that role or vision capability | Add `roles:["caption"]` to a suitable entry & re-run.
`HTTP 404 /chat/completions` | Backend process not running on port in `backend_config.port` | Start backend or update port.
`Capability error` | Model lacks required capability (`vision`) | Select a multimodal model or adjust capabilities after verifying they are real.

### 11.8 Adding a New Role
1. Add role name to one or more entries' `roles` list.
2. (Optional) Introduce a new CLI flag if it's a common stage (otherwise reuse `--caption-role` etc. with a custom value).
3. Run tagger with `--<stage>-role <new-role>`.

---
*Role-based selection section added after migration to unified registry with dynamic role resolution.*
