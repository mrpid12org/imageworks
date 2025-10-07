# Model Loader Architecture & Integration Guide

This document provides an in-depth overview of the unified **model_loader** subsystem: data model, registry lifecycle, download integration, hashing/locking, role resolution, probes, and extension points. It complements `../reference/model-downloader.md` (acquisition) and `deterministic-model-serving.md` (serving rationale).

---
## 1. Goals
The model loader layer unifies *logical* model metadata and *download provenance* into a single deterministic registry (`configs/model_registry.json`). It enables:
- Stable variant naming & selection
- Multi-backend compatibility (vLLM, LMDeploy, Ollama, GGUF loaders, future engines)
- Artifact integrity & version locking
- Role-based functional routing (caption, keywords, description, embedding, etc.)
- Progressive enrichment (performance samples, probes, chat templates)

## 2. Core Data Structures
Defined in `model_loader/models.py` (all dataclasses):

| Class | Purpose |
|-------|---------|
| `RegistryEntry` | Canonical record for a model variant (logical + download state). |
| `BackendConfig` | Launch/runtime hints (port, model path, extra args). |
| `Artifacts` / `ArtifactFile` | Source-of-truth hashing block (aggregate + per-file sha256). |
| `ChatTemplate` | Where chat templating originates (embedded or external). |
| `VersionLock` | Lock + expectation for deterministic reproducibility. |
| `PerformanceSummary` / `PerformanceLastSample` | Rolling latency / throughput stats (future adaptive scheduling). |
| `Probes` / `VisionProbe` | Health signals (e.g. vision pipeline readiness). |
| `SelectedModel` | Resolution result returned to higher-level selectors. |

### 2.1 Key `RegistryEntry` Fields Summary
Logical Identity:
- `name` (deterministic variant key)
- `family` (normalized base lineage, e.g. `llava-v1.5-13b`)
- `backend` (e.g. `vllm`, `lmdeploy`, `ollama`)
- `served_model_id` (backend-native identifier; colon tags preserved for Ollama)
- `model_aliases` (legacy / alternate references)

Classification & Roles:
- `quantization` (e.g. `q4_k_m`, `awq`, `fp16`)
- `download_format` (artifact packaging: `gguf`, `awq`, `safetensors`, etc.)
- `roles` + `role_priority` (functional routing with ordering)
- `capabilities` (booleans: text, vision, embedding, audio) – heuristic initially

Download Provenance (unified from downloader):
- `download_path`, `download_location`, `download_files`, `download_size_bytes`, `download_directory_checksum`
- `downloaded_at`, `last_accessed`
- `source_provider` (`hf`, `ollama`, `other`)

Integrity & Lifecycle:
- `artifacts` (aggregate + file-level hashes)
- `version_lock` (lock mode + expected aggregate hash + last verification timestamp)
- `deprecated` (exclusion from default selection / listing)

Enrichment:
- `performance` (rolling metrics)
- `probes` (e.g. vision readiness)
- `metadata` (misc user or tooling annotations)

## 3. Registry Persistence & Access
Implemented in `registry.py`:
- **Load**: `load_registry(path=None, force=False)` reads JSON → dataclasses, caches in-memory. Root must be a list.
- **Save**: `save_registry()` writes atomic JSON (temp file then replace). Always pretty printed + trailing newline for stable diffs.
- **Update**: `update_entries([...])` merges mutated entries.
- **Remove**: `remove_entry(name)` deletes by key.
- **Cache Strategy**: Single global cache `_REGISTRY_CACHE`; `force=True` invalidates.

JSON schema follows dataclass field names; missing optional fields default gracefully.

## 4. Download Integration (Adapter Layer)
`download_adapter.py` consolidates creation/update of download metadata replacing any legacy fragment registries:
- `record_download()` – idempotent enrichment or creation:
  1. Determine `family` (from HF id tail or `family_override`).
  2. Materialize a `ModelIdentity` via `naming.build_identity(...)` which yields both the slug
     (`family-backend-format-quant`) and the human label used by UIs.
  3. Populate file list, size, directory checksum if path exists.
  4. Compute artifact hashes if absent (delegates to `compute_artifact_hashes`).
  5. Merge roles / role_priority if provided.
  6. Persist changes by delegating to `update_entries(..., save=True)` (no extra `save_registry()` call required).
- `list_downloads(only_installed=False)` – enumerates entries with a *non-null* `download_path`. `only_installed=True` filters by path existence.
- `remove_download(name, keep_entry=True)` – clears download_* fields (retains logical shell) or purges entire entry on demand.

### 4.1 Variant Naming Contract
`ModelIdentity` centralizes normalization:

```
from imageworks.model_loader.naming import build_identity

identity = build_identity(
    family="qwen2.5-vl-7b-instruct",
    backend="vllm",
    format_type="awq",
    quantization="q4_k_m",
)
identity.slug         # -> "qwen2.5-vl-7b-instruct-vllm-awq-q4_k_m"
identity.display_name # -> "Qwen2.5 VL 7B Instruct (AWQ Q4 K M, vLLM)"
```

All importers and download flows should call `build_identity` before persisting data so the registry never contains mismatched naming variants.

### 4.2 Capability Inference
Minimal heuristic `_infer_capabilities` (presence of tokens) – expected to be refined or replaced by static curated capability declarations.

## 5. Ollama Integration (Bridge Summary)
Importer: `scripts/import_ollama_models.py`
- Uses `ModelIdentity` for slug + presentation naming; importer no longer constructs ad-hoc display strings.
- Quant detection via regex: `^(q\d(?:_k(?:_m)?)?|int4|int8|fp16|f16)$`.
- Fallback plaintext parser for older CLI output.
- `served_model_id` retains colon-annotated identifier.
- Synthetic path mapping if store not located.
Backfill: `imageworks-download backfill-ollama-paths` populates `download_path` for previously logical-only entries.

## 6. Hashing & Integrity
In `hashing.py`:
- `compute_artifact_hashes(entry)` – full recursive hash of all files under `backend_config.model_path` (relative path + sha256; aggregated deterministically).
- `update_entry_artifacts(entry)` – targeted minimal hashing (legacy support) of key config/tokenizer artifacts.
- `update_entry_artifacts_from_download(entry)` – uses `download_files` list to hash only originally recorded files (stable subset). Preferred when deterministic download file capture is required.
- `verify_model(entry, enforce_lock=True)` – updates artifacts via chosen path, enforces version lock if enabled, stamps `last_verified`.
- `VersionLockViolation` raised if locked and mismatch occurs.
- `verify_entry_hash(entry)` – returns tuple `(ok, updated_entry)` for batch checking without raising.

### 6.1 Version Lock Lifecycle
1. Entry created: `version_lock.locked = False`, no expected hash.
2. User enables lock & verifies: expected aggregate snapshot stored.
3. Subsequent drift triggers violation (unless lock disabled).
4. Regenerating artifacts via normalization or explicit verification keeps deterministic ordering (sort by `<hash>  <rel>` lines).

## 7. Role Selection & Serving (Overview)
Though not fully detailed here, `role_selection.py` (and related service code) resolves a candidate model for a functional role:
- Filters by required capabilities & non-deprecated.
- Applies `role_priority` overrides (lower int = higher precedence).
- Potential tie-breakers: performance metrics (future), quantization preference, format.
Outputs a `SelectedModel` with `logical_name`, `backend`, `internal_model_id` (e.g. `served_model_id` or variant name), and an endpoint URL assembled by higher service layers.

## 8. Probes & Performance
- `probe.py` & performance metrics modules can enrich `performance` and `probes` fields asynchronously.
- `performance.last_sample` stores last TTFT / throughput measurement enabling rolling averages.
- Vision probe may validate that image embedding or projector assets are present.

## 9. Deprecation Semantics
- `deprecated=True` hides entry from default CLI lists and role resolution (unless explicitly surfaced with flags).
- Deprecation used for: superseded placeholder variants, missing-pruned downloads (when not removed), or intentionally retired families.
- Purge step physically removes entries once safe (command: `purge-deprecated`).

## 10. Normalization & Rebuild Interplay
The model downloader’s `normalize-formats` command (in downloader CLI) interacts with registry entries that have a `download_path`:
- Re-detects `download_format` and `quantization` using shared heuristics.
- Optionally rebuilds dynamic size/file/checksum fields.
- Marks missing paths as deprecated or prunes (with `--prune-missing`).
- Avoids mutating entries without a `download_path` (logical-only entries like early Ollama imports unless backfilled).

## 11. Typical Lifecycle Flow
### 11.1 Quick Start (Programmatic)
```python
from imageworks.model_loader.download_adapter import record_download, list_downloads
entry = record_download(
  hf_id="example/Model-1B",
  backend="vllm",
  format_type="awq",
  quantization="w4g128",
  path="~/ai-models/weights/example/Model-1B",
  location="linux_wsl",
  source_provider="hf",
)
for e in list_downloads():
  print(e.name, e.download_format, e.quantization)
```

### 11.2 Lifecycle (Mermaid)
```mermaid
flowchart TD
  A[Acquire (download/import)] --> B[record_download]
  B --> C[RegistryEntry persisted]
  C --> D[normalize-formats (optional)]
  D --> E[verify_model / lock]
  E --> F[Assign roles / priorities]
  F --> G[role_selection -> SelectedModel]
  G --> H[Serve / Monitor]
  H --> I[Performance & Probes update]
  I --> J[Deprecate / Remove if obsolete]
```

### 11.3 Linear Steps
1. Download (or import) → record_download → RegistryEntry created/enriched
2. Normalization (optional) → re-detect format/quant & rebuild dynamic fields
3. Verification → hashing + version lock capture (if enabled)
4. Role assignment (manual or scripted) → roles & role_priority updated
5. Serving layer selects variant (role_selection) → produces SelectedModel
6. Performance / probes update aggregated metrics
7. Removal or deprecation if superseded or invalidated

## 12. Extension Points
| Area | How to Extend | Notes |
|------|---------------|-------|
| Additional Backends | Introduce new backend identifier + adapt server start script; reuse `record_download`. | Ensure unique `backend` and update capabilities mapping if needed. |
| Capability Detection | Replace `_infer_capabilities` with curated static map or model card parser. | Avoid false positives (e.g. substring collisions). |
| Quant Detection | Centralize logic currently partly in downloader heuristics. | Promote to shared utility for multi-source parity. |
| Performance Metrics | Feed new samples into `performance` and recompute rolling averages. | Could store window history externally if growth concerns arise. |
| Probes | Add new probe dataclasses (audio, embedding). | Extend `Probes` structure. |
| Artifact Policies | Hook post-`record_download` to enforce presence of required files. | Raise before entry persisted for deterministic failure. |

## 13. Error Handling & Safety
- Atomic writes: temp file rename prevents partial corruption.
- Duplicate names during load raise `RegistryLoadError` early.
- Missing registry file → explicit load failure (caller decides bootstrap path).
- Hashing does not throw if path missing; leaves entry untouched.
- `record_download` gracefully skips size/file enumeration if path absent (e.g. synthetic Ollama path).

## 14. Performance Considerations
- Registry JSON kept small through concise arrays; artifacts list only stored for hashed entries.
- Full hashing is O(n files * file size). For large weight repos, prefer incremental hashing (future) or subset hashing until stabilized.
- `compute_directory_checksum` (lightweight) used for quick drift detection pre full re-hash.

## 15. Security / Integrity Notes
- No remote network access in loader itself (isolation of concerns). Acquisition layer (downloader/import scripts) handles network.
- Future: can sign aggregate hashes or maintain provenance chain (e.g., supply chain attestation) by extending `metadata`.

## 16. Known Limitations / Tech Debt
| Limitation | Impact | Mitigation Path |
|-----------|--------|-----------------|
| Capability inference heuristic | Possible misclassification | Introduce explicit capability map or registry augmentation tool. |
| Mixed source quant detection spread | Duplication & drift risk | Consolidate into shared detection module. |
| Large artifact lists in JSON | Registry size growth | Add optional hash manifest externalization. |
| Lack of concurrency controls | Race on simultaneous writes | Introduce file lock or transactional write queue if multi-process usage grows. |
| Synthetic Ollama paths lack per-file hashing | Reduced integrity signals | Add export or on-demand extraction pipeline. |

## 17. Frequently Asked Questions
**Q: Why separate `family` from `name`?**
A: `name` is the full variant key; `family` allows grouping & future selection (e.g. pick best quant for a family-role pair).

**Q: When does `artifacts.aggregate_sha256` populate?**
A: On first hash pass (`record_download` if path present & hashing invoked) or via explicit verification / normalization.

**Q: How do I lock a model version?**
1. Ensure files settled. 2. Run verification locking command (future CLI) or programmatically set `version_lock.locked=True` then call `verify_model(entry)`. 3. Registry stores expected hash.

**Q: What if a download directory is manually modified?**
A: Re-verify → aggregate hash drift; if locked, violation raised; if unlocked, baseline updates.

**Q: How do deprecated entries differ from removed ones?**
A: Deprecated remain in JSON (auditable); removed entries are physically deleted from the registry list.

**Q: Do I need to drop and re-import models to pick up new capability synonyms?**
A: No. Capability normalization happens every time the registry is loaded or a caller invokes `require_capabilities`, so restarting the process is enough to pick up the expanded synonyms. The underlying JSON does not need to be rewritten unless you want the normalized map persisted on disk.

## 18. Programmatic Usage Examples
### Recording a Download
```python
from imageworks.model_loader.download_adapter import record_download
entry = record_download(
    hf_id="mistralai/Mistral-7B-Instruct-v0.2",
    backend="vllm",
    format_type="awq",
    quantization=None,
    path="~/ai-models/weights/mistralai/Mistral-7B-Instruct-v0.2",
    location="linux_wsl",
    source_provider="hf",
)
```

### Listing Downloads
```python
from imageworks.model_loader.download_adapter import list_downloads
for e in list_downloads():
    print(e.name, e.download_size_bytes)
```

### Verifying & Locking
```python
from imageworks.model_loader.registry import load_registry
from imageworks.model_loader.hashing import verify_model
reg = load_registry(force=True)
entry = reg["mistral-7b-instruct-v0.2-vllm-awq"]
entry.version_lock.locked = True
verify_model(entry)  # populates expected_aggregate_sha256
```

### Clearing Download Metadata
```python
from imageworks.model_loader.download_adapter import remove_download
remove_download("mistral-7b-instruct-v0.2-vllm-awq", keep_entry=True)
```

## 19. Cross References
- Acquisition: `../reference/model-downloader.md`
- Ollama specifics: `../runbooks/ollama-summary-and-actions.md`
- Serving rationale: `deterministic-model-serving.md`
- Future personal tagger registry usage: `../domains/personal-tagger/model-registry.md`

---
**End of Model Loader Architecture Overview**
