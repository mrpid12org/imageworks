# Model Loader Reference

The model loader package provides a deterministic way to read and update the
ImageWorks model registry. It is a Python library with a Typer-powered CLI
(`imageworks-loader`) rather than a standalone web service.

## Registry Layout

The active registry is kept in a layered structure under `configs/`:

- `model_registry.curated.json` – hand-maintained baseline metadata (display
  names, roles, licensing, version locks, etc.).
- `model_registry.discovered.json` – dynamic overlay written by tooling
  (downloader, importers, normalization commands). Runtime fields such as
  `download_path`, `download_size_bytes`, `performance`, and probe data live
  here.
- `model_registry.json` – merged snapshot regenerated on load/save for backward
  compatibility with scripts that still expect a single file.

`imageworks.model_loader.registry.load_registry()` reads the curated and
discovered layers, overlays them in memory, and caches the merged result.
`save_registry()` only rewrites the discovered layer (and refreshes the merged
snapshot) so curated edits remain stable.

## CLI (`imageworks-loader`)

Run the commands with `uv run imageworks-loader …` if you use the project’s
virtual environment. The CLI operates directly on the layered registry.

### `list`

```
uv run imageworks-loader list [--role ROLE]
```

Loads the layered registry, applies optional role filtering, and prints a JSON
array with basic fields (`name`, `backend`, `locked`, `vision`, `roles`,
`hash`).

### `select`

```
uv run imageworks-loader select <logical_name> [--require-vision]
```

Returns a JSON descriptor containing:

- `logical_name`
- `endpoint` (resolved from backend configuration)
- `backend`
- `internal_model_id` (falls back to the logical name if no explicit override)
- `capabilities` (normalized boolean map)

Missing capabilities raise `CapabilityError` with an exit code of `1`.

### `verify`

```
uv run imageworks-loader verify <logical_name>
```

Recomputes artifact hashes for the entry and compares them with the stored
version lock. When `version_lock.locked` is true and the hashes diverge the
command exits with code `2`. On success it prints the aggregate sha256 and the
timestamp captured in `version_lock.last_verified`.

### `lock` / `unlock`

```
uv run imageworks-loader lock <logical_name> [--set-expected]
uv run imageworks-loader unlock <logical_name>
```

`lock` marks the entry as locked. With `--set-expected` it first computes the
aggregate hash (if missing) and stores it in
`version_lock.expected_aggregate_sha256`. `unlock` simply clears the locked
flag. Both commands persist via `save_registry()`.

### `probe-vision`

```
uv run imageworks-loader probe-vision <logical_name> <image_path>
```

Invokes `model_loader.probe.run_vision_probe`, forwards the image to the target
backend, and prints the resulting JSON payload (vision_ok flag, latency, notes
and probe version). Errors are surfaced with exit code `1`.

### `metrics`

```
uv run imageworks-loader metrics [<logical_name>]
```

Shows lightweight operational data derived from the registry:

- When called without a name it lists all entries with their lock state and a
  truncated artifact hash.
- When a name is provided it prints the full aggregate hash and lock status for
  that entry.

## Python API Highlights

Import the helpers directly from `imageworks.model_loader`:

```python
from imageworks.model_loader.registry import load_registry, get_entry, save_registry
from imageworks.model_loader.service import select_model, CapabilityError
from imageworks.model_loader.hashing import verify_model, VersionLockViolation
```

- `load_registry(force=True)` refreshes the layered cache.
- `get_entry(name)` returns a `RegistryEntry` dataclass; raising `KeyError` when
  absent.
- `select_model(name, require_capabilities=None)` resolves backend endpoint
  URLs and returns a `SelectedModel`.
- `verify_model(entry, enforce_lock=True)` recomputes hashes and updates
  `artifacts.aggregate_sha256` and `version_lock.last_verified`. Set
  `enforce_lock=False` to update hashes without enforcing the lock.

For a deeper architectural explanation (data classes, layering policy, download
interactions) see `docs/architecture/model-loader-overview.md` and
`docs/architecture/layered-registry.md`.
