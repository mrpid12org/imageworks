# Model Loader Reference

The model loader package provides a deterministic way to read and update the
ImageWorks model registry. It is a Python library with a Typer-powered CLI
(`imageworks-loader`) rather than a standalone web service.

## Registry Layout

The active registry is layered under `configs/`:

- `model_registry.curated.json` – human-maintained overrides: display names,
  deliberate launch flags, backend host overrides, role tweaks, etc. Curated
  entries now explicitly carry `metadata.registry_layer="curated"`.
- `model_registry.discovered.json` – dynamic state written by tooling
  (downloaders, rediscovery, probes). This is where runtime fields such as
  `download_path`, file hashes, performance samples, and timestamps live.
- `model_registry.json` – merged snapshot regenerated on load/save for legacy
  scripts that still expect a single file.

`imageworks.model_loader.registry.load_registry()` reads both layers, merges the
dynamic data with curated overrides, and caches the result. `save_registry()`
rewrites the discovered overlay and regenerates the merged snapshot while
preserving the curated file. The curated overlay purposefully omits dynamic
fields so rediscovery can continue to refresh them without clobbering manual
tuning.

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

### `activate-model`

```
uv run imageworks-loader activate-model <logical_name>
uv run imageworks-loader activate-model --stop
```

Controls the single-port vLLM orchestrator used by the chat proxy. When passed
a vLLM-backed logical name the command stops any running instance, launches the
requested model on the canonical port, waits for the `/v1/health` probe, and
writes the new metadata to the orchestrator state file. `--stop` (or passing the
literal value `none`) shuts down the active instance without starting a new
model. The command exits with a non-zero status if orchestration is disabled or
the switch fails.

The downloader/importer layer also sets sensible defaults for Ollama entries.
When an Ollama model is recorded and no host is provided, the tooling now
injects `backend_config.host = "host.docker.internal"` (overridable with
`IMAGEWORKS_OLLAMA_HOST`) so proxy containers can talk to the host daemon
without additional manual curation.

### `current-model`

```
uv run imageworks-loader current-model
```

Prints the JSON descriptor stored in the orchestrator state file (`null` when no
active vLLM process is registered). This is handy for verifying which model is
currently bound to the canonical port.

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
