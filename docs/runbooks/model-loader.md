# Model Loader Runbook

Manage deterministic registry operations and vLLM orchestration using the
`imageworks-loader` CLI.

## 1. Inspect registry state
```bash
uv run imageworks-loader list --role vision
```
- Lists logical entries filtered by capability role, including backend, lock
  status, and aggregate hash.【F:src/imageworks/model_loader/cli_sync.py†L36-L120】
- Add `--include-non-installed` when troubleshooting entries missing filesystem
  paths.【F:src/imageworks/model_loader/cli_sync.py†L59-L88】

## 2. Select models for clients
```bash
uv run imageworks-loader select qwen2-vl-2b
```
- Returns resolved endpoints, served IDs, and capability flags. Exits with
  `CapabilityError` when requirements (e.g. vision) aren’t satisfied.【F:src/imageworks/model_loader/cli_sync.py†L121-L200】
- Downstream tools (chat proxy, tagger, narrator) rely on this output when
  `--use-loader` or registry integration is enabled.

## 3. Verify and lock models
```bash
uv run imageworks-loader verify qwen2-vl-2b
uv run imageworks-loader lock qwen2-vl-2b --set-expected
```
- `verify` recomputes artifact hashes and updates `version_lock.last_verified`,
  exiting with code `2` if the stored hash diverges.【F:src/imageworks/model_loader/cli_sync.py†L271-L340】
- `lock`/`unlock` toggle the deterministic lock status to protect against
  unexpected weight changes.【F:src/imageworks/model_loader/cli_sync.py†L341-L420】

## 4. Operate vLLM orchestrator
```bash
uv run imageworks-loader activate-model qwen2-vl-2b
uv run imageworks-loader current-model
uv run imageworks-loader activate-model --stop
```
- Switches the active vLLM model bound to `CHAT_PROXY_VLLM_PORT`, persisting state
  in `_staging/active_vllm.json`. Failures raise runtime errors for quick
  feedback.【F:src/imageworks/model_loader/cli_sync.py†L202-L270】

## 5. Diagnose layered registry issues
- Registries live under `configs/` (`.curated`, `.discovered`, merged snapshot).
  Re-run `uv run imageworks-download normalize-formats --dry-run` to reconcile
  stale metadata when hashes drift.【F:src/imageworks/model_loader/registry.py†L1-L200】【F:docs/reference/model-downloader.md†L1-L160】
- If layered overrides conflict, consult ADR-0001 (ensure curated entries exclude
  dynamic fields) and the `layered-registry` troubleshooting notes.【F:docs/reference/troubleshooting/layered-registry-curated-override-issue.md†L1-L80】

## 6. Troubleshooting
| Symptom | Checks |
| --- | --- |
| `CapabilityError` | Verify registry roles are correct and that `capabilities` includes requested flags. Update `model_registry.curated.json` if needed.【F:src/imageworks/model_loader/service.py†L1-L180】 |
| `verify` exits with code 2 | Inspect file changes in the weights directory; rerun with `--set-expected` after confirming the new hash is valid.【F:src/imageworks/model_loader/hashing.py†L1-L200】 |
| Orchestrator stuck in previous model | Remove `_staging/active_vllm.json` and restart the proxy; ensure `CHAT_PROXY_VLLM_SINGLE_PORT=1` in the environment.【F:src/imageworks/model_loader/cli_sync.py†L202-L270】 |
| Registry edits lost | Make changes in `.curated` only and rerun tooling; `.json` is regenerated on save. Use version control for curated file changes.【F:src/imageworks/model_loader/registry.py†L1-L200】 |

Archive CLI output in ops logs when making registry changes for auditability.
