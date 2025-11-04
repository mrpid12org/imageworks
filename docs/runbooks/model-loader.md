# Model Loader Runbook

Operational procedures for maintaining the ImageWorks model registry and deterministic loader.

---
## 1. Daily Checks

1. Run `uv run imageworks-models list` to ensure registry loads without errors.
2. Inspect `configs/model_registry.curated.json` for pending PRs or manual edits awaiting merge.
3. Confirm `_staging/active_vllm.json` aligns with expected logical model (if single-port mode is in use).
4. Review `logs/model_loader.log` (if enabled) for lock violations or probe failures.

---
## 2. Adding or Updating a Model

### 2.1 CLI steps
1. Download/import weights: `uv run imageworks-download download owner/repo --location linux_wsl`.
2. Edit `configs/model_registry.curated.json`:
   - Add entry with `backend`, `served_model_id`, `roles`, `capabilities`, and `download_path`.
   - Supply `backend_config` (launch command for vLLM, base URL for remote, etc.).
3. Validate schema: `uv run python -m imageworks.model_loader.registry --validate` (optional helper script or run unit tests).
4. Load registry: `uv run imageworks-models list`.
5. Verify hash: `uv run imageworks-models verify <logical-name>`.
6. Lock entry: `uv run imageworks-models lock <logical-name> --set-expected`.
7. Reload chat proxy / GUI to pick up new entry.

### 2.2 GUI steps
1. Open **Models** page → **Registry** tab.
2. Use **Import from Download History** (pulls discovered overlay) and fill metadata fields.
3. Trigger **Verify Hash** button (executes CLI verify).
4. Toggle **Lock version** once verification succeeds.

---
## 3. Handling Hash Drift

1. Alert triggered (CLI exit 2 or GUI warning badge).
2. Inspect `download_path` for unexpected changes (`ls -R` within the directory).
3. If drift expected (new quant, patch release):
   - Re-run `uv run imageworks-models verify <name>`.
   - Accept new hash by re-running with `--set-expected` on the next `lock` command.
4. If drift unexpected:
   - Re-download assets using downloader with `--force`.
   - Compare README/license from registry history for tampering.
   - File incident report before unlocking in production.

---
## 4. Managing vLLM Single-Port Instance

1. Start: `uv run imageworks-models activate-model <logical-name>`.
2. Stop: `uv run imageworks-models activate-model --stop`.
3. Status: `uv run imageworks-models current-model`.
4. GUI: Models page → **Active Model** panel.
5. Troubleshooting: If activation fails, check `_staging/active_vllm.json` and GPU utilization; adjust `CHAT_PROXY_VLLM_*` env vars if necessary.

---
## 5. Vision Probe Validation

1. Prepare canonical probe image.
2. Run `uv run imageworks-models probe-vision <logical-name> path/to/image.jpg`.
3. Confirm output shows `vision_ok: true` and includes reasoning text.
4. GUI: Models page includes **Vision Probe** button invoking the same command; review modal output.

---
## 6. Registry Hot Reload

1. When curated/discovered files change, chat proxy reloads automatically.
2. For manual reload: `uv run python -c "from imageworks.model_loader.registry import load_registry; load_registry(force=True)"`.
3. GUI: Settings → Backends → **Reload registry** button (wraps the command above).

---
## 7. Incident Response Matrix

| Event | Response |
|-------|----------|
| Registry fails to load | Validate JSON syntax (`jq . configs/model_registry.curated.json`). Roll back last edit if malformed. |
| `CapabilityError` from downstream tool | Update entry capabilities/roles to match actual backend features, or adjust tool configuration. |
| vLLM activation stuck | Review `autostart` logs, ensure `backend_config.launch` command is valid, verify GPU availability. |
| GUI shows missing artifacts | Run downloader `normalize-formats --rebuild` to refresh metadata; fix `download_path`. |
| API `/v1/select` 404 | Ensure correct logical name, reload registry, confirm not filtered by profile/test flag. |

---
## 8. Change Control Checklist

- [ ] Registry diff reviewed and approved.
- [ ] Hash verification executed and logged.
- [ ] Version lock toggled as appropriate.
- [ ] Downstream services (chat proxy, GUI) restarted or reloaded.
- [ ] Documentation updated (capability matrix, runbook reference).

