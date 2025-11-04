# GUI Models Page Guide

The Models page in the ImageWorks GUI provides a visual interface to inspect and manage the deterministic model registry, orchestrate vLLM activation, and view backend health.

---
## 1. Layout

1. **Registry Overview Table** (top left)
   - Columns: Logical Name, Display Name, Backend, Roles, Capabilities (icons), Lock status, Hash snippet, Installed?, Active?.
   - Filters: role multi-select, backend select, lock status toggle, include testing models.
   - Search: fuzzy search across logical/display names and aliases.

2. **Action Drawer** (right panel)
   - Contextual actions for selected row: Activate/Stop via proxy, Verify hash, Toggle lock, Edit registry entry (opens modal), Copy endpoint info, View JSON entry.
   - Displays registry provenance (curated vs discovered), download path, served model id, backend config snippet.

3. **Diff Viewer**
   - Shows last change in curated/discovered layers; supports git-style diff with added/removed keys.

4. **Backend Summary Cards**
   - Aggregated status by backend (vLLM, LMDeploy, Ollama) with counts of installed/not installed/locked.

5. **Recent Actions Log**
   - Timeline of operations executed via GUI (activation, verify, lock toggles) with operator name and timestamp.

---
## 2. Workflows

### 2.1 Activate a vLLM model
1. Select row for desired model (backend must be `vllm`).
2. Click **Activate via proxy**; confirm dialog shows CLI equivalent (`imageworks-models activate-model <name>`).
3. Status updates to “Activating…” with spinner; upon success, Active column shows ✅.
4. Check Dashboard active model card for confirmation.

### 2.2 Verify and lock
1. Select model → **Verify hash** to trigger CLI verify.
2. Upon success, hash snippet updates; if lock disabled, click **Lock version**.
3. Lock toggle calls `imageworks-models lock/unlock` and updates registry file.

### 2.3 Edit registry metadata
1. Select model → **Edit entry**.
2. Modal exposes editable fields (display name, roles, capabilities, served id, backend config, metadata).
3. Save writes to `model_registry.curated.json` and displays diff preview; operator must confirm before commit.

### 2.4 Import discovered entries
1. Use **Import from downloads** button (top right) to open wizard.
2. Choose discovered entries lacking curated overlay; fill missing fields (roles, backend, served id).
3. Review summary and confirm to append to curated registry.

### 2.5 Toggle testing visibility
- Use “Include testing models” toggle; interacts with `model_loader.testing_filters.is_testing_entry`.
- Useful for staging environment to surface experimental entries.

---
## 3. Backend Health Integration

- vLLM status derived from `imageworks-models current-model` and `_staging/active_vllm.json`.
- Ollama health via chat proxy autostart map (shows loaded tag, online/offline).
- Remote endpoints (LMDeploy/Triton) pinged via HTTP; results shown as coloured indicators next to backend summary cards.

---
## 4. Keyboard Shortcuts

- `f`: focus search box.
- `r`: reload registry (equivalent to Settings action).
- `l`: toggle lock on selected entry.
- `a`: activate selected model (if eligible).

---
## 5. Audit & Export

- **Export table**: download CSV of current filter view.
- **Copy registry entry**: copies JSON to clipboard for change requests.
- **Action log**: stored in `_staging/gui_registry_actions.json` with timestamps; accessible via Results page.

---
## 6. Troubleshooting

| Issue | Explanation | Fix |
|-------|-------------|-----|
| Row missing download path | Discovered entry lacks local files. | Run downloader or adjust registry `download_path`. |
| Activate button disabled | Backend not vLLM or entry locked to non-vLLM backend. | Choose vLLM-capable entry or adjust backend. |
| Diff viewer blank | No changes since last reload. | Force reload via shortcut `r` or Settings page. |
| Save fails with permission error | Registry file read-only. | Adjust filesystem permissions or open PR if repo-managed. |

