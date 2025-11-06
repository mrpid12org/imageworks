# GUI Runbook

Operational playbook for the ImageWorks Streamlit interface.

---
## 1. Launching the GUI

1. Ensure chat proxy and required backends are available (or be prepared to start them via Settings).
2. Start Streamlit:
   ```bash
   uv run imageworks-gui
   ```
3. Access the UI at `http://localhost:8501` (default Streamlit port).
4. Log in via reverse proxy/SAML if deployed behind an auth gateway (out of scope for local runs).

---
## 2. Daily Operator Checklist

1. **Dashboard** – verify chat proxy status, GPU utilisation, and recent job results.
2. **Models** – confirm critical models remain locked and active entry matches expected production model.
3. **Mono Checker / Color Narrator** – review last run timestamps and JSONL output from Results page.
4. **Settings** – ensure deployment profile correct (production vs staging) and registry reload succeeded overnight.

---
## 3. Running Jobs from the GUI

1. Navigate to module page (Mono Checker, Image Similarity, Personal Tagger, Color Narrator).
2. Choose preset or configure inputs manually.
3. Review command preview for accuracy.
4. Click **Run** and monitor Process Runner output; failures trigger toast + log expansion.
5. Use Results page to inspect outputs across modules (JSONL/Markdown download buttons).

### 3.1 VRAM Estimator tab

- Open **⚡ VRAM Estimator** for forward and inverse VRAM planning.
- Optional: expand “GPU Overview” to display `nvidia-smi` results.
- Select an overhead profile (or click *Auto-detect profile* to use GPU heuristics).
- Fill in model parameters and context window to estimate total VRAM. Results are shown as metrics and JSON for export.
- Use the inverse form to discover the maximum context (k tokens) that fits within a VRAM budget.

---
## 4. Managing Presets

1. Adjust fields on module page to desired defaults.
2. Click **Save preset as default** (if available) to persist to `_staging/gui_presets.json`.
3. For temporary overrides, rely on session state; changes reset after page refresh unless saved.

---
## 5. Chat Proxy Control via Settings

1. Open Settings → Backends.
2. Use **Start Chat Proxy** or **Stop Chat Proxy** buttons to control service.
3. Use **Reload registry** to force loader cache refresh.
4. Profile selector updates `ProfileManager`; confirm by revisiting Models page.

---
## 6. Troubleshooting

| Issue | Resolution |
|-------|-----------|
| Process runner stuck on “Starting...” | Underlying CLI waiting for input or hung; click **Cancel run**, inspect logs, rerun in terminal if needed. |
| Command fails with permission error | Ensure directories referenced in overrides are writable by current user. |
| GUI loses settings after restart | Presets not saved; reapply configuration and use “Save preset”. |
| Chat proxy status red | Start service via Settings or check logs in `logs/chat_proxy.jsonl`. |
| Registry table empty | Reload registry (`Settings → Reload registry` or shortcut `r` on Models page). |

---
## 7. Maintenance & Updates

1. Pull latest code and restart Streamlit to apply UI updates.
2. Clear `_staging/gui_process_history.json` periodically to control disk usage.
3. Review logs in `logs/gui.log` for repeated errors; escalate to engineering if persistent.
4. Update this runbook when new pages or workflows are added.
