# GUI Operations Guide

The ImageWorks Streamlit GUI provides an operator console over the core modules (model downloader, registry, mono checker, color narrator, similarity checker, personal tagger, settings). It wraps CLI commands with presets, exposes telemetry, and stores operator preferences.

---
## 1. Architecture Overview

| Component | Details |
|-----------|---------|
| App entry | `imageworks.gui.app:main` launches Streamlit, registers pages under `src/imageworks/gui/pages`. |
| State management | `gui.state` initialises session state (`init_session_state`) with shared caches (paths, presets, process status). |
| Configuration | `gui.config` reads defaults from `pyproject.toml` and `.env.local`, exposes getters/setters for per-app overrides. |
| Components | Shared UI pieces (preset selector, process runner, results viewer, charts) reside under `gui.components`. |
| CLI integration | `gui.utils.cli_wrapper` builds Typer commands and invokes them via subprocess, streaming logs into the UI. |
| Persistence | User overrides stored in Streamlit `session_state` and optionally JSON config in `_staging/gui_settings.json`. |

---
## 2. Page Catalogue

1. **Dashboard (`1_🏠_Dashboard.py`)**
   - Displays system status (active chat proxy model, GPU utilisation, recent job history).
   - Widgets to start/stop chat proxy, view last downloader runs, quick links to runbooks.

2. **Models (`2_🎯_Models.py`)**
   - Registry inventory table with filters (role, backend, lock state).
   - Actions: activate/stop vLLM model, verify hash, toggle locks, open registry editor.
   - Registry diff viewer (side panel) and audit log of recent changes.
   - See dedicated guide in `gui-models-page.md` for workflow details.

3. **Mono Checker (`3_🖼️_Mono_Checker.py`)**
   - Preset selector + overrides for thresholds.
   - Process runner executing `imageworks-mono check`.
   - Verdict tabs, overlay toggles, Lightroom tips, JSONL viewer.

4. **Image Similarity (`4_🖼️_Image_Similarity.py`)**
   - Candidate uploader, library configuration, strategy toggles, result gallery.

5. **Personal Tagger (`5_🖼️_Personal_Tagger.py`)**
   - Directory selection, backend/registry toggles, metadata preview grid.

6. **Color Narrator (`6_🖼️_Color_Narrator.py`)**
   - Backend diagnostics, prompt selection, run status charts, narration results table.

7. **Results (`7_📊_Results.py`)**
   - Unified viewer for JSONL/Markdown outputs across modules; supports filtering and export.

8. **Settings (`8_⚙️_Settings.py`)**
   - Environment checks, chat proxy controls, registry reload, profile selection, GUI theme.

---
## 3. CLI Parity & Command Mapping

| Page | CLI Command(s) |
|------|----------------|
| Models | `imageworks-models list/select/verify/activate-model` |
| Mono Checker | `imageworks-mono check`, `imageworks-mono visualize` |
| Image Similarity | `imageworks-image-similarity check` |
| Personal Tagger | `imageworks-personal-tagger run` |
| Color Narrator | `imageworks-color-narrator narrate`, `diagnose-backend` |
| Settings | `imageworks-chat-proxy` start/stop (via supervisor), registry reload helpers |

Process runner surfaces the exact command executed; logs persist in `_staging/gui_process_history.json` for auditing.

---
## 4. Presets & Overrides

- Presets defined under `gui.presets` (e.g., `MONO_CHECKER_PRESETS`) provide named configurations.
- Operators can customise fields; updates stored in session state and optionally saved as default via “Save preset” buttons (writes to `_staging/gui_presets.json`).
- Overrides feed into CLI command builder functions ensuring reproducibility.

---
## 5. Integration with Backend Services

- Chat proxy health polled via `/v1/models` (status chips on Dashboard/Settings).
- Model registry operations rely on loader CLI/REST; run results cached for diffing.
- Download history consumed from `logs/model_downloader.log` (Dashboard widget).
- GPU metrics pulled via `nvidia-smi` or `psutil` wrappers.

---
## 6. Logging & Diagnostics

- GUI logs to `logs/gui.log` (configured via `IMAGEWORKS_LOG_DIR`).
- Process runner captures stdout/stderr; transcripts saved per run for debugging.
- Error toasts include “copy to clipboard” button with tracebacks.

---
## 7. Extending the GUI

1. Add new page script under `src/imageworks/gui/pages` following naming convention `<order>_<emoji>_Name.py`.
2. Register new presets in `gui.presets` and state keys in `gui.state`.
3. Use shared components (process runner, results viewer) for consistent UX.
4. Update this guide and runbook when adding modules or altering workflows.

