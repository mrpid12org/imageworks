# GUI Control Center Runbook

Operate the Streamlit control center to orchestrate ImageWorks workflows.

## 1. Launch the app
```bash
uv run imageworks-gui
```
- The entry point initialises directories and session state before rendering the
  sidebar and welcome screen.【F:src/imageworks/gui/app.py†L1-L110】
- On first run Streamlit prints the localhost URL (default `http://localhost:8501`).

## 2. Verify environment status
- Sidebar GPU panel queries `GPUDetector` to confirm VRAM availability; enable the
  "🐛 Debug" toggle to surface stack traces when detection fails.【F:src/imageworks/gui/app.py†L40-L80】
- Backend panel shows expected chat proxy, vLLM, and Ollama endpoints. Update the
  defaults in `gui/config.py` if your deployment uses alternative hosts.【F:src/imageworks/gui/config.py†L41-L70】

## 3. Run workflows
- Navigate to **Workflows → Mono Checker** (and others) to configure parameters.
  Each page mirrors CLI options and displays the generated command preview for
  reproducibility.【F:src/imageworks/gui/pages/mono.py†L150-L210】
- Use the "Queue Job" buttons to launch commands asynchronously. Output logs are
  captured via `utils.jobs.run_subprocess` and streamed to the UI.【F:src/imageworks/gui/utils/jobs.py†L15-L140】
- Job history persists in `outputs/gui_job_history.json`, letting you rerun recent
  configurations from the UI.【F:src/imageworks/gui/config.py†L11-L40】【F:src/imageworks/gui/pages/results.py†L1-L120】

## 4. Inspect results
- The **Results** page loads JSONL artifacts using pagination and filtering tools.
  It links to generated overlays, summaries, and metric files produced by the CLIs.【F:src/imageworks/gui/pages/results.py†L120-L210】
- Toggle the session "Debug" checkbox to expose raw session state for support
  scenarios.【F:src/imageworks/gui/app.py†L80-L110】

## 5. Troubleshooting
| Symptom | Checks |
| --- | --- |
| "No GPU detected" warning | Confirm NVIDIA drivers are visible to the Python environment. In WSL ensure `nvidia-smi` works; toggle Debug for detailed errors.【F:src/imageworks/gui/app.py†L40-L80】 |
| Commands never start | Verify `uv` is installed and available on PATH. The job runner relies on `subprocess.Popen`; check `outputs/logs` for captured stderr.【F:src/imageworks/gui/utils/jobs.py†L15-L140】 |
| Registry data missing | Run `uv run imageworks-loader list` to generate merged registry files. The GUI reads from `configs/model_registry.json` on start.【F:src/imageworks/gui/config.py†L1-L40】 |
| Streamlit session resets unexpectedly | Ensure browser caching allows third-party cookies; session state is stored server-side but requires stable connections. Export configuration using the job history as a backup.【F:src/imageworks/gui/state.py†L1-L60】 |

Document GUI-driven runs alongside CLI equivalents to keep audit logs consistent.
