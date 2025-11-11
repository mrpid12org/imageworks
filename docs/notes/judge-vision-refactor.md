# Judge Vision Refactor Status

## Objective
- Separate competition judging (compliance → technical → critique → pairwise) from the Personal Tagger pipeline so tagging focuses solely on captions/keywords/descriptions.
- Introduce a dedicated Judge Vision backend module and CLI (`imageworks-judge-vision`) that the GUI can call directly.
- Add progress tracking (per-image updates + current filename) that the GUI can poll during a run.
- Remove critique/pairwise code paths, CLI flags, and UI elements from the Personal Tagger module and documentation.

## Planned Work
1. **New Judge Vision backend** – create modules for config, prompts, models, inference, runner, progress tracking, and Typer CLI.
2. **Move judging helpers** – relocate critique prompts, inference logic, rubric + compliance dataclasses, pairwise aggregator out of personal_tagger.
3. **Strip critique from Personal Tagger** – remove critique-related config/CLI fields, dataclasses, inference steps, summary output, and tests.
4. **GUI + progress** – run Judge Vision via the new CLI, show registry presets + input browser, display live progress (`st.progress`).
5. **Docs/tests** – document the split and update unit/integration tests.

## Progress To Date
- Judge Vision modules implemented: `config.py`, `models.py`, `prompts.py`, `progress.py`, `inference.py`, `runner.py`, `cli/main.py`; exported via `imageworks.apps.judge_vision.__all__` and wired into `pyproject.toml` as the `imageworks-judge-vision` script.
- CLI wrapper now has `build_judge_command`; Judge Vision Streamlit page uses it, builds presets from the competition registry, and shows rule context/notes.
- Progress tracker JSON file created per run and polled by the GUI; `st.progress` displays processed/total + current image name.
- Competition schema enhanced (rules include colour-space/class/entries, config supports `display_name` and `notes`).
- Personal Tagger is now critique-free end-to-end: prompts/models/inference slimmed to caption/keyword/description, CLI/runner/tests/GUI no longer expose critique or pairwise knobs.
- Judge Vision exports appear on the 📊 Results page (defaults wired to `outputs/results/judge_vision.jsonl` and matching summary), job history/stats understand the new module, runner tests assert JSONL/progress writes, and runbooks + the GUI user guide document the standalone CLI plus progress indicator.

## Remaining Tasks
1. Explore incremental JSONL writes or checkpointing for Judge Vision so long runs survive interruptions (currently everything is flushed at the end).
2. Prototype richer analytics (charts/deltas) for Judge Vision history once enough runs accumulate in the 📊 Results browser.

## Challenges / Notes
- Streamlit caching caused `CompetitionConfig` objects without `display_name` after schema changes; handled via `getattr`, but full GUI restart may still be needed after major updates.
- Old runs wrote data to personal tagger outputs; with the new CLI we must ensure Judge Vision writes to its own files consistently.
- Interrupted runs previously lost in-memory results because JSONL was only written at the end; consider incremental writes in the Judge Vision runner if we need resilience later.

This log should let us resume the refactor in a fresh conversation without losing current context.
