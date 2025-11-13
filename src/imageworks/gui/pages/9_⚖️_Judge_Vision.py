"""Judge Vision Streamlit page."""

from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from functools import partial

from streamlit_autorefresh import st_autorefresh

from imageworks.apps.judge_vision.progress import normalise_progress_path
from imageworks.apps.judge_vision.pairwise_playoff import recommended_pairwise_rounds
from imageworks.gui.components.file_browser import render_path_browser
from imageworks.gui.components.preset_selector import render_preset_selector
from imageworks.gui.components.process_runner import render_process_runner
from imageworks.gui.components.results_viewer import parse_jsonl
from imageworks.gui.config import (
    ALL_IMAGE_EXTENSIONS,
    DEFAULT_INPUT_DIR,
    JUDGE_VISION_DEFAULT_COMPETITION_CONFIG,
    JUDGE_VISION_DEFAULT_OUTPUT_JSONL,
    JUDGE_VISION_DEFAULT_SUMMARY_PATH,
    JUDGE_VISION_DEFAULT_IQA_CACHE,
    get_app_setting,
    reset_app_settings,
    set_app_setting,
)
from imageworks.gui.pages.judge_vision_presets import build_competition_presets
from imageworks.gui.state import init_session_state
from imageworks.gui.utils.cli_wrapper import build_judge_command
from imageworks.gui.components.sidebar_footer import render_sidebar_footer

APP_KEY = "judge"
PROGRESS_INIT_STATE_KEY = "judge_progress_initialised_paths"
PROGRESS_AUTORELOAD_INTERVAL_MS = 4000
RESULTS_SELECTION_KEY = "judge_results_selected_index"
RESULTS_TABLE_KEY = "judge_results_table"
_IMAGE_EXT_SET = {ext.lower() for ext in ALL_IMAGE_EXTENSIONS}


def _load_iqa_cache_records(cache_path: Optional[str]) -> List[Dict[str, Any]]:
    if not cache_path:
        return []
    cache_file = Path(cache_path).expanduser()
    if not cache_file.exists():
        return []
    try:
        return parse_jsonl(str(cache_file), cache_file.stat().st_mtime)
    except Exception:
        return []


def _index_iqa_cache(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    mapping: Dict[str, Dict[str, Any]] = {}
    for entry in records:
        image = entry.get("image")
        signals = entry.get("technical_signals")
        if not image or not signals:
            continue
        mapping[str(Path(image))] = signals
    return mapping


def _merge_technical_signals(
    primary: Dict[str, Any], fallback: Dict[str, Any]
) -> Dict[str, Any]:
    if not fallback:
        return primary
    merged: Dict[str, Any] = {}
    fallback_metrics = dict(fallback.get("metrics") or {})
    primary_metrics = dict(primary.get("metrics") or {})
    combined_metrics = dict(fallback_metrics)
    for key, value in primary_metrics.items():
        if value is not None:
            combined_metrics[key] = value
    merged["metrics"] = combined_metrics
    merged["notes"] = primary.get("notes") or fallback.get("notes") or ""
    merged["tonal_summary"] = primary.get("tonal_summary") or fallback.get(
        "tonal_summary"
    )
    return merged


def _render_iqa_cache_table(records: List[Dict[str, Any]]) -> None:
    if not records:
        st.info("IQA cache is empty. Run Stage 1 to populate deterministic scores.")
        return
    st.success(f"Loaded {len(records)} IQA entries from cache")
    rows = []
    for entry in records:
        image_path = entry.get("image")
        technical = entry.get("technical_signals") or {}
        metrics = technical.get("metrics") or {}
        rows.append(
            {
                "Filename": Path(image_path).name if image_path else image_path,
                "MUSIQ MOS": metrics.get("musiq_mos"),
                "NIMA aesthetic": metrics.get("nima_aesthetic_mean"),
                "NIMA technical": metrics.get("nima_technical_mean"),
                "Notes": technical.get("notes") or "",
            }
        )
    st.dataframe(rows, hide_index=True, use_container_width=True)


def _count_images_in_directory(directory: str, recursive: bool) -> int:
    try:
        base = Path(directory).expanduser()
        if not base.is_dir():
            return 0
    except Exception:
        return 0
    iterator = base.rglob("*") if recursive else base.iterdir()
    count = 0
    try:
        for entry in iterator:
            if entry.is_file() and entry.suffix.lower() in _IMAGE_EXT_SET:
                count += 1
    except Exception:
        return count
    return count


def _hydrate_prefill_from_tagger() -> None:
    payload = st.session_state.pop("judge_prefill", None)
    if not payload:
        return

    inputs = payload.get("input") or []
    if isinstance(inputs, list) and inputs:
        set_app_setting(st.session_state, APP_KEY, "input_mode", "Directory")
        set_app_setting(st.session_state, APP_KEY, "input_dir", inputs[0])

    mappings = {
        "output_jsonl": "output_jsonl",
        "summary": "summary",
        "competition": "competition",
        "competition_config": "competition_config",
        "pairwise_rounds": "pairwise_rounds",
        "pairwise_threshold": "pairwise_threshold",
        "critique_title_template": "critique_title_template",
        "critique_category": "critique_category",
        "critique_notes": "critique_notes",
    }
    for source, target in mappings.items():
        if payload.get(source) is not None:
            set_app_setting(st.session_state, APP_KEY, target, payload[source])


def render_judge_overrides(
    preset, session_key_prefix: str, registry=None
) -> Dict[str, Any]:  # pragma: no cover - Streamlit UI
    overrides: Dict[str, Any] = {}
    _hydrate_prefill_from_tagger()

    competition_id = preset.flags.get("competition", "")
    previous_competition = st.session_state.get("judge_active_competition")
    if competition_id and competition_id != previous_competition:
        for key, default in (
            ("pairwise_rounds", preset.flags.get("pairwise_rounds")),
            ("pairwise_threshold", preset.flags.get("pairwise_threshold")),
            ("critique_category", preset.flags.get("critique_category")),
            ("critique_notes", preset.flags.get("critique_notes")),
            ("critique_title_template", preset.flags.get("critique_title_template")),
        ):
            if default is not None:
                set_app_setting(st.session_state, APP_KEY, key, default)
        st.session_state["judge_active_competition"] = competition_id

    col_reset, _ = st.columns([1, 3])
    with col_reset:
        if st.button(
            "🔄 Reset to Defaults",
            key=f"{session_key_prefix}_reset",
            help="Reset Judge Vision overrides to preset defaults.",
        ):
            reset_app_settings(st.session_state, APP_KEY)
            st.success("✅ Judge Vision settings restored.")
            st.rerun()

    competition_name = preset.flags.get(
        "competition_name", competition_id or "Competition"
    )
    rules_text = preset.flags.get("rules_text")
    st.markdown(f"### Competition: {competition_name}")
    if rules_text:
        st.caption(rules_text)
    if preset.flags.get("competition_notes"):
        st.info(preset.flags["competition_notes"])

    current_mode = get_app_setting(st.session_state, APP_KEY, "input_mode", "Directory")
    input_mode = st.radio(
        "Input Selection Mode",
        options=["Directory", "Single File"],
        index=0 if current_mode == "Directory" else 1,
        key=f"{session_key_prefix}_input_mode",
        horizontal=True,
    )
    if input_mode != current_mode:
        set_app_setting(st.session_state, APP_KEY, "input_mode", input_mode)

    st.markdown("### Execution Strategy")
    stage1_default = get_app_setting(st.session_state, APP_KEY, "stage1_enabled", False)
    stage2_default = get_app_setting(st.session_state, APP_KEY, "stage2_enabled", True)
    stage3_default = get_app_setting(st.session_state, APP_KEY, "stage3_enabled", False)
    stage1 = st.checkbox(
        "Pass 1 – IQA",
        value=stage1_default,
        key=f"{session_key_prefix}_stage1",
        help="Stage 1 produces deterministic MUSIQ/NIMA metrics and cache.",
    )
    set_app_setting(st.session_state, APP_KEY, "stage1_enabled", stage1)
    stage2 = st.checkbox(
        "Pass 2 – Critique",
        value=stage2_default,
        key=f"{session_key_prefix}_stage2",
        help="Stage 2 runs the VLM critique using cached IQA metrics.",
    )
    set_app_setting(st.session_state, APP_KEY, "stage2_enabled", stage2)
    stage3 = st.checkbox(
        "Pass 3 – Pairwise playoff",
        value=stage3_default,
        key=f"{session_key_prefix}_stage3",
        help="Run the playoff to separate the high scores.",
    )
    set_app_setting(st.session_state, APP_KEY, "stage3_enabled", stage3)
    if stage3 and stage1 and not stage2:
        st.warning(
            "Pairwise requires Stage 2 results. Enable Pass 2 or run Pairwise only."
        )
        stage3 = False
        set_app_setting(st.session_state, APP_KEY, "stage3_enabled", False)
    stage_value = None
    if stage1 and stage2:
        stage_value = "two-pass"
    elif stage1:
        stage_value = "iqa"
    elif stage2:
        stage_value = "critique"
    elif stage3:
        stage_value = "pairwise"
    else:
        st.warning("Select at least one pass to run. Defaulting to Stage 2 only.")
        stage_value = "critique"
        stage2 = True
    set_app_setting(st.session_state, APP_KEY, "stage", stage_value)
    overrides["stage"] = stage_value

    device_options = ["cpu", "gpu"]
    current_device = get_app_setting(st.session_state, APP_KEY, "iqa_device", "gpu")
    try:
        device_index = device_options.index(current_device)
    except ValueError:
        device_index = device_options.index("gpu")
    iqa_device = st.selectbox(
        "IQA device",
        options=device_options,
        index=device_index,
        key=f"{session_key_prefix}_iqa_device_select",
        help="GPU mode runs TensorFlow IQA models (NIMA/MUSIQ) in a containerized environment with GPU acceleration. CPU mode runs locally (slower).",
    )
    set_app_setting(st.session_state, APP_KEY, "iqa_device", iqa_device)
    overrides["iqa_device"] = iqa_device

    st.markdown("### Pass 3 – Pairwise playoff")
    st.caption(
        "Set comparisons per finalist. Leave at 0 to skip Pass 3; try 4 when you want the "
        "LLM playoff."
    )
    threshold_default = int(
        get_app_setting(
            st.session_state,
            APP_KEY,
            "pairwise_threshold",
            17,
        )
    )
    pairwise_threshold = st.number_input(
        "Pairwise threshold",
        min_value=0,
        max_value=20,
        value=threshold_default,
        step=1,
        key=f"{session_key_prefix}_pairwise_threshold",
        help="Only images at or above this Stage 2 score enter the playoff.",
    )
    set_app_setting(
        st.session_state, APP_KEY, "pairwise_threshold", int(pairwise_threshold)
    )
    overrides["pairwise_threshold"] = int(pairwise_threshold)
    overrides["pairwise_enabled"] = bool(stage3)
    override_default = get_app_setting(
        st.session_state,
        APP_KEY,
        "pairwise_override_enabled",
        False,
    )
    pairwise_override = False
    pairwise_rounds_value = None
    if stage3:
        pairwise_override = st.checkbox(
            "Override pairwise comparisons per image",
            value=override_default,
            key=f"{session_key_prefix}_pairwise_override_toggle",
        )
        set_app_setting(
            st.session_state, APP_KEY, "pairwise_override_enabled", pairwise_override
        )
        if pairwise_override:
            current_override = int(
                get_app_setting(
                    st.session_state,
                    APP_KEY,
                    "pairwise_rounds_override",
                    int(preset.flags.get("pairwise_rounds", 4)),
                )
            )
            pairwise_rounds_value = st.number_input(
                "Manual comparisons per finalist",
                min_value=1,
                max_value=10,
                step=1,
                value=current_override,
                key=f"{session_key_prefix}_pairwise_rounds_override",
            )
            set_app_setting(
                st.session_state,
                APP_KEY,
                "pairwise_rounds_override",
                int(pairwise_rounds_value),
            )
    overrides["pairwise_rounds"] = (
        int(pairwise_rounds_value) if pairwise_rounds_value else None
    )

    current_input_dir = get_app_setting(
        st.session_state, APP_KEY, "input_dir", DEFAULT_INPUT_DIR
    )
    current_input_file = get_app_setting(st.session_state, APP_KEY, "input_file", "")

    start_path = (
        current_input_file
        if input_mode == "Single File" and current_input_file
        else current_input_dir
    )

    browser_state = render_path_browser(
        key_prefix=f"{session_key_prefix}_input_browser",
        start_path=start_path,
        allow_file_selection=input_mode == "Single File",
        file_types=ALL_IMAGE_EXTENSIONS,
        initial_file=current_input_file if current_input_file else None,
        show_file_listing=input_mode == "Single File",
    )

    selected_dir = browser_state["selected_dir"]
    selected_file = browser_state.get("selected_file")

    if selected_dir and selected_dir != current_input_dir:
        set_app_setting(st.session_state, APP_KEY, "input_dir", selected_dir)

    current_recursive = get_app_setting(
        st.session_state, APP_KEY, "include_recursive", False
    )
    if input_mode == "Directory":
        include_recursive = st.checkbox(
            "Include subdirectories",
            value=current_recursive,
            key=f"{session_key_prefix}_include_recursive",
            help="Process every folder under the selected directory.",
        )
        if include_recursive != current_recursive:
            set_app_setting(
                st.session_state, APP_KEY, "include_recursive", include_recursive
            )
        overrides["recursive"] = include_recursive
    else:
        overrides["recursive"] = False

    skip_preflight = st.checkbox(
        "Skip backend preflight checks",
        value=get_app_setting(st.session_state, APP_KEY, "skip_preflight", False),
        key=f"{session_key_prefix}_skip_preflight",
        help="Enable if connectivity checks fail but you know the backend is running.",
    )
    set_app_setting(st.session_state, APP_KEY, "skip_preflight", skip_preflight)
    overrides["skip_preflight"] = skip_preflight

    if input_mode == "Directory":
        if selected_dir:
            include_recursive = overrides.get("recursive", current_recursive)
            total_images = _count_images_in_directory(selected_dir, include_recursive)
            image_summary = (
                f"{total_images} image{'s' if total_images != 1 else ''}"
                if total_images
                else "no supported images"
            )
            st.info(f"Using directory: `{selected_dir}` ({image_summary} detected)")
            overrides["input"] = [selected_dir]
        else:
            st.warning("Select a directory to continue.")
    else:
        if selected_file:
            set_app_setting(st.session_state, APP_KEY, "input_file", selected_file)
            overrides["input"] = [selected_file]
            st.info(f"Using file: `{selected_file}`")
        else:
            st.warning("Select a file to continue.")

    col1, col2 = st.columns(2)
    with col1:
        current_jsonl = get_app_setting(
            st.session_state,
            APP_KEY,
            "output_jsonl",
            str(JUDGE_VISION_DEFAULT_OUTPUT_JSONL),
        )
        output_jsonl = st.text_input(
            "Output JSONL",
            value=current_jsonl,
            key=f"{session_key_prefix}_output_jsonl",
        )
        if output_jsonl != current_jsonl:
            set_app_setting(st.session_state, APP_KEY, "output_jsonl", output_jsonl)
        overrides["output_jsonl"] = output_jsonl

    with col2:
        current_summary = get_app_setting(
            st.session_state,
            APP_KEY,
            "summary",
            str(JUDGE_VISION_DEFAULT_SUMMARY_PATH),
        )
        summary_path = st.text_input(
            "Summary Markdown",
            value=current_summary,
            key=f"{session_key_prefix}_summary",
        )
        if summary_path != current_summary:
            set_app_setting(st.session_state, APP_KEY, "summary", summary_path)
        overrides["summary"] = summary_path

    title_template = st.text_input(
        "Critique title template",
        value=get_app_setting(
            st.session_state,
            APP_KEY,
            "critique_title_template",
            preset.flags.get("critique_title_template", "{stem}"),
        ),
        key=f"{session_key_prefix}_critique_title",
        help="Supports placeholders {stem}, {name}, {caption}, {parent}.",
    )
    set_app_setting(
        st.session_state, APP_KEY, "critique_title_template", title_template
    )
    overrides["critique_title_template"] = title_template or None

    available_categories = preset.flags.get("available_categories") or [
        "Open",
        "Mono",
    ]
    default_category = available_categories[0] if available_categories else ""
    current_category = get_app_setting(
        st.session_state,
        APP_KEY,
        "critique_category",
        default_category,
    )
    try:
        category_index = available_categories.index(current_category)
    except ValueError:
        category_index = 0
    category = st.selectbox(
        "Default category",
        options=available_categories,
        index=category_index,
        key=f"{session_key_prefix}_critique_category",
    )
    set_app_setting(st.session_state, APP_KEY, "critique_category", category)
    overrides["critique_category"] = category or None

    current_notes = get_app_setting(
        st.session_state,
        APP_KEY,
        "critique_notes",
        preset.flags.get("critique_notes", ""),
    )
    notes = st.text_area(
        "Judge notes / brief",
        value=current_notes,
        key=f"{session_key_prefix}_critique_notes",
        help="Optional context appended to each critique prompt.",
        height=120,
    )
    if notes != current_notes:
        set_app_setting(st.session_state, APP_KEY, "critique_notes", notes)
    overrides["critique_notes"] = notes or None

    with st.expander("Deterministic scoring (NIMA / MUSIQ)", expanded=False):
        enable_musiq = st.checkbox(
            "Include MUSIQ perceptual quality score",
            value=get_app_setting(st.session_state, APP_KEY, "enable_musiq", True),
            key=f"{session_key_prefix}_enable_musiq",
            help="Runs the TensorFlow MUSIQ model to capture a deterministic MOS-style score.",
        )
        set_app_setting(st.session_state, APP_KEY, "enable_musiq", enable_musiq)
        overrides["enable_musiq"] = enable_musiq

        enable_nima = st.checkbox(
            "Include NIMA aesthetic & technical scores",
            value=get_app_setting(st.session_state, APP_KEY, "enable_nima", True),
            key=f"{session_key_prefix}_enable_nima",
            help="Adds MobileNet-based NIMA predictions for both aesthetic and technical quality.",
        )
        set_app_setting(st.session_state, APP_KEY, "enable_nima", enable_nima)
        overrides["enable_nima"] = enable_nima

        cache_default = str(
            get_app_setting(
                st.session_state,
                APP_KEY,
                "iqa_cache",
                str(JUDGE_VISION_DEFAULT_IQA_CACHE),
            )
        )
        iqa_cache = st.text_input(
            "IQA cache path",
            value=cache_default,
            key=f"{session_key_prefix}_iqa_cache",
            help="Per-image deterministic metrics are written here during the IQA stage.",
        )
        set_app_setting(st.session_state, APP_KEY, "iqa_cache", iqa_cache)
        overrides["iqa_cache"] = iqa_cache

    return overrides


def _build_judge_command(config: Dict[str, Any]) -> List[str]:
    normalized = dict(config)
    normalized.setdefault("progress_file", "outputs/metrics/judge_vision_progress.json")
    return build_judge_command(normalized)


def _clear_progress_file(progress_path: str, *, force: bool = False) -> None:
    raw = Path(progress_path).expanduser()
    actual = normalise_progress_path(raw)
    if not raw.exists() and not actual.exists():
        return
    if force:
        actual.unlink(missing_ok=True)
        if raw.is_dir():
            shutil.rmtree(raw, ignore_errors=True)
        return
    try:
        data = json.loads(actual.read_text())
    except Exception:
        actual.unlink(missing_ok=True)
        if raw.is_dir():
            shutil.rmtree(raw, ignore_errors=True)
        return
    status = (data.get("status") or "").lower()
    if status != "running":
        actual.unlink(missing_ok=True)
        if raw.is_dir():
            shutil.rmtree(raw, ignore_errors=True)


def _prime_progress_refresh(progress_path: str) -> None:
    _clear_progress_file(progress_path, force=True)
    st.session_state["judge_run_active"] = True
    cleared_paths = st.session_state.setdefault(PROGRESS_INIT_STATE_KEY, set())
    cleared_paths.add(str(normalise_progress_path(Path(progress_path).expanduser())))


def _ensure_progress_initialised(progress_path: str) -> None:
    cleared_paths = st.session_state.setdefault(PROGRESS_INIT_STATE_KEY, set())
    resolved = str(normalise_progress_path(Path(progress_path).expanduser()))
    if resolved in cleared_paths:
        return
    _clear_progress_file(resolved, force=False)
    cleared_paths.add(resolved)


def _render_progress(progress_path: str) -> None:
    raw_path = Path(progress_path).expanduser()
    path = normalise_progress_path(raw_path)
    process_running = bool(st.session_state.get("judge_runner_process_state"))
    run_active = bool(st.session_state.get("judge_run_active"))
    if not path.exists():
        if process_running or run_active:
            st.caption("Waiting for Judge Vision to report progress…")
            st_autorefresh(
                interval=PROGRESS_AUTORELOAD_INTERVAL_MS,
                key=f"judge-progress-wait-{hash(progress_path)}",
            )
        return
    try:
        data = json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        st.caption("Progress file is updating…")
        st_autorefresh(
            interval=PROGRESS_AUTORELOAD_INTERVAL_MS,
            key=f"judge-progress-retry-{hash(str(path))}",
        )
        return

    total = max(int(data.get("total", 0)), 0)
    processed = max(int(data.get("processed", 0)), 0)
    current = data.get("current_image")
    status = (data.get("status") or "running").lower()
    message = data.get("message")
    phase = data.get("phase")
    history = data.get("history") or []

    if history:
        st.markdown("#### Completed Stages")
        for entry in history:
            entry_phase = entry.get("phase") or "Previous Stage"
            entry_status = entry.get("status", "complete").title()
            entry_total = entry.get("total") or 0
            entry_processed = entry.get("processed") or entry_total
            entry_text = (
                f"{entry_phase}: {entry_status} ({entry_processed}/{entry_total})"
            )
            st.progress(1.0, text=entry_text)
            if entry.get("timestamp"):
                st.caption(f"Finished at {entry.get('timestamp')}")
            if entry.get("message"):
                st.caption(entry.get("message"))

    if phase:
        progress_value = min(1.0, processed / total) if total else 0.0
        label = phase
        progress_text = (
            f"{label}: {processed}/{total}"
            if total
            else f"{label}: waiting for progress…"
        )
        st.progress(progress_value, text=progress_text)
    else:
        progress_value = min(1.0, processed / total) if total else 0.0
        st.progress(
            progress_value,
            text=(
                f"Processing {processed}/{total}" if total else "Waiting for progress…"
            ),
        )

    should_refresh = process_running or status == "running"
    if should_refresh:
        st.caption("Auto-refreshing progress…")
        st_autorefresh(
            interval=PROGRESS_AUTORELOAD_INTERVAL_MS,
            key=f"judge-progress-refresh-{hash(str(path))}",
        )
    elif status in {"complete", "error"}:
        st.session_state["judge_run_active"] = False

    if status == "complete":
        st.success(f"✅ Completed {processed}/{total} image(s).")
    elif status == "error":
        detail = message or "Unknown error"
        st.error(f"❌ Failed after {processed}/{total} image(s): {detail}")
    if current:
        st.caption(f"Current image: {Path(current).name}")


def _build_table_rows(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for rec in records:
        compliance = rec.get("compliance") or {}
        rows.append(
            {
                "Filename": rec.get("filename"),
                "Image Title": rec.get("image_title"),
                "Competition": rec.get("competition_category") or "Uncategorised",
                "Style": rec.get("style") or "",
                "Total": rec.get("total"),
                "Pairwise Baseline": rec.get("pairwise_score_initial"),
                "Raw Score": rec.get("total_initial"),
                "Award": rec.get("award"),
                "IQA": rec.get("technical_summary") or "",
                "Compliance": "PASS" if compliance.get("passed") else "Check",
            }
        )
    return rows


def _render_record_detail(
    record: Dict[str, Any], heading: str | None = None
) -> None:  # pragma: no cover - UI helper
    if heading:
        st.subheader(heading)
    col_img, col_meta = st.columns([1, 1])
    with col_img:
        image_path = record.get("image_path")
        if image_path and Path(image_path).exists():
            caption = record.get("image_title") or record.get("filename")
            st.image(image_path, caption=caption)
        else:
            st.info("Image preview unavailable for this entry.")

    with col_meta:
        st.write(f"**Filename:** {record.get('filename') or 'n/a'}")
        st.write(f"**Image title:** {record.get('image_title') or 'n/a'}")
        st.write(
            f"**Competition category:** {record.get('competition_category') or 'n/a'}"
        )
        st.write(f"**Style:** {record.get('style') or 'n/a'}")
        st.write(f"**Award suggestion:** {record.get('award') or 'n/a'}")
        st.metric("Total score", record.get("total") or "n/a")
        compliance = record.get("compliance") or {}
        if compliance:
            st.write(
                f"**Compliance:** {'PASS' if compliance.get('passed') else 'Check flags'}"
            )
            if compliance.get("messages"):
                with st.expander("Compliance notes", expanded=False):
                    st.write("\n".join(compliance.get("messages")))
        if record.get("critique"):
            with st.expander("Critique", expanded=True):
                st.write(record.get("critique"))


def _baseline_score(rec: Dict[str, Any]) -> Optional[float]:
    for key in ("pairwise_score_initial", "total_initial", "total"):
        value = rec.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def _category_selection_key(category: str) -> str:
    slug = (category or "uncategorised").strip().lower().replace(" ", "_")
    return f"{RESULTS_SELECTION_KEY}_{slug or 'uncategorised'}"


def _category_sort_key(name: str) -> tuple[int, str]:
    normalized = (name or "").strip().lower()
    if normalized == "colour":
        return (0, normalized)
    if normalized == "mono":
        return (1, normalized)
    return (2, normalized)


def _render_pairwise_section(
    results: List[Dict[str, Any]], category_label: str
) -> None:
    playoff_rows = [
        rec
        for rec in results
        if rec.get("pairwise_wins") is not None and rec.get("pairwise_comparisons")
    ]
    if not playoff_rows:
        return

    st.markdown(f"### 🎯 Pairwise Adjustments – {category_label}")
    table = [
        {
            "Filename": Path(rec.get("image_path") or rec.get("filename") or "").name,
            "Category": rec.get("competition_category"),
            "Final Score": rec.get("total"),
            "Pairwise Baseline": rec.get("pairwise_score_initial"),
            "Win Ratio": rec.get("pairwise_win_ratio"),
            "Wins": rec.get("pairwise_wins"),
            "Comparisons": rec.get("pairwise_comparisons"),
        }
        for rec in playoff_rows
    ]
    st.dataframe(table, hide_index=True, use_container_width=True)


def _render_category_results_view(
    category: str, records: List[Dict[str, Any]], threshold_value: int
) -> None:
    if not records:
        st.info(f"No records available for {category}.")
        return

    st.markdown(
        f"### 📊 Rubric Scores – {category} "
        f"({len(records)} image{'s' if len(records) != 1 else ''})"
    )
    rows = _build_table_rows(records)
    st.dataframe(rows, use_container_width=True)

    labels = [
        f"{i + 1:02d} · {rec['filename'] or rec['image_path']}"
        for i, rec in enumerate(records)
    ]
    selection_key = _category_selection_key(category)
    selector_key = f"{selection_key}_selector"
    if labels:
        current_index = st.session_state.get(selection_key, 0) or 0
        current_index = min(max(current_index, 0), len(labels) - 1)
        selected_label = st.selectbox(
            "Select image",
            options=labels,
            index=current_index,
            key=selector_key,
        )
        st.session_state[selection_key] = labels.index(selected_label)
    else:
        st.info("No images to select for this category.")

    baseline_counter: Counter[str] = Counter()
    final_counter: Counter[str] = Counter()
    for rec in records:
        baseline = _baseline_score(rec)
        if baseline is not None:
            baseline_counter[f"{baseline:.1f}"] += 1
        total = rec.get("total")
        if total is not None:
            try:
                final_counter[f"{float(total):.1f}"] += 1
            except (TypeError, ValueError):
                continue

    if baseline_counter:
        with st.expander(
            f"📈 Score histogram (pre-pairwise) – {category}", expanded=False
        ):
            histogram_rows = [
                {"Score": score, "Images": count}
                for score, count in sorted(
                    baseline_counter.items(), key=lambda item: float(item[0])
                )
            ]
            st.dataframe(histogram_rows, hide_index=True, use_container_width=True)
    if final_counter:
        with st.expander(f"📊 Score histogram (final) – {category}", expanded=False):
            histogram_rows = [
                {"Score": score, "Images": count}
                for score, count in sorted(
                    final_counter.items(), key=lambda item: float(item[0])
                )
            ]
            st.dataframe(histogram_rows, hide_index=True, use_container_width=True)

    candidate_count = sum(
        1 for rec in records if (_baseline_score(rec) or 0) >= threshold_value
    )
    if candidate_count:
        recommended_rounds = recommended_pairwise_rounds(candidate_count)
        st.info(
            f"{category}: Recommended pairwise rounds → {recommended_rounds} "
            f"({candidate_count} image{'s' if candidate_count != 1 else ''} ≥ {threshold_value})."
        )

    bucket_map: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        total = rec.get("total")
        if total is None:
            continue
        try:
            bucket = int(round(float(total)))
        except (TypeError, ValueError):
            continue
        bucket_map[bucket].append(rec)

    st.markdown("### 🖼️ Score Bucket Review")
    priority_buckets = [20, 19, 18]
    for bucket_score in priority_buckets:
        bucket_records = bucket_map.get(bucket_score, [])
        st.markdown(
            f"#### Score {bucket_score} ({len(bucket_records)} image{'s' if len(bucket_records) != 1 else ''})"
        )
        if not bucket_records:
            st.caption("No images in this bucket.")
            continue
        for idx, record in enumerate(bucket_records, start=1):
            with st.container():
                _render_record_detail(
                    record, heading=f"{idx:02d}. {record.get('filename') or ''}"
                )
                st.divider()

    other_buckets = sorted(
        score for score in bucket_map.keys() if score not in set(priority_buckets)
    )
    if other_buckets:
        st.markdown("#### Other Buckets")
        bucket_selection = st.selectbox(
            "Select another bucket",
            other_buckets,
            format_func=lambda value: f"Score {value} ({len(bucket_map[value])} image{'s' if len(bucket_map[value]) != 1 else ''})",
            key=f"{selection_key}_bucket_selector",
        )
        for idx, record in enumerate(bucket_map.get(bucket_selection, []), start=1):
            with st.container():
                _render_record_detail(
                    record, heading=f"{idx:02d}. {record.get('filename') or ''}"
                )
                st.divider()

    st.markdown("### 🖼️ Image & Critique Viewer")
    selected_index = st.session_state.get(selection_key, 0) or 0
    if selected_index < 0 or selected_index >= len(records):
        selected_index = 0
    selected_record = records[selected_index]
    _render_record_detail(selected_record)

    _render_pairwise_section(records, category)


def render_judge_results(config: Dict[str, Any]) -> None:
    cache_path = config.get("iqa_cache") or str(JUDGE_VISION_DEFAULT_IQA_CACHE)
    cache_records = _load_iqa_cache_records(cache_path)
    cache_index = _index_iqa_cache(cache_records)
    stage = (config.get("stage") or "").lower()

    if stage == "iqa":
        st.markdown("### 🧮 Stage 1 IQA Cache")
        if cache_path:
            st.caption(f"Source: {cache_path}")
        _render_iqa_cache_table(cache_records)
        return

    jsonl_path = config.get("output_jsonl") or str(JUDGE_VISION_DEFAULT_OUTPUT_JSONL)
    if not jsonl_path:
        st.info("Specify an output JSONL path to load results.")
        return

    jsonl = Path(jsonl_path).expanduser()
    if not jsonl.exists():
        st.info(f"No Judge Vision run found at `{jsonl}` yet.")
        return

    results = parse_jsonl(jsonl_path, jsonl.stat().st_mtime)
    if not results:
        st.info("JSONL file is empty.")
        return

    display_records: List[Dict[str, Any]] = []
    for item in results:
        image_path = item.get("image") or ""
        subscores = item.get("critique_subscores") or {}
        compliance = item.get("compliance") or {}
        technical = item.get("technical_signals") or {}
        cache_payload = cache_index.get(str(Path(image_path))) if image_path else None
        if cache_payload:
            technical = _merge_technical_signals(technical, cache_payload)
        metrics = technical.get("metrics") or {}
        summary_parts: List[str] = []
        if metrics.get("musiq_mos") is not None:
            summary_parts.append(f"MUSIQ {metrics['musiq_mos']:.1f}")
        if metrics.get("nima_aesthetic_mean") is not None:
            summary_parts.append(f"NIMA aesthetic {metrics['nima_aesthetic_mean']:.2f}")
        if metrics.get("nima_technical_mean") is not None:
            summary_parts.append(f"NIMA technical {metrics['nima_technical_mean']:.2f}")
        display_records.append(
            {
                "image_path": image_path,
                "filename": Path(image_path).name if image_path else "",
                "image_title": item.get("image_title")
                or item.get("critique_title")
                or "",
                "competition_category": item.get("competition_category")
                or item.get("critique_category")
                or "",
                "style": item.get("style_inference") or item.get("category") or "",
                "total_initial": item.get("critique_total_initial"),
                "pairwise_score_initial": item.get("pairwise_score_initial"),
                "total": item.get("critique_total"),
                "award": item.get("critique_award") or "",
                "pairwise_wins": item.get("pairwise_wins"),
                "pairwise_comparisons": item.get("pairwise_comparisons"),
                "pairwise_win_ratio": item.get("pairwise_win_ratio"),
                "critique": item.get("critique") or "",
                "subscores": subscores,
                "compliance": compliance,
                "technical_summary": "; ".join(summary_parts) if summary_parts else "",
                "technical_signals": technical,
            }
        )

    if not display_records:
        st.info("No records parsed from JSONL.")
        return

    threshold_value = int(config.get("pairwise_threshold") or 17)
    category_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in display_records:
        category = rec.get("competition_category") or "Uncategorised"
        category_map[category].append(rec)

    ordered_categories = sorted(category_map.keys(), key=_category_sort_key)
    if len(ordered_categories) == 1:
        category = ordered_categories[0]
        _render_category_results_view(category, category_map[category], threshold_value)
    else:
        tab_labels = [
            f"{category} ({len(category_map[category])})"
            for category in ordered_categories
        ]
        tabs = st.tabs(tab_labels)
        for tab, category in zip(tabs, ordered_categories):
            with tab:
                _render_category_results_view(
                    category, category_map[category], threshold_value
                )


def main():  # pragma: no cover - Streamlit entry point
    st.set_page_config(page_title="⚖️ Judge Vision", layout="wide")
    init_session_state()
    with st.sidebar:
        render_sidebar_footer()

    st.title("⚖️ Judge Vision")
    st.caption(
        "Run full compliance, rubric scoring, and tournament analysis separate from Personal Tagger."
    )

    tabs = st.tabs(["📋 Brief & Rules", "▶️ Run Pipeline", "📈 Results"])

    with tabs[0]:
        registry_setting = get_app_setting(
            st.session_state,
            APP_KEY,
            "competition_config",
            str(JUDGE_VISION_DEFAULT_COMPETITION_CONFIG),
        )
        registry_path = st.text_input(
            "Competition Registry (TOML)",
            value=registry_setting,
            key="judge_registry_path",
            help="Path to the TOML file containing club competitions.",
        )
        if registry_path != registry_setting:
            set_app_setting(
                st.session_state, APP_KEY, "competition_config", registry_path
            )

        presets, registry, registry_errors = build_competition_presets(registry_path)
        for message in registry_errors:
            st.warning(message)

        config = render_preset_selector(
            presets,
            session_key_prefix=APP_KEY,
            custom_override_renderer=partial(render_judge_overrides, registry=registry),
        )
        st.info(
            "Judge Vision always runs in dry-run/analysis mode. Metadata writing stays disabled."
        )
        if not config.get("input"):
            st.warning("Add at least one input path to continue.")
        st.session_state["judge_active_config"] = config

    with tabs[1]:
        config = st.session_state.get("judge_active_config") or st.session_state.get(
            f"{APP_KEY}_config", {}
        )
        if not config:
            st.info("Configure Judge Vision on the first tab.")
        else:
            progress_path = (
                config.get("progress_file")
                or "outputs/metrics/judge_vision_progress.json"
            )
            _ensure_progress_initialised(progress_path)
            render_process_runner(
                button_label="Run Judge Vision",
                command_builder=_build_judge_command,
                config=config,
                key_prefix="judge_runner",
                result_key="judge_runner_result",
                timeout=3600,
                on_execute=lambda path=progress_path: _prime_progress_refresh(path),
                async_mode=True,
            )
            _render_progress(progress_path)

    with tabs[2]:
        config = st.session_state.get("judge_active_config") or st.session_state.get(
            f"{APP_KEY}_config", {}
        )
        if not config:
            st.info("Configure Judge Vision first to pick result paths.")
        else:
            render_judge_results(config)


if __name__ == "__main__":
    main()
