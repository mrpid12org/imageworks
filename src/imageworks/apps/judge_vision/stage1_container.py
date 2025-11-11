"""TensorFlow Stage 1 container orchestration."""

from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
import textwrap
import threading
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from imageworks.apps.judge_vision.config import JudgeVisionConfig

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
_HOST_MODEL_ENV = Path(
    os.environ.get("IMAGEWORKS_MODEL_ROOT", str(Path.home() / "ai-models"))
).expanduser()


def _normalized_host_model_root(path: Path) -> Path:
    """
    Ensure we mount the parent directory that actually contains ``weights/``.

    Users often point ``IMAGEWORKS_MODEL_ROOT`` directly at ``~/ai-models/weights``.
    Inside the container we expect ``IMAGEWORKS_MODEL_ROOT`` to refer to the parent
    (so Judge Vision can look up ``<root>/weights/...``). If the env already ends
    with ``weights`` we mount its parent so the container sees the same layout as
    the chat proxy.
    """

    resolved = path
    if resolved.name == "weights":
        return resolved.parent
    return resolved


HOST_MODEL_ROOT = _normalized_host_model_root(_HOST_MODEL_ENV)
CONTAINER_MODEL_ROOT = Path("/root/ai-models")
DEFAULT_IMAGE = "nvcr.io/nvidia/tensorflow:24.02-tf2-py3"
TF_RUNTIME_ROOT = Path(
    os.environ.get("JUDGE_VISION_TF_RUNTIME_ROOT", PROJECT_ROOT / ".tf-backend")
)
TF_ENV_DIR = os.environ.get("JUDGE_VISION_TF_ENV_DIR", ".venv-tf")
TF_CACHE_DIR = os.environ.get(
    "JUDGE_VISION_TF_CACHE_DIR", str(TF_RUNTIME_ROOT / "cache")
)
TF_UV_BIN = os.environ.get("JUDGE_VISION_TF_UV_BIN", str(TF_RUNTIME_ROOT / "bin/uv"))
TF_EXTRA_MOUNTS = os.environ.get("JUDGE_VISION_TF_EXTRA_MOUNTS", "")
TF_FORCE_PULL = os.environ.get("JUDGE_VISION_TF_FORCE_PULL", "0") in {
    "1",
    "true",
    "TRUE",
}
TF_SKIP_PULL = os.environ.get("JUDGE_VISION_TF_SKIP_PULL", "0") in {"1", "true", "TRUE"}
TF_CONTAINER_ENABLED = os.environ.get("JUDGE_VISION_TF_CONTAINER_ENABLED", "1") not in {
    "0",
    "false",
    "FALSE",
}
TF_DOCKER_ARGS = os.environ.get("JUDGE_VISION_TF_DOCKER_ARGS", "")
CONFIG_SUBDIR = "judge_stage1"

_PULLED_IMAGES: Set[str] = set()
_ACTIVE_PROCESS: Optional[subprocess.Popen] = None


class Stage1ContainerError(RuntimeError):
    """Raised when the TensorFlow container orchestration fails."""


def _resolve_path(path: Path | str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = (PROJECT_ROOT / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return candidate


def _ensure_stage_setting(config: JudgeVisionConfig) -> JudgeVisionConfig:
    if (config.stage or "").lower() != "iqa":
        config.stage = "iqa"
    return config


def serialize_config(config: JudgeVisionConfig, *, root: Optional[Path] = None) -> Path:
    """Write the Stage 1 configuration to a JSON payload."""

    config = _ensure_stage_setting(config)
    payload = config.to_dict()
    base_root = root or PROJECT_ROOT
    target_dir = base_root / "tmp" / CONFIG_SUBDIR
    target_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time() * 1000)
    path = target_dir / f"config_{timestamp}_{os.getpid()}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _candidate_mounts(config: JudgeVisionConfig) -> List[Tuple[Path, Path]]:
    mounts: Dict[str, Tuple[Path, Path]] = {}

    def _register(host: Path, container: Optional[Path] = None) -> None:
        host = host.resolve()
        target = container if container is not None else host
        mounts[f"{host}->{target}"] = (host, target)

    _register(PROJECT_ROOT)
    _register(_resolve_path(Path.home()))
    _register(HOST_MODEL_ROOT, CONTAINER_MODEL_ROOT)

    def add_path(value: Optional[Path]) -> None:
        if not value:
            return
        absolute = Path(value).expanduser()
        if not absolute.is_absolute():
            absolute = (PROJECT_ROOT / absolute).resolve()
        else:
            absolute = absolute.resolve()
        if absolute.is_file():
            absolute = absolute.parent
        _register(absolute)

    for input_path in config.input_paths:
        add_path(input_path)

    add_path(config.iqa_cache_path)
    add_path(config.progress_path)
    add_path(config.output_jsonl)
    add_path(config.summary_path)
    add_path(config.competition_config)

    if TF_EXTRA_MOUNTS:
        for raw in TF_EXTRA_MOUNTS.split(","):
            raw = raw.strip()
            if raw:
                _register(_resolve_path(raw))

    # Ensure deterministic backend cache directories exist on the host
    for host, _ in list(mounts.values()):
        if not host.exists():
            try:
                host.mkdir(parents=True, exist_ok=True)
            except Exception:  # noqa: BLE001 - best effort
                continue

    return [pair for pair in mounts.values() if pair[0] != Path("/")]


def _format_mounts(paths: Iterable[Tuple[Path, Path]]) -> List[str]:
    args: List[str] = []
    seen: Set[Tuple[str, str]] = set()
    for host, container in paths:
        pair = (str(host), str(container))
        if pair in seen:
            continue
        seen.add(pair)
        args.extend(["-v", f"{pair[0]}:{pair[1]}"])
    return args


def _container_script(config_path: Path) -> str:
    sanitized = shlex.quote(str(config_path))
    script = f"""
set -euo pipefail
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=${{TF_CPP_MIN_LOG_LEVEL:-2}}
# Use container's Python 3.10 with pre-installed GPU TensorFlow
export PYTHONPATH={str(PROJECT_ROOT)}/src${{PYTHONPATH:+:$PYTHONPATH}}
# Remove Python bytecode cache to ensure fresh imports
find {str(PROJECT_ROOT)}/src -type d -name __pycache__ -exec rm -rf {{}} + 2>/dev/null || true
# Install minimal dependencies needed for Stage 1 (IQA)
# CRITICAL: Install numpy<2.0 FIRST to prevent opencv from upgrading to numpy 2.x
# TensorFlow 2.15 requires numpy<2.0 and will break with numpy 2.x
python3 -m pip install --no-cache-dir --quiet 'numpy<2.0' 2>&1 | grep -v "already satisfied" || true
python3 -m pip install --no-cache-dir --quiet tomli opencv-python-headless 2>&1 | grep -v "already satisfied" || true
# Verify GPU detection
echo "[tf-stage1] Checking TensorFlow GPU availability..."
python3 -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print(f'GPU devices: {{gpus}}')"
# Run Stage 1 entry point
python3 -m imageworks.apps.judge_vision.stage1_entry {sanitized}
"""
    return textwrap.dedent(script).strip()


def build_docker_command(
    config: JudgeVisionConfig,
    payload_path: Path,
    *,
    image: Optional[str] = None,
) -> Tuple[str, List[str]]:
    """Construct the docker command used to run Stage 1 inside the TF container."""

    chosen_image = image or os.environ.get("JUDGE_VISION_TF_IMAGE", DEFAULT_IMAGE)
    mounts = _format_mounts(_candidate_mounts(config))
    extra_args: List[str] = shlex.split(TF_DOCKER_ARGS) if TF_DOCKER_ARGS else []

    command: List[str] = ["docker", "run", "--rm", "--gpus", "all"]
    command.extend(mounts)
    command.extend(["-w", str(PROJECT_ROOT)])
    # UV_PROJECT_ENVIRONMENT is set dynamically in the container script
    command.extend(["-e", f"UV_CACHE_DIR={TF_CACHE_DIR}"])
    if TF_DOCKER_ARGS:
        command.extend(extra_args)
    command.extend(["-e", f"IMAGEWORKS_MODEL_ROOT={CONTAINER_MODEL_ROOT}"])
    command.append(chosen_image)
    command.extend(["bash", "-lc", _container_script(payload_path)])
    return chosen_image, command


def _capture_command_output(cmd: Sequence[str], *, timeout: int = 5) -> str:
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001
        return f"{' '.join(cmd)} failed: {exc}"
    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        return f"{' '.join(cmd)} exited {proc.returncode}: {stderr or '<no stderr>'}"
    output = (proc.stdout or "").strip()
    return output or "<no output>"


def _gpu_diag_worker(
    stop_event: threading.Event, image_name: str | None, interval: int = 10
) -> None:
    while not stop_event.wait(interval):
        gpu_output = _capture_command_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader",
            ]
        )
        logger.info("[tf-stage1][diag] nvidia-smi apps: %s", gpu_output)

        docker_cmd: list[str] = [
            "docker",
            "ps",
            "--format",
            "{{.ID}} {{.Image}} {{.Status}} {{.Names}}",
        ]
        if image_name:
            docker_cmd.insert(2, f"--filter=ancestor={image_name}")
        docker_output = _capture_command_output(docker_cmd)
        logger.info("[tf-stage1][diag] docker ps: %s", docker_output)


def _run_subprocess(cmd: Sequence[str], *, diag_image: str | None = None) -> None:
    global _ACTIVE_PROCESS

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError as exc:  # pragma: no cover - depends on runtime
        raise Stage1ContainerError("docker CLI not found in PATH") from exc
    except Exception as exc:  # noqa: BLE001
        raise Stage1ContainerError(f"Failed to start docker process: {exc}") from exc

    captured: list[str] = []
    _ACTIVE_PROCESS = proc
    stop_event = threading.Event()
    diag_thread = threading.Thread(
        target=_gpu_diag_worker,
        args=(stop_event, diag_image),
        daemon=True,
    )
    diag_thread.start()
    try:
        assert proc.stdout is not None  # for mypy
        for line in proc.stdout:
            line = line.rstrip()
            captured.append(line)
            logger.info("[tf-stage1] %s", line)
        returncode = proc.wait()
    finally:
        _ACTIVE_PROCESS = None
        stop_event.set()
        diag_thread.join(timeout=2)
    if returncode != 0:
        tail = "\n".join(captured[-10:])
        raise Stage1ContainerError(
            f"Docker exited with code {returncode}. Output tail:\n{tail}"
        )


def _ensure_image_available(image: str) -> None:
    if TF_SKIP_PULL:
        return
    if not TF_FORCE_PULL and image in _PULLED_IMAGES:
        return
    logger.info("Pulling TensorFlow container image: %s", image)
    try:
        subprocess.run(
            ["docker", "pull", image],
            check=True,
        )
    except FileNotFoundError as exc:  # pragma: no cover - runtime dependency
        raise Stage1ContainerError("docker CLI not found in PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise Stage1ContainerError(f"docker pull failed: {exc}") from exc
    _PULLED_IMAGES.add(image)


def run_stage1_in_container(config: JudgeVisionConfig) -> None:
    """Serialize config, ensure image availability, then run Stage 1 inside docker."""

    if not TF_CONTAINER_ENABLED:
        raise Stage1ContainerError(
            "TensorFlow container execution disabled via env var"
        )

    payload = serialize_config(config)
    try:
        image, command = build_docker_command(config, payload)
        _ensure_image_available(image)
        logger.info(
            "Launching TensorFlow Stage 1 container (%s)",
            image,
        )
        _run_subprocess(command, diag_image=image)
    finally:
        try:
            payload.unlink()
        except FileNotFoundError:
            pass


def terminate_active_container() -> None:
    """Terminate the active docker process (used when the CLI is interrupted)."""

    global _ACTIVE_PROCESS
    proc = _ACTIVE_PROCESS
    if not proc:
        return
    logger.info("Terminating TensorFlow container (signal forwarded)")
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        logger.warning("TensorFlow container did not exit after SIGTERM; killing")
        proc.kill()
    finally:
        _ACTIVE_PROCESS = None


__all__ = [
    "Stage1ContainerError",
    "build_docker_command",
    "run_stage1_in_container",
    "serialize_config",
    "terminate_active_container",
]
