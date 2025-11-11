"""Wrapper to call TensorFlow inference in container and return results to host."""

from __future__ import annotations

import base64
import json
import logging
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional

import requests

logger = logging.getLogger(__name__)

DEFAULT_IMAGE = "nvcr.io/nvidia/tensorflow:24.02-tf2-py3"
PROJECT_ROOT = Path(__file__).resolve().parents[4]
SERVICE_URL = os.environ.get("JUDGE_VISION_TF_SERVICE_URL", "http://127.0.0.1:5105")
SERVICE_HEALTH_ENDPOINT = f"{SERVICE_URL}/health"
SERVICE_INFER_ENDPOINT = f"{SERVICE_URL}/infer"
SERVICE_SHUTDOWN_ENDPOINT = f"{SERVICE_URL}/shutdown"
SERVICE_START_SCRIPT = PROJECT_ROOT / "scripts" / "start_tf_iqa_service.sh"
DEFAULT_PIP_PACKAGES = (
    "numpy<2.0",
    "tensorflow-hub==0.15.0",
    "tomli",
)


def _call_http_service(image_path: Path, use_gpu: bool) -> Optional[Dict]:
    """Send inference request to long-running service."""
    try:
        image_bytes = image_path.read_bytes()
    except FileNotFoundError:
        logger.warning("Image not found for TF service: %s", image_path)
        return None
    except OSError as exc:
        logger.warning("Unable to read %s for TF service: %s", image_path, exc)
        return None

    payload = {
        "image_path": str(image_path),
        "image_name": image_path.name,
        "use_gpu": use_gpu,
        "image_b64": base64.b64encode(image_bytes).decode("ascii"),
    }
    try:
        resp = requests.post(SERVICE_INFER_ENDPOINT, json=payload, timeout=120)
        if resp.status_code != 200:
            logger.warning("TF service returned %s: %s", resp.status_code, resp.text)
            return None
        return resp.json()
    except requests.RequestException as exc:
        logger.debug("TF service unavailable: %s", exc)
        return None


def _service_available() -> bool:
    try:
        resp = requests.get(SERVICE_HEALTH_ENDPOINT, timeout=2)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def ensure_tf_service_running(wait_seconds: float = 20.0) -> bool:
    """Start the long-running TF IQA service if needed."""
    if os.environ.get("JUDGE_VISION_USE_TF_SERVICE", "1") != "1":
        return False
    if _service_available():
        return True
    if not SERVICE_START_SCRIPT.exists():
        logger.warning(
            "TF IQA service start script missing at %s", SERVICE_START_SCRIPT
        )
        return False
    logger.info(
        "TensorFlow IQA service not running; launching via %s", SERVICE_START_SCRIPT
    )
    try:
        subprocess.run(["bash", str(SERVICE_START_SCRIPT)], check=True)
    except subprocess.CalledProcessError as exc:
        logger.warning("Failed to launch TF IQA service: %s", exc)
        return False
    deadline = time.monotonic() + max(1.0, wait_seconds)
    while time.monotonic() < deadline:
        if _service_available():
            logger.info("TensorFlow IQA service is healthy")
            return True
        time.sleep(0.5)
    logger.warning(
        "TensorFlow IQA service failed to become healthy within %.1fs", wait_seconds
    )
    return False


def shutdown_tf_service(timeout: float = 5.0) -> bool:
    """Attempt to shut down the long-running TF service via HTTP."""
    if os.environ.get("JUDGE_VISION_USE_TF_SERVICE", "1") != "1":
        return False
    try:
        resp = requests.post(SERVICE_SHUTDOWN_ENDPOINT, timeout=timeout)
        if resp.status_code == 200:
            logger.info("TensorFlow IQA service acknowledged shutdown request")
            return True
        logger.debug("TF service shutdown returned %s: %s", resp.status_code, resp.text)
    except requests.RequestException as exc:
        logger.debug("TF service shutdown request failed: %s", exc)
    return False


def call_tf_container_inference(
    image_path: Path,
    use_gpu: bool = True,
    container_image: str = DEFAULT_IMAGE,
    model_root: Optional[Path] = None,
) -> Dict:
    """
    Call TensorFlow container to run NIMA/MUSIQ inference on a single image.

    Args:
        image_path: Path to image file
        use_gpu: Whether to use GPU
        container_image: Docker image to use
        model_root: Model weights directory (defaults to IMAGEWORKS_MODEL_ROOT)

    Returns:
        Dict with inference results: {nima_aesthetic, nima_technical, musiq_spaq}
    """
    from imageworks.tools.model_downloader.config import get_config

    if model_root is None:
        model_root = get_config().linux_wsl.root

    # Normalize model root
    if model_root.name == "weights":
        model_root = model_root.parent

    container_model_root = Path("/root/ai-models")
    image_path = image_path.resolve()

    # First attempt HTTP service if available
    if os.environ.get("JUDGE_VISION_USE_TF_SERVICE", "1") == "1":
        result = _call_http_service(image_path, use_gpu=use_gpu)
        if result is not None:
            return result

    # Path to the inference script for one-off runs
    inference_script = (
        PROJECT_ROOT
        / "src"
        / "imageworks"
        / "apps"
        / "judge_vision"
        / "tf_inference_service.py"
    )

    musiq_cache = (
        container_model_root / "weights" / "judge-iqa" / "musiq" / "tfhub-cache"
    )

    pip_packages = os.environ.get("JUDGE_VISION_TF_PIP_PACKAGES")
    if pip_packages:
        pip_cmd = pip_packages
    else:
        pip_cmd = "pip install -q --no-cache-dir " + " ".join(
            shlex.quote(pkg) for pkg in DEFAULT_PIP_PACKAGES
        )

    python_cmd = "python3 {script} run {image}{cpu}".format(
        script=shlex.quote(str(inference_script)),
        image=shlex.quote(str(image_path)),
        cpu=" --cpu" if not use_gpu else "",
    )

    cmd = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all" if use_gpu else "none",
        "-v",
        f"{PROJECT_ROOT}:{PROJECT_ROOT}",
        "-v",
        f"{model_root}:{container_model_root}",
        "-v",
        f"{image_path.parent}:{image_path.parent}",
        "-w",
        str(PROJECT_ROOT),
        "-e",
        f"PYTHONPATH={PROJECT_ROOT}/src",
        "-e",
        f"IMAGEWORKS_MODEL_ROOT={container_model_root}",
        "-e",
        f"TFHUB_CACHE_DIR={musiq_cache}",
        "-e",
        f"TFHUB_CACHE={musiq_cache}",
        "-e",
        "TF_CPP_MIN_LOG_LEVEL=2",
        "-e",
        "JUDGE_VISION_INSIDE_CONTAINER=1",
        container_image,
        "bash",
        "-c",
        f"{pip_cmd} && {python_cmd}",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            logger.error("Container inference failed: %s", result.stderr)
            return {"error": f"Container failed: {result.stderr}"}

        # Parse JSON output from last line
        output_lines = [line for line in result.stdout.strip().split("\n") if line]
        if not output_lines:
            return {"error": "No output from container"}

        return json.loads(output_lines[-1])

    except subprocess.TimeoutExpired:
        return {"error": "Container inference timed out"}
    except json.JSONDecodeError as e:
        logger.error("Failed to parse container output: %s", result.stdout)
        return {"error": f"Invalid JSON from container: {e}"}
    except Exception as e:
        logger.error("Container inference exception: %s", e)
        return {"error": str(e)}
