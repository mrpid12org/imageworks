"""Deterministic image quality predictors (NIMA + MUSIQ)."""

from __future__ import annotations

import contextlib
import io
import logging
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from PIL import Image

from imageworks.tools.model_downloader.config import get_config

logger = logging.getLogger(__name__)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

_MODEL_ROOT = get_config().linux_wsl.root / "weights" / "judge-iqa"
_NIMA_DIR = _MODEL_ROOT / "nima"
_MUSIQ_CACHE = _MODEL_ROOT / "musiq" / "tfhub-cache"
_MUSIQ_CACHE.mkdir(parents=True, exist_ok=True)
# TensorFlow Hub switched from TFHUB_CACHE to TFHUB_CACHE_DIR; set both so cached
# modules land under the shared judge-iqa directory even inside containers.
os.environ.setdefault("TFHUB_CACHE_DIR", str(_MUSIQ_CACHE))
os.environ.setdefault("TFHUB_CACHE", str(_MUSIQ_CACHE))

_NIMA_FILENAMES: Dict[str, str] = {
    "aesthetic": "weights_mobilenet_aesthetic_0.07.hdf5",
    "technical": "weights_mobilenet_technical_0.11.hdf5",
}

_MUSIQ_URLS = {
    "spaq": "https://tfhub.dev/google/musiq/spaq/1",
    "koniq": "https://tfhub.dev/google/musiq/koniq/1",
    "ava": "https://tfhub.dev/google/musiq/ava/1",
}

_TF_MODULES = None
_TF_LOCK = threading.Lock()
_TF_GPU_MODE: Optional[bool] = None
_CONTAINER_CACHE: "OrderedDict[str, Dict]" = OrderedDict()
_CONTAINER_CACHE_LOCK = threading.Lock()
_CONTAINER_CACHE_LIMIT = 16


def _encode_rgb_jpeg(image_path: Path) -> bytes:
    """Load an image, force RGB8, and return encoded JPEG bytes."""
    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=95)
        return buf.getvalue()


def _nima_directory() -> Path:
    return _NIMA_DIR


def _resolve_nima_weight(flavor: str) -> Path:
    filename = _NIMA_FILENAMES[flavor]
    path = _nima_directory() / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/download_judge_iqa_models.py before executing Judge Vision."
        )
    return path


def _ensure_musiq_cache_ready() -> None:
    if not _MUSIQ_CACHE.exists() or not any(_MUSIQ_CACHE.iterdir()):
        raise FileNotFoundError(
            "MUSIQ cache is empty. Run scripts/download_judge_iqa_models.py before executing Judge Vision."
        )


def _require_tensorflow(use_gpu: bool):
    global _TF_MODULES
    global _TF_GPU_MODE
    if _TF_MODULES is not None:
        if _TF_GPU_MODE != use_gpu:
            raise RuntimeError(
                "TensorFlow already initialised with a different GPU mode"
            )
        return _TF_MODULES
    with _TF_LOCK:
        if _TF_MODULES is None:
            _TF_GPU_MODE = use_gpu
            buf = io.StringIO()
            with contextlib.redirect_stderr(buf):
                import tensorflow as tf  # type: ignore
                import tensorflow_hub as hub  # type: ignore
            if not use_gpu:
                try:
                    tf.config.set_visible_devices([], "GPU")
                except Exception:  # noqa: BLE001
                    pass
            if buf.getvalue().strip():
                logger.debug("TensorFlow import logs: %s", buf.getvalue().strip())
            gpu_devices = tf.config.list_physical_devices("GPU")
            if gpu_devices:
                device_str = ", ".join(device.name for device in gpu_devices)
                logger.info("TensorFlow detected GPU device(s): %s", device_str)
            else:
                logger.warning(
                    "TensorFlow GPU device list is empty; using CPU execution."
                )
            _TF_MODULES = (tf, hub)
    return _TF_MODULES


class _NimaClient:
    _lock = threading.Lock()
    _models: Dict[str, object] = {}

    @classmethod
    def score(cls, flavor: str, image_path: Path, use_gpu: bool) -> Dict[str, float]:
        tf, _ = _require_tensorflow(use_gpu)
        with cls._lock:
            model = cls._models.get(flavor)
            if model is None:
                model = cls._build_model(tf, flavor)
                cls._models[flavor] = model
        return cls._predict(tf, model, image_path)

    @classmethod
    def _build_model(cls, tf_mod, flavor: str):
        weights = _resolve_nima_weight(flavor)
        logger.info("Building NIMA %s model from %s", flavor, weights)
        base_model = tf_mod.keras.applications.MobileNet(
            input_shape=(224, 224, 3), weights=None, include_top=False, pooling="avg"
        )
        x = tf_mod.keras.layers.Dropout(0.75)(base_model.output)
        outputs = tf_mod.keras.layers.Dense(10, activation="softmax")(x)
        model = tf_mod.keras.Model(base_model.inputs, outputs)
        model.load_weights(str(weights), by_name=True, skip_mismatch=True)
        logger.info("NIMA %s weights loaded", flavor)
        return model

    @staticmethod
    def _predict(tf_mod, model, image_path: Path) -> Dict[str, float]:
        img = tf_mod.keras.utils.load_img(str(image_path), target_size=(224, 224))
        arr = tf_mod.keras.utils.img_to_array(img)
        arr = tf_mod.keras.applications.mobilenet.preprocess_input(arr)
        preds = model(tf_mod.expand_dims(arr, axis=0), training=False).numpy()[0]
        scores = np.arange(1, 11, dtype=np.float32)
        mean = float(np.dot(preds, scores))
        std = float(np.sqrt(np.dot(preds, (scores - mean) ** 2)))
        return {
            "mean": mean,
            "std": std,
        }


class _MusiqClient:
    _lock = threading.Lock()
    _signatures: Dict[str, object] = {}

    @classmethod
    def score(cls, variant: str, image_path: Path, use_gpu: bool) -> float:
        _ensure_musiq_cache_ready()
        tf, hub = _require_tensorflow(use_gpu)
        with cls._lock:
            signature = cls._signatures.get(variant)
            if signature is None:
                model = hub.load(_MUSIQ_URLS[variant])
                signature = model.signatures["serving_default"]
                cls._signatures[variant] = signature
        image_bytes = _encode_rgb_jpeg(image_path)
        tensor = tf.convert_to_tensor(image_bytes, dtype=tf.string)
        outputs = signature(image_bytes_tensor=tensor)
        return float(outputs["output_0"].numpy())


def score_nima(
    image_path: Path, flavor: str, use_gpu: bool = False
) -> Optional[Dict[str, float]]:
    if flavor not in _NIMA_FILENAMES:
        raise ValueError(f"Unknown NIMA flavor '{flavor}'")

    inside_container = os.environ.get("JUDGE_VISION_INSIDE_CONTAINER") == "1"
    if inside_container:
        try:
            return _NimaClient.score(flavor, image_path, use_gpu)
        except Exception as exc:  # noqa: BLE001
            logger.warning("NIMA %s inference failed inside container: %s", flavor, exc)
            return None

    try:
        results = _get_container_results(image_path, use_gpu=use_gpu)
    except Exception as exc:  # noqa: BLE001
        logger.error("Container inference failed for NIMA %s: %s", flavor, exc)
        return None
    if not results:
        logger.error("Empty response from TensorFlow IQA container for %s", flavor)
        return None
    if "error" in results:
        logger.error("Container inference error for %s: %s", flavor, results["error"])
        return None
    payload = results.get(f"nima_{flavor}")
    if payload:
        return payload
    logger.error("NIMA %s not present in container response: %s", flavor, results)
    return None


def score_musiq(
    image_path: Path, variant: str = "spaq", use_gpu: bool = False
) -> Optional[float]:
    if variant not in _MUSIQ_URLS:
        raise ValueError(f"Unknown MUSIQ variant '{variant}'")

    inside_container = os.environ.get("JUDGE_VISION_INSIDE_CONTAINER") == "1"
    if inside_container:
        try:
            return _MusiqClient.score(variant, image_path, use_gpu)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "MUSIQ (%s) inference failed inside container: %s", variant, exc
            )
            return None

    try:
        results = _get_container_results(image_path, use_gpu=use_gpu)
    except Exception as exc:  # noqa: BLE001
        logger.error("Container inference failed for MUSIQ %s: %s", variant, exc)
        return None
    if not results:
        logger.error(
            "Empty response from TensorFlow IQA container for MUSIQ %s", variant
        )
        return None
    if "error" in results:
        logger.error(
            "Container inference error for MUSIQ %s: %s", variant, results["error"]
        )
        return None
    payload = results.get(f"musiq_{variant}")
    if payload is not None:
        return payload
    logger.error("MUSIQ %s not present in container response: %s", variant, results)
    return None


__all__ = ["score_nima", "score_musiq"]


def _get_container_results(image_path: Path, use_gpu: bool) -> Optional[Dict]:
    """Fetch (and cache) container inference results for an image."""
    image_path = image_path.resolve()
    key = _container_cache_key(image_path)
    with _CONTAINER_CACHE_LOCK:
        cached = _CONTAINER_CACHE.get(key)
        if cached is not None:
            _CONTAINER_CACHE.move_to_end(key)
            return cached

    from imageworks.apps.judge_vision.tf_container_wrapper import (
        call_tf_container_inference,
    )

    results = call_tf_container_inference(image_path, use_gpu=use_gpu)
    if results and "error" not in results:
        with _CONTAINER_CACHE_LOCK:
            _CONTAINER_CACHE[key] = results
            _CONTAINER_CACHE.move_to_end(key)
            while len(_CONTAINER_CACHE) > _CONTAINER_CACHE_LIMIT:
                _CONTAINER_CACHE.popitem(last=False)
    return results


def _container_cache_key(image_path: Path) -> str:
    try:
        stat = image_path.stat()
        marker = f"{stat.st_mtime_ns}:{stat.st_size}"
    except FileNotFoundError:
        marker = "missing"
    return f"{str(image_path)}:{marker}"
