"""Minimal TensorFlow inference service for NIMA/MUSIQ.

This module can:
1. Run inference once from the command line
2. Serve a lightweight HTTP API (POST /infer) for repeated calls

Only the TensorFlow-heavy models (NIMA + MUSIQ) execute here; the host
handles all other technical signals.
"""

from __future__ import annotations

import base64
import json
import argparse
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Import aesthetic_models directly - the __init__.py now handles missing cv2 gracefully
import imageworks.libs.vision.aesthetic_models as aesthetic_models

_GPU_PROBED = False
_SHUTDOWN_REQUESTED = False


def _log_gpu_devices(use_gpu: bool) -> None:
    global _GPU_PROBED
    if _GPU_PROBED:
        return
    event = {"event": "tf_devices", "use_gpu": use_gpu}
    if not use_gpu:
        event["devices"] = []
        print(json.dumps(event))
        _GPU_PROBED = True
        return
    try:
        import tensorflow as tf  # type: ignore

        devices = [dev.name for dev in tf.config.list_physical_devices("GPU")]
        event["devices"] = devices
    except Exception as exc:  # noqa: BLE001
        event = {"event": "tf_devices_error", "error": str(exc)}
    print(json.dumps(event))
    _GPU_PROBED = True


@contextmanager
def _temp_image_file(image_bytes: bytes, image_name: str | None = None):
    suffix = Path(image_name or "image.jpg").suffix or ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(image_bytes)
        tmp.flush()
        tmp.close()
        yield Path(tmp.name)
    finally:
        Path(tmp.name).unlink(missing_ok=True)


def run_inference(
    image_path: str | Path | None = None,
    *,
    image_bytes: bytes | None = None,
    image_name: str | None = None,
    use_gpu: bool = True,
) -> dict:
    """Run NIMA and MUSIQ inference on a single image.

    Returns dict with:
        nima_aesthetic: {mean, std}
        nima_technical: {mean, std}
        musiq_spaq: float
    """
    if image_bytes is not None:
        with _temp_image_file(image_bytes, image_name=image_name) as tmp_path:
            return _run_inference_path(tmp_path, use_gpu=use_gpu)
    if image_path is None:
        raise ValueError("image_path or image_bytes must be provided")
    return _run_inference_path(Path(image_path), use_gpu=use_gpu)


def _run_inference_path(img_path: Path, use_gpu: bool) -> dict:

    results = {}

    # NIMA aesthetic
    try:
        nima_aesthetic = aesthetic_models.score_nima(
            img_path, flavor="aesthetic", use_gpu=use_gpu
        )
        results["nima_aesthetic"] = nima_aesthetic
    except Exception as e:
        results["nima_aesthetic_error"] = str(e)

    # NIMA technical
    try:
        nima_technical = aesthetic_models.score_nima(
            img_path, flavor="technical", use_gpu=use_gpu
        )
        results["nima_technical"] = nima_technical
    except Exception as e:
        results["nima_technical_error"] = str(e)

    # MUSIQ
    try:
        musiq_result = aesthetic_models.score_musiq(
            img_path, variant="spaq", use_gpu=use_gpu
        )
        results["musiq_spaq"] = musiq_result
    except Exception as e:
        results["musiq_error"] = str(e)

    return results


class _InferenceHandler(BaseHTTPRequestHandler):
    use_gpu = True

    def log_message(self, format: str, *args) -> None:  # noqa: D401
        """Silence default logging; container logs are noisy otherwise."""
        return

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    def _write_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: D401
        """Health endpoint."""
        if self.path == "/health":
            self._write_json(HTTPStatus.OK, {"status": "ok", "use_gpu": self.use_gpu})
        else:
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: D401
        """Handle inference requests."""
        if self.path != "/infer":
            if self.path == "/shutdown" and self.command == "POST":
                self._handle_shutdown()
            else:
                self._write_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        payload = self._read_json()
        image_path = payload.get("image_path")
        image_b64 = payload.get("image_b64")
        image_name = payload.get("image_name")
        if not image_path and not image_b64:
            self._write_json(
                HTTPStatus.BAD_REQUEST, {"error": "image_path or image_b64 required"}
            )
            return
        use_gpu = bool(payload.get("use_gpu", self.use_gpu))
        if image_b64:
            try:
                image_bytes = base64.b64decode(image_b64)
            except Exception:  # noqa: BLE001
                self._write_json(HTTPStatus.BAD_REQUEST, {"error": "invalid image_b64"})
                return
            results = run_inference(
                None,
                image_bytes=image_bytes,
                image_name=image_name or image_path,
                use_gpu=use_gpu,
            )
        else:
            results = run_inference(image_path, use_gpu=use_gpu)
        self._write_json(HTTPStatus.OK, results)

    def _handle_shutdown(self) -> None:
        global _SHUTDOWN_REQUESTED
        if _SHUTDOWN_REQUESTED:
            self._write_json(HTTPStatus.ACCEPTED, {"status": "shutting_down"})
            return
        _SHUTDOWN_REQUESTED = True
        self._write_json(HTTPStatus.OK, {"status": "terminating"})

        def _delayed_exit():
            time.sleep(0.2)
            os._exit(0)

        threading.Thread(target=_delayed_exit, daemon=True).start()


def run_http_server(host: str, port: int, use_gpu: bool = True) -> None:
    _InferenceHandler.use_gpu = use_gpu
    server = ThreadingHTTPServer((host, port), _InferenceHandler)
    print(
        json.dumps(
            {"status": "listening", "host": host, "port": port, "use_gpu": use_gpu}
        )
    )
    _log_gpu_devices(use_gpu)
    server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TensorFlow IQA inference service")
    subparsers = parser.add_subparsers(dest="mode")

    run_parser = subparsers.add_parser("run", help="Run inference once")
    run_parser.add_argument("image_path", type=str, help="Path to the image")
    run_parser.add_argument("--cpu", action="store_true", help="Force CPU inference")

    serve_parser = subparsers.add_parser("serve", help="Start HTTP inference server")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=5005)
    serve_parser.add_argument("--cpu", action="store_true", help="Force CPU inference")

    args = parser.parse_args()

    if args.mode == "serve":
        run_http_server(args.host, args.port, use_gpu=not args.cpu)
    elif args.mode == "run":
        results = run_inference(args.image_path, use_gpu=not args.cpu)
        print(json.dumps(results))
    else:
        parser.error("Please specify 'run' or 'serve'")
