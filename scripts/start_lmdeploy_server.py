#!/usr/bin/env python3
"""Helper script to launch an LMDeploy OpenAI-compatible server.

The script defaults to the Qwen2.5-VL-7B vision-language model and enables
PyTorch eager mode to avoid CUDA Graph allocations (lower VRAM footprint).
Adjust arguments as needed for alternative deployments.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Mapping, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from imageworks.logging_utils import configure_logging

DEFAULT_MODEL_NAME = "Qwen2.5-VL-7B-AWQ"
DEFAULT_MODEL_REPO = Path("qwen-vl") / "Qwen2.5-VL-7B-Instruct-AWQ"

LOG_PATH = configure_logging("lmdeploy_server")
logger = logging.getLogger(__name__)
logger.info("LMDeploy startup logging initialised → %s", LOG_PATH)

ESSENTIAL_FILES = ("config.json", "tokenizer_config.json")
TOKENIZER_ALTERNATIVES = ("tokenizer.json", "tokenizer.model")
SUPPORTING_FILES = (
    "generation_config.json",
    "chat_template.json",
    "quantization_config.json",
)
WEIGHT_GLOBS = ("*.safetensors", "*.bin", "*.pt", "*.awq", "*.gguf")


def validate_model_directory(model_path: Path) -> List[str]:
    """Validate that essential model assets exist before launching LMDeploy.

    Returns a list of warning messages for optional-but-recommended files that are
    missing. Raises ``FileNotFoundError`` if the directory itself is absent and
    ``RuntimeError`` when required assets are missing.
    """

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model path '{model_path}' does not exist. Use --model-path to point to a valid directory."
        )

    if not model_path.is_dir():
        raise RuntimeError(
            f"Model path '{model_path}' is not a directory. Provide a directory that contains the exported weights."
        )

    missing_required: List[str] = []
    for filename in ESSENTIAL_FILES:
        candidate = model_path / filename
        if not candidate.is_file():
            missing_required.append(filename)

    if not any((model_path / alt).is_file() for alt in TOKENIZER_ALTERNATIVES):
        missing_required.append("tokenizer.json or tokenizer.model")

    if missing_required:
        raise RuntimeError(
            "Missing required model files: " + ", ".join(missing_required)
        )

    warnings: List[str] = []
    for filename in SUPPORTING_FILES:
        if not (model_path / filename).is_file():
            warnings.append(
                f"Optional file '{filename}' not found — responses may be degraded or require manual template overrides."
            )

    if not any(model_path.glob(pattern) for pattern in WEIGHT_GLOBS):
        warnings.append(
            "No weight files detected (*.safetensors/*.bin/*.pt). Ensure the download completed successfully."
        )

    return warnings


def resolve_default_model_root(
    env: Optional[Mapping[str, str]] = None,
    home: Optional[Path] = None,
) -> Path:
    """Resolve the base weights directory used for LMDeploy models.

    When ``IMAGEWORKS_MODEL_ROOT`` points at the ``weights`` directory (the
    default behaviour of the model downloader), use it verbatim. Otherwise append
    a ``weights`` suffix so both ``~/ai-models`` and ``~/ai-models/weights`` work
    without additional flags.
    """

    environ = env or os.environ
    candidate = environ.get("IMAGEWORKS_MODEL_ROOT")
    if candidate:
        root = Path(candidate).expanduser()
        if root.name.lower() == "weights":
            return root
        return root / "weights"

    base_home = home or Path.home()
    return base_home / "ai-models" / "weights"


def resolve_default_model_path(
    env: Optional[Mapping[str, str]] = None,
    home: Optional[Path] = None,
) -> Path:
    """Return the default path to the bundled Qwen AWQ checkpoint."""

    return resolve_default_model_root(env=env, home=home) / DEFAULT_MODEL_REPO


def build_command(args: argparse.Namespace) -> List[str]:
    """Construct the lmdeploy CLI command."""

    resolved_path = args.model_path or str(resolve_default_model_path())
    model_path = Path(resolved_path).expanduser()
    command = [
        "lmdeploy",
        "serve",
        "api_server",
        str(model_path),
        "--server-name",
        args.host,
        "--server-port",
        str(args.port),
        "--model-name",
        args.model_name,
        "--backend",
        args.backend,
        "--device",
        args.device,
    ]

    if args.vision_max_batch_size is not None:
        command.extend(
            [
                "--vision-max-batch-size",
                str(args.vision_max_batch_size),
            ]
        )

    if args.max_batch_size is not None:
        command.extend(["--max-batch-size", str(args.max_batch_size)])

    if args.eager:
        command.append("--eager-mode")

    if args.disable_fastapi_docs:
        command.append("--disable-fastapi-docs")

    if args.api_keys:
        command.extend(["--api-keys", args.api_keys])

    if args.extra:
        command.extend(args.extra)

    return command


def start_server() -> None:
    parser = argparse.ArgumentParser(
        description="Start an LMDeploy OpenAI-compatible API server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    default_path = resolve_default_model_path()
    parser.add_argument(
        "--model-path",
        default=None,
        help=(
            "HuggingFace repo or local path to the model weights "
            f"(defaults to {default_path})"
        ),
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Model identifier exposed to the OpenAI API",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host interface for the server",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=23333,
        help="Port for the API server",
    )
    parser.add_argument(
        "--backend",
        default="pytorch",
        choices=["pytorch", "turbomind"],
        help="Inference backend to use",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Target device for inference (e.g. cuda, cuda:0, cpu)",
    )
    parser.add_argument(
        "--vision-max-batch-size",
        type=int,
        default=1,
        help="Maximum batch size for vision inputs",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=None,
        help="Override LMDeploy max batch size",
    )
    parser.add_argument(
        "--eager",
        dest="eager",
        action="store_true",
        help="Enable eager mode (disables CUDA Graphs to reduce VRAM usage)",
    )
    parser.add_argument(
        "--no-eager",
        dest="eager",
        action="store_false",
        help="Disable eager mode",
    )
    parser.set_defaults(eager=True)
    parser.add_argument(
        "--disable-fastapi-docs",
        action="store_true",
        help="Disable FastAPI OpenAPI/Swagger endpoints",
    )
    parser.add_argument(
        "--api-keys",
        default=None,
        help="Comma-separated API keys for authentication",
    )
    parser.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Additional arguments forwarded to lmdeploy",
    )

    args = parser.parse_args()
    if args.model_path is None:
        args.model_path = str(default_path)

    model_path = Path(args.model_path).expanduser()
    try:
        warnings = validate_model_directory(model_path)
    except FileNotFoundError as exc:
        logger.error("❌ %s", exc)
        sys.exit(2)
    except RuntimeError as exc:
        logger.error(
            "❌ Missing critical model assets detected. %s. Ensure the downloader completed successfully or copy the files manually.",
            exc,
        )
        sys.exit(2)

    if warnings:
        for warning in warnings:
            logger.warning("⚠️  %s", warning)

    args.model_path = str(model_path)

    if shutil.which("lmdeploy") is None:
        logger.error(
            "lmdeploy CLI not found. Install via 'uv add lmdeploy' or 'pip install lmdeploy'."
        )
        sys.exit(1)

    command = build_command(args)
    logger.info("🚀 Launching LMDeploy server with command: %s", " ".join(command))

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        logger.error("LMDeploy server exited with status %s", exc.returncode)
        sys.exit(exc.returncode)


if __name__ == "__main__":
    start_server()
