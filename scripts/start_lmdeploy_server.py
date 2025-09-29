#!/usr/bin/env python3
"""Helper script to launch an LMDeploy OpenAI-compatible server.

The script defaults to the Qwen2.5-VL-7B vision-language model and enables
PyTorch eager mode to avoid CUDA Graph allocations (lower VRAM footprint).
Adjust arguments as needed for alternative deployments.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

DEFAULT_MODEL_NAME = "Qwen2.5-VL-7B-AWQ"
DEFAULT_MODEL_PATH = str(
    Path(os.environ.get("IMAGEWORKS_MODEL_ROOT", Path.home() / "ai-models" / "weights"))
    / "Qwen2.5-VL-7B-Instruct-AWQ"
)


def build_command(args: argparse.Namespace) -> List[str]:
    """Construct the lmdeploy CLI command."""
    model_path = Path(args.model_path).expanduser()
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
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="HuggingFace repo or local path to the model weights",
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

    if shutil.which("lmdeploy") is None:
        sys.stderr.write(
            "lmdeploy CLI not found. Install via 'uv add lmdeploy' or 'pip install lmdeploy'.\n"
        )
        sys.exit(1)

    command = build_command(args)
    print("🚀 Launching LMDeploy server with command:")
    print(" ".join(command))

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        sys.stderr.write(f"LMDeploy server exited with status {exc.returncode}\n")
        sys.exit(exc.returncode)


if __name__ == "__main__":
    start_server()
