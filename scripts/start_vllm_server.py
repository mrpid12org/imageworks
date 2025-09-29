#!/usr/bin/env python3
"""Helper script to launch a vLLM OpenAI-compatible server."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List


def build_command(args: argparse.Namespace) -> List[str]:
    command = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(Path(args.model).expanduser()),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--served-model-name",
        args.served_model_name,
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-model-len",
        str(args.max_model_len),
        "--dtype",
        args.dtype,
    ]

    if args.max_num_seqs is not None:
        command.extend(["--max-num-seqs", str(args.max_num_seqs)])
    if args.max_num_batched_tokens is not None:
        command.extend(["--max-num-batched-tokens", str(args.max_num_batched_tokens)])

    if args.enforce_eager:
        command.append("--enforce-eager")

    if args.trust_remote_code:
        command.append("--trust-remote-code")

    if args.api_keys:
        command.extend(["--api-keys", args.api_keys])

    if args.extra:
        command.extend(args.extra)

    return command


def start_server() -> None:
    parser = argparse.ArgumentParser(
        description="Start a vLLM OpenAI-compatible API server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="./models/Qwen2-VL-2B-Instruct",
        help="Path or repo ID for the model",
    )
    parser.add_argument(
        "--served-model-name",
        default="vllm-model",
        help="Model name exposed through the OpenAI API",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host interface")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of tensor parallel partitions",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
        help="Fraction of GPU memory to allocate",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="Maximum sequence length",
    )
    parser.add_argument("--dtype", default="auto", help="Torch dtype to use")
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=None,
        help="Maximum number of sequences batched together",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help="Maximum tokens processed in a single batch",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable CUDA graphs (lower VRAM, slower throughput)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow execution of custom HF model code",
    )
    parser.add_argument(
        "--api-keys",
        default=None,
        help="Comma-separated API keys to require for access",
    )
    parser.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Additional arguments forwarded to vLLM",
    )

    args = parser.parse_args()

    if shutil.which("python") is None:
        sys.stderr.write("Python interpreter not found in PATH.\n")
        sys.exit(1)

    command = build_command(args)
    print("🚀 Launching vLLM server with command:")
    print(" ".join(command))

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        sys.stderr.write(f"vLLM server exited with status {exc.returncode}\n")
        sys.exit(exc.returncode)


if __name__ == "__main__":
    start_server()
