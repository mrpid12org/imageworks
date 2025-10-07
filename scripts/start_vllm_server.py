#!/usr/bin/env python3
"""Helper to launch a vLLM OpenAI-compatible server.

Provides convenience defaults for the Qwen2-VL-2B model and resolves weights
from ``$IMAGEWORKS_MODEL_ROOT`` when present, while still allowing explicit
paths or Hugging Face repo identifiers.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, TextIO

DEFAULT_MODEL_SUBDIR = "Qwen2-VL-2B-Instruct"
DEFAULT_SERVED_MODEL_NAME = "Qwen2-VL-2B-Instruct"


def _candidate_model_root() -> Path:
    return Path(
        os.environ.get("IMAGEWORKS_MODEL_ROOT", Path.home() / "ai-models" / "weights")
    )


def resolve_model_argument(raw_model: str | None) -> str:
    """Resolve the model argument to a path when possible.

    - If *raw_model* is provided and refers to an existing file system path, the
      expanded path is returned.
    - If *raw_model* is empty, fall back to ``$IMAGEWORKS_MODEL_ROOT`` (or the
      default weights directory) joined with ``DEFAULT_MODEL_SUBDIR`` when that
      path exists.
    - Otherwise, return the original string so Hugging Face repos remain valid.
    """

    if raw_model:
        expanded = Path(raw_model).expanduser()
        if expanded.exists():
            return str(expanded)
        return raw_model

    candidate = _candidate_model_root() / DEFAULT_MODEL_SUBDIR
    if candidate.exists():
        return str(candidate)

    # Default to relative path used in documentation for local runs
    return str(Path("./models") / DEFAULT_MODEL_SUBDIR)


def _extract_option_value(tokens: List[str], option: str) -> Optional[str]:
    """Return the value assigned to *option* from *tokens* when present."""

    for idx, token in enumerate(tokens):
        if token == option:
            if idx + 1 < len(tokens):
                return tokens[idx + 1]
            return ""
        if token.startswith(f"{option}="):
            return token.split("=", 1)[1]
    return None


def build_command(args: argparse.Namespace) -> List[str]:
    """Construct the vLLM server command."""

    model_arg = resolve_model_argument(args.model)

    # Use the currently running interpreter (sys.executable) instead of a hardcoded
    # 'python' binary. Some minimal distributions only expose 'python3'.
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_arg,
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
    # Optional memory tuning flags
    if getattr(args, "kv_cache_dtype", None):
        command.extend(["--kv-cache-dtype", args.kv_cache_dtype])
    if getattr(args, "swap_space", None) is not None:
        command.extend(["--swap-space", str(args.swap_space)])

    if args.enforce_eager:
        command.append("--enforce-eager")

    if args.trust_remote_code:
        command.append("--trust-remote-code")

    if args.api_keys:
        command.extend(["--api-keys", args.api_keys])

    # Attach chat template if provided (or auto-detected below)
    if args.chat_template:
        command.extend(["--chat-template", args.chat_template])

    if args.extra:
        forwarded = [token for token in args.extra if token and token != "--"]
        if forwarded:
            parser_value = _extract_option_value(forwarded, "--tool-call-parser")
            chat_format_value = _extract_option_value(
                forwarded, "--chat-template-content-format"
            )

            if parser_value == "openai" and chat_format_value is None:
                command.extend(["--chat-template-content-format", "openai"])

            command.extend(forwarded)

    return command


def start_server() -> None:
    parser = argparse.ArgumentParser(
        description="Start a vLLM OpenAI-compatible API server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Path or repo ID for the model. Defaults to $IMAGEWORKS_MODEL_ROOT/"
            f"{DEFAULT_MODEL_SUBDIR} when available."
        ),
    )
    parser.add_argument(
        "--served-model-name",
        default=DEFAULT_SERVED_MODEL_NAME,
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
        "--kv-cache-dtype",
        default=None,
        choices=[
            "auto",
            "fp8",
            "fp8_e5m2",
            "fp8_e4m3",
            "fp16",
            "bfloat16",
            "bf16",
        ],
        help=(
            "Data type for KV cache. Using FP8 (e5m2) can significantly reduce VRAM "
            "usage on consumer GPUs."
        ),
    )
    parser.add_argument(
        "--swap-space",
        type=int,
        default=None,
        help=("CPU swap space in GiB for paged attention (helps when VRAM is tight)."),
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
        "--chat-template",
        default=None,
        help=(
            "Path to a Jinja2 chat template file. If omitted and the resolved model "
            "appears to be a LLaVA/Vicuna variant, an attempt will be made to auto-"
            "select 'llava15_vicuna.jinja' from the current working directory if present."
        ),
    )
    parser.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Additional arguments forwarded to vLLM",
    )
    parser.add_argument(
        "--background",
        action="store_true",
        help="Start the server in the background (do not block).",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="If provided, redirect stdout/stderr to this file when running in background.",
    )

    args = parser.parse_args()

    # Best-effort auto detection for common LLaVA models lacking an embedded chat template.
    # Prefer packaged template over CWD; fallback to CWD if provided explicitly.
    if not args.chat_template:
        model_id_lower = (args.model or "").lower()
        if "llava" in model_id_lower or "vicuna" in model_id_lower:
            try:
                from importlib.resources import files as _res_files

                pkg_path = (
                    _res_files("imageworks.chat_templates") / "llava15_vicuna.jinja"
                )
                args.chat_template = str(pkg_path)
                print(
                    f"[auto-chat-template] Using packaged template {args.chat_template} for model '{args.model or 'auto-resolved'}'"
                )
            except Exception:
                candidate = Path("llava15_vicuna.jinja")
                if candidate.exists():
                    args.chat_template = str(candidate.resolve())
                    print(
                        f"[auto-chat-template] Using {args.chat_template} for model '{args.model or 'auto-resolved'}'"
                    )

    if shutil.which("python") is None:
        sys.stderr.write("Python interpreter not found in PATH.\n")
        sys.exit(1)

    command = build_command(args)
    print("🚀 Launching vLLM server with command:")
    print(" ".join(command))

    if args.background:
        # Start without blocking. Redirect to log file if provided, otherwise inherit.
        stdout: Optional[TextIO]
        stderr: Optional[TextIO]
        if args.log_file:
            log_path = Path(args.log_file).expanduser()
            log_path.parent.mkdir(parents=True, exist_ok=True)
            f = open(log_path, "a", buffering=1)
            stdout = f
            stderr = f
            print(f"[vllm-launcher] Background mode. Logging to {log_path}")
        else:
            stdout = subprocess.DEVNULL
            stderr = subprocess.STDOUT
            print("[vllm-launcher] Background mode. Output suppressed (no --log-file).")
        proc = subprocess.Popen(command, stdout=stdout, stderr=stderr)
        print(f"[vllm-launcher] Started PID {proc.pid} on port {args.port}.")
        # Do not wait; caller can check liveness separately
        return
    else:
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as exc:
            sys.stderr.write(f"vLLM server exited with status {exc.returncode}\n")
            sys.exit(exc.returncode)


if __name__ == "__main__":
    start_server()
