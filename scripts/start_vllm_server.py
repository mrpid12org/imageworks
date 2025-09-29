#!/usr/bin/env python3
"""Helper to launch the default Qwen2-VL-2B vLLM server."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_MODEL_SUBDIR = "Qwen2-VL-2B-Instruct"


def resolve_model_path(raw_path: str | None) -> Path:
    """Resolve the model directory, preferring the shared model root."""

    if raw_path:
        candidate = Path(raw_path).expanduser()
    else:
        model_root = Path(
            os.environ.get(
                "IMAGEWORKS_MODEL_ROOT", Path.home() / "ai-models" / "weights"
            )
        )
        candidate = model_root / DEFAULT_MODEL_SUBDIR

    if not candidate.exists():
        raise FileNotFoundError(
            f"Model directory not found: {candidate}. "
            "Set IMAGEWORKS_MODEL_ROOT or pass --model-path to point at the weights."
        )

    return candidate


def build_server_args(config: dict[str, object]) -> list[str]:
    args: list[str] = ["python", "-m", "vllm.entrypoints.openai.api_server"]
    for key, value in config.items():
        if value is None:
            continue
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                args.append(flag)
        else:
            args.extend([flag, str(value)])
    return args


def start_vllm_server(port: int, gpu_memory: float, model_path: str | None) -> None:
    """Start vLLM server with optimal configuration for Qwen2-VL-2B."""

    resolved_path = resolve_model_path(model_path)

    config = {
        "model": str(resolved_path),
        "host": "0.0.0.0",
        "port": port,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": gpu_memory,
        "max_model_len": 8192,
        "dtype": "auto",
        "max_num_seqs": 16,
        "max_num_batched_tokens": 4096,
        "served_model_name": "Qwen2-VL-2B-Instruct",
        "trust_remote_code": True,
        "enforce_eager": False,
    }

    print("🚀 Starting vLLM server for Qwen2-VL-2B-Instruct")
    print(f"📁 Model path: {resolved_path}")
    print(f"🌐 Server: http://localhost:{config['port']}")
    print(f"🎯 GPU memory: {config['gpu_memory_utilization']*100:.0f}%")
    print(f"📏 Max sequence length: {config['max_model_len']}")

    try:
        args = build_server_args(config)
        print(f"🔧 vLLM command: {' '.join(args)}")
        print("⏳ Loading model and starting server...")
        subprocess.run(args, check=True)
    except FileNotFoundError as exc:
        print(f"❌ {exc}")
        sys.exit(1)
    except ImportError as exc:
        print(f"❌ vLLM import error: {exc}")
        print("💡 Make sure vLLM is installed: uv add vllm")
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        print(f"❌ Server startup error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start vLLM server for Qwen2-VL-2B")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument(
        "--gpu-memory", type=float, default=0.8, help="GPU memory utilization"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Override model directory"
        " (defaults to $IMAGEWORKS_MODEL_ROOT/Qwen2-VL-2B-Instruct)",
    )

    cli_args = parser.parse_args()
    start_vllm_server(cli_args.port, cli_args.gpu_memory, cli_args.model_path)
