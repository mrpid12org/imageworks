#!/usr/bin/env python3
"""
vLLM server startup script for Qwen2-VL-7B-Instruct.
Optimized for vision-language tasks with OpenAI API compatibility.
"""

import argparse
import sys


def start_vllm_server():
    """Start vLLM server with optimal configuration for Qwen2-VL-7B."""

    # Model path
    model_path = "./models/Qwen2-VL-7B-Instruct"

    # Server configuration
    config = {
        "model": model_path,
        "host": "0.0.0.0",
        "port": 8000,
        "api_key": "EMPTY",
        # GPU and memory settings
        "tensor_parallel_size": 1,  # Single GPU
        "gpu_memory_utilization": 0.9,  # Use 90% of GPU memory
        "max_model_len": 8192,  # Context length
        # Vision settings
        "image_input_type": "pixel_values",
        "image_token_id": 151646,
        "image_feature_size": 4096,
        # Performance settings
        "max_num_seqs": 16,
        "max_num_batched_tokens": 4096,
        # API compatibility
        "served_model_name": "Qwen2-VL-7B-Instruct",
        "chat_template": None,  # Use model's built-in template
        # Safety and reliability
        "trust_remote_code": True,
        "enforce_eager": False,
        "disable_log_stats": False,
    }

    print("🚀 Starting vLLM server for Qwen2-VL-7B-Instruct")
    print(f"📁 Model path: {model_path}")
    print(f"🌐 Server: http://localhost:{config['port']}")
    print(f"🎯 GPU memory: {config['gpu_memory_utilization']*100}%")
    print(f"📏 Max sequence length: {config['max_model_len']}")

    # Import and start vLLM
    try:
        from vllm.entrypoints.openai.api_server import main as vllm_api_server

        # Build command line arguments
        args = []
        for key, value in config.items():
            if value is not None:
                args.extend([f"--{key.replace('_', '-')}", str(value)])

        print(f"🔧 vLLM args: {' '.join(args)}")
        print("⏳ Loading model and starting server...")

        # Start the server
        sys.argv = ["vllm.entrypoints.openai.api_server"] + args
        vllm_api_server()

    except ImportError as e:
        print(f"❌ vLLM import error: {e}")
        print("💡 Make sure vLLM is installed: uv add vllm")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Server startup error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start vLLM server for Qwen2-VL-7B")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument(
        "--gpu-memory", type=float, default=0.9, help="GPU memory utilization"
    )

    args = parser.parse_args()
    start_vllm_server()
