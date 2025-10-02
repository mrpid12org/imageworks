#!/usr/bin/env python3
"""Launch a lightweight OpenAI-compatible HTTP server for a GGUF model.

This provides a fallback when vLLM / LMDeploy cannot host a desired
quantized variant but a GGUF file is readily available.

Implementation strategy:
- Use llama-cpp-python's built-in fastapi server when available (preferred)
- Fallback to a minimal custom FastAPI wrapper around llama_cpp.Llama for
  a subset of the Chat Completions API (only what imageworks currently uses)

Key constraints:
- Single-process, single-GPU (or CPU) inference
- Vision inputs are NOT supported in this fallback (text-only). If the
  caller supplies image content, we raise an error to avoid silent failures.

Environment variables:
  IMAGEWORKS_MODEL_ROOT  Base directory for weights (default ~/ai-models/weights)

Example:
  uv run python scripts/start_gguf_server.py \
     --model TheBloke/Llama-2-7B-Chat-GGUF \
     --quant Q4_K_M \
     --port 8600

If a local path is supplied and ends with .gguf it is used directly.
Otherwise we attempt to resolve a registry or typical folder
convention: <root>/<owner>/<repo>/<file>.gguf.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn
except Exception:  # noqa: BLE001
    sys.stderr.write(
        "Missing FastAPI dependencies. Install with 'uv add fastapi uvicorn'.\n"
    )
    raise

try:
    from llama_cpp import Llama
except Exception:  # noqa: BLE001
    sys.stderr.write(
        "llama-cpp-python not installed. Install with 'uv add llama-cpp-python'.\n"
    )
    raise


DEFAULT_MODEL = "TheBloke/Llama-2-7B-Chat-GGUF"
DEFAULT_QUANT = "Q4_K_M"  # widely available compromise quant


def resolve_model_path(model: str, quant: str) -> Path:
    p = Path(model).expanduser()
    if p.is_file() and p.suffix == ".gguf":
        return p
    # If it's a directory containing exactly one gguf file (or quant match), pick it
    if p.is_dir():
        candidates = list(p.glob("*.gguf"))
        if not candidates:
            raise FileNotFoundError(f"No .gguf files found in directory {p}")
        if quant:
            quant_matches = [c for c in candidates if quant.lower() in c.name.lower()]
            if quant_matches:
                return quant_matches[0]
        return candidates[0]

    # Otherwise treat as owner/repo pattern referencing local weights root
    root = Path(
        os.environ.get("IMAGEWORKS_MODEL_ROOT", Path.home() / "ai-models" / "weights")
    ).expanduser()
    if "/" in model:
        owner, repo = model.split("/", 1)
        repo_dir = root / owner / repo
        if not repo_dir.exists():
            raise FileNotFoundError(
                f"Repo path {repo_dir} not found. Download the GGUF variant first."
            )
        # Choose quant file
        quant_files = [
            f for f in repo_dir.glob("*.gguf") if quant.lower() in f.name.lower()
        ]
        if not quant_files:
            all_files = list(repo_dir.glob("*.gguf"))
            if not all_files:
                raise FileNotFoundError(f"No GGUF files under {repo_dir}")
            # fallback to first
            return all_files[0]
        return sorted(quant_files)[0]

    raise FileNotFoundError(f"Could not resolve model path for {model}")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    max_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.95
    stream: bool = False


class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    model: str
    choices: List[ChatChoice]
    usage: dict


def build_app(llm: Llama, served_model_name: str) -> FastAPI:
    app = FastAPI()

    @app.get("/v1/models")
    def list_models():  # noqa: D401
        return {"data": [{"id": served_model_name, "object": "model"}]}

    @app.post("/v1/chat/completions", response_model=ChatResponse)
    def chat(req: ChatRequest):  # noqa: D401
        # minimal validation
        if any("<image" in m.content.lower() for m in req.messages):
            raise HTTPException(
                400, "Vision inputs not supported by GGUF fallback server"
            )
        # Build a simple prompt (OpenAI style -> plain conversation)
        rendered = []
        for m in req.messages:
            prefix = (
                "User:"
                if m.role == "user"
                else (
                    "Assistant:" if m.role == "assistant" else f"{m.role.capitalize():}"
                )
            )
            rendered.append(f"{prefix} {m.content.strip()}")
        rendered.append("Assistant:")
        prompt = "\n".join(rendered)

        output = llm(
            prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            stop=["User:", "Assistant:"],
        )
        text = output["choices"][0]["text"].strip()
        usage = output.get("usage", {})

        return ChatResponse(
            id="chatcmpl-gguf-fallback",
            model=served_model_name,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=text),
                    finish_reason="stop",
                )
            ],
            usage=usage
            or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )

    return app


def main():  # noqa: D401
    parser = argparse.ArgumentParser(
        description="Start a minimal OpenAI-compatible server for GGUF models"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Owner/Repo or local path to GGUF (file, dir, or repo ID)",
    )
    parser.add_argument(
        "--quant",
        default=DEFAULT_QUANT,
        help="Quantization substring to pick GGUF file (e.g. Q4_K_M, Q5_K_S)",
    )
    parser.add_argument(
        "--served-model-name",
        default=None,
        help="Model name reported by API (defaults to GGUF filename)",
    )
    parser.add_argument("--ctx-size", type=int, default=4096, help="Context length")
    parser.add_argument("--port", type=int, default=8600, help="HTTP port")
    parser.add_argument("--threads", type=int, default=8, help="Threads for llama.cpp")
    parser.add_argument(
        "--gpu-layers",
        type=int,
        default=-1,
        help="Number of layers to offload to GPU (-1 = all, 0 = CPU only)",
    )
    parser.add_argument(
        "--n-batch", type=int, default=512, help="Batch size for prompt processing"
    )
    parser.add_argument(
        "--no-mmap", action="store_true", help="Disable mmap (slightly higher RAM use)"
    )
    parser.add_argument("extra", nargs=argparse.REMAINDER, help="Reserved for future")
    args = parser.parse_args()

    try:
        gguf_path = resolve_model_path(args.model, args.quant)
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"❌ {exc}\n")
        sys.exit(2)

    served_name = args.served_model_name or gguf_path.stem
    print(f"✅ Using GGUF file: {gguf_path}")
    print(f"   Served model name: {served_name}")

    llm = Llama(
        model_path=str(gguf_path),
        n_ctx=args.ctx_size,
        n_threads=args.threads,
        n_gpu_layers=args.gpu_layers,
        n_batch=args.n_batch,
        use_mmap=not args.no_mmap,
        verbose=False,
    )

    app = build_app(llm, served_name)
    print(f"🚀 Starting GGUF fallback server on port {args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":  # pragma: no cover
    main()
