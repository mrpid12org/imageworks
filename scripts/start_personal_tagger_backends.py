#!/usr/bin/env python3
"""Launch all backend servers required for the Personal Tagger."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import urlparse

from imageworks.apps.personal_tagger.core.config import load_config
from imageworks.libs.vlm import VLMBackend

DEFAULT_LMDEPLOY_SCRIPT = Path("scripts/start_lmdeploy_server.py")
DEFAULT_VLLM_SCRIPT = Path("scripts/start_vllm_server.py")


@dataclass(frozen=True)
class StageConfig:
    name: str
    backend: VLMBackend
    base_url: str
    model: str
    model_path: str | None


def parse_backend(value: str) -> VLMBackend:
    try:
        return VLMBackend(value.lower())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Unknown backend '{value}'") from exc


def parse_base_url(url: str) -> Tuple[str, int]:
    if not url:
        raise ValueError("Base URL must not be empty")
    parsed = urlparse(url if "://" in url else f"http://{url}")
    host = parsed.hostname or "localhost"
    port = parsed.port
    if port is None:
        port = 443 if parsed.scheme == "https" else 80
    return host, port


def stage_from_args(prefix: str, args: argparse.Namespace) -> StageConfig:
    backend: VLMBackend = getattr(args, f"{prefix}_backend")
    model: str = getattr(args, f"{prefix}_model")
    base_url: str = getattr(args, f"{prefix}_base_url")
    model_path: str | None = getattr(args, f"{prefix}_model_path")
    return StageConfig(prefix, backend, base_url, model, model_path)


def build_command(
    stage: StageConfig,
    lmdeploy_script: Path,
    vllm_script: Path,
) -> Tuple[List[str], Tuple[str, int]]:
    host, port = parse_base_url(stage.base_url)
    if stage.backend == VLMBackend.LMDEPLOY:
        cmd = [
            sys.executable,
            str(lmdeploy_script),
            "--model-name",
            stage.model,
            "--port",
            str(port),
        ]
        if host != "0.0.0.0":
            cmd.extend(["--host", host])
        if stage.model_path:
            cmd.extend(["--model-path", stage.model_path])
        return cmd, (host, port)

    if stage.backend == VLMBackend.VLLM:
        model_arg = stage.model_path or stage.model
        cmd = [
            sys.executable,
            str(vllm_script),
            "--model",
            model_arg,
            "--served-model-name",
            stage.model,
            "--port",
            str(port),
            "--host",
            host,
            "--trust-remote-code",
        ]
        return cmd, (host, port)

    raise ValueError(f"Backend {stage.backend.value} is not supported by this script")


def main() -> None:
    defaults = load_config(Path.cwd())

    parser = argparse.ArgumentParser(
        description="Start backend servers required for personal tagger",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Stage-specific overrides
    parser.add_argument(
        "--caption-backend",
        type=parse_backend,
        default=defaults.default_backend,
        help="Backend for caption generation",
    )
    parser.add_argument(
        "--caption-model",
        default=defaults.caption_model,
        help="Model identifier for caption generation",
    )
    parser.add_argument(
        "--caption-base-url",
        default=str(defaults.default_base_url),
        help="Base URL where the caption backend will listen",
    )
    parser.add_argument(
        "--caption-model-path",
        default=None,
        help="Optional local path to the caption model (for local backends)",
    )

    parser.add_argument(
        "--keyword-backend",
        type=parse_backend,
        default=defaults.default_backend,
        help="Backend for keyword generation",
    )
    parser.add_argument(
        "--keyword-model",
        default=defaults.keyword_model,
        help="Model identifier for keyword generation",
    )
    parser.add_argument(
        "--keyword-base-url",
        default=str(defaults.default_base_url),
        help="Base URL where the keyword backend will listen",
    )
    parser.add_argument(
        "--keyword-model-path",
        default=None,
        help="Optional local path to the keyword model",
    )

    parser.add_argument(
        "--description-backend",
        type=parse_backend,
        default=defaults.default_backend,
        help="Backend for description generation",
    )
    parser.add_argument(
        "--description-model",
        default=defaults.description_model,
        help="Model identifier for description generation",
    )
    parser.add_argument(
        "--description-base-url",
        default=str(defaults.default_base_url),
        help="Base URL where the description backend will listen",
    )
    parser.add_argument(
        "--description-model-path",
        default=None,
        help="Optional local path to the description model",
    )

    parser.add_argument(
        "--lmdeploy-script",
        type=Path,
        default=DEFAULT_LMDEPLOY_SCRIPT,
        help="Path to the LMDeploy startup helper",
    )
    parser.add_argument(
        "--vllm-script",
        type=Path,
        default=DEFAULT_VLLM_SCRIPT,
        help="Path to the vLLM startup helper",
    )

    parser.add_argument(
        "--launch",
        action="store_true",
        help="Launch the backend processes instead of printing commands",
    )

    parser.add_argument(
        "--allow-shared-port",
        action="store_true",
        help="Allow multiple stages to share the same base URL even if they use different models",
    )

    args = parser.parse_args()

    stages = [
        stage_from_args("caption", args),
        stage_from_args("keyword", args),
        stage_from_args("description", args),
    ]

    combos: Dict[Tuple[VLMBackend, str, str], List[str]] = {}
    for stage in stages:
        key = (stage.backend, stage.base_url, stage.model)
        combos.setdefault(key, []).append(stage.name)

    commands: Dict[Tuple[VLMBackend, str, str], Tuple[List[str], Tuple[str, int]]] = {}
    for stage in stages:
        key = (stage.backend, stage.base_url, stage.model)
        if key not in commands:
            commands[key] = build_command(stage, args.lmdeploy_script, args.vllm_script)

    print("📋 Personal Tagger backend plan:")
    for key, stage_names in combos.items():
        backend, base_url, model = key
        print(f"  • {', '.join(stage_names)} → {backend.value} ({model}) @ {base_url}")

    if not args.allow_shared_port:
        seen_ports: Dict[Tuple[str, int], Tuple[VLMBackend, str]] = {}
        for key, (_, host_port) in commands.items():
            backend, base_url, model = key
            if host_port in seen_ports and seen_ports[host_port] != (backend, model):
                raise SystemExit(
                    f"Port conflict: {base_url} is already assigned to {seen_ports[host_port]}"
                )
            seen_ports[host_port] = (backend, model)

    if not args.launch:
        print("\n🔧 Run the following commands to start the servers:")
        for (backend, base_url, model), (cmd, _) in commands.items():
            print(f"\n# {backend.value} → {model} @ {base_url}")
            print(" ".join(cmd))
        return

    processes = []
    try:
        for key, (cmd, _) in commands.items():
            backend, base_url, model = key
            print(f"\n🚀 Launching {backend.value} for {model} @ {base_url}")
            print("  command:", " ".join(cmd))
            proc = subprocess.Popen(cmd)
            processes.append(proc)

        print("\n✅ All backend processes started. Press Ctrl+C to terminate.")
        for proc in processes:
            proc.wait()
    except KeyboardInterrupt:
        print("\n🛑 Received interrupt, terminating child processes...")
    finally:
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()


if __name__ == "__main__":
    main()
