"""Color-Narrator CLI - Main command line interface.

This module provides the primary CLI for color-narrator functionality,
following the same patterns as the mono-checker CLI.
"""

import os
import re
import typer
from typing import Optional, Dict, Any, List
from pathlib import Path
from urllib.parse import urlparse
import tomllib  # Built-in since Python 3.11
import logging
import json
import requests
from PIL import Image
import time
import subprocess
from datetime import datetime
from collections import Counter

from ..core import prompts
from ..core.metadata import XMPMetadataWriter
from ..core.narrator import ColorNarrator, NarrationConfig, ProcessingResult
from ..core.region_based_vlm import (
    RegionBasedVLMAnalyzer,
)
from ..core.vlm import VLMBackend, VLMClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = typer.Typer(help="Color-Narrator - VLM-guided color localization")


DEFAULT_VLM_SETTINGS: Dict[VLMBackend, Dict[str, Any]] = {
    VLMBackend.VLLM: {
        "base_url": "http://localhost:8000/v1",
        "model": "Qwen2-VL-2B-Instruct",
        "api_key": "EMPTY",
        "timeout": 120,
    },
    VLMBackend.LMDEPLOY: {
        "base_url": "http://localhost:24001/v1",
        "model": "Qwen2.5-VL-7B-AWQ",
        "api_key": "EMPTY",
        "timeout": 120,
    },
    VLMBackend.TRITON: {
        "base_url": "http://localhost:9000/v1",
        "model": "Qwen2-VL-2B-Instruct",
        "api_key": "EMPTY",
        "timeout": 120,
    },
}


def _to_bool(value: Any, default: bool) -> bool:
    """Coerce a configuration value into a boolean."""

    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    try:
        return bool(int(value))
    except Exception:
        return bool(value)


def _find_pyproject(start_path: Path) -> Optional[Path]:
    """Find pyproject.toml by walking up the directory tree."""
    current = start_path.resolve()
    for parent in [current] + list(current.parents):
        candidate = parent / "pyproject.toml"
        if candidate.exists():
            return candidate
    return None


def load_config(start_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load merged imageworks configuration for color narrator + mono defaults."""

    cfg_path = _find_pyproject(start_path or Path.cwd())
    if not cfg_path:
        return {}

    try:
        with cfg_path.open("rb") as fp:
            data = tomllib.load(fp)
    except Exception as exc:
        logger.warning("Failed to load config from %s: %s", cfg_path, exc)
        return {}

    imageworks_cfg = (
        data.get("tool", {}).get("imageworks", {}) if isinstance(data, dict) else {}
    )

    merged: Dict[str, Any] = {}

    mono_cfg = imageworks_cfg.get("mono")
    if isinstance(mono_cfg, dict):
        merged.update(mono_cfg)

    narrator_cfg = imageworks_cfg.get("color_narrator")
    if isinstance(narrator_cfg, dict):
        merged.update(narrator_cfg)

    # Include other scalar entries directly under [tool.imageworks] when available
    for key, value in imageworks_cfg.items():
        if key in {"mono", "color_narrator"}:
            continue
        merged.setdefault(key, value)

    # Environment overrides: IMAGEWORKS_COLOR_NARRATOR__FOO=bar → foo: "bar"
    prefix = "IMAGEWORKS_COLOR_NARRATOR__"
    for env_key, env_value in os.environ.items():
        if not env_key.startswith(prefix):
            continue
        config_key = env_key[len(prefix) :].lower()
        merged[config_key] = env_value

    return merged


def _backend_connection_info(base_url: str) -> Dict[str, Any]:
    """Parse base URL and derive connection metadata."""

    trimmed = base_url.rstrip("/")
    parsed = urlparse(trimmed if "://" in trimmed else f"http://{trimmed}")

    scheme = parsed.scheme or "http"
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if scheme == "https" else 80)

    api_base = trimmed
    if not parsed.path or parsed.path == "/":
        api_base = f"{scheme}://{host}"
        if port not in (80, 443):
            api_base = f"{api_base}:{port}"
        api_base = f"{api_base}/v1"

    api_root = api_base
    if api_root.endswith("/v1"):
        api_root = api_root[: -len("/v1")]

    return {
        "scheme": scheme,
        "host": host,
        "port": port,
        "api_root": api_root,
        "api_base": api_base,
    }


def _resolve_vlm_runtime_settings(
    cfg: Dict[str, Any],
    *,
    backend: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    timeout: Optional[int] = None,
    api_key: Optional[str] = None,
) -> tuple[VLMBackend, str, str, int, str, Optional[Dict[str, Any]]]:
    """Resolve runtime VLM settings from configuration and overrides."""

    backend_value_raw = backend or cfg.get("vlm_backend")
    backend_value = str(backend_value_raw or VLMBackend.LMDEPLOY.value).strip().lower()

    try:
        backend_enum = VLMBackend(backend_value)
    except ValueError as exc:  # noqa: B904 - keep original context for caller
        raise ValueError(f"Unknown VLM backend '{backend_value}'.") from exc

    def resolve_backend_setting(suffix: str, default: Any) -> Any:
        for key in (
            f"vlm_{backend_enum.value}_{suffix}",
            f"{backend_enum.value}_{suffix}",
            f"vlm_{suffix}",
            suffix,
        ):
            if key in cfg and cfg[key] not in (None, ""):
                return cfg[key]
        return default

    defaults = DEFAULT_VLM_SETTINGS.get(backend_enum, {})

    resolved_base_url = base_url or resolve_backend_setting(
        "base_url", defaults.get("base_url", "http://localhost:8000/v1")
    )
    resolved_model = model or resolve_backend_setting(
        "model", defaults.get("model", "Qwen2-VL-2B-Instruct")
    )

    raw_timeout = (
        timeout
        if timeout is not None
        else resolve_backend_setting("timeout", defaults.get("timeout", 120))
    )
    try:
        resolved_timeout = int(raw_timeout)
    except (TypeError, ValueError):
        resolved_timeout = int(defaults.get("timeout", 120))

    resolved_api_key = (
        api_key
        if api_key is not None
        else resolve_backend_setting("api_key", defaults.get("api_key", "EMPTY"))
    )

    backend_options = resolve_backend_setting("options", None)
    if not isinstance(backend_options, dict):
        backend_options = None

    resolved_base_url = str(resolved_base_url).strip()
    resolved_model = str(resolved_model).strip()
    resolved_api_key = "" if resolved_api_key is None else str(resolved_api_key)

    return (
        backend_enum,
        resolved_base_url,
        resolved_model,
        resolved_timeout,
        resolved_api_key,
        backend_options,
    )


def _start_vllm_server(port: Optional[int] = None) -> subprocess.Popen:
    """Start the vLLM server in the background.

    Returns:
        subprocess.Popen: The server process
    """
    try:
        # Check if start_vllm_server.py exists
        server_script = Path("scripts/start_vllm_server.py")
        if not server_script.exists():
            server_script = Path("start_vllm_server.py")
        if not server_script.exists():
            raise FileNotFoundError(f"Server script not found: {server_script}")

        # Start server using uv run
        typer.echo("🔧 Starting vLLM server process...")
        command = ["uv", "run", "python", str(server_script)]
        if port is not None:
            command.extend(["--port", str(port)])
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=Path.cwd(),
        )

        # Give it a moment to start
        time.sleep(3)

        # Check if process is still running
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            typer.echo("❌ Server process exited immediately:")
            if stdout:
                typer.echo(f"stdout: {stdout[:500]}")
            if stderr:
                typer.echo(f"stderr: {stderr[:500]}")
            raise RuntimeError("Server process failed to start")

        typer.echo("✅ Server process started successfully")
        return process
    except Exception as e:
        logger.error(f"Failed to start vLLM server: {e}")
        raise


def _start_lmdeploy_server(
    model_name: str,
    port: Optional[int] = None,
    host: str = "0.0.0.0",
    eager: bool = True,
) -> subprocess.Popen:
    """Start the LMDeploy OpenAI-compatible server in the background."""

    try:
        server_script = Path("scripts/start_lmdeploy_server.py")
        if not server_script.exists():
            server_script = Path("start_lmdeploy_server.py")
        if not server_script.exists():
            raise FileNotFoundError(f"Server script not found: {server_script}")

        typer.echo("🔧 Starting LMDeploy server process...")
        command = [
            "uv",
            "run",
            "python",
            str(server_script),
            "--model-name",
            model_name,
        ]
        if port is not None:
            command.extend(["--port", str(port)])
        if host:
            command.extend(["--host", host])
        if eager:
            command.append("--eager")

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=Path.cwd(),
        )

        time.sleep(3)

        if process.poll() is not None:
            stdout, stderr = process.communicate()
            typer.echo("❌ LMDeploy server exited immediately:")
            if stdout:
                typer.echo(f"stdout: {stdout[:500]}")
            if stderr:
                typer.echo(f"stderr: {stderr[:500]}")
            raise RuntimeError("LMDeploy server process failed to start")

        typer.echo("✅ LMDeploy server process started successfully")
        return process
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to start LMDeploy server: %s", exc)
        raise


def _wait_for_server(
    base_url: str = "http://localhost:8000/v1", timeout: int = 120
) -> bool:
    """Wait for the vLLM server to be ready.

    Args:
        base_url: Server URL to check
        timeout: Maximum time to wait in seconds

    Returns:
        bool: True if server is ready, False if timeout
    """
    info = _backend_connection_info(base_url)
    api_base = info["api_base"].rstrip("/")

    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{api_base}/models", timeout=5)
            if response.status_code == 200:
                return True
        except (requests.RequestException, Exception):
            pass
        time.sleep(2)
    return False


def _get_vlm_client_with_autostart(
    backend: str | VLMBackend = VLMBackend.VLLM,
    *,
    base_url: Optional[str] = None,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: Optional[int] = None,
    backend_options: Optional[Dict[str, Any]] = None,
    auto_start: bool = True,
    eager: bool = True,
) -> VLMClient:
    """Get VLM client with optional automatic server startup.

    Args:
        backend: Backend identifier (string or enum)
        base_url: API base URL for the backend
        model_name: Model identifier
        api_key: API key for authenticated backends
        timeout: Request timeout in seconds
        auto_start: Whether to auto-start server if not running
        eager: Use eager mode when starting LMDeploy server (disables CUDA graphs)

    Returns:
        VLMClient: Connected VLM client

    Raises:
        typer.Exit: If client initialization fails
    """
    try:
        backend_enum = (
            backend if isinstance(backend, VLMBackend) else VLMBackend(backend)
        )
        defaults = DEFAULT_VLM_SETTINGS.get(backend_enum, {})

        resolved_base_url = (
            base_url or defaults.get("base_url", "http://localhost:8000/v1")
        ).rstrip("/")
        resolved_model = model_name or defaults.get("model", "Qwen2-VL-2B-Instruct")
        resolved_api_key = api_key or defaults.get("api_key", "EMPTY")
        resolved_timeout = timeout or defaults.get("timeout", 120)

        connection = _backend_connection_info(resolved_base_url)

        vlm_client = VLMClient(
            base_url=connection["api_base"],
            model_name=resolved_model,
            api_key=resolved_api_key,
            timeout=resolved_timeout,
            backend=backend_enum,
            backend_options=backend_options,
        )

        # Test VLM connection and auto-start if needed
        if not vlm_client.health_check():
            if auto_start:
                typer.echo("🚀 VLM server not running, starting automatically...")
                typer.echo("💡 This may take 30-60 seconds for model loading")

                try:
                    server_process: Optional[subprocess.Popen[Any]] = None
                    if backend_enum == VLMBackend.VLLM:
                        server_process = _start_vllm_server(connection["port"])
                    elif backend_enum == VLMBackend.LMDEPLOY:
                        server_process = _start_lmdeploy_server(
                            model_name=resolved_model,
                            port=connection["port"],
                            host=connection["host"],
                            eager=eager,
                        )
                    else:
                        typer.echo(
                            "❌ Automatic startup not supported for this backend."
                        )
                        raise typer.Exit(1)

                    # Wait for server to be ready
                    typer.echo("⏳ Waiting for server to be ready...")
                    api_base = connection["api_base"]
                    if _wait_for_server(api_base, timeout=resolved_timeout):
                        typer.echo("✅ VLM server started successfully")
                    else:
                        typer.echo("❌ Server failed to start within 120 seconds")
                        if backend_enum == VLMBackend.VLLM:
                            typer.echo(
                                "💡 Try starting manually: uv run python scripts/start_vllm_server.py"
                            )
                        elif backend_enum == VLMBackend.LMDEPLOY:
                            typer.echo(
                                "💡 Try starting manually: uv run python scripts/start_lmdeploy_server.py"
                            )
                        typer.echo("💡 Or use --no-auto-start to disable auto-start")
                        if server_process is not None:
                            server_process.terminate()
                        raise typer.Exit(1)

                    # Test connection again
                    if not vlm_client.health_check():
                        typer.echo("❌ Server started but health check failed")
                        raise typer.Exit(1)

                except Exception as e:
                    typer.echo(f"❌ Failed to auto-start server: {e}")
                    if backend_enum == VLMBackend.VLLM:
                        typer.echo(
                            "💡 Try starting manually: uv run python scripts/start_vllm_server.py"
                        )
                    elif backend_enum == VLMBackend.LMDEPLOY:
                        typer.echo(
                            "💡 Try starting manually: uv run python scripts/start_lmdeploy_server.py"
                        )
                    typer.echo("💡 Or use --no-auto-start to disable auto-start")
                    raise typer.Exit(1)
            else:
                typer.echo(f"❌ VLM server not available at {connection['api_base']}")
                if backend_enum == VLMBackend.VLLM:
                    typer.echo(
                        "💡 Start server with: uv run python scripts/start_vllm_server.py"
                    )
                elif backend_enum == VLMBackend.LMDEPLOY:
                    typer.echo(
                        "💡 Start server with: uv run python scripts/start_lmdeploy_server.py"
                    )
                typer.echo("💡 Or use --auto-start to enable automatic startup")
                raise typer.Exit(1)

        typer.echo(
            f"🤖 VLM: Connected to {backend_enum.value} / {vlm_client.model_name}"
        )
        return vlm_client

    except Exception as e:
        if "Failed to initialize VLM client" not in str(e):
            typer.echo(f"❌ Failed to initialize VLM client: {e}")
        raise typer.Exit(1)


def _load_defaults() -> Dict[str, Any]:
    """Backward-compatible helper that returns merged imageworks defaults."""
    return load_config()


def _generate_enhancement_summary(enhanced_results: list, output_path: Path) -> None:
    """Generate human-readable markdown summary of enhancement results."""

    # Group by verdict
    by_verdict = {}
    for result in enhanced_results:
        verdict = result["verdict"]
        if verdict not in by_verdict:
            by_verdict[verdict] = []
        by_verdict[verdict].append(result)

    # Calculate stats
    total = len(enhanced_results)
    pass_count = len(by_verdict.get("pass", []))
    query_count = len(by_verdict.get("pass_with_query", []))
    fail_count = len(by_verdict.get("fail", []))

    avg_processing_time = (
        sum(r["vlm_processing_time"] for r in enhanced_results) / total
        if total > 0
        else 0
    )

    # Generate summary
    lines = [
        "# VLM-Enhanced Mono Analysis Summary",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"**Summary:** PASS={pass_count}  QUERY={query_count}  FAIL={fail_count}  (Total: {total})",
        f"**Average VLM Processing Time:** {avg_processing_time:.2f}s",
        "",
    ]

    # Add sections for each verdict type
    verdict_order = ["fail", "pass_with_query", "pass"]
    verdict_labels = {"fail": "FAIL", "pass_with_query": "QUERY", "pass": "PASS"}

    for verdict in verdict_order:
        if verdict not in by_verdict:
            continue

        results = by_verdict[verdict]
        if not results:
            continue

        lines.append(f"## {verdict_labels[verdict]} ({len(results)})")
        lines.append("")

        for result in results:
            lines.extend(
                [
                    f"**{result['title']}** by {result['author']}",
                    f"- File: {result['image_name']}",
                    f"- Verdict: {result['verdict']} ({result['mode']})",
                    f"- Dominant Color: {result['dominant_color']} (std dev: {result['hue_std_deg']:.1f}°)",
                    f"- Colorfulness: {result['colorfulness']:.2f} | Max Chroma: {result['chroma_max']:.2f}",
                    f"- VLM Processing: {result['vlm_processing_time']:.2f}s",
                    "",
                    "**VLM Enhanced Description:**",
                    f"{result['vlm_description']}",
                    "",
                    f"**Technical Context:** {result.get('original_reason', 'No technical details available')}",
                    "",
                    "---",
                    "",
                ]
            )

    # Write summary
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def _write_narration_summary(
    results: List[ProcessingResult], output_path: Path
) -> Dict[str, int]:
    """Persist human-readable narration summary and return verdict counts."""

    counts: Counter[str] = Counter()
    successes: List[ProcessingResult] = []
    failed: List[ProcessingResult] = []

    for result in results:
        mono_verdict = str(result.item.mono_data.get("verdict", "unknown")).lower()
        counts[mono_verdict] += 1

        if result.error:
            failed.append(result)
        elif result.vlm_response:
            successes.append(result)

    total = len(results)
    fail_total = counts.get("fail", 0)
    query_total = counts.get("pass_with_query", 0)
    pass_total = counts.get("pass", 0)
    other_total = total - (fail_total + query_total + pass_total)

    lines = [
        "# Color Narration Summary",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        (
            "**Totals:** "
            f"FAIL={fail_total}  QUERY={query_total}  PASS={pass_total}"
            + (f"  OTHER={other_total}" if other_total else "")
            + f"  (processed: {total})"
        ),
        "",
    ]

    verdict_sections = {
        "fail": "FAIL",
        "pass_with_query": "QUERY",
        "pass": "PASS",
    }

    for verdict_key, heading in verdict_sections.items():
        section_items = [
            result
            for result in successes
            if str(result.item.mono_data.get("verdict", "")).lower() == verdict_key
        ]
        if not section_items:
            continue

        lines.append(f"## {heading} ({len(section_items)})")
        lines.append("")

        for result in section_items:
            mono = result.item.mono_data
            image_name = result.item.image_path.name
            title = mono.get("title") or result.item.image_path.stem
            author = mono.get("author") or "Unknown"
            dominant_color = mono.get("dominant_color") or "unknown"
            hue = mono.get("dominant_hue_deg")
            hue_text = f" (hue {hue:.1f}°)" if isinstance(hue, (int, float)) else ""
            colorfulness = mono.get("colorfulness")
            chroma_max = mono.get("chroma_max")
            reason = mono.get("reason_summary") or mono.get("failure_reason") or ""

            lines.append(f"**{title}** by {author}")
            lines.append(f"- File: {image_name}")
            lines.append(
                f"- Verdict: {mono.get('verdict', 'unknown')} ({mono.get('mode', 'unknown')})"
            )
            lines.append(f"- Dominant colour: {dominant_color}{hue_text}")
            if colorfulness is not None or chroma_max is not None:
                cf = (
                    f"{colorfulness:.2f}"
                    if isinstance(colorfulness, (int, float))
                    else "?"
                )
                cm = (
                    f"{chroma_max:.2f}" if isinstance(chroma_max, (int, float)) else "?"
                )
                lines.append(f"- Colourfulness: {cf} | Max chroma: {cm}")
            if reason:
                lines.append(f"- Mono summary: {reason}")

            description = (result.vlm_response.description or "").strip()
            if description:
                lines.append("")
                lines.append("**Narration:**")
                lines.append(description)

            lines.append("")

    if failed:
        lines.append("## Errors")
        for result in failed[:10]:
            image_name = result.item.image_path.name
            lines.append(f"- {image_name}: {result.error}")
        lines.append("")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")

    return dict(counts)


def _sanitize_file_part(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", value)
    sanitized = re.sub(r"-+", "-", sanitized).strip("-._")
    return sanitized or "run"


def _duplicate_summary_with_label(
    summary_path: Path, prompt_definition: prompts.PromptDefinition, vlm_model: str
) -> Optional[Path]:
    summary_path = Path(summary_path)
    label = f"{vlm_model}_{prompt_definition.name}"
    sanitized = _sanitize_file_part(label)
    new_path = summary_path.with_name(
        f"{summary_path.stem}_{sanitized}{summary_path.suffix}"
    )

    if new_path == summary_path:
        return None

    new_path.parent.mkdir(parents=True, exist_ok=True)
    new_path.write_text(summary_path.read_text(encoding="utf-8"), encoding="utf-8")
    return new_path


def _write_results_json(
    results: List[ProcessingResult],
    output_path: Path,
    prompt_definition: prompts.PromptDefinition,
    narration_config: NarrationConfig,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat()

    with output_path.open("w", encoding="utf-8") as handle:
        for result in results:
            mono = result.item.mono_data
            vlm_response = result.vlm_response
            record = {
                "timestamp": timestamp,
                "image": str(result.item.image_path),
                "overlay": (
                    str(result.item.overlay_path) if result.item.overlay_path else None
                ),
                "prompt_id": prompt_definition.id,
                "prompt_name": prompt_definition.name,
                "vlm_model": narration_config.vlm_model,
                "vlm_base_url": narration_config.vlm_base_url,
                "verdict": mono.get("verdict"),
                "mode": mono.get("mode"),
                "dominant_color": mono.get("dominant_color"),
                "dominant_hue_deg": mono.get("dominant_hue_deg"),
                "colorfulness": mono.get("colorfulness"),
                "chroma_max": mono.get("chroma_max"),
                "reason_summary": mono.get("reason_summary"),
                "description": vlm_response.description if vlm_response else None,
                "confidence": (
                    getattr(vlm_response, "confidence", None) if vlm_response else None
                ),
                "metadata_written": result.metadata_written,
                "processing_time": result.processing_time,
                "error": result.error,
            }

            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


@app.command()
def narrate(
    images_dir: Optional[Path] = typer.Option(
        None, "--images", "-i", help="Directory containing JPEG originals"
    ),
    overlays_dir: Optional[Path] = typer.Option(
        None, "--overlays", "-o", help="Directory containing LAB overlay PNGs"
    ),
    mono_jsonl: Optional[Path] = typer.Option(
        None, "--mono-jsonl", "-j", help="Mono-checker results JSONL file"
    ),
    batch_size: Optional[int] = typer.Option(
        None, "--batch-size", "-b", help="VLM batch size (defaults to config)"
    ),
    no_meta: bool = typer.Option(
        False,
        "--no-meta",
        "--dry-run",
        help="Skip writing metadata (formerly --dry-run)",
    ),
    debug: bool = typer.Option(
        False,
        "--debug/--no-debug",
        help="Use bundled sample assets when enabled (otherwise rely on configured paths)",
    ),
    summary: Optional[Path] = typer.Option(
        None,
        "--summary",
        "-s",
        help="Path for Markdown summary (defaults to narrate_summary.md)",
    ),
    results_json: Optional[Path] = typer.Option(
        None,
        "--results-json",
        help="Path for JSONL results (defaults to narrate_results.jsonl)",
    ),
    prompt_id: Optional[int] = typer.Option(
        None,
        "--prompt",
        "-p",
        help="Prompt template id (use --list-prompts to inspect options)",
    ),
    list_prompts: bool = typer.Option(
        False, "--list-prompts", help="List available prompt templates and exit"
    ),
    regions: bool = typer.Option(
        False,
        "--regions/--no-regions",
        help="Enable spatial grid guidance when supported by the prompt",
    ),
    require_overlays_flag: Optional[bool] = typer.Option(
        None,
        "--require-overlays/--allow-missing-overlays",
        help="Override overlay requirement (default comes from configuration)",
    ),
    vlm_backend_opt: Optional[str] = typer.Option(
        None,
        "--vlm-backend",
        help="Override VLM backend (vllm, lmdeploy, triton)",
    ),
    vlm_base_url_opt: Optional[str] = typer.Option(
        None,
        "--vlm-base-url",
        help="Override VLM base URL (defaults depend on backend)",
    ),
    vlm_model_opt: Optional[str] = typer.Option(
        None,
        "--vlm-model",
        help="Override VLM model identifier",
    ),
    vlm_timeout_opt: Optional[int] = typer.Option(
        None,
        "--vlm-timeout",
        help="Override VLM request timeout (seconds)",
    ),
    vlm_api_key_opt: Optional[str] = typer.Option(
        None,
        "--vlm-api-key",
        help="Override VLM API key",
    ),
) -> None:
    """Generate colour narration metadata for competition JPEGs.

    Examples:
        imageworks-color-narrator narrate -i ./images -o ./overlays -j ./mono.jsonl
    """

    if list_prompts:
        typer.echo("Available prompts:")
        for definition in prompts.list_prompt_definitions():
            region_note = " (supports regions)" if definition.supports_regions else ""
            typer.echo(
                f"  {definition.id}: {definition.name}{region_note} - {definition.description}"
            )
        raise typer.Exit(0)

    cfg = load_config()

    available_prompts = prompts.list_prompt_definitions()
    available_ids = {definition.id for definition in available_prompts}

    cfg_prompt = cfg.get("default_prompt_id", prompts.DEFAULT_PROMPT_ID)
    default_prompt_id = prompts.get_prompt_definition(cfg_prompt).id

    selected_prompt_id = prompt_id if prompt_id is not None else default_prompt_id

    if selected_prompt_id not in available_ids:
        typer.echo(
            f"❌ Unknown prompt id: {selected_prompt_id}. Use --list-prompts to view options."
        )
        raise typer.Exit(1)

    prompt_definition = prompts.get_prompt_definition(selected_prompt_id)

    use_regions_flag = regions
    if use_regions_flag and not prompt_definition.supports_regions:
        typer.echo(
            f"⚠️ Prompt {prompt_definition.id} does not support region guidance; ignoring --regions"
        )
        use_regions_flag = False

    # Allow debug mode to use shared test assets when paths are omitted
    if debug and images_dir is None and overlays_dir is None and mono_jsonl is None:
        shared_images = Path("tests/shared/sample_production_images")
        shared_overlays = shared_images
        shared_jsonl = Path(
            "tests/shared/sample_production_mono_json_output/production_sample.jsonl"
        )
        if (
            shared_images.exists()
            and shared_overlays.exists()
            and shared_jsonl.exists()
        ):
            typer.echo("🐞 Debug mode: using tests/shared assets")
            images_dir = shared_images
            overlays_dir = shared_overlays
            mono_jsonl = shared_jsonl

    cfg_for_paths = cfg

    def resolve_path(option: Optional[Path], keys: List[str]) -> Optional[Path]:
        if option is not None:
            return option
        for key in keys:
            value = cfg_for_paths.get(key)
            if isinstance(value, str) and value:
                return Path(value)
        return None

    images_path = resolve_path(images_dir, ["default_images_dir", "default_folder"])
    overlays_path = resolve_path(overlays_dir, ["default_overlays_dir"])
    mono_jsonl_path = resolve_path(mono_jsonl, ["default_mono_jsonl", "default_jsonl"])

    if images_path is None or not images_path.exists():
        typer.echo(
            "❌ Images directory not found. Use --images or set default_images_dir/default_folder in pyproject.toml"
        )
        raise typer.Exit(1)

    if overlays_path is None or not overlays_path.exists():
        candidates = [
            images_path / "overlays",
            images_path.parent / "overlays",
            images_path,
        ]
        overlays_path = next(
            (candidate for candidate in candidates if candidate.exists()), overlays_path
        )

    if overlays_path is None or not overlays_path.exists():
        typer.echo(
            "❌ Overlays directory not found. Use --overlays or set default_overlays_dir in pyproject.toml"
        )
        raise typer.Exit(1)

    if mono_jsonl_path is None or not mono_jsonl_path.exists():
        typer.echo(
            "❌ Mono-checker JSONL not found. Use --mono-jsonl or set default_mono_jsonl/default_jsonl in pyproject.toml"
        )
        raise typer.Exit(1)

    if summary is not None:
        summary_path = summary
    else:
        summary_default = cfg.get(
            "narrate_summary_path", "outputs/summaries/narrate_summary.md"
        )
        summary_path = Path(summary_default)
        if debug:
            summary_path = Path("tests/test_output/narrate_summary.md")

    if results_json is not None:
        results_path = results_json
    else:
        results_default = cfg.get(
            "narrate_results_path", "outputs/results/narrate_results.jsonl"
        )
        results_path = Path(results_default)
        if debug:
            results_path = Path("tests/test_output/narrate_results.jsonl")

    batch = (
        batch_size if batch_size is not None else int(cfg.get("default_batch_size", 4))
    )
    min_contamination = float(cfg.get("min_contamination_level", 0.1))
    if require_overlays_flag is not None:
        require_overlays = require_overlays_flag
    else:
        require_overlays = _to_bool(cfg.get("require_overlays", True), True)
    overwrite_existing = _to_bool(cfg.get("overwrite_existing_metadata", False), False)
    backup_originals = _to_bool(cfg.get("backup_original_files", True), True)

    try:
        (
            backend_enum,
            vlm_base_url,
            vlm_model,
            vlm_timeout,
            vlm_api_key,
            backend_options,
        ) = _resolve_vlm_runtime_settings(
            cfg,
            backend=vlm_backend_opt,
            base_url=vlm_base_url_opt,
            model=vlm_model_opt,
            timeout=vlm_timeout_opt,
            api_key=vlm_api_key_opt,
        )
    except ValueError as exc:
        typer.echo(f"❌ {exc}")
        valid_backends = ", ".join(b.value for b in VLMBackend)
        typer.echo(f"   Available options: {valid_backends}")
        raise typer.Exit(1)

    typer.echo("🎨 Color Narrator — generating competition metadata")
    if no_meta:
        typer.echo("🔍 No-meta mode: files will not be modified")

    typer.echo(f"📁 Images: {images_path}")
    typer.echo(f"📁 Overlays: {overlays_path}")
    typer.echo(f"📄 Mono data: {mono_jsonl_path}")
    typer.echo(f"🧮 Batch size: {batch}")
    typer.echo(f"🎯 Contamination threshold: {min_contamination}")
    typer.echo(f"🤖 VLM: {backend_enum.value} / {vlm_model} at {vlm_base_url}")
    typer.echo(f"🗒️ Prompt {prompt_definition.id}: {prompt_definition.name}")
    if use_regions_flag:
        typer.echo("📐 Region guidance enabled")
    typer.echo(f"📊 Results JSON: {results_path}")

    narration_config = NarrationConfig(
        images_dir=images_path,
        overlays_dir=overlays_path,
        mono_jsonl=mono_jsonl_path,
        vlm_base_url=vlm_base_url,
        vlm_model=vlm_model,
        vlm_timeout=vlm_timeout,
        vlm_backend=backend_enum.value,
        vlm_api_key=str(vlm_api_key),
        vlm_backend_options=backend_options,
        batch_size=batch,
        min_contamination_level=min_contamination,
        require_overlays=require_overlays,
        dry_run=no_meta,
        debug=debug,
        overwrite_existing=overwrite_existing,
        prompt_id=prompt_definition.id,
        use_regions=use_regions_flag,
        allowed_verdicts={"fail", "pass_with_query"},
    )

    narrator = ColorNarrator(narration_config)
    narrator.metadata_writer.backup_original = backup_originals

    typer.echo("🚀 Starting narration pipeline...")
    try:
        results = narrator.process_all()
    except Exception as exc:
        typer.echo(f"❌ Narration pipeline failed: {exc}")
        raise typer.Exit(1)

    total = len(results)
    failures = [r for r in results if r.error]

    counts = _write_narration_summary(results, summary_path)
    fail_total = counts.get("fail", 0)
    query_total = counts.get("pass_with_query", 0)
    processed_total = fail_total + query_total

    typer.echo("\n📊 Narration Summary")
    typer.echo(f"   Processed: {total}")
    typer.echo(f"   Successful: {total - len(failures)}")
    typer.echo(f"   With errors: {len(failures)}")
    typer.echo(
        f"Summary: FAIL={fail_total}  QUERY={query_total}  TOTAL={processed_total}"
    )
    typer.echo(f"📝 Summary written to {summary_path}")

    alt_summary = _duplicate_summary_with_label(
        summary_path, prompt_definition, narration_config.vlm_model
    )
    if alt_summary:
        typer.echo(f"📝 Summary copy written to {alt_summary}")

    _write_results_json(results, results_path, prompt_definition, narration_config)
    typer.echo(f"💾 Results JSON written to {results_path}")

    if failures:
        typer.echo("\n⚠️  Sample errors (first 5):")
        for failure in failures[:5]:
            typer.echo(f"   • {failure.item.image_path.name}: {failure.error}")

    typer.echo("\n✅ Color narration complete")


@app.command()
def compare_prompts(
    images_dir: Optional[Path] = typer.Option(
        None, "--images", "-i", help="Directory containing JPEG originals"
    ),
    overlays_dir: Optional[Path] = typer.Option(
        None, "--overlays", "-o", help="Directory containing LAB overlay PNGs"
    ),
    mono_jsonl: Optional[Path] = typer.Option(
        None, "--mono-jsonl", "-j", help="Mono-checker results JSONL file"
    ),
    count: int = typer.Option(
        5,
        "--count",
        "-c",
        help="Number of most recent prompt templates to compare (default: 5)",
    ),
    prompt_ids: Optional[List[int]] = typer.Option(
        None,
        "--prompts",
        "-p",
        help="Specific prompt ids to compare (overrides --count)",
    ),
    summary: Optional[Path] = typer.Option(
        None,
        "--summary",
        "-s",
        help="Path for comparison Markdown output (defaults to prompt_comparison_<timestamp>.md)",
    ),
    batch_size: Optional[int] = typer.Option(
        None, "--batch-size", "-b", help="VLM batch size (defaults to config)"
    ),
    debug: bool = typer.Option(False, "--debug", help="Enable verbose logging"),
    regions: bool = typer.Option(
        False,
        "--regions/--no-regions",
        help="Enable spatial grid guidance when supported by prompts",
    ),
) -> None:
    """Compare narrator outputs across the most recent prompt templates."""

    cfg = load_config()

    available_prompts = prompts.list_prompt_definitions()
    if prompt_ids:
        selected_prompts = [p for p in available_prompts if p.id in set(prompt_ids)]
        missing = set(prompt_ids) - {p.id for p in selected_prompts}
        if missing:
            typer.echo(f"❌ Unknown prompt id(s): {sorted(missing)}")
            raise typer.Exit(1)
    else:
        ordered = sorted(available_prompts, key=lambda p: p.id, reverse=True)
        default_prompt = next(
            (p for p in available_prompts if p.id == prompts.DEFAULT_PROMPT_ID),
            None,
        )
        selected_prompts = []
        if default_prompt:
            selected_prompts.append(default_prompt)
        for definition in ordered:
            if default_prompt and definition.id == default_prompt.id:
                continue
            if len(selected_prompts) >= max(count, 1):
                break
            selected_prompts.append(definition)

    if not selected_prompts:
        typer.echo("❌ No prompt templates selected for comparison")
        raise typer.Exit(1)

    def to_bool(value: Any, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        try:
            return bool(int(value))
        except Exception:
            return bool(value)

    cfg_for_paths = cfg

    def resolve_path(option: Optional[Path], keys: List[str]) -> Optional[Path]:
        if option is not None:
            return option
        for key in keys:
            value = cfg_for_paths.get(key)
            if isinstance(value, str) and value:
                return Path(value)
        return None

    sample_images = Path("tests/shared/sample_production_images")
    sample_overlays = sample_images
    sample_jsonl = Path(
        "tests/shared/sample_production_mono_json_output/production_sample.jsonl"
    )

    if debug and images_dir is None and overlays_dir is None and mono_jsonl is None:
        if sample_images.exists() and sample_jsonl.exists():
            typer.echo("🐞 Debug mode: using tests/shared assets")
            images_dir = sample_images
            overlays_dir = sample_overlays
            mono_jsonl = sample_jsonl

    images_path = resolve_path(images_dir, ["default_images_dir", "default_folder"])
    overlays_path = resolve_path(overlays_dir, ["default_overlays_dir"])
    mono_jsonl_path = resolve_path(mono_jsonl, ["default_mono_jsonl", "default_jsonl"])

    if images_path is None or not images_path.exists():
        if sample_images.exists():
            typer.echo("🐞 Falling back to sample images for comparison")
            images_path = sample_images
            overlays_path = sample_overlays if overlays_dir is None else overlays_path
        else:
            typer.echo(
                "❌ Images directory not found. Use --images or set default_images_dir/default_folder in pyproject.toml"
            )
            raise typer.Exit(1)

    if overlays_path is None or not overlays_path.exists():
        candidates = [
            images_path / "overlays",
            images_path.parent / "overlays",
            images_path,
        ]
        overlays_path = next(
            (candidate for candidate in candidates if candidate.exists()), overlays_path
        )

    if overlays_path is None or not overlays_path.exists():
        if sample_overlays.exists():
            overlays_path = sample_overlays
        else:
            typer.echo(
                "❌ Overlays directory not found. Use --overlays or set default_overlays_dir in pyproject.toml"
            )
            raise typer.Exit(1)

    if mono_jsonl_path is None or not mono_jsonl_path.exists():
        if sample_jsonl.exists():
            mono_jsonl_path = sample_jsonl
        else:
            typer.echo(
                "❌ Mono-checker JSONL not found. Use --mono-jsonl or set default_mono_jsonl/default_jsonl in pyproject.toml"
            )
            raise typer.Exit(1)

    batch = (
        batch_size if batch_size is not None else int(cfg.get("default_batch_size", 4))
    )
    min_contamination = float(cfg.get("min_contamination_level", 0.1))
    require_overlays = to_bool(cfg.get("require_overlays", True), True)
    vlm_base_url = cfg.get("vlm_base_url", "http://localhost:8000/v1")
    vlm_model = cfg.get("vlm_model", "Qwen2-VL-2B-Instruct")
    vlm_timeout = int(cfg.get("vlm_timeout", 120))

    typer.echo("🎯 Comparing narrator prompts")
    typer.echo(f"📁 Images: {images_path}")
    typer.echo(f"📁 Overlays: {overlays_path}")
    typer.echo(f"📄 Mono data: {mono_jsonl_path}")
    typer.echo(f"🤖 Model: {vlm_model} at {vlm_base_url}")

    comparison_results: Dict[int, Dict[str, Any]] = {}

    for prompt_definition in selected_prompts:
        use_regions_flag = regions and prompt_definition.supports_regions
        if regions and not prompt_definition.supports_regions:
            typer.echo(
                f"⚠️ Prompt {prompt_definition.id} does not support region guidance; ignoring regions"
            )

        typer.echo(
            f"🚀 Evaluating prompt {prompt_definition.id}: {prompt_definition.name}"
        )

        narration_config = NarrationConfig(
            images_dir=images_path,
            overlays_dir=overlays_path,
            mono_jsonl=mono_jsonl_path,
            vlm_base_url=vlm_base_url,
            vlm_model=vlm_model,
            vlm_timeout=vlm_timeout,
            batch_size=batch,
            min_contamination_level=min_contamination,
            require_overlays=require_overlays,
            dry_run=True,
            debug=debug,
            overwrite_existing=False,
            prompt_id=prompt_definition.id,
            use_regions=use_regions_flag,
            allowed_verdicts={"fail", "pass_with_query"},
        )

        narrator = ColorNarrator(narration_config)
        narrator.metadata_writer.backup_original = False

        try:
            results = narrator.process_all()
            comparison_results[prompt_definition.id] = {
                "definition": prompt_definition,
                "results": results,
                "error": None,
                "use_regions": use_regions_flag,
            }
        except Exception as exc:
            typer.echo(f"❌ Prompt {prompt_definition.id} failed: {exc}")
            comparison_results[prompt_definition.id] = {
                "definition": prompt_definition,
                "results": [],
                "error": str(exc),
                "use_regions": use_regions_flag,
            }

    if summary is not None:
        comparison_path = summary
    else:
        default_summary = cfg.get(
            "prompt_comparison_summary_path",
            "outputs/summaries/prompt_comparison.md",
        )
        base_path = Path(default_summary)
        if debug:
            base_path = Path("tests/test_output/prompt_comparison.md")
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        comparison_path = base_path.with_name(
            f"{base_path.stem}_{timestamp}{base_path.suffix}"
        )

    lines: List[str] = [
        "# Prompt Comparison",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"Model: {vlm_model}",
        "",
        "Prompts compared:",
    ]

    ordered_prompts = selected_prompts
    for definition in ordered_prompts:
        region_note = (
            " (regions)" if comparison_results[definition.id]["use_regions"] else ""
        )
        lines.append(f"- {definition.id}: {definition.name}{region_note}")

    lines.append("")

    image_map: Dict[str, Dict[int, ProcessingResult]] = {}
    for definition in ordered_prompts:
        results = comparison_results[definition.id]["results"]
        for result in results:
            image_name = result.item.image_path.name
            image_map.setdefault(image_name, {})[definition.id] = result

    if not image_map:
        lines.append("_No narration results available._")
    else:
        for image_name in sorted(image_map.keys()):
            any_result = next(iter(image_map[image_name].values()))
            mono = any_result.item.mono_data if any_result else {}
            title = mono.get("title") or Path(image_name).stem
            lines.append(f"## {title} ({image_name})")
            lines.append("")
            for definition in ordered_prompts:
                record = comparison_results[definition.id]
                result = image_map[image_name].get(definition.id)
                header = f"### Prompt {definition.id} — {definition.name}"
                if record["error"]:
                    lines.append(header)
                    lines.append(f"_Failed_: {record['error']}")
                    lines.append("")
                    continue
                if not result:
                    lines.append(header)
                    lines.append("_No narration produced_")
                    lines.append("")
                    continue
                description = (result.vlm_response.description or "").strip()
                if not description and result.error:
                    description = f"_Error_: {result.error}"
                elif not description:
                    description = "_Empty narration_"
                lines.append(header)
                lines.append(description)
                lines.append("")

    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_path.write_text("\n".join(lines), encoding="utf-8")

    typer.echo(f"📝 Prompt comparison written to {comparison_path}")


@app.command()
def validate(
    images_dir: Optional[Path] = typer.Option(
        None, "--images", "-i", help="Directory containing JPEG originals"
    ),
    mono_jsonl: Optional[Path] = typer.Option(
        None, "--mono-jsonl", "-j", help="Mono-checker results JSONL file"
    ),
) -> None:
    """Validate existing color narrations against mono-checker data.

    Reads existing XMP metadata from JPEG files and validates the color descriptions
    against mono-checker analysis data to ensure accuracy and consistency.
    """
    typer.echo("🔍 Color-Narrator - Validate command")

    # Load configuration and set defaults
    config = load_config()
    images_path = images_dir or Path(config.get("default_images_dir", ""))
    mono_jsonl_path = mono_jsonl or Path(config.get("default_mono_jsonl", ""))

    # Basic validation with graceful fallback when defaults are absent
    if not images_path or not str(images_path):
        typer.echo("⚠️ Images directory not configured; skipping validation.")
        typer.echo("✅ Validation complete")
        return

    if not images_path.exists():
        if images_dir is not None:
            typer.echo(f"❌ Images directory not found: {images_path}")
            raise typer.Exit(1)
        typer.echo(f"⚠️ Images directory not found: {images_path}")
        typer.echo("✅ Validation complete")
        return

    if not mono_jsonl_path or not str(mono_jsonl_path):
        typer.echo("⚠️ Mono-checker JSONL path not configured; skipping validation.")
        typer.echo("✅ Validation complete")
        return

    if not mono_jsonl_path.exists():
        if mono_jsonl is not None:
            typer.echo(f"❌ Mono-checker JSONL file not found: {mono_jsonl_path}")
            raise typer.Exit(1)
        typer.echo(f"⚠️ Mono-checker JSONL file not found: {mono_jsonl_path}")
        typer.echo("✅ Validation complete")
        return

    typer.echo(f"📁 Images: {images_path}")
    typer.echo(f"📄 Mono data: {mono_jsonl_path}")

    # Create metadata reader
    metadata_writer = XMPMetadataWriter()

    # Count validation results
    validated_count = 0
    with_metadata = 0
    without_metadata = 0
    validation_errors = []

    try:
        with open(mono_jsonl_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    image_path = Path(data["path"])
                    image_name = image_path.name

                    # Check if we have local copies in test directory
                    local_image = images_path / image_name

                    if local_image.exists():
                        # Check for existing metadata
                        metadata = metadata_writer.read_metadata(local_image)

                        if metadata:
                            with_metadata += 1
                            typer.echo(f"✅ {image_name}: Has color narration metadata")
                            typer.echo(
                                f"   📝 Description: {metadata.description[:80]}..."
                            )
                            typer.echo(f"   🎯 Confidence: {metadata.confidence_score}")
                            typer.echo(
                                f"   🎨 Colors: {', '.join(metadata.color_regions[:3])}"
                            )

                            # Basic validation checks
                            mono_colors = data.get("top_colors", [])
                            metadata_colors = metadata.color_regions

                            if mono_colors and metadata_colors:
                                # Check if main colors match
                                main_mono_color = mono_colors[0] if mono_colors else ""
                                if main_mono_color in metadata_colors:
                                    typer.echo(
                                        f"   ✅ Color consistency: {main_mono_color} found in both"
                                    )
                                else:
                                    typer.echo(
                                        f"   ⚠️  Color mismatch: mono='{main_mono_color}', meta='{metadata_colors}'"
                                    )
                                    validation_errors.append(
                                        f"{image_name}: Color mismatch"
                                    )
                        else:
                            without_metadata += 1
                            typer.echo(
                                f"⚪ {image_name}: No color narration metadata found"
                            )

                        validated_count += 1

                except json.JSONDecodeError as e:
                    typer.echo(f"⚠️  Skipping invalid JSON on line {line_num}: {e}")
                    continue
                except KeyError as e:
                    typer.echo(f"⚠️  Missing required field on line {line_num}: {e}")
                    continue

    except Exception as e:
        typer.echo(f"❌ Error processing mono data: {e}")
        raise typer.Exit(1)

    typer.echo("\n📊 Validation Summary:")
    typer.echo(f"   Total images: {validated_count}")
    typer.echo(f"   With metadata: {with_metadata}")
    typer.echo(f"   Without metadata: {without_metadata}")

    if validation_errors:
        typer.echo(f"   ⚠️  Validation errors: {len(validation_errors)}")
        for error in validation_errors[:3]:  # Show first 3
            typer.echo(f"      • {error}")
        if len(validation_errors) > 3:
            typer.echo(f"      ... and {len(validation_errors) - 3} more")
    else:
        typer.echo("   ✅ No validation errors detected")

    typer.echo("\n✅ Validation complete")


@app.command()
def interpret_mono(
    mono_jsonl: Path = typer.Option(
        ..., "-j", "--mono-jsonl", help="Mono-checker results JSONL file"
    ),
    images_dir: Path = typer.Option(
        ..., "-i", "--images", help="Directory containing JPEG originals"
    ),
    output_jsonl: Optional[Path] = typer.Option(
        None, "-o", "--output", help="Output JSONL for VLM interpretations"
    ),
    limit: int = typer.Option(
        5, "--limit", help="Limit number of images to process (for testing)"
    ),
):
    """
    Interpret mono-checker data using VLM analysis.

    Takes technical mono-checker data and images, uses VLM to generate
    independent verdicts and descriptions for comparison with original analysis.
    """
    from ..core.vlm_mono_interpreter import VLMMonoInterpreter

    typer.echo("🔬 VLM Mono-Interpreter")

    if not mono_jsonl.exists():
        typer.echo(f"❌ Mono JSONL file not found: {mono_jsonl}")
        raise typer.Exit(1)

    if not images_dir.exists():
        typer.echo(f"❌ Images directory not found: {images_dir}")
        raise typer.Exit(1)

    # Initialize VLM interpreter
    try:
        interpreter = VLMMonoInterpreter()
        typer.echo(f"🤖 VLM: {interpreter.model} at {interpreter.base_url}")
    except Exception as e:
        typer.echo(f"❌ Failed to initialize VLM interpreter: {e}")
        raise typer.Exit(1)

    # Load mono data
    mono_results = []
    try:
        with open(mono_jsonl, "r") as f:
            for line in f:
                if line.strip():
                    mono_results.append(json.loads(line))
    except Exception as e:
        typer.echo(f"❌ Failed to load mono JSONL: {e}")
        raise typer.Exit(1)

    typer.echo(f"📄 Loaded {len(mono_results)} mono results")

    # Process limited set for testing
    process_count = min(limit, len(mono_results))
    typer.echo(f"🔄 Processing {process_count} images...")

    vlm_interpretations = []
    comparisons = []

    for i, mono_data in enumerate(mono_results[:process_count]):
        try:
            # Find corresponding image
            original_path = Path(mono_data["path"])
            image_name = original_path.name
            image_path = images_dir / image_name

            if not image_path.exists():
                typer.echo(f"⚠️  Image not found: {image_name}")
                continue

            typer.echo(f"📷 Processing {i+1}/{process_count}: {image_name}")

            # Get VLM interpretation
            vlm_result = interpreter.interpret_mono_data(mono_data, image_path)

            # Store result
            vlm_interpretation = {
                "image_path": str(image_path),
                "image_name": image_name,
                "title": mono_data.get("title", "Unknown"),
                "author": mono_data.get("author", "Unknown"),
                "vlm_verdict": vlm_result.verdict,
                "vlm_technical": vlm_result.technical_reasoning,
                "vlm_visual": vlm_result.visual_description,
                "vlm_summary": vlm_result.professional_summary,
                "vlm_processing_time": vlm_result.processing_time,
                "vlm_model": vlm_result.vlm_model,
                "timestamp": datetime.now().isoformat(),
            }
            vlm_interpretations.append(vlm_interpretation)

            # Compare with original mono verdict
            mono_verdict = mono_data.get("verdict", "unknown")
            comparison = {
                "image": image_name,
                "mono_verdict": mono_verdict,
                "vlm_verdict": vlm_result.verdict,
                "verdict_match": mono_verdict == vlm_result.verdict,
                "mono_reason": mono_data.get("reason_summary", ""),
                "vlm_summary": vlm_result.professional_summary,
            }
            comparisons.append(comparison)

            # Show comparison
            match_icon = "✅" if comparison["verdict_match"] else "❌"
            typer.echo(
                f"   {match_icon} Mono: {mono_verdict} | VLM: {vlm_result.verdict}"
            )
            typer.echo(f"   📝 VLM: {vlm_result.professional_summary[:100]}...")

        except Exception as e:
            typer.echo(f"❌ Error processing {image_name}: {e}")
            continue

    # Save results
    if output_jsonl:
        try:
            with open(output_jsonl, "w") as f:
                for result in vlm_interpretations:
                    f.write(json.dumps(result) + "\n")
            typer.echo(f"💾 VLM interpretations saved to: {output_jsonl}")
        except Exception as e:
            typer.echo(f"❌ Failed to save results: {e}")

    # Summary statistics
    total_processed = len(comparisons)
    matches = sum(1 for c in comparisons if c["verdict_match"])
    match_rate = (matches / total_processed * 100) if total_processed > 0 else 0

    typer.echo("\n📊 Comparison Summary:")
    typer.echo(f"   Total processed: {total_processed}")
    typer.echo(f"   Verdict matches: {matches} ({match_rate:.1f}%)")
    typer.echo(f"   Verdict mismatches: {total_processed - matches}")

    # Show mismatches for analysis
    mismatches = [c for c in comparisons if not c["verdict_match"]]
    if mismatches:
        typer.echo("\n🔍 Verdict Mismatches:")
        for mm in mismatches[:3]:  # Show first 3
            typer.echo(f"   📷 {mm['image']}")
            typer.echo(f"      Mono: {mm['mono_verdict']} | VLM: {mm['vlm_verdict']}")
            typer.echo(f"      VLM reasoning: {mm['vlm_summary'][:120]}...")

    typer.echo("\n✅ VLM interpretation complete")


@app.command()
def enhance_mono(*args, **kwargs):
    """Deprecated entrypoint preserved for backward compatibility."""
    typer.echo(
        "⚠️  'enhance-mono' has been consolidated into 'narrate'. "
        "Use 'imageworks-color-narrator narrate' with the appropriate options."
    )
    raise typer.Exit(1)


@app.command()
def analyze_regions(
    image_path: Path = typer.Option(
        ..., "--image", "-i", help="Path to JPEG image to analyze"
    ),
    output_json: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output JSON file for structured results"
    ),
    use_grid_regions: bool = typer.Option(
        True,
        "--grid-regions/--no-grid-regions",
        help="Use simple 3x3 grid regions (default) or analyze whole image",
    ),
    auto_start_server: bool = typer.Option(
        True,
        "--auto-start/--no-auto-start",
        help="Automatically start VLM server if not running (default: true)",
    ),
    demo_mode: bool = typer.Option(
        False,
        "--demo",
        help="Use demo mono-checker data (for testing without real mono analysis)",
    ),
    debug: bool = typer.Option(False, "--debug", help="Enable debug output"),
    vlm_backend_opt: Optional[str] = typer.Option(
        None,
        "--vlm-backend",
        help="Override VLM backend (vllm, lmdeploy, triton)",
    ),
    vlm_base_url_opt: Optional[str] = typer.Option(
        None,
        "--vlm-base-url",
        help="Override VLM base URL",
    ),
    vlm_model_opt: Optional[str] = typer.Option(
        None,
        "--vlm-model",
        help="Override VLM model identifier",
    ),
    vlm_timeout_opt: Optional[int] = typer.Option(
        None,
        "--vlm-timeout",
        help="Override VLM request timeout (seconds)",
    ),
    vlm_api_key_opt: Optional[str] = typer.Option(
        None,
        "--vlm-api-key",
        help="Override VLM API key",
    ),
    vlm_eager_opt: Optional[bool] = typer.Option(
        None,
        "--vlm-eager/--vlm-no-eager",
        help="Force eager mode when auto-starting LMDeploy (defaults to config)",
    ),
) -> None:
    """[DEPRECATED] Analyze color regions using hallucination-resistant VLM prompts.

    ⚠️  DEPRECATION NOTICE: This command is deprecated in favor of
    `narrate --prompt <id> --regions`, which now offers the richer prompting
    and optional grid guidance. Use the main narrate workflow instead.

    This command uses the new simplified approach with optional 3x3 grid regions:
    - Human-readable spatial regions (top-left, center, etc.)
    - Optional region data (can analyze whole image if no regions)
    - No priming examples to avoid bias
    - Structured JSON output for validation
    - Uncertainty handling with confidence scores

    Example:
        imageworks-color-narrator analyze-regions --image photo.jpg --demo --debug
        imageworks-color-narrator analyze-regions --image photo.jpg --no-grid-regions
    """
    typer.echo("🔬 Region-Based VLM Analysis")
    typer.echo(
        "⚠️  WARNING: This command is deprecated. Use 'imageworks-color-narrator narrate --regions' instead."
    )
    typer.echo()

    cfg = load_config()

    try:
        (
            backend_enum,
            vlm_base_url,
            vlm_model,
            vlm_timeout,
            vlm_api_key,
            backend_options,
        ) = _resolve_vlm_runtime_settings(
            cfg,
            backend=vlm_backend_opt,
            base_url=vlm_base_url_opt,
            model=vlm_model_opt,
            timeout=vlm_timeout_opt,
            api_key=vlm_api_key_opt,
        )
    except ValueError as exc:
        typer.echo(f"❌ {exc}")
        valid_backends = ", ".join(b.value for b in VLMBackend)
        typer.echo(f"   Available options: {valid_backends}")
        raise typer.Exit(1)

    lmdeploy_eager_cfg = _to_bool(cfg.get("vlm_lmdeploy_eager", True), True)
    eager_mode = vlm_eager_opt if vlm_eager_opt is not None else lmdeploy_eager_cfg

    typer.echo(f"🤖 VLM backend: {backend_enum.value} / {vlm_model} at {vlm_base_url}")

    if not image_path.exists():
        typer.echo(f"❌ Image not found: {image_path}")
        raise typer.Exit(1)

    if debug:
        typer.echo(f"📁 Image: {image_path}")
        typer.echo(f"🗂️ Grid regions: {use_grid_regions}")
        typer.echo(f"🧪 Demo mode: {demo_mode}")
        typer.echo(
            f"🤖 VLM backend: {backend_enum.value} / {vlm_model} at {vlm_base_url}"
        )

    # Initialize VLM client with auto-start capability
    vlm_client = _get_vlm_client_with_autostart(
        backend_enum,
        base_url=vlm_base_url,
        model_name=vlm_model,
        api_key=vlm_api_key,
        timeout=vlm_timeout,
        backend_options=backend_options,
        auto_start=auto_start_server,
        eager=eager_mode,
    )

    # Load image to get dimensions
    try:
        with Image.open(image_path) as img:
            image_width, image_height = img.size
        typer.echo(f"� Image dimensions: {image_width}×{image_height}")
    except Exception as e:
        typer.echo(f"❌ Failed to load image: {e}")
        raise typer.Exit(1)

    # Get mono-checker data (real or demo)
    if demo_mode:
        # Demo mono-checker data
        mono_data = {
            "verdict": "pass_with_query",
            "dominant_color": "green",
            "dominant_hue_deg": 88.0,
            "chroma_score": 5.2,
        }
        typer.echo("🎭 Using demo mono-checker data")
    else:
        try:
            # Run mono-checker with grid regions
            from imageworks.libs.vision.mono import check_monochrome

            result = check_monochrome(
                str(image_path), include_grid_regions=use_grid_regions
            )
            mono_data = {
                "verdict": result.verdict,
                "dominant_color": result.dominant_color or "unknown",
                "dominant_hue_deg": result.dominant_hue_deg or 0.0,
                "chroma_score": result.chroma_max or 0.0,
            }
            typer.echo(f"� Mono-checker result: {result.verdict}")
        except Exception as e:
            typer.echo(f"❌ Failed to run mono-checker: {e}")
            raise typer.Exit(1)

    # Show grid region info if enabled
    if use_grid_regions and debug:
        from imageworks.apps.color_narrator.core.grid_regions import ImageGridAnalyzer

        grid_regions = ImageGridAnalyzer.analyze_color_in_regions(
            mono_data, image_width, image_height
        )
        if grid_regions:
            typer.echo(f"🗂️ Found color in {len(grid_regions)} grid regions:")
            for region in grid_regions:
                typer.echo(
                    f"   {region.grid_region.value}: {region.dominant_color} "
                    f"({region.area_pct:.1f}% affected)"
                )
        else:
            typer.echo("🗂️ No significant color detected in grid regions")

    # Initialize region-based analyzer
    analyzer = RegionBasedVLMAnalyzer(vlm_client)

    # Perform analysis
    region_type = "grid" if use_grid_regions else "none"
    typer.echo(f"⚡ Running VLM analysis (region type: {region_type})...")
    try:
        analysis = analyzer.analyze_with_regions(
            image_path=image_path,
            mono_data=mono_data,
            region_data=None,  # We're not using technical region data anymore
            use_grid_regions=use_grid_regions,
            image_dimensions=(image_width, image_height) if use_grid_regions else None,
        )

        typer.echo(f"✅ Analysis complete ({len(analysis.findings)} findings)")

        # Show validation results
        if analysis.validation_errors:
            typer.echo(f"⚠️  {len(analysis.validation_errors)} validation issues:")
            for error in analysis.validation_errors:
                typer.echo(f"   • {error}")
        else:
            typer.echo("✅ No validation errors")

        # Generate human-readable summary
        summary = analyzer.generate_human_readable_summary(analysis)
        typer.echo("\n" + "=" * 60)
        typer.echo(summary)
        typer.echo("=" * 60)

        # Show findings with confidence scores
        if analysis.findings:
            typer.echo("\n📊 Detailed Findings:")
            for finding in analysis.findings:
                confidence = finding.get("confidence", 0.0)
                confidence_icon = (
                    "🟢" if confidence >= 0.8 else "🟡" if confidence >= 0.5 else "🔴"
                )
                location = (
                    f" - {finding.get('location', '')}"
                    if finding.get("location")
                    else ""
                )
                typer.echo(
                    f"   {confidence_icon} {finding.get('color_family', 'unknown')} "
                    f"on {finding.get('object_part', 'unknown')}{location} "
                    f"({finding.get('tonal_zone', 'unknown')}, confidence: {confidence:.2f})"
                )

        # Save structured results if requested
        if output_json:
            try:
                results = {
                    "file_name": analysis.file_name,
                    "dominant_color": analysis.dominant_color,
                    "dominant_hue_deg": analysis.dominant_hue_deg,
                    "region_type": analysis.region_type,
                    "findings": analysis.findings,
                    "validation_errors": analysis.validation_errors,
                    "raw_response": analysis.raw_response if debug else None,
                    "timestamp": datetime.now().isoformat(),
                }

                output_json.parent.mkdir(parents=True, exist_ok=True)
                with open(output_json, "w") as f:
                    json.dump(results, f, indent=2)

                typer.echo(f"💾 Structured results saved to: {output_json}")

            except Exception as e:
                typer.echo(f"⚠️  Failed to save results: {e}")

        # Show raw VLM response if debug
        if debug:
            typer.echo("\n🤖 Raw VLM Response:")
            typer.echo("-" * 40)
            typer.echo(analysis.raw_response)
            typer.echo("-" * 40)

    except Exception as e:
        typer.echo(f"❌ Analysis failed: {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
