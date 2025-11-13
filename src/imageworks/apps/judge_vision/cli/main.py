"""Command line interface for Judge Vision."""

from __future__ import annotations

import logging
import signal
import sys
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Optional

import httpx
import typer

from imageworks.logging_utils import configure_logging

from imageworks.apps.judge_vision.config import JudgeVisionConfig, build_runtime_config
from imageworks.apps.judge_vision.runner import (
    JudgeVisionRunner,
    load_records_from_jsonl,
)
from imageworks.apps.judge_vision.tf_container_wrapper import (
    ensure_tf_service_running,
    shutdown_tf_service,
)

GPU_LEASE_TIMEOUT_S = float(os.getenv("JUDGE_VISION_GPU_LEASE_TIMEOUT", "90"))
GPU_LEASE_RETRIES = int(os.getenv("JUDGE_VISION_GPU_LEASE_RETRIES", "3"))
GPU_LEASE_RETRY_DELAY = float(os.getenv("JUDGE_VISION_GPU_LEASE_RETRY_DELAY", "5"))
SELF_LEASE_MAX_AGE = float(os.getenv("JUDGE_VISION_GPU_SELF_LEASE_MAX_AGE", "240"))
AUTO_SHUTDOWN_TF = os.getenv("JUDGE_VISION_TF_AUTO_SHUTDOWN", "1") not in {
    "0",
    "false",
    "False",
}

LOG_PATH = configure_logging("judge_vision")
logger = logging.getLogger(__name__)

app = typer.Typer(help="Run Judge Vision critiques and tournaments.")


_ACTIVE_LEASE: Optional["GpuLeaseHandle"] = None


def _set_active_lease(handle: Optional["GpuLeaseHandle"]) -> None:
    global _ACTIVE_LEASE
    _ACTIVE_LEASE = handle


def _handle_termination(signum, frame):  # noqa: ARG001
    """Ensure GPU leases are released if the process is interrupted."""
    try:
        sig_name = signal.Signals(signum).name  # type: ignore[arg-type]
    except Exception:  # noqa: BLE001
        sig_name = str(signum)
    logger.warning(
        "Termination signal (%s) received; stopping Judge Vision run", sig_name
    )
    handle = _ACTIVE_LEASE
    if handle:
        try:
            logger.info("Attempting to release GPU lease before exit")
            handle.client.release(handle.token, restart_model=False)
        except Exception as exc:  # noqa: BLE001
            logger.warning("GPU lease release during termination failed: %s", exc)
        finally:
            handle.client.close()
            _set_active_lease(None)
    # No need to terminate container - new architecture uses short-lived inference calls
    raise SystemExit(128 + signum)


def _configure_signal_handlers() -> None:
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handle_termination)
        except Exception:  # noqa: BLE001
            pass


_configure_signal_handlers()


class GpuLeaseUnavailable(RuntimeError):
    """Raised when the GPU lease endpoint is not reachable."""


@dataclass
class GpuLeaseHandle:
    token: str
    client: "GpuLeaseClient"


class GpuLeaseClient:
    """Minimal HTTP client for the chat proxy GPU lease endpoints."""

    def __init__(self, base_url: str):
        if not base_url:
            raise GpuLeaseUnavailable("Base URL missing")
        root = base_url.rstrip("/")
        if root.endswith("/v1"):
            root = root[: -len("/v1")]
        self._root = root or base_url
        self._session = httpx.Client(timeout=GPU_LEASE_TIMEOUT_S)

    def acquire(self, *, owner: str, reason: str | None = None) -> str:
        url = f"{self._root}/v1/gpu/lease"
        payload = {"owner": owner, "reason": reason, "restart_model": True}
        try:
            resp = self._session.post(url, json=payload)
        except httpx.HTTPError as exc:
            raise GpuLeaseUnavailable(str(exc)) from exc
        if resp.status_code == 404:
            raise GpuLeaseUnavailable("GPU lease endpoint not available")
        if resp.status_code == 409:
            raise GpuLeaseUnavailable(resp.json().get("detail", "GPU busy"))
        if resp.status_code >= 400:
            raise GpuLeaseUnavailable(resp.text)
        data = resp.json()
        token = data.get("token")
        if not token:
            raise GpuLeaseUnavailable("Lease response missing token")
        return token

    def release(self, token: str, *, restart_model: bool = True) -> None:
        url = f"{self._root}/v1/gpu/release"
        payload = {"token": token, "restart_model": restart_model}
        try:
            resp = self._session.post(url, json=payload)
        except httpx.HTTPError as exc:
            logger.warning("GPU lease release failed: %s", exc)
            return
        if resp.status_code >= 400:
            logger.warning(
                "GPU lease release returned %s: %s", resp.status_code, resp.text
            )

    def close(self) -> None:
        self._session.close()


def _proxy_root_from_base(base_url: str | None) -> str | None:
    if not base_url:
        return None
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    return root or base_url


def _log_gpu_status(proxy_root: str) -> None:
    try:
        resp = httpx.get(f"{proxy_root}/v1/gpu/status", timeout=5)
        if resp.status_code == 200:
            logger.info("GPU lease status: %s", resp.json())
        else:
            logger.warning(
                "GPU status check returned %s: %s", resp.status_code, resp.text
            )
    except httpx.HTTPError as exc:
        logger.debug("Unable to query GPU status: %s", exc)


def _maybe_force_release_stale_self_lease(proxy_root: str) -> bool:
    if not proxy_root or SELF_LEASE_MAX_AGE <= 0:
        return False
    try:
        status_resp = httpx.get(f"{proxy_root}/v1/gpu/status", timeout=5)
    except httpx.HTTPError as exc:
        logger.debug("Unable to query GPU status for stale release: %s", exc)
        return False
    if status_resp.status_code != 200:
        return False
    data = status_resp.json()
    lease = data.get("lease")
    if not data.get("leased") or not lease:
        return False
    if lease.get("owner") != "judge-vision":
        return False
    granted_at = lease.get("granted_at")
    if granted_at is None:
        return False
    age = time.time() - float(granted_at)
    if age < SELF_LEASE_MAX_AGE:
        return False
    logger.warning(
        "Force-releasing stale Judge Vision GPU lease (age %.1fs, token=%s…)",
        age,
        str(lease.get("token", ""))[:8],
    )
    payload = {
        "owner": "judge-vision",
        "max_age": SELF_LEASE_MAX_AGE,
        "restart_model": False,
    }
    try:
        resp = httpx.post(
            f"{proxy_root}/v1/gpu/force_release", json=payload, timeout=10
        )
    except httpx.HTTPError as exc:
        logger.warning("Force-release request failed: %s", exc)
        return False
    if resp.status_code >= 400:
        logger.warning(
            "Force-release request returned %s: %s", resp.status_code, resp.text
        )
        return False
    logger.info("Stale Judge Vision GPU lease cleared.")
    return True


def _maybe_acquire_gpu_lease(
    config: JudgeVisionConfig,
    *,
    reason: str,
) -> tuple[JudgeVisionConfig, Optional[GpuLeaseHandle]]:
    if config.iqa_device.lower() != "gpu":
        return config, None
    if (config.backend or "").lower() != "vllm":
        logger.info("GPU lease skipped (backend %s)", config.backend)
        return config, None
    proxy_root = _proxy_root_from_base(config.base_url)
    if not proxy_root:
        logger.warning("GPU lease skipped: base URL missing")
        return config, None

    last_error: Optional[GpuLeaseUnavailable] = None
    for attempt in range(1, max(1, GPU_LEASE_RETRIES) + 1):
        client = GpuLeaseClient(config.base_url or proxy_root)
        try:
            token = client.acquire(owner="judge-vision", reason=reason)
            handle = GpuLeaseHandle(token=token, client=client)
            _set_active_lease(handle)
            logger.info("GPU lease granted (token=%s…)", token[:8])
            config.gpu_lease_token = token
            return config, handle
        except GpuLeaseUnavailable as exc:
            last_error = exc
            client.close()
            busy = "busy" in str(exc).lower() or "leased" in str(exc).lower()
            if busy and attempt < GPU_LEASE_RETRIES:
                logger.warning(
                    "GPU lease busy (attempt %d/%d); retrying in %.1fs",
                    attempt,
                    GPU_LEASE_RETRIES,
                    GPU_LEASE_RETRY_DELAY,
                )
                _log_gpu_status(proxy_root)
                if _maybe_force_release_stale_self_lease(proxy_root):
                    continue
                time.sleep(max(0.1, GPU_LEASE_RETRY_DELAY))
                continue
            break
    msg = str(last_error) if last_error else "unknown error"
    logger.warning("GPU lease unavailable (%s); falling back to CPU", msg)
    _set_active_lease(None)
    return replace(config, iqa_device="cpu"), None


def _release_gpu_lease(
    handle: GpuLeaseHandle | None, *, restart_model: bool = True
) -> None:
    if not handle:
        return
    try:
        try:
            handle.client.release(handle.token, restart_model=restart_model)
        except TypeError:
            handle.client.release(handle.token)
        logger.info("GPU lease released (no automatic model restart)")
    finally:
        handle.client.close()
        if _ACTIVE_LEASE is handle:
            _set_active_lease(None)


def _acquire_gpu_lease_strict(
    config: JudgeVisionConfig,
    *,
    reason: str,
) -> Optional[GpuLeaseHandle]:
    """Acquire a GPU lease for vLLM stages, exiting if unavailable."""

    if (config.backend or "").lower() != "vllm":
        return None
    proxy_root = _proxy_root_from_base(config.base_url)
    if not proxy_root:
        typer.echo("❌ GPU lease requires chat proxy base URL")
        raise typer.Exit(code=1)

    last_error: Optional[GpuLeaseUnavailable] = None
    for attempt in range(1, max(1, GPU_LEASE_RETRIES) + 1):
        client = GpuLeaseClient(config.base_url or proxy_root)
        try:
            token = client.acquire(owner="judge-vision", reason=reason)
            handle = GpuLeaseHandle(token=token, client=client)
            _set_active_lease(handle)
            logger.info("GPU lease granted for %s (token=%s…)", reason, token[:8])
            config.gpu_lease_token = token
            return handle
        except GpuLeaseUnavailable as exc:
            last_error = exc
            client.close()
            busy = "busy" in str(exc).lower() or "leased" in str(exc).lower()
            if busy and attempt < GPU_LEASE_RETRIES:
                logger.warning(
                    "GPU lease busy for %s (attempt %d/%d); retrying in %.1fs",
                    reason,
                    attempt,
                    GPU_LEASE_RETRIES,
                    GPU_LEASE_RETRY_DELAY,
                )
                _log_gpu_status(proxy_root)
                if _maybe_force_release_stale_self_lease(proxy_root):
                    continue
                time.sleep(max(0.1, GPU_LEASE_RETRY_DELAY))
                continue
            break
    msg = str(last_error) if last_error else "unknown error"
    typer.echo(f"❌ GPU lease unavailable for {reason}: {msg}")
    _set_active_lease(None)
    raise typer.Exit(code=1)


def _run_chained_two_pass(config: JudgeVisionConfig) -> None:
    """Execute deterministic IQA followed by critique using shared cache."""

    cache_path = config.iqa_cache_path
    logger.info("Two-pass mode: Stage 1 → IQA (cache: %s)", cache_path)
    stage1_config = replace(config, stage="iqa")
    _execute_stage1(stage1_config, reason="Judge Vision Stage 1 (two-pass)")

    if not cache_path.exists():
        raise FileNotFoundError(
            f"IQA cache not found at {cache_path}. Stage 1 did not produce results."
        )

    logger.info("Two-pass mode: Stage 2 → Critique (reusing cache)")
    stage2_config = replace(config, stage="critique")
    lease_handle = _acquire_gpu_lease_strict(
        stage2_config, reason="Judge Vision Stage 2 (two-pass)"
    )
    try:
        JudgeVisionRunner(stage2_config).run()
    finally:
        _release_gpu_lease(lease_handle)
        stage2_config.gpu_lease_token = None


def _run_pairwise_stage(config: JudgeVisionConfig) -> None:
    if not config.pairwise_enabled:
        raise ValueError("Pairwise stage requested but pairwise is disabled")
    records = load_records_from_jsonl(config.output_jsonl)
    runner = JudgeVisionRunner(config)
    if not records:
        runner.progress.reset(total=1, phase="Stage 3 – Pairwise")
        runner.progress.update(processed=1, current_image="No images available")
        runner.progress.complete()
        return

    plans = runner._build_pairwise_plans(records)
    if not plans:
        runner.progress.reset(total=1, phase="Stage 3 – Pairwise")
        runner.progress.update(
            processed=1,
            current_image=f"No eligible ≥{config.pairwise_threshold} images",
        )
        runner.progress.complete()
    else:
        model_name = config.model or config.critique_role or "judge"
        for plan in plans:
            runner._run_pairwise_plan(plan, model_name=model_name)
    runner._rewrite_jsonl(records)
    runner._write_summary(records)


def _execute_stage1(config: JudgeVisionConfig, *, reason: str) -> None:
    """Acquire GPU lease (if required) and run Stage 1 through the correct backend."""

    adjusted, lease_handle = _maybe_acquire_gpu_lease(config, reason=reason)
    try:
        _maybe_start_tf_service(adjusted)
        _run_stage1_backend(adjusted)
    finally:
        if AUTO_SHUTDOWN_TF:
            _maybe_shutdown_tf_service()
        _release_gpu_lease(lease_handle)
        adjusted.gpu_lease_token = None


def _run_stage1_backend(config: JudgeVisionConfig) -> None:
    device = (config.iqa_device or "cpu").lower()
    if device == "gpu":
        logger.info(
            "Running Stage 1 with GPU-accelerated IQA (containerized TensorFlow)"
        )
        run_stage1_in_container(config)
        return
    logger.info("Running Stage 1 with CPU-based IQA")
    JudgeVisionRunner(config).run()


def run_stage1_in_container(config: JudgeVisionConfig) -> None:
    """Hook for running Stage 1 inside the managed TensorFlow container."""
    JudgeVisionRunner(config).run()


def _maybe_start_tf_service(config: JudgeVisionConfig) -> None:
    if config.iqa_device.lower() != "gpu":
        return
    if ensure_tf_service_running():
        logger.info("TensorFlow IQA service ready for Stage 1")
    else:
        logger.warning(
            "TensorFlow IQA service unavailable; Stage 1 will use per-image containers"
        )


def _maybe_shutdown_tf_service() -> None:
    try:
        if shutdown_tf_service():
            logger.info("TensorFlow IQA service shut down to free GPU resources")
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to shut down TensorFlow IQA service: %s", exc)


@app.command()
def run(  # noqa: PLR0913
    input_dir: List[Path] = typer.Option(
        ..., "--input-dir", "-i", help="Directories or files to judge."
    ),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive"),
    backend: Optional[str] = typer.Option(None, "--backend"),
    base_url: Optional[str] = typer.Option(None, "--base-url"),
    api_key: Optional[str] = typer.Option(None, "--api-key"),
    timeout: Optional[int] = typer.Option(None, "--timeout"),
    max_new_tokens: Optional[int] = typer.Option(None, "--max-new-tokens"),
    temperature: Optional[float] = typer.Option(None, "--temperature"),
    top_p: Optional[float] = typer.Option(None, "--top-p"),
    model: Optional[str] = typer.Option(
        None, "--model", help="Explicit model name for critique stage."
    ),
    use_registry: bool = typer.Option(True, "--use-registry/--no-registry"),
    critique_role: Optional[str] = typer.Option("judge", "--critique-role"),
    skip_preflight: bool = typer.Option(False, "--skip-preflight"),
    competition_config: Optional[Path] = typer.Option(None, "--competition-config"),
    competition: Optional[str] = typer.Option(None, "--competition"),
    pairwise_rounds: Optional[int] = typer.Option(
        None,
        "--pairwise-rounds",
        help="Override comparisons per image for pairwise playoffs.",
    ),
    pairwise: bool = typer.Option(
        False, "--pairwise/--no-pairwise", help="Enable Stage 3 pairwise playoff."
    ),
    pairwise_threshold: int = typer.Option(
        17,
        "--pairwise-threshold",
        help="Minimum Stage 2 score to enter the pairwise playoff.",
    ),
    critique_title_template: str = typer.Option("{stem}", "--critique-title-template"),
    critique_category: Optional[str] = typer.Option(None, "--critique-category"),
    critique_notes: str = typer.Option("", "--critique-notes"),
    output_jsonl: Optional[Path] = typer.Option(None, "--output-jsonl"),
    summary: Optional[Path] = typer.Option(None, "--summary"),
    progress: Optional[Path] = typer.Option(None, "--progress-file"),
    enable_musiq: bool = typer.Option(True, "--enable-musiq/--disable-musiq"),
    enable_nima: bool = typer.Option(True, "--enable-nima/--disable-nima"),
    iqa_cache: Optional[Path] = typer.Option(None, "--iqa-cache"),
    stage: str = typer.Option(
        "two-pass",
        "--stage",
        help="Stages: full|two-pass|iqa|critique|pairwise",
    ),
    iqa_device: str = typer.Option("gpu", "--iqa-device", help="IQA device: cpu|gpu"),
):
    typer.echo(f"Judge Vision log → {LOG_PATH}")
    if sys.argv:
        logger.info("Judge Vision CLI command: %s", " ".join(sys.argv))
    config = build_runtime_config(
        input_dirs=input_dir,
        recursive=recursive,
        backend=backend,
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        model=model,
        use_registry=use_registry,
        critique_role=critique_role,
        skip_preflight=skip_preflight,
        competition_config=competition_config,
        competition_id=competition,
        pairwise_rounds=pairwise_rounds,
        pairwise_enabled=pairwise,
        pairwise_threshold=pairwise_threshold,
        critique_title_template=critique_title_template,
        critique_category=critique_category,
        critique_notes=critique_notes,
        output_jsonl=output_jsonl,
        summary_path=summary,
        progress_path=progress,
        enable_musiq=enable_musiq,
        enable_nima=enable_nima,
        iqa_cache_path=iqa_cache,
        stage=stage,
        iqa_device=iqa_device,
    )

    intended_stage = config.stage.lower()
    if intended_stage == "pairwise":
        config.pairwise_enabled = True
    if intended_stage == "critique":
        cache_path = config.iqa_cache_path
        if not cache_path.exists():
            typer.echo(
                f"❌ Cannot run Stage 2 (critique) without Stage 1 IQA cache at {cache_path}. "
                "Re-run Stage 1 or two-pass first."
            )
            raise typer.Exit(code=1)
    if intended_stage == "two-pass":
        _run_chained_two_pass(config)
    elif intended_stage == "iqa":
        _execute_stage1(config, reason="Judge Vision Stage 1 (IQA)")
    elif intended_stage == "pairwise":
        lease_handle = _acquire_gpu_lease_strict(
            config, reason="Judge Vision Stage 3 (Pairwise)"
        )
        try:
            _run_pairwise_stage(config)
        finally:
            _release_gpu_lease(lease_handle)
            config.gpu_lease_token = None
    elif intended_stage == "critique":
        lease_handle = _acquire_gpu_lease_strict(
            config, reason="Judge Vision Stage 2 (Critique)"
        )
        try:
            runner = JudgeVisionRunner(config)
            runner.run()
        finally:
            _release_gpu_lease(lease_handle)
            config.gpu_lease_token = None
    else:
        lease_handle = None
        if intended_stage == "full":
            lease_handle = _acquire_gpu_lease_strict(
                config, reason="Judge Vision (Full)"
            )
        runner = JudgeVisionRunner(config)
        try:
            runner.run()
        finally:
            _release_gpu_lease(lease_handle)
            config.gpu_lease_token = None


if __name__ == "__main__":
    app()
