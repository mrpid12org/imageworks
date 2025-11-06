"""VRAM estimation utilities and CLI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import typer
from rich.console import Console
from rich.table import Table

from imageworks.libs.hardware.gpu_detector import GPUDetector

BYTES_PER_PARAM: Dict[str, float] = {
    "fp16": 2.0,
    "bf16": 2.0,
    "fp8": 1.0,
    "int8": 1.0,
    "int4": 0.55,
    "fp4": 0.50,
}

# KV cache element size in bytes (per value) for common formats.
KV_BYTES_ELEM: Dict[str, float] = {
    "fp16": 2.0,
    "bf16": 2.0,
    "fp8": 1.0,
}

DEFAULT_OVERHEAD_PROFILES: Dict[str, Dict[str, float]] = {
    "default": {"overhead_gib": 1.5, "vision_gib": 0.0, "frag_factor": 1.10},
    "ada_16gb": {"overhead_gib": 1.6, "vision_gib": 0.5, "frag_factor": 1.12},
    "hopper_80gb": {"overhead_gib": 2.0, "vision_gib": 1.0, "frag_factor": 1.08},
    "studio_dual": {"overhead_gib": 2.5, "vision_gib": 1.0, "frag_factor": 1.15},
}

OVERHEAD_FILE = Path(__file__).with_name("overhead_profiles.json")

console = Console()
app = typer.Typer(help="Estimate VRAM requirements for vLLM models.")


@dataclass
class VRAMEstimate:
    weights_gib: float
    kv_gib: float
    overhead_gib: float
    vision_gib: float
    frag_factor: float

    @property
    def total_gib(self) -> float:
        return (
            self.weights_gib + self.kv_gib + self.overhead_gib + self.vision_gib
        ) * self.frag_factor

    def to_dict(self) -> Dict[str, float]:
        return {
            "weights_gib": self.weights_gib,
            "kv_gib": self.kv_gib,
            "overhead_gib": self.overhead_gib,
            "vision_gib": self.vision_gib,
            "frag_factor": self.frag_factor,
            "total_gib": self.total_gib,
        }


def load_overhead_profiles() -> Dict[str, Dict[str, float]]:
    """Load overhead profiles from JSON if present else defaults."""

    if OVERHEAD_FILE.exists():
        try:
            data = json.loads(OVERHEAD_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {**DEFAULT_OVERHEAD_PROFILES, **data}
        except Exception:  # pragma: no cover - graceful fallback
            console.log(
                "[yellow]Failed to read overhead_profiles.json; using defaults."
            )
    return DEFAULT_OVERHEAD_PROFILES.copy()


def _resolve_profile(profile: Optional[str]) -> Dict[str, float]:
    profiles = load_overhead_profiles()
    if profile:
        if profile not in profiles:
            raise typer.BadParameter(
                f"Unknown profile '{profile}'. Available: {', '.join(sorted(profiles))}"
            )
        return profiles[profile]
    return profiles["default"]


def estimate_vram_gib(
    *,
    params_billion: float,
    quant: str,
    layers: int,
    kv_heads: int,
    head_dim: int,
    kv_dtype: str,
    context_k: float,
    batch: int,
    overhead_gib: float,
    vision_gib: float,
    frag_factor: float,
    bytes_per_param_override: Optional[float] = None,
    kv_bytes_override: Optional[float] = None,
) -> VRAMEstimate:
    """Forward estimator returning a VRAMEstimate."""

    bpp = (
        bytes_per_param_override
        if bytes_per_param_override is not None
        else BYTES_PER_PARAM.get(quant.lower(), 1.0)
    )
    kvb = (
        kv_bytes_override
        if kv_bytes_override is not None
        else KV_BYTES_ELEM.get(kv_dtype.lower(), 2.0)
    )

    weights_gib = (params_billion * 1e9 * bpp) / (1024**3)
    tokens_active = context_k * 1024 * batch
    kv_gib = (2 * layers * kv_heads * head_dim * kvb * tokens_active) / (1024**3)

    return VRAMEstimate(
        weights_gib=weights_gib,
        kv_gib=kv_gib,
        overhead_gib=overhead_gib,
        vision_gib=vision_gib,
        frag_factor=frag_factor,
    )


def estimate_max_context_k(
    *,
    total_vram_gib: float,
    params_billion: float,
    quant: str,
    layers: int,
    kv_heads: int,
    head_dim: int,
    kv_dtype: str,
    batch: int,
    overhead_gib: float,
    vision_gib: float,
    frag_factor: float,
    max_context_k: float = 256.0,
) -> Dict[str, float]:
    """Inverse estimator - find maximum context (in thousands) that fits."""

    low = 0.0
    high = max_context_k
    best = 0.0

    while high - low > 0.01:
        mid = (low + high) / 2.0
        estimate = estimate_vram_gib(
            params_billion=params_billion,
            quant=quant,
            layers=layers,
            kv_heads=kv_heads,
            head_dim=head_dim,
            kv_dtype=kv_dtype,
            context_k=mid,
            batch=batch,
            overhead_gib=overhead_gib,
            vision_gib=vision_gib,
            frag_factor=frag_factor,
        )
        if estimate.total_gib <= total_vram_gib:
            best = mid
            low = mid
        else:
            high = mid

    final_estimate = estimate_vram_gib(
        params_billion=params_billion,
        quant=quant,
        layers=layers,
        kv_heads=kv_heads,
        head_dim=head_dim,
        kv_dtype=kv_dtype,
        context_k=best,
        batch=batch,
        overhead_gib=overhead_gib,
        vision_gib=vision_gib,
        frag_factor=frag_factor,
    )
    return {
        "context_k": best,
        "context_tokens": best * 1024,
        "kv_gib": final_estimate.kv_gib,
        "total_gib": final_estimate.total_gib,
    }


def _print_estimate_table(estimate: VRAMEstimate) -> None:
    table = Table(title="VRAM Estimate", show_header=False)
    table.add_row("Weights (GiB)", f"{estimate.weights_gib:.2f}")
    table.add_row("KV Cache (GiB)", f"{estimate.kv_gib:.2f}")
    table.add_row("Overhead (GiB)", f"{estimate.overhead_gib:.2f}")
    table.add_row("Vision Tower (GiB)", f"{estimate.vision_gib:.2f}")
    table.add_row("Fragmentation Factor", f"{estimate.frag_factor:.2f}")
    table.add_row("Total (GiB)", f"{estimate.total_gib:.2f}")
    console.print(table)


def auto_profile(profile: Optional[str]) -> Dict[str, float]:
    if profile:
        return _resolve_profile(profile)

    detector = GPUDetector()
    gpus = detector.detect_gpus()
    if not gpus:
        return _resolve_profile("default")

    # Simple heuristic: choose profile based on GPU name / VRAM.
    gpu = gpus[0]
    name = gpu.name.lower()
    if "4090" in name or "4080" in name:
        return _resolve_profile("ada_16gb")
    if gpu.vram_total_mb >= 80000:
        return _resolve_profile("hopper_80gb")
    return _resolve_profile("default")


@app.command()
def profiles() -> None:
    """List available overhead profiles."""

    profiles = load_overhead_profiles()
    table = Table(title="Overhead Profiles", show_header=True, header_style="bold")
    table.add_column("Profile")
    table.add_column("Overhead GiB")
    table.add_column("Vision GiB")
    table.add_column("Fragmentation")
    for name, data in sorted(profiles.items()):
        table.add_row(
            name,
            f"{data.get('overhead_gib', 0.0):.2f}",
            f"{data.get('vision_gib', 0.0):.2f}",
            f"{data.get('frag_factor', 1.0):.2f}",
        )
    console.print(table)


@app.command()
def estimate(
    params_billion: float = typer.Option(
        ..., help="Model size in billions of parameters."
    ),
    quant: str = typer.Option(
        "fp8", help="Quantisation scheme (fp16, bf16, fp8, int8, int4, fp4)."
    ),
    layers: int = typer.Option(..., help="Number of decoder layers."),
    kv_heads: int = typer.Option(..., help="KV heads per layer."),
    head_dim: int = typer.Option(..., help="Dimension per attention head."),
    kv_dtype: str = typer.Option("fp8", help="KV cache precision (fp16, bf16, fp8)."),
    context_k: float = typer.Option(
        8.0, help="Context window in thousands (e.g., 8 = 8192 tokens)."
    ),
    batch: int = typer.Option(1, help="Concurrent sequences in batch."),
    profile: Optional[str] = typer.Option(
        None, help="Overhead profile name (use `profiles` to list)."
    ),
    overhead: Optional[float] = typer.Option(
        None, help="Override overhead GiB (weights, buffers, runtime)."
    ),
    vision: Optional[float] = typer.Option(None, help="Override vision tower GiB."),
    frag: Optional[float] = typer.Option(None, help="Override fragmentation factor."),
    json_output: bool = typer.Option(
        False, "--json", help="Emit JSON instead of table output."
    ),
) -> None:
    """Forward VRAM estimation."""

    profile_data = auto_profile(profile)
    overhead_gib = (
        overhead if overhead is not None else profile_data.get("overhead_gib", 1.5)
    )
    vision_gib = vision if vision is not None else profile_data.get("vision_gib", 0.0)
    frag_factor = frag if frag is not None else profile_data.get("frag_factor", 1.10)

    estimate_obj = estimate_vram_gib(
        params_billion=params_billion,
        quant=quant,
        layers=layers,
        kv_heads=kv_heads,
        head_dim=head_dim,
        kv_dtype=kv_dtype,
        context_k=context_k,
        batch=batch,
        overhead_gib=overhead_gib,
        vision_gib=vision_gib,
        frag_factor=frag_factor,
    )

    if json_output:
        console.print_json(data=estimate_obj.to_dict())
    else:
        _print_estimate_table(estimate_obj)


@app.command("max-context")
def max_context(
    total_vram_gib: float = typer.Option(..., help="VRAM budget in GiB."),
    params_billion: float = typer.Option(
        ..., help="Model size in billions of parameters."
    ),
    quant: str = typer.Option("fp8"),
    layers: int = typer.Option(...),
    kv_heads: int = typer.Option(...),
    head_dim: int = typer.Option(...),
    kv_dtype: str = typer.Option("fp8"),
    batch: int = typer.Option(1),
    profile: Optional[str] = typer.Option(None),
    overhead: Optional[float] = typer.Option(None),
    vision: Optional[float] = typer.Option(None),
    frag: Optional[float] = typer.Option(None),
    max_context_k: float = typer.Option(
        256.0, help="Upper bound for search (thousands of tokens)."
    ),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Calculate the maximum context (in thousands of tokens) that fits in a VRAM budget."""

    profile_data = auto_profile(profile)
    overhead_gib = (
        overhead if overhead is not None else profile_data.get("overhead_gib", 1.5)
    )
    vision_gib = vision if vision is not None else profile_data.get("vision_gib", 0.0)
    frag_factor = frag if frag is not None else profile_data.get("frag_factor", 1.10)

    payload = estimate_max_context_k(
        total_vram_gib=total_vram_gib,
        params_billion=params_billion,
        quant=quant,
        layers=layers,
        kv_heads=kv_heads,
        head_dim=head_dim,
        kv_dtype=kv_dtype,
        batch=batch,
        overhead_gib=overhead_gib,
        vision_gib=vision_gib,
        frag_factor=frag_factor,
        max_context_k=max_context_k,
    )

    if json_output:
        console.print_json(data=payload)
    else:
        table = Table(title="Max Context Estimate", show_header=False)
        table.add_row("Context (k tokens)", f"{payload['context_k']:.2f}")
        table.add_row("Context (tokens)", f"{payload['context_tokens']:.0f}")
        table.add_row("KV Cache (GiB)", f"{payload['kv_gib']:.2f}")
        table.add_row("Estimated Total (GiB)", f"{payload['total_gib']:.2f}")
        console.print(table)


if __name__ == "__main__":  # pragma: no cover
    app()
