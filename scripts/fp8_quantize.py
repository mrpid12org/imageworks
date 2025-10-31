#!/usr/bin/env python
"""Reusable FP8 quantization helper for Qwen (and compatible) models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from llmcompressor import oneshot
from llmcompressor.modeling import replace_modules_for_calibration
from llmcompressor.modifiers.quantization import QuantizationModifier
from transformers import AutoModelForVision2Seq, AutoProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a BF16 HuggingFace-style model directory to FP8 using "
            "LLM Compressor's FP8 Dynamic (W8A8) quantization."
        )
    )
    parser.add_argument(
        "source_model",
        type=Path,
        help="Path to the source model directory (e.g. prithivMLmods/Qwen3-VL-8B-Instruct-abliterated)",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Destination directory for the FP8 export (will be created if missing)",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length for calibration metadata (default: 2048)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty output directory",
    )
    return parser.parse_args()


def ensure_paths(source: Path, output: Path, overwrite: bool) -> None:
    if not source.exists():
        sys.exit(f"Source model not found: {source}")
    if not source.is_dir():
        sys.exit(f"Source path must be a directory: {source}")

    if output.exists() and any(output.iterdir()) and not overwrite:
        sys.exit(
            f"Output directory {output} is not empty. Pass --overwrite to reuse it."
        )
    output.mkdir(parents=True, exist_ok=True)


def run_quantization(source: Path, output: Path, max_seq_length: int) -> None:
    print(f"Starting FP8 quantization\n  Source : {source}\n  Output : {output}")
    print("Loading base model and processor...", flush=True)
    model = AutoModelForVision2Seq.from_pretrained(
        str(source),
        trust_remote_code=True,
        torch_dtype="auto",
    )
    processor = AutoProcessor.from_pretrained(
        str(source),
        trust_remote_code=True,
    )
    model = replace_modules_for_calibration(model)
    model.eval()

    quant_recipe = QuantizationModifier(
        targets="Linear",
        scheme="FP8_DYNAMIC",
        ignore=[
            "re:.*lm_head",
            "re:visual.*",
            "re:model.visual.*",
        ],
    )

    oneshot(
        model=model,
        processor=processor,
        dataset=None,
        recipe=[quant_recipe],
        output_dir=str(output),
        max_seq_length=max_seq_length,
        num_calibration_samples=0,
        trust_remote_code_model=True,
        clear_sparse_session=True,
        save_compressed=True,
    )

    processor.save_pretrained(str(output))

    print("\n---")
    print(f"Successfully wrote FP8 export to: {output}")
    print("---")


def main() -> None:
    args = parse_args()
    ensure_paths(args.source_model, args.output_dir, args.overwrite)
    run_quantization(args.source_model, args.output_dir, args.max_seq_length)


if __name__ == "__main__":
    main()
