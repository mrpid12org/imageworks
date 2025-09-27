"""Color-Narrator CLI - Main command line interface.

This module provides the primary CLI for color-narrator functionality,
following the same patterns as the mono-checker CLI.
"""

import typer
from typing import Optional
from pathlib import Path
import tomllib  # Built-in since Python 3.11
import logging
import json
import requests
import base64
from PIL import Image
import io
import time
from datetime import datetime
from typing import Dict, Any

from ..core.metadata import XMPMetadataWriter, ColorNarrationMetadata
from ..core.region_based_vlm import (
    RegionBasedVLMAnalyzer,
    create_demo_regions,
)
from ..core.vlm import VLMClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = typer.Typer(help="Color-Narrator - VLM-guided color localization")


def _find_pyproject(start_path: Path) -> Optional[Path]:
    """Find pyproject.toml by walking up the directory tree."""
    current = start_path.resolve()
    for parent in [current] + list(current.parents):
        candidate = parent / "pyproject.toml"
        if candidate.exists():
            return candidate
    return None


def _load_defaults() -> Dict[str, Any]:
    """Load defaults from [tool.imageworks] in pyproject.toml (if present)."""
    try:
        cfg_p = _find_pyproject(Path.cwd())
        if not cfg_p:
            return {}
        data = tomllib.loads(cfg_p.read_text())

        # Get both mono and color_narrator configs
        imageworks = data.get("tool", {}).get("imageworks", {})
        mono_config = imageworks.get("mono", {})
        cn_config = imageworks.get("color_narrator", {})

        # Merge configs, with color_narrator taking precedence
        merged = {}
        merged.update(mono_config)
        merged.update(cn_config)

        return merged if isinstance(merged, dict) else {}
    except Exception:
        return {}


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


def resize_image_for_vlm(image_path: Path, max_size: int = 1024) -> bytes:
    """Resize image to reasonable size for VLM processing."""
    with Image.open(image_path) as img:
        # Convert to RGB if needed
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Calculate new size maintaining aspect ratio
        width, height = img.size
        if max(width, height) > max_size:
            if width > height:
                new_width = max_size
                new_height = int(height * max_size / width)
            else:
                new_height = max_size
                new_width = int(width * max_size / height)

            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Save to bytes buffer
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        return buffer.getvalue()


def encode_image(image_path: Path) -> str:
    """Encode image to base64 with size optimization."""
    image_bytes = resize_image_for_vlm(image_path)
    return base64.b64encode(image_bytes).decode("utf-8")


def call_vlm(image_path: Path, mono_data: dict, vlm_url: str, model: str) -> str:
    """Call VLM to get color description."""
    try:
        # Prepare the prompt
        prompt = f"""You are analyzing a photograph that should be monochrome but contains residual color.

Mono-checker analysis shows:
- Dominant color: {mono_data.get('dominant_color', 'unknown')}
- Color cast: {mono_data.get('reason_summary', 'No summary available')}
- Verdict: {mono_data.get('verdict', 'unknown')}

Please describe in natural language where you observe residual color in this image. Be concise and professional."""

        # Encode the image
        base64_image = encode_image(image_path)

        # Prepare API request
        headers = {"Content-Type": "application/json"}
        data = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 300,
            "temperature": 0.1,
        }

        # Make API call
        response = requests.post(
            f"{vlm_url}/chat/completions", headers=headers, json=data, timeout=30
        )
        response.raise_for_status()

        result = response.json()
        return result["choices"][0]["message"]["content"]

    except Exception as e:
        return f"Error calling VLM: {e}"


def load_config() -> dict:
    """Load configuration from pyproject.toml."""
    try:
        with open("pyproject.toml", "rb") as f:
            config = tomllib.load(f)
        return config.get("tool", {}).get("imageworks", {}).get("color_narrator", {})
    except Exception as e:
        logger.warning(f"Could not load configuration: {e}")
        return {}


@app.command()
def narrate(
    images_dir: Optional[Path] = typer.Option(
        None, "--images", "-i", help="Directory containing JPEG originals"
    ),
    overlays_dir: Optional[Path] = typer.Option(
        None, "--overlays", "-o", help="Directory containing lab overlay PNGs"
    ),
    mono_jsonl: Optional[Path] = typer.Option(
        None, "--mono-jsonl", "-j", help="Mono-checker results JSONL file"
    ),
    batch_size: Optional[int] = typer.Option(
        None, "--batch-size", "-b", help="VLM batch size (default from config)"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be processed without making changes"
    ),
    debug: bool = typer.Option(
        False, "--debug", help="Enable debug output and save debug images"
    ),
) -> None:
    """Generate natural language color descriptions for monochrome competition images.

    Processes JPEG images using mono-checker analysis data and VLM inference to create
    natural language descriptions of where residual color appears. Results are embedded
    as XMP metadata in the original JPEG files.

    Example:
        imageworks-color-narrator narrate -i ./originals -o ./overlays -j ./mono_results.jsonl
    """
    typer.echo("🎨 Color-Narrator - Narrate command")

    if dry_run:
        typer.echo("🔍 DRY RUN MODE - No files will be modified")

    # Load configuration
    config = load_config()
    if debug:
        typer.echo(f"📋 Loaded config: {json.dumps(config, indent=2)}")

    # Use provided paths or defaults from config
    images_path = images_dir or Path(config.get("default_images_dir", ""))
    overlays_path = overlays_dir or Path(config.get("default_overlays_dir", ""))
    mono_jsonl_path = mono_jsonl or Path(config.get("default_mono_jsonl", ""))

    # Basic validation
    if not images_path.exists():
        typer.echo(f"❌ Images directory not found: {images_path}")
        raise typer.Exit(1)

    if not overlays_path.exists():
        typer.echo(f"❌ Overlays directory not found: {overlays_path}")
        raise typer.Exit(1)

    if not mono_jsonl_path.exists():
        typer.echo(f"❌ Mono-checker JSONL file not found: {mono_jsonl_path}")
        raise typer.Exit(1)

    # Create metadata writer
    backup_originals = config.get("backup_original_files", True)
    metadata_writer = XMPMetadataWriter(backup_original=backup_originals)

    # Basic processing simulation for now
    typer.echo(f"📁 Images: {images_path}")
    typer.echo(f"📁 Overlays: {overlays_path}")
    typer.echo(f"📄 Mono data: {mono_jsonl_path}")

    vlm_url = config.get("vlm_base_url", "http://localhost:8000/v1")
    vlm_model = config.get("vlm_model", "Qwen/Qwen2-VL-7B-Instruct")
    typer.echo(f"🤖 VLM: {vlm_model} at {vlm_url}")
    typer.echo(
        f"💾 XMP metadata backup: {'enabled' if backup_originals else 'disabled'}"
    )

    # Load and process mono data
    processed_count = 0
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
                        typer.echo(f"📷 Processing: {image_name}")
                        if debug:
                            typer.echo(f"   Verdict: {data.get('verdict', 'unknown')}")
                            typer.echo(
                                f"   Color: {data.get('dominant_color', 'unknown')}"
                            )
                            typer.echo(
                                f"   Reason: {data.get('reason_summary', 'No summary')}"
                            )

                        if not dry_run:
                            # Actual VLM processing
                            typer.echo("   🤖 Calling VLM for color description...")
                            start_time = time.time()
                            description = call_vlm(
                                local_image, data, vlm_url, vlm_model
                            )
                            processing_time = time.time() - start_time
                            typer.echo(f"   📝 VLM Response: {description}")

                            # Create metadata object
                            metadata = ColorNarrationMetadata(
                                description=description or "No description available",
                                confidence_score=0.85,  # Mock confidence for now
                                color_regions=data.get("top_colors", []) or [],
                                processing_timestamp=datetime.now().isoformat(),
                                mono_contamination_level=float(
                                    data.get("colorfulness", 0.0) or 0.0
                                ),
                                vlm_model=vlm_model or "unknown",
                                vlm_processing_time=processing_time,
                                hue_analysis=f"Dominant: {data.get('dominant_color', 'unknown')}",
                                chroma_analysis=f"Max chroma: {data.get('chroma_max', 0.0)}",
                                validation_status="unvalidated",
                            )

                            # Write metadata to image
                            typer.echo("   💾 Writing metadata to image...")
                            success = metadata_writer.write_metadata(
                                local_image, metadata
                            )
                            if success:
                                typer.echo("   ✅ Metadata written successfully")
                            else:
                                typer.echo("   ⚠️  Failed to write metadata")
                        else:
                            typer.echo(
                                "   🎨 [DRY RUN] Would call VLM and write metadata"
                            )
                        processed_count += 1
                    else:
                        if debug:
                            typer.echo(
                                f"⏭️  Skipping {image_name} (not in local images)"
                            )

                except json.JSONDecodeError as e:
                    typer.echo(f"⚠️  Skipping invalid JSON on line {line_num}: {e}")
                    continue
                except KeyError as e:
                    typer.echo(f"⚠️  Missing required field on line {line_num}: {e}")
                    continue

    except Exception as e:
        typer.echo(f"❌ Error processing mono data: {e}")
        raise typer.Exit(1)

    typer.echo(f"✅ Processed {processed_count} images")
    typer.echo("⚠️  Full implementation coming soon - basic validation complete")


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

    # Basic validation
    if not images_path.exists():
        typer.echo(f"❌ Images directory not found: {images_path}")
        raise typer.Exit(1)

    if not mono_jsonl_path.exists():
        typer.echo(f"❌ Mono-checker JSONL file not found: {mono_jsonl_path}")
        raise typer.Exit(1)

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
def enhance_mono(
    mono_jsonl: Optional[Path] = typer.Option(
        None,
        "-j",
        "--mono-jsonl",
        help="Mono-checker results JSONL file (uses config default if not specified)",
    ),
    images_dir: Optional[Path] = typer.Option(
        None,
        "-i",
        "--images",
        help="Directory containing JPEG originals (uses config default if not specified)",
    ),
    output_jsonl: Optional[Path] = typer.Option(
        None, "-o", "--output", help="Output JSONL for enhanced results"
    ),
    summary_md: Optional[Path] = typer.Option(
        None, "-s", "--summary", help="Output markdown summary file"
    ),
    limit: int = typer.Option(
        5, "--limit", help="Limit number of images to process (for testing)"
    ),
    interesting_only: bool = typer.Option(
        True,
        "--interesting-only/--all",
        help="Process only fails and queries (not pure passes)",
    ),
):
    """
    Enhance mono-checker descriptions with VLM analysis.

    Combines mono-checker's accurate verdicts with VLM's rich contextual descriptions
    for professional competition documentation.
    """
    from ..core.hybrid_mono_enhancer import HybridMonoEnhancer

    defaults = _load_defaults()

    typer.echo("🎨 Hybrid Mono Enhancement - VLM Descriptions + Mono Verdicts")

    # Determine summary path if not specified
    if summary_md is None:
        if "default_summary" in defaults:
            # Use same pattern as mono but for enhancements
            default_summary_path = Path(defaults["default_summary"])
            summary_md = default_summary_path.parent / "enhancement_summary.md"
        else:
            summary_md = Path("outputs/summaries/enhancement_summary.md")

    # Determine mono JSONL path
    if mono_jsonl is None:
        if "default_mono_jsonl" in defaults:
            mono_jsonl = Path(defaults["default_mono_jsonl"])
        elif "default_jsonl" in defaults:
            mono_jsonl = Path(defaults["default_jsonl"])
        else:
            typer.echo(
                "❌ No mono JSONL specified and no default configured. Use -j or set default_mono_jsonl in pyproject.toml"
            )
            raise typer.Exit(1)

    # Determine images directory
    if images_dir is None:
        if "default_images_dir" in defaults:
            images_dir = Path(defaults["default_images_dir"])
        elif "default_folder" in defaults:
            images_dir = Path(defaults["default_folder"])
        else:
            typer.echo(
                "❌ No images directory specified and no default configured. Use -i or set default_folder in pyproject.toml"
            )
            raise typer.Exit(1)

    if not mono_jsonl.exists():
        typer.echo(f"❌ Mono JSONL file not found: {mono_jsonl}")
        raise typer.Exit(1)

    # Check if we can access images (either through images_dir or full paths in mono data)
    images_dir_valid = images_dir.exists() if images_dir.name != "dummy" else False
    if not images_dir_valid:
        typer.echo(f"📁 Using full paths from mono results (images_dir: {images_dir})")
    else:
        typer.echo(f"📁 Images directory: {images_dir}")

    typer.echo(f"📄 Mono JSONL: {mono_jsonl}")

    # Initialize hybrid enhancer
    try:
        enhancer = HybridMonoEnhancer()
        typer.echo(f"🤖 VLM: {enhancer.model} at {enhancer.base_url}")
    except Exception as e:
        typer.echo(f"❌ Failed to initialize enhancer: {e}")
        raise typer.Exit(1)

    # Load mono data
    mono_results = []
    try:
        with open(mono_jsonl, "r") as f:
            for line in f:
                if line.strip():
                    mono_data = json.loads(line)
                    # Filter for interesting cases if requested
                    if interesting_only and mono_data.get("verdict") == "pass":
                        continue
                    mono_results.append(mono_data)
    except Exception as e:
        typer.echo(f"❌ Failed to load mono JSONL: {e}")
        raise typer.Exit(1)

    total_loaded = sum(1 for line in open(mono_jsonl) if line.strip())
    filter_msg = " (filtered to fails and queries)" if interesting_only else ""
    typer.echo(
        f"📄 Loaded {len(mono_results)} mono results from {total_loaded} total{filter_msg}"
    )

    # Process limited set
    process_count = min(limit, len(mono_results))
    typer.echo(f"🔄 Processing {process_count} images...")

    enhanced_results = []

    for i, mono_data in enumerate(mono_results[:process_count]):
        try:
            # Use the full path from mono results
            original_path = Path(mono_data["path"])
            image_path = original_path
            image_name = original_path.name

            if not image_path.exists():
                typer.echo(f"⚠️  Image not found: {image_name}")
                continue
                continue

            title = mono_data.get("title", "Unknown")
            author = mono_data.get("author", "Unknown")
            typer.echo(f'📷 Processing {i+1}/{process_count}: "{title}" by {author}')

            # Get enhanced result
            enhanced = enhancer.enhance_mono_result(mono_data, image_path)

            # Store result
            enhanced_data = {
                "image_path": enhanced.image_path,
                "image_name": image_name,
                "title": enhanced.title,
                "author": enhanced.author,
                # Original mono-checker verdict (authoritative)
                "verdict": enhanced.original_verdict,
                "mode": enhanced.original_mode,
                "original_reason": enhanced.original_reason,
                # VLM enhancement (descriptive)
                "vlm_description": enhanced.vlm_description,
                "vlm_model": enhanced.vlm_model,
                "vlm_processing_time": enhanced.vlm_processing_time,
                # Key technical metrics
                "dominant_color": enhanced.dominant_color,
                "colorfulness": enhanced.colorfulness,
                "chroma_max": enhanced.chroma_max,
                "hue_std_deg": enhanced.hue_std_deg,
                "timestamp": datetime.now().isoformat(),
            }
            enhanced_results.append(enhanced_data)

            # Show result preview
            verdict_icon = {"pass": "✅", "pass_with_query": "⚠️", "fail": "❌"}.get(
                enhanced.original_verdict, "❓"
            )
            typer.echo(
                f"   {verdict_icon} Verdict: {enhanced.original_verdict} ({enhanced.original_mode})"
            )
            typer.echo(f"   📝 VLM: {enhanced.vlm_description[:100]}...")
            typer.echo(f"   ⏱️  Processing: {enhanced.vlm_processing_time:.2f}s")

        except Exception as e:
            typer.echo(f"❌ Error processing {image_name}: {e}")
            continue

    # Save results
    if output_jsonl:
        try:
            with open(output_jsonl, "w") as f:
                for result in enhanced_results:
                    f.write(json.dumps(result) + "\n")
            typer.echo(f"💾 Enhanced results saved to: {output_jsonl}")
        except Exception as e:
            typer.echo(f"❌ Failed to save results: {e}")

    # Summary
    typer.echo("\n📊 Enhancement Summary:")
    typer.echo(f"   Total processed: {len(enhanced_results)}")
    typer.echo(
        f"   Average VLM processing time: {sum(r['vlm_processing_time'] for r in enhanced_results) / len(enhanced_results):.2f}s"
    )

    # Show distribution of verdicts
    verdict_counts = {}
    for result in enhanced_results:
        verdict = result["verdict"]
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1

    typer.echo("   Verdict distribution:")
    for verdict, count in verdict_counts.items():
        verdict_icon = {"pass": "✅", "pass_with_query": "⚠️", "fail": "❌"}.get(
            verdict, "❓"
        )
        typer.echo(f"      {verdict_icon} {verdict}: {count}")

    # Generate human-readable summary
    if enhanced_results:
        try:
            _generate_enhancement_summary(enhanced_results, summary_md)
            typer.echo(f"📋 Summary report saved to: {summary_md}")
        except Exception as e:
            typer.echo(f"⚠️  Failed to generate summary: {e}")

    typer.echo("\n✅ Hybrid mono enhancement complete")


@app.command()
def analyze_regions(
    image_path: Path = typer.Option(
        ..., "--image", "-i", help="Path to JPEG image to analyze"
    ),
    output_json: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output JSON file for structured results"
    ),
    demo_mode: bool = typer.Option(
        False,
        "--demo",
        help="Use demo regions (for testing without mono-checker regions)",
    ),
    debug: bool = typer.Option(False, "--debug", help="Enable debug output"),
) -> None:
    """Analyze color regions using hallucination-resistant VLM prompts.

    This command demonstrates the new approach suggested for better VLM responses:
    - No priming examples to avoid bias
    - Structured JSON output for validation
    - Grounded in technical region data
    - Uncertainty handling with confidence scores

    Example:
        imageworks-color-narrator analyze-regions --image photo.jpg --demo --debug
    """
    typer.echo("🔬 Region-Based VLM Analysis")

    if not image_path.exists():
        typer.echo(f"❌ Image not found: {image_path}")
        raise typer.Exit(1)

    if debug:
        typer.echo(f"📁 Image: {image_path}")
        typer.echo(f"🧪 Demo mode: {demo_mode}")

    # Initialize VLM client
    try:
        vlm_client = VLMClient(
            base_url="http://localhost:8000/v1",
            model_name="Qwen2-VL-2B-Instruct",
            timeout=120,
        )

        # Test VLM connection
        if not vlm_client.health_check():
            typer.echo("❌ VLM server not available at http://localhost:8000")
            typer.echo("💡 Start server with: uv run python start_vllm_server.py")
            raise typer.Exit(1)

        typer.echo(f"🤖 VLM: Connected to {vlm_client.model_name}")

    except Exception as e:
        typer.echo(f"❌ Failed to initialize VLM client: {e}")
        raise typer.Exit(1)

    # Load and encode image
    try:
        with open(image_path, "rb") as f:
            image_data = f.read()
        image_base64 = base64.b64encode(image_data).decode("utf-8")
        typer.echo("📸 Image encoded")
    except Exception as e:
        typer.echo(f"❌ Failed to load image: {e}")
        raise typer.Exit(1)

    # Create regions (demo mode for now, until mono-checker provides real regions)
    if demo_mode:
        regions = create_demo_regions()
        dominant_color = "yellow-green"
        dominant_hue_deg = 88.0
        typer.echo("🎭 Using demo regions (2 synthetic color regions)")
    else:
        typer.echo(
            "❌ Real region analysis requires mono-checker integration (not yet implemented)"
        )
        typer.echo("💡 Use --demo flag to test with synthetic regions")
        raise typer.Exit(1)

    if debug:
        typer.echo("🔍 Regions to analyze:")
        for region in regions:
            typer.echo(
                f"   Region {region.index}: {region.hue_name} "
                f"({region.area_pct:.1f}% area, L*={region.mean_L:.1f})"
            )

    # Initialize region-based analyzer
    analyzer = RegionBasedVLMAnalyzer(vlm_client)

    # Perform analysis
    typer.echo("⚡ Running VLM analysis...")
    try:
        analysis = analyzer.analyze_regions(
            file_name=image_path.name,
            regions=regions,
            dominant_color=dominant_color,
            dominant_hue_deg=dominant_hue_deg,
            image_base64=image_base64,
            # overlay_hue_base64=None,      # Would be provided by mono-checker
            # overlay_chroma_base64=None    # Would be provided by mono-checker
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
                confidence_icon = (
                    "🟢"
                    if finding.confidence >= 0.8
                    else "🟡" if finding.confidence >= 0.5 else "🔴"
                )
                location = (
                    f" {finding.location_phrase}" if finding.location_phrase else ""
                )
                typer.echo(
                    f"   {confidence_icon} Region {finding.region_index}: "
                    f"{finding.color_family} on {finding.object_part}{location} "
                    f"({finding.tonal_zone}, confidence: {finding.confidence:.2f})"
                )

        # Save structured results if requested
        if output_json:
            try:
                results = {
                    "file_name": analysis.file_name,
                    "dominant_color": analysis.dominant_color,
                    "dominant_hue_deg": analysis.dominant_hue_deg,
                    "findings": [
                        {
                            "region_index": f.region_index,
                            "object_part": f.object_part,
                            "color_family": f.color_family,
                            "tonal_zone": f.tonal_zone,
                            "location_phrase": f.location_phrase,
                            "confidence": f.confidence,
                        }
                        for f in analysis.findings
                    ],
                    "validation_errors": analysis.validation_errors,
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
