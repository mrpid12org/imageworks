#!/usr/bin/env python3
"""Regenerate Personal Tagger summary from existing JSONL results."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List

import typer

app = typer.Typer(help="Regenerate Personal Tagger summary from JSONL results")


@app.command()
def regenerate(
    jsonl_path: Path = typer.Argument(
        ..., help="Path to existing JSONL results file", exists=True
    ),
    output_path: Path = typer.Option(
        None, "--output", "-o", help="Output summary path (defaults to input dir)"
    ),
    backend: str = typer.Option("unknown", help="Backend name for summary header"),
) -> None:
    """Regenerate summary markdown from JSONL results."""

    if output_path is None:
        output_path = jsonl_path.parent / "regenerated_summary.md"

    # Load records from JSONL
    records = []
    try:
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError as e:
                    print(
                        f"Warning: Skipping invalid JSON on line {line_num}: {e}",
                        file=sys.stderr,
                    )
    except Exception as e:
        print(f"Error reading JSONL file: {e}", file=sys.stderr)
        raise typer.Exit(1)

    if not records:
        print("No valid records found in JSONL file", file=sys.stderr)
        raise typer.Exit(1)

    # Generate summary
    total = len(records)
    lines: List[str] = [
        "# Personal Tagger Summary (Regenerated)",
        f"Generated: {datetime.now(UTC).isoformat().replace('+00:00', 'Z')}",
        f"Source: {jsonl_path}",
        "",
        f"Processed {total} image(s).",
        f"Backend: {backend}",
        "",
    ]

    # Group by directory
    grouped: Dict[Path, List[dict]] = defaultdict(list)
    for record in records:
        image_path = Path(record["image"])
        grouped[image_path.parent].append(record)

    # Generate entries for each directory
    for directory in sorted(grouped.keys()):
        entries = grouped[directory]
        lines.append(f"## {directory} ({len(entries)} image(s))")
        lines.append("")

        for record in entries:
            image_path = Path(record["image"])

            # Extract keywords (handle both list of dicts and simple list formats)
            keywords_data = record.get("keywords", [])
            if keywords_data and isinstance(keywords_data[0], dict):
                # New format with keyword objects
                all_keywords = [kw["keyword"].title() for kw in keywords_data]
            else:
                # Simple list format
                all_keywords = [str(kw).title() for kw in keywords_data]

            keywords = ", ".join(all_keywords) if all_keywords else "<none>"

            # Extract caption and description
            caption = record.get("caption", "").strip() or "<none>"
            description = record.get("description", "").strip() or "<none>"

            # Determine status
            metadata_written = record.get("metadata_written", False)
            notes = record.get("notes", "")

            status = "metadata written" if metadata_written else "metadata pending"
            if notes:
                status = f"{status} ({notes})"

            lines.append(f"- **{image_path.name}** — {status}")
            lines.append(f"  - Keywords: {keywords}")
            lines.append(f"  - Caption: {caption}")
            lines.append(f"  - Description: {description}")

        lines.append("")

    # Write summary
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"✅ Regenerated summary written to {output_path}")
    except Exception as e:
        print(f"Error writing summary: {e}", file=sys.stderr)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
