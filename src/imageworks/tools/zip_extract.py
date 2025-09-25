import zipfile
from pathlib import Path
from typing import List, Optional
import typer
from rich.console import Console
from rich.table import Table
import tomllib
import re
import subprocess

import xml.etree.ElementTree as ET

app = typer.Typer()
console = Console()

# Read defaults from pyproject.toml [tool.imageworks.zip-extract]
PYPROJECT_PATH = Path(__file__).parents[3] / "pyproject.toml"
DEFAULT_OUTPUT_FILE = "zip_extract_summary.md"
IMAGE_EXTS = {".jpg", ".jpeg", ".tif", ".tiff", ".png"}


def get_zip_extract_defaults():
    try:
        with open(PYPROJECT_PATH, "rb") as f:
            data = tomllib.load(f)
        section = data.get("tool", {}).get("imageworks", {}).get("zip-extract", {})
        zip_dir = section.get("default_zip_dir", "./zips")
        extract_root = section.get("default_extract_root", "./_staging")
        return zip_dir, extract_root
    except Exception:
        return "./zips", "./_staging"


def is_image_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in IMAGE_EXTS


def extract_zip(
    zip_path: Path,
    extract_root: Path,
    include_xmp: bool = False,
    update_all_metadata: bool = False,
):
    """
    Extract images (and optionally .xmp files) from zip_path into a subdirectory of extract_root, named after the text between brackets in the zip filename.
    For each extracted image, if a matching .xmp exists in the zip, read its description and creator and update the jpg's metadata using exiftool.
    Returns (extracted_files, skipped_files, errors)
    """
    # Find text between brackets (parentheses)
    m = re.search(r"\(([^)]+)\)", zip_path.stem)
    if m:
        subdir = m.group(1).strip()
    else:
        subdir = zip_path.stem
    target_dir = extract_root / subdir
    extracted = []
    skipped = []
    meta_updated = []
    errors = []

    def should_extract(filename: str) -> bool:
        ext = Path(filename).suffix.lower()
        if ext in IMAGE_EXTS:
            return True
        if include_xmp and ext == ".xmp":
            return True
        return False

    # Always create the directory if it doesn't exist, but do not skip if it does
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        errors.append(f"Failed to create directory {target_dir}: {e}")
        return extracted, skipped, errors
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        # Build a map of xmp sidecars in the zip
        xmp_map = {Path(m).stem: m for m in members if m.lower().endswith(".xmp")}
        for member in members:
            if should_extract(member):
                out_path = target_dir / member
                file_existed = out_path.exists()
                if not file_existed:
                    try:
                        zf.extract(member, target_dir)
                        extracted.append(str(out_path))
                    except Exception as e:
                        errors.append(f"Failed to extract {member}: {e}")
                        continue
                else:
                    skipped.append(f"File exists: {out_path}")
                # Update metadata if XMP present and allowed by flag
                do_metadata = (not file_existed) or update_all_metadata
                if Path(member).suffix.lower() in IMAGE_EXTS:
                    # Always add 'monopy' keyword
                    try:
                        subprocess.run(
                            [
                                "exiftool",
                                "-overwrite_original",
                                "-keywords+=monopy",
                                str(out_path),
                            ],
                            check=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                        )
                    except Exception as e:
                        errors.append(
                            f"Failed to add 'monopy' keyword to {out_path}: {e}"
                        )
                    if do_metadata:
                        stem = Path(member).stem
                        xmp_member = xmp_map.get(stem)
                        if xmp_member:
                            try:
                                xmp_bytes = zf.read(xmp_member)
                                desc, creator = _parse_xmp_for_title_author(xmp_bytes)
                                if desc or creator:
                                    _update_jpg_metadata_exiftool(
                                        out_path, desc, creator
                                    )
                                    meta_updated.append(str(out_path))
                            except Exception as e:
                                errors.append(
                                    f"Failed to update metadata for {out_path}: {e}"
                                )
    return extracted, skipped, meta_updated, errors


def write_summary(summary_path: Path, results: List[dict]):
    with open(summary_path, "w") as f:
        f.write("# ZIP Extraction Summary\n\n")
        f.write(
            "Detailed errors and logs for each ZIP are included below. This file serves as the log for this run.\n\n"
        )
        total_zips = len(results)
        total_images = sum(len(r["images"]) for r in results)
        total_meta = sum(len(r.get("meta_updated", [])) for r in results)
        f.write(
            f"Processed {total_zips} ZIP files. Extracted {total_images} images. Updated metadata for {total_meta} images.\n\n"
        )
        for r in results:
            f.write(f"## {r['zip']}\n")
            if r["images"]:
                f.write(f"Extracted {len(r['images'])} images:\n")
                for img in r["images"]:
                    f.write(f"- {img}\n")
            else:
                f.write("No images extracted.\n")
            if r.get("meta_updated"):
                f.write("Metadata updated for:\n")
                for m in r["meta_updated"]:
                    f.write(f"- {m}\n")
            if r.get("skipped"):
                f.write("Skipped actions:\n")
                for s in r["skipped"]:
                    f.write(f"- {s}\n")
            if r["errors"]:
                f.write("Errors:\n")
                for err in r["errors"]:
                    f.write(f"- {err}\n")
            f.write("\n")


@app.command()
def run(
    zip_dir: Optional[str] = typer.Option(None, help="Directory containing ZIP files."),
    extract_root: Optional[str] = typer.Option(
        None, help="Directory to extract images to."
    ),
    output_file: str = typer.Option(
        DEFAULT_OUTPUT_FILE, help="Detailed summary output file."
    ),
    include_xmp: bool = typer.Option(False, help="Also extract .xmp files from ZIPs."),
    metadata: bool = typer.Option(
        False, "-m", help="Update metadata for all files, not just new ones."
    ),
):
    """Extract images (and optionally .xmp files) from ZIP files for further processing.\n\nBy default, metadata is only updated for newly extracted files. Use -m/--metadata to update metadata for all files."""
    # Use defaults from pyproject.toml if not provided
    default_zip_dir, default_extract_root = get_zip_extract_defaults()
    zip_dir = Path(zip_dir or default_zip_dir)
    extract_root = Path(extract_root or default_extract_root)
    output_file = Path(output_file)
    extract_root.mkdir(parents=True, exist_ok=True)

    zip_files = list(zip_dir.glob("*.zip"))
    results = []
    total_images = 0
    for zip_path in zip_files:
        images = []
        skipped = []
        meta_updated = []
        errors = []
        try:
            images, skipped, meta_updated, errors = extract_zip(
                zip_path,
                extract_root,
                include_xmp=include_xmp,
                update_all_metadata=metadata,
            )
            total_images += len(images)
        except Exception as e:
            errors.append(str(e))
        results.append(
            {
                "zip": zip_path.name,
                "images": images,
                "skipped": skipped,
                "meta_updated": meta_updated,
                "errors": errors,
            }
        )
        # Print progress after each ZIP
        console.print(
            f"[cyan]Processed: {zip_path.name} | Extracted: {len(images)} | Metadata updated: {len(meta_updated)} | Skipped: {len(skipped)} | Errors: {len(errors)}"
        )

    # Brief summary to terminal
    table = Table(title="ZIP Extraction Summary")
    table.add_column("ZIP File")
    table.add_column("Extracted", justify="right")
    table.add_column("Metadata Updated", justify="right")
    table.add_column("Skipped", justify="right")
    table.add_column("Errors", justify="right")
    for r in results:
        table.add_row(
            r["zip"],
            str(len(r["images"])),
            str(len(r.get("meta_updated", []))),
            str(len(r.get("skipped", []))),
            str(len(r["errors"])),
        )
    console.print(table)
    # Remove detailed per-file output from terminal; only print summary table
    console.print(
        f"[bold green]Processed {len(zip_files)} ZIP files. Extracted {total_images} images."
    )

    write_summary(output_file, results)


def _parse_xmp_for_title_author(xmp_bytes: bytes):
    """Parse XMP bytes for description and creator fields."""
    try:
        text = xmp_bytes.decode("utf-8", "ignore")
    except Exception:
        text = str(xmp_bytes)
    marker = text.find("<x:xmpmeta")
    if marker != -1:
        text = text[marker:]
    try:
        root = ET.fromstring(text)
    except Exception:
        return None, None
    ns = {
        "dc": "http://purl.org/dc/elements/1.1/",
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    }
    # Read 'title' from dc:description (not dc:title)
    title = None
    for node in root.findall(".//dc:description", ns):
        alt = node.find("rdf:Alt", ns)
        if alt is not None:
            for li in alt.findall("rdf:li", ns):
                t = (li.text or "").strip()
                if t:
                    title = t
                    break
        if title:
            break
    creator = None
    creator_node = root.find(".//dc:creator", ns)
    if creator_node is not None:
        seq = creator_node.find("rdf:Seq", ns)
        if seq is not None:
            for li in seq.findall("rdf:li", ns):
                t = (li.text or "").strip()
                if t:
                    creator = t
                    break
    return title, creator


def _update_jpg_metadata_exiftool(
    jpg_path: Path, desc: Optional[str], creator: Optional[str]
):
    """Update XMP-dc:Title and XMP-dc:Creator (and IPTC/EXIF equivalents) in a jpg file using exiftool."""
    cmd = ["exiftool", "-overwrite_original"]
    if desc:
        cmd.append(f"-XMP-dc:Title={desc}")
        cmd.append(f"-IPTC:ObjectName={desc}")
    if creator:
        cmd.append(f"-XMP-dc:Creator={creator}")
        cmd.append(f"-EXIF:Artist={creator}")
    cmd.append(str(jpg_path))
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as e:
        raise RuntimeError(f"Could not update metadata for {jpg_path}: {e}")


if __name__ == "__main__":
    app()
