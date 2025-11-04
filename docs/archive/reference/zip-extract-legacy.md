# ZIP Extractor Reference

Automate extraction of competition ZIP submissions, enforce keyword hygiene, and
synchronise XMP metadata with JPEGs.

## Configuration
- Defaults read from `[tool.imageworks.zip-extract]` in `pyproject.toml`, covering
  ZIP source directory, extraction root, and overlay behaviour.【F:src/imageworks/tools/zip_extract.py†L1-L52】【F:pyproject.toml†L226-L257】
- CLI options override defaults and support metadata toggles: `--include-xmp`,
  `--update-all-metadata`, and keyword enforcement.【F:src/imageworks/tools/zip_extract.py†L100-L200】

## Extraction pipeline
- ZIP names are parsed for text inside parentheses to determine target
  subdirectories under the extract root.【F:src/imageworks/tools/zip_extract.py†L31-L70】
- Only image files (and optional XMP sidecars) are extracted; existing files are
  skipped but recorded for auditing.【F:src/imageworks/tools/zip_extract.py†L52-L90】
- `ccc` keyword enforcement uses ExifTool to append IPTC metadata to every image
  processed, ensuring Lightroom categorisation.【F:src/imageworks/tools/zip_extract.py†L70-L140】
- XMP sidecars are parsed for title/author data and applied to matching JPEGs when
  available and allowed by `--include-xmp` and `--update-all-metadata`.【F:src/imageworks/tools/zip_extract.py†L70-L140】

## Reporting
- `write_summary` writes Markdown summaries containing per-ZIP success counts,
  metadata updates, skipped files, and errors.【F:src/imageworks/tools/zip_extract.py†L142-L190】
- Console output (Rich tables) provides a quick overview of processed archives
  during CLI execution.【F:src/imageworks/tools/zip_extract.py†L200-L340】

## CLI (`imageworks-zip`)
- `run` processes a directory of ZIP files, allowing dry runs and selective
  metadata updates. Each archive produces an entry in the Markdown summary.
- Additional subcommands expose diagnostic helpers for listing available ZIPs and
  verifying metadata consistency (see CLI module for options).【F:src/imageworks/tools/zip_extract.py†L200-L340】

## Integration
- Output directories match GUI expectations so extracted images appear in the
  Control Center workflows automatically.【F:src/imageworks/gui/config.py†L1-L40】
- Mono checker consumes the extracted folders as input for verification runs.
