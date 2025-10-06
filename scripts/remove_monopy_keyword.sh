#!/usr/bin/env bash
# Remove the 'monopy' keyword from images in a directory (recursively).
# Usage:
#   bash scripts/remove_monopy_keyword.sh "/path/to/folder"

set -euo pipefail

DIR="${1:-}"
if [[ -z "${DIR}" ]]; then
  echo "Usage: $(basename "$0") <folder>" >&2
  exit 1
fi

# If the repo exiftool config exists, export it so custom namespaces still resolve
if [[ -f "configs/exiftool/.ExifTool_config" ]]; then
  export EXIFTOOL_HOME="configs/exiftool"
fi

# Remove the 'monopy' keyword from common image types, recursively.
# -keywords-=monopy updates XMP-dc:Subject/Keywords synonyms and IPTC Keywords where applicable.
exiftool \
  -overwrite_original \
  -r \
  -ext jpg -ext jpeg -ext tif -ext tiff -ext png \
  -keywords-=monopy \
  "${DIR}"

echo "✅ Removed 'monopy' keyword where present under: ${DIR}"
