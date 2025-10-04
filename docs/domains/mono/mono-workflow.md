# Competition Image Processing Workflow

This document outlines the complete workflow for processing competition images, from initial zip download to monochrome validation.

## Overview

The workflow consists of three main stages:

1. **Preprocessing (Zip Extraction)**: Extract competition images from downloaded zip files and prepare them for Lightroom
2. **Lightroom Organization**: Images are automatically imported and organized in Lightroom
3. **Monochrome Validation**: Images are checked for monochrome compliance

## Stage 1: Preprocessing - Zip Extraction

The zip extraction process (`zip_extract.py`) handles the initial preparation of competition images:

1. **Input**: Competition zip files downloaded from the competition website
2. **Process**:
   - Extracts JPG images from zip files
   - Reads associated XMP files for title and author metadata
   - Embeds metadata into JPGs
   - Creates organized directory structure in Lightroom-watched folder
3. **Output**:
   - JPGs with embedded metadata in Lightroom-watched directories
   - XMP files are processed but not moved (no longer needed)

### Directory Structure Creation

The tool creates directories following this pattern:
```
[Lightroom Watched Folder]/
└── Competition Round Name/
    └── Extracted JPGs with embedded metadata
```

### Metadata Handling
- Title and author information is read from XMP files
- This metadata is embedded directly in the JPG files
- Original XMP files are not moved to the Lightroom directories

## Stage 2: Lightroom Organization

Lightroom's Auto-Import feature manages the images:

1. Watches the directory created by zip extraction
2. Automatically imports new images
3. Maintains organization by competition rounds
4. Provides viewing and basic image management

## Stage 3: Monochrome Validation

The monochrome checking tools (`mono.py`) operate on the Lightroom-managed directories:

1. **Input**: Images from Lightroom-managed directories
2. **Process**:
   - Analyzes images for monochrome compliance
   - Generates overlays for flagged images
3. **Output**:
   - Analysis results (PASS/QUERY/FAIL)
   - Overlay files for visual verification
   - Summary reports in various formats

### Default Configuration
The monochrome checker is configured to use Lightroom-managed directories as its default input location, creating a seamless workflow from zip extraction through validation.

## Complete Workflow Example

1. Download competition zip file (e.g., "Round2_Entries.zip")
2. Run zip extraction:
   ```bash
   imageworks-zip-extract --zip-dir ~/Downloads --watched ~/Pictures/Competition
   ```
3. Lightroom auto-imports images to:
   ```
   ~/Pictures/Competition/Round 2/
   ```
4. Run monochrome validation:
   ```bash
   imageworks-mono check "~/Pictures/Competition/Round 2"
   ```
5. Review results and overlays in Lightroom
