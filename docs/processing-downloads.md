# Process Zip Downloads Logic

This document outlines the step-by-step logic of the `process_zip_downloads.py` module, which automates the ingestion of competition image zip files into a Lightroom-watched folder. This process is designed to streamline the workflow for competition organizers by preparing image submissions for Lightroom's auto-import feature.

## Plain English Summary

This module acts as a smart assistant for handling image submissions that arrive as zip files. Its main job is to:

1.  **Find Zip Files:** Scan a designated folder for all `.zip` archives.
2.  **Extract to a Temporary Area:** Unpack each zip file into a temporary staging location.
3.  **Figure Out the Destination Name:** Look at the zip file's name (specifically, text within parentheses like "(Serial PDI 2024-2025 Round 2)") to determine the final folder name for the extracted images.
4.  **Clean Up the Name:** Make sure the derived folder name is safe and valid for all operating systems.
5.  **Move to Lightroom's Watch Folder:** Transfer the extracted images into a new folder (named as determined in step 3) within your Lightroom Auto-Import "Watched Folder."
6.  **Handle Conflicts:** If a folder with the same name already exists, it can either create a unique new folder (e.g., "Round 2 (2)"), merge the new images into the existing one, or skip the process for that zip.
7.  **Clean Up (Optional):** Delete the original zip file after successful processing and remove any temporary extraction folders.

This process ensures that Lightroom can automatically import the images, keeping them organized by competition round or series, without you having to manually extract and sort each zip file.

---

## Detailed Logic Flow

The `process_zip_downloads.py` module executes the following sequence of operations:

### 1. Setup and Configuration

*   **Argument Parsing:** The script uses `argparse` to define and parse command-line arguments, allowing users to specify:
    *   `--zip-dir`: The source directory containing the `.zip` files.
    *   `--watched`: The path to the Lightroom Auto-Import "Watched Folder."
    *   `--extract-root`: An optional staging directory for extraction. If not provided, a temporary directory is used.
    *   `--pattern`: A regular expression to extract the destination folder name from the zip filename (defaults to capturing text in the last parentheses).
    *   `--conflict`: How to handle naming conflicts in the watched folder (`unique`, `merge`, `skip`).
    *   `--delete-zip-after`: A flag to delete the original zip file after successful processing.
    *   `--dry-run`: A flag to simulate the process without making any actual changes.
    *   `--verbose`: A flag for more detailed output.
*   **Input Validation:** Checks if the provided `--zip-dir` and `--watched` paths exist and are valid directories.

### 2. Scanning for Zip Files

*   The script scans the `--zip-dir` (non-recursively) for all files ending with `.zip`.
*   If no zip files are found, it logs a message and exits.

### 3. Processing Each Zip File

For each `.zip` file found:

*   **Derive Destination Folder Name:**
    *   It uses the `extract_bracketed_name` function with the configured `--pattern` (defaulting to `r"(([^)]+))"`) to find text within the last set of parentheses in the zip file's stem (filename without extension).
    *   Example: "ImageZip 166416 (Serial PDI 2024-2025 Round 2).zip" → "Serial PDI 2024-2025 Round 2".
    *   If no match is found, it falls back to using the zip file's stem as the folder name.
*   **Sanitize Folder Name:**
    *   The `sanitise_folder_name` function cleans the derived name to ensure it's valid across different operating systems (e.g., removing illegal characters for Windows, collapsing whitespace, trimming).
*   **Determine Final Destination Path:**
    *   The sanitized folder name is combined with the `--watched` folder path to form the `final_dest_path`.

### 4. Extraction

*   **Staging Area:**
    *   If `--extract-root` is provided, a subdirectory within it (named after the zip file's stem) is created for extraction.
    *   If `--extract-root` is omitted, a temporary directory is created using `tempfile.mkdtemp()`.
*   **Extract Contents:**
    *   The `extract_zip` function unpacks the zip file's contents into the chosen staging area.
    *   It includes a safeguard against "Zip Slip" attacks by checking for `..` in member paths.
    *   Handles `BadZipFile` errors.

### 5. Moving to Lightroom Watched Folder

*   **Conflict Resolution (`move_extracted_into_dest` function):**
    *   **`skip`:** If `final_dest_path` already exists, the current zip file is skipped.
    *   **`unique`:** If `final_dest_path` exists, `ensure_unique_path` is called to append a numeric suffix (e.g., "(2)", "(3)") until a unique path is found.
    *   **`merge`:** If `final_dest_path` exists, the extracted contents are moved into it. If file/directory names collide, files get a unique numeric suffix, and directories are merged recursively (`dirs_exist_ok=True`).
*   **Content Movement:**
    *   The script intelligently handles cases where the zip file contains a single top-level directory (e.g., `myzip.zip` contains `myfolder/image.jpg`). In such cases, it moves the *contents* of `myfolder` directly into `final_dest_path` to avoid redundant nesting.
    *   Otherwise, all extracted items are moved directly into `final_dest_path`.
    *   `shutil.move` is used for efficient file system operations.

### 6. Cleanup

*   **Delete Zip (Optional):** If `--delete-zip-after` is specified and the process was successful (and not a dry run), the original `.zip` file is deleted.
*   **Cleanup Staging Area:** Any temporary extraction directories are removed. If `--extract-root` was used, the created subdirectory within it is removed.

### 7. Dry Run Mode

*   If `--dry-run` is enabled, the script logs all actions it *would* take without actually modifying any files or directories. This is crucial for testing and verification.

---

## How to Use (Lightroom Integration)

This module is designed to work in conjunction with Lightroom's Auto-Import feature.

1.  **Configure Lightroom Auto-Import:**
    *   In Lightroom Classic, go to `File > Auto Import > Auto Import Settings...`.
    *   Set the "Watched Folder" to the path you will provide to this script via `--watched`.
    *   Configure the "Destination" within Lightroom to your preferred library location.
    *   **Important:** Do NOT enable any file renaming in Lightroom at this stage, as this script does not rename individual image files.

2.  **Run the Script:**
    *   Save the script as `process_zip_downloads.py` (or `ingest_zips_for_lightroom.py` if you prefer the original name).
    *   **Test with `--dry-run` first:**
        ```bash
        python -m imageworks.tools.process_zip_downloads filter \
            --zip-dir "/path/to/your/zips" \
            --watched "/path/to/LightroomWatch" \
            --extract-root "/path/to/_staging" \
            --conflict unique \
            --dry-run \
            --verbose
        ```
    *   **Run for real when satisfied:**
        ```bash
        python -m imageworks.tools.process_zip_downloads filter \
            --zip-dir "/path/to/your/zips" \
            --watched "/path/to/LightroomWatch" \
            --extract-root "/path/to/_staging" \
            --conflict unique \
            --delete-zip-after \
            --verbose
        ```

---

## Notes & Options

*   **Conflict Handling:**
    *   `unique` (default): Appends `(2)`, `(3)`, etc., to folder names if a conflict occurs.
    *   `merge`: Merges new content into an existing folder. Colliding files get a numeric suffix.
    *   `skip`: Skips processing a zip if its destination folder already exists.
*   **Safety & Cleanliness:**
    *   Individual image files are never renamed by this script.
    *   Destination folder names are sanitized for cross-platform compatibility.
    *   Includes protection against "Zip Slip" vulnerabilities.
*   **Pattern Customization:** The `--pattern` argument allows flexibility if your zip file naming conventions change.
*   **Staging Location:** `--extract-root` provides control over where temporary extractions occur.
*   **Scope:** Currently processes `.zip` files only in the top level of the `--zip-dir`.
