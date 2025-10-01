"""
Main model downloader implementation.

Coordinates URL analysis, format detection, directory routing, and download execution
using aria2c for optimal performance.
"""

import os
import subprocess
import tempfile
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .config import get_config, DownloaderConfig
from .registry import get_registry, ModelRegistry, ModelEntry
from .url_analyzer import URLAnalyzer
from .formats import FormatDetector


@dataclass
class DownloadProgress:
    """Progress information for a download."""

    total_files: int
    completed_files: int
    total_bytes: int
    downloaded_bytes: int
    current_speed: str
    eta: str


class ModelDownloader:
    """Main model downloader class."""

    def __init__(
        self,
        config: Optional[DownloaderConfig] = None,
        registry: Optional[ModelRegistry] = None,
    ):
        self.config = config or get_config()
        self.registry = registry or get_registry()
        self.url_analyzer = URLAnalyzer()
        self.format_detector = FormatDetector()

        # Progress tracking
        self._current_log_file: Optional[Path] = None
        self._current_file_count: int = 0

        # Ensure aria2c is available
        self._check_aria2c()

    def _check_aria2c(self) -> None:
        """Check if aria2c is available."""
        try:
            subprocess.run(
                ["aria2c", "--version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(
                "aria2c not found. Please install aria2c for optimal download performance.\n"
                "Ubuntu/Debian: sudo apt install aria2\n"
                "macOS: brew install aria2"
            )

    def _parse_aria2c_progress(
        self, log_file: Path, total_files: int
    ) -> Dict[str, Any]:
        """Parse aria2c progress by monitoring actual file sizes and aria2c activity."""

        # Initialize tracking variables if not present
        if not hasattr(self, "_progress_tracker"):
            self._progress_tracker = {
                "last_size_check": {},
                "last_check_time": time.time(),
                "last_speed": 0,
            }

        current_time = time.time()
        tracker = self._progress_tracker
        time_diff = current_time - tracker["last_check_time"]

        # Don't check too frequently for performance
        if time_diff < 0.5:
            return {
                "active_files": total_files,
                "total_files": total_files,
                "total_size_bytes": 0,
                "downloaded_bytes": 0,
                "speed_bps": tracker["last_speed"],
                "eta_seconds": 0,
            }

        try:
            total_size_change = 0
            completed_files = 0
            active_files = 0
            downloaded_bytes = 0

            # Monitor actual file sizes if we have target directory
            if hasattr(self, "_current_target_dir") and self._current_target_dir:
                for file_path in self._current_target_dir.rglob("*"):
                    if file_path.is_file():
                        file_key = str(file_path)
                        current_size = file_path.stat().st_size

                        # Skip aria2 control files for size calculation
                        if file_path.suffix == ".aria2":
                            active_files += 1
                            continue

                        # Calculate size change for speed estimation
                        last_size = tracker["last_size_check"].get(file_key, 0)
                        size_change = max(0, current_size - last_size)
                        total_size_change += size_change
                        tracker["last_size_check"][file_key] = current_size
                        downloaded_bytes += current_size

                        # Check if file is complete (no corresponding .aria2 file)
                        aria2_file = file_path.parent / f"{file_path.name}.aria2"
                        if not aria2_file.exists() and current_size > 0:
                            completed_files += 1
                        elif aria2_file.exists():
                            active_files += 1

            # Calculate speed in bytes per second
            speed_bps = int(total_size_change / time_diff) if time_diff > 0 else 0
            tracker["last_speed"] = speed_bps
            tracker["last_check_time"] = current_time

            return {
                "active_files": active_files,
                "total_files": total_files,
                "total_size_bytes": 0,
                "downloaded_bytes": downloaded_bytes,
                "speed_bps": speed_bps,
                "eta_seconds": 0,
            }

        except Exception:
            return {
                "active_files": total_files,
                "total_files": total_files,
                "total_size_bytes": 0,
                "downloaded_bytes": 0,
                "speed_bps": 0,
                "eta_seconds": 0,
            }

    def get_progress(self) -> Dict[str, Any]:
        """Get current download progress if a download is active."""
        if self._current_log_file and self._current_file_count > 0:
            return self._parse_aria2c_progress(
                self._current_log_file, self._current_file_count
            )
        return {
            "active_files": 0,
            "total_files": 0,
            "total_size_bytes": 0,
            "downloaded_bytes": 0,
            "speed_bps": 0,
            "eta_seconds": 0,
        }

    def _verify_download_complete(self, files: List[Any], target_dir: Path) -> bool:
        """Verify that all downloaded files are complete and valid."""
        for file_info in files:
            file_path = target_dir / file_info.path

            # Check if file exists
            if not file_path.exists():
                print(f"⚠️  Missing file: {file_info.path}")
                return False

            # Check for aria2 control file indicating incomplete download
            aria2_control = file_path.parent / f"{file_path.name}.aria2"
            if aria2_control.exists():
                print(
                    f"⚠️  Incomplete download detected: {file_info.path} (aria2 control file exists)"
                )
                return False

            # Check file has non-zero size
            if file_path.stat().st_size == 0:
                print(f"⚠️  Zero-size file: {file_info.path}")
                return False

        return True

    def download(
        self,
        model_identifier: str,
        format_preference: Optional[List[str]] = None,
        location_override: Optional[str] = None,
        include_optional: bool = False,
        force_redownload: bool = False,
        interactive: bool = True,
    ) -> ModelEntry:
        """Download a model from a URL or HuggingFace identifier."""

        # Step 1: Analyze the URL/identifier
        analysis = self.url_analyzer.analyze_url(model_identifier)

        # Step 2: Check if already downloaded
        repository_id = f"{analysis.repository.owner}/{analysis.repository.repo}"
        if not force_redownload:
            existing_models = self.registry.find_model(repository_id)
            if existing_models:
                existing = existing_models[0]  # Use first match

                # Check if the model directory actually exists
                if existing.path_obj.exists():
                    print(f"✅ Model already downloaded: {existing.model_name}")
                    return existing
                else:
                    print(
                        f"⚠️  Model '{existing.model_name}' found in registry but directory missing: {existing.path}"
                    )
                    if interactive:
                        try:
                            response = (
                                input(
                                    "Directory not found. Do you want to re-download the model? (y/N): "
                                )
                                .strip()
                                .lower()
                            )
                            if response not in ["y", "yes"]:
                                print("❌ Download cancelled by user")
                                return existing  # Return existing entry without downloading
                            print("🔄 Proceeding with re-download...")
                        except (KeyboardInterrupt, EOFError):
                            print("\n❌ Download cancelled by user")
                            return existing
                        # Remove stale registry entry to avoid confusion
                        self.registry.remove_model(
                            existing.model_name, existing.format_type, existing.location
                        )
                    else:
                        print("🔄 Non-interactive mode: proceeding with re-download...")
                        # Remove stale registry entry
                        self.registry.remove_model(
                            existing.model_name, existing.format_type, existing.location
                        )

        # Step 3: Detect format from file list
        all_files = []
        for file_list in analysis.files.values():
            all_files.extend(file_list)
        formats = self.format_detector.detect_from_filelist([f.path for f in all_files])
        if not formats:
            print("⚠️  Could not detect model format, using default routing")
            primary_format = "unknown"
        else:
            # Use format preference or first detected
            if format_preference:
                for pref in format_preference:
                    for fmt in formats:
                        if fmt.format_type == pref:
                            primary_format = pref
                            break
                    else:
                        continue
                    break
                else:
                    primary_format = formats[0].format_type
            else:
                primary_format = formats[0].format_type

        print(f"🔧 Detected format: {primary_format}")

        # Step 4: Determine target directory
        if location_override:
            target_dir = Path(location_override) / repository_id
        else:
            target_dir = self.config.get_target_directory(primary_format, repository_id)

        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Target directory: {target_dir}")

        # Step 5: Filter files based on preferences
        required_files = []
        optional_files = []
        for category, file_list in analysis.files.items():
            if category in ["model_weights", "config", "tokenizer"]:
                required_files.extend(file_list)
            elif category in ["optional", "large_optional"]:
                optional_files.extend(file_list)

        if not required_files:
            raise RuntimeError("No essential model files found")

        # Step 6: Download files
        base_url = f"https://huggingface.co/{repository_id}/resolve/main"

        # Prepare aria2c download
        print(f"📥 Downloading {len(required_files)} required files...")
        if required_files:
            self._download_files_with_aria2c(required_files, base_url, target_dir)

        if include_optional and optional_files:
            print(f"📥 Downloading {len(optional_files)} optional files...")
            self._download_files_with_aria2c(optional_files, base_url, target_dir)

        # Step 7: Verify download completion before registry registration
        all_files = required_files + (optional_files if include_optional else [])
        if not self._verify_download_complete(all_files, target_dir):
            raise RuntimeError(
                "Download verification failed - some files are missing, incomplete, or corrupted. "
                "Please try the download again."
            )

        print("✅ Download verification passed - all files complete")

        # Step 8: Calculate total size (now guaranteed to be accurate)
        total_size = sum((target_dir / f.path).stat().st_size for f in all_files)

        # Step 9: Register in registry (now guaranteed to be accurate)
        # Get list of downloaded files
        downloaded_files = []
        for f in all_files:
            downloaded_files.append(f.path)

        entry = ModelEntry(
            model_name=repository_id,
            format_type=primary_format,
            path=str(target_dir),
            size_bytes=total_size,
            location=(
                "linux_wsl" if "ai-models" in str(target_dir) else "windows_lmstudio"
            ),
            files=downloaded_files,
            downloaded_at="",  # Will be set by registry
            metadata={
                "model_type": analysis.repository.model_type,
                "library": analysis.repository.library_name,
                "huggingface_id": repository_id,
                "files_downloaded": len(all_files),
                "verified_complete": True,  # Mark as verified
            },
        )

        # Step 10: Add to registry with error handling
        try:
            registry_key = self.registry.add_model(
                model_name=entry.model_name,
                format_type=entry.format_type,
                location=entry.location,
                path=entry.path,
                size_bytes=entry.size_bytes,
                files=entry.files,
                metadata=entry.metadata,
            )
            print(f"✅ Added to registry: {registry_key}")
        except Exception as e:
            print(f"⚠️  Warning: Could not add to registry: {e}")

        return entry

    def _download_files_with_aria2c(
        self, files: List[Any], base_url: str, target_dir: Path
    ) -> None:
        """Download files using aria2c."""

        if not files:
            return

        # Store target directory for progress monitoring
        self._current_target_dir = target_dir

        # Create temporary input file for aria2c
        with tempfile.NamedTemporaryFile(mode="w", suffix=".aria2", delete=False) as f:
            aria_input_file = f.name

            for file_info in files:
                file_url = f"{base_url}/{file_info.path}"
                local_path = target_dir / file_info.path

                # Ensure local directory exists
                local_path.parent.mkdir(parents=True, exist_ok=True)

                # Write aria2c input line with relative path (since we use cwd)
                f.write(f"{file_url}\n")
                f.write(f"  out={file_info.path}\n")

        try:
            # Build aria2c command
            log_file = target_dir / ".aria2c.log"
            aria_cmd = [
                "aria2c",
                "--input-file",
                aria_input_file,
                "--max-connection-per-server",
                str(self.config.max_connections_per_server),
                "--max-concurrent-downloads",
                str(self.config.max_concurrent_downloads),
                "--continue",
                "true" if self.config.enable_resume else "false",
                "--auto-file-renaming",
                "false",
                "--allow-overwrite",
                "true",
                "--summary-interval",
                "0",
                "--truncate-console-readout",
                "true",
                "--human-readable",
                "true",
                "--console-log-level",
                "warn",
                # Clean, minimal progress display - warn level hides NOTICE messages
            ]

            # Execute aria2c with progress monitoring
            print(f"🚀 Starting parallel download of {len(files)} files...")

            # Store progress callback for CLI integration
            self._current_log_file = log_file
            self._current_file_count = len(files)

            # Execute aria2c from target directory to avoid path issues
            result = subprocess.run(
                aria_cmd, capture_output=False, check=True, cwd=str(target_dir)
            )

            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, aria_cmd)

            # Clean up log file after successful download
            if log_file.exists():
                log_file.unlink()

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"aria2c download failed: {e}")
        finally:
            # Clean up aria2c input file
            try:
                os.unlink(aria_input_file)
            except OSError:
                pass

            # Reset progress tracking
            self._current_log_file = None
            self._current_file_count = 0

    def verify_model(self, model_name: str) -> bool:
        """Verify model integrity."""
        entries = self.registry.find_model(model_name)
        if not entries:
            return False
        entry = entries[0]  # Use first match

        # Check if path exists
        if not entry.path.exists():
            return False

        # Basic file existence check
        # Could be extended with checksum verification
        return True

    def remove_model(self, model_name: str, delete_files: bool = False) -> bool:
        """Remove model from registry and optionally delete files."""
        entries = self.registry.find_model(model_name)
        if not entries:
            return False
        entry = entries[0]  # Use first match

        # Remove from registry
        success = self.registry.remove(model_name)

        # Delete files if requested
        if delete_files and entry.path.exists():
            try:
                shutil.rmtree(entry.path)
                print(f"🗑️  Deleted files: {entry.path}")
            except Exception as e:
                print(f"⚠️  Could not delete files: {e}")
                return False

        return success

    def list_models(self) -> List[ModelEntry]:
        """List all downloaded models."""
        return self.registry.get_all_models()

    def get_stats(self) -> Dict[str, Any]:
        """Get download statistics."""
        models = self.registry.get_all_models()

        total_size = sum(m.size_bytes for m in models)
        format_counts = {}
        location_counts = {}

        for model in models:
            format_counts[model.format_type] = (
                format_counts.get(model.format_type, 0) + 1
            )
            location_counts[model.location] = location_counts.get(model.location, 0) + 1

        return {
            "total_models": len(models),
            "total_size_bytes": total_size,
            "by_format": format_counts,
            "by_location": location_counts,
        }
