"""
Main model downloader implementation.

Coordinates URL analysis, format detection, directory routing, and download execution
using aria2c for optimal performance.
"""

import logging
import os
import subprocess
import tempfile
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Sequence, Union
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


logger = logging.getLogger(__name__)


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

    def _log(self, message: str, *, level: int = logging.INFO) -> None:
        """Log a message and mirror it to stdout for CLI parity."""

        logger.log(level, message)
        print(message)

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
                self._log(f"⚠️  Missing file: {file_info.path}", level=logging.WARNING)
                return False

            # Check for aria2 control file indicating incomplete download
            aria2_control = file_path.parent / f"{file_path.name}.aria2"
            if aria2_control.exists():
                self._log(
                    f"⚠️  Incomplete download detected: {file_info.path} (aria2 control file exists)",
                    level=logging.WARNING,
                )
                return False

            # Check file has non-zero size
            if file_path.stat().st_size == 0:
                self._log(f"⚠️  Zero-size file: {file_info.path}", level=logging.WARNING)
                return False

        return True

    def download(
        self,
        model_identifier: str,
        format_preference: Optional[Union[Sequence[str], str]] = None,
        location_override: Optional[str] = None,
        include_optional: bool = False,
        force_redownload: bool = False,
        interactive: bool = True,
    ) -> ModelEntry:
        """Download a model from a URL or HuggingFace identifier."""

        # Step 1: Analyze the URL/identifier
        analysis = self.url_analyzer.analyze_url(model_identifier)

        # Normalise repository metadata for downstream routing
        owner = analysis.repository.owner
        repo_name = analysis.repository.repo
        repository_id = f"{owner}/{repo_name}"

        branch = analysis.repository.branch or "main"
        storage_repo_name = (
            repo_name if branch == "main" else f"{repo_name}@{branch.replace('/', '_')}"
        )
        registry_model_name = (
            repository_id if branch == "main" else f"{repository_id}@{branch}"
        )

        # Normalise format preferences – accept single strings from callers
        preferred_formats: Optional[List[str]]
        if isinstance(format_preference, str):
            preferred_formats = [format_preference]
        elif format_preference is None:
            preferred_formats = None
        else:
            preferred_formats = [str(fmt) for fmt in format_preference]

        # Step 2: Check if already downloaded
        if not force_redownload:
            existing_models = self.registry.find_model(registry_model_name)
            if existing_models:
                existing = existing_models[0]  # Use first match

                # Check if the model directory actually exists
                if existing.path_obj.exists():
                    self._log(f"✅ Model already downloaded: {existing.model_name}")
                    return existing
                else:
                    self._log(
                        f"⚠️  Model '{existing.model_name}' found in registry but directory missing: {existing.path}",
                        level=logging.WARNING,
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
                                self._log(
                                    "❌ Download cancelled by user",
                                    level=logging.WARNING,
                                )
                                return existing  # Return existing entry without downloading
                            self._log("🔄 Proceeding with re-download...")
                        except (KeyboardInterrupt, EOFError):
                            self._log(
                                "\n❌ Download cancelled by user", level=logging.WARNING
                            )
                            return existing
                        # Remove stale registry entry to avoid confusion
                        self.registry.remove_model(
                            existing.model_name, existing.format_type, existing.location
                        )
                    else:
                        self._log(
                            "🔄 Non-interactive mode: proceeding with re-download..."
                        )
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
            self._log(
                "⚠️  Could not detect model format, using default routing",
                level=logging.WARNING,
            )
            primary_format = "unknown"
        else:
            # Use format preference or first detected
            if preferred_formats:
                for pref in preferred_formats:
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

        self._log(f"🔧 Detected format: {primary_format}")

        # Step 4: Determine target directory
        if location_override:
            normalized_override = location_override.strip()
            if normalized_override in {"linux_wsl", "windows_lmstudio"}:
                if normalized_override == "linux_wsl":
                    base_dir = self.config.linux_wsl.root / "weights"
                else:
                    base_dir = self.config.windows_lmstudio.root

                target_dir = base_dir / owner / storage_repo_name
            else:
                target_dir = (
                    Path(normalized_override).expanduser() / owner / storage_repo_name
                )
        else:
            target_dir = self.config.get_target_directory(
                primary_format,
                storage_repo_name,
                publisher=owner,
            )

        target_dir.mkdir(parents=True, exist_ok=True)
        self._log(f"📁 Target directory: {target_dir}")

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
        base_url = f"https://huggingface.co/{repository_id}/resolve/{branch}"

        # Prepare aria2c download
        self._log(f"📥 Downloading {len(required_files)} required files...")
        if required_files:
            self._download_files_with_aria2c(required_files, base_url, target_dir)

        if include_optional and optional_files:
            self._log(f"📥 Downloading {len(optional_files)} optional files...")
            self._download_files_with_aria2c(optional_files, base_url, target_dir)

        # Step 7: Verify download completion before registry registration
        all_files = required_files + (optional_files if include_optional else [])
        if not self._verify_download_complete(all_files, target_dir):
            raise RuntimeError(
                "Download verification failed - some files are missing, incomplete, or corrupted. "
                "Please try the download again."
            )

        self._log("✅ Download verification passed - all files complete")

        # Chat template detection & root file harvesting
        # 1. Embedded template inside tokenizer_config.json
        tokenizer_cfg_path = target_dir / "tokenizer_config.json"
        has_embedded_template = False
        embedded_template_snippet = None
        if tokenizer_cfg_path.exists():
            try:
                import json

                data = json.loads(tokenizer_cfg_path.read_text(encoding="utf-8"))
                tmpl = data.get("chat_template")
                if tmpl and isinstance(tmpl, str) and "{{" in tmpl:
                    has_embedded_template = True
                    embedded_template_snippet = tmpl[:160] + (
                        "..." if len(tmpl) > 160 else ""
                    )
            except Exception:  # noqa: BLE001
                pass
        else:
            self._log(
                "ℹ️  tokenizer_config.json not present; cannot inspect chat template",
                level=logging.INFO,
            )

        # 2. External template files placed at repo root (common patterns: *.jinja, chat_template*.json, *_chat_template*, *template*vicuna*)
        template_files: List[str] = []
        candidate_patterns = [
            ".jinja",
            "chat_template.json",
            "chat_template",  # generic prefix
            "template",  # broad safety net
        ]
        try:
            for p in target_dir.iterdir():
                if not p.is_file():
                    continue
                name_lower = p.name.lower()
                if any(pattern in name_lower for pattern in candidate_patterns):
                    # Ignore extremely large non-text files (heuristic)
                    if (
                        p.stat().st_size < 2_000_000
                    ):  # 2MB safety limit for template-like files
                        # simple text sniff
                        try:
                            head = p.read_text(encoding="utf-8", errors="ignore")[:400]
                            if "{{" in head and "}}" in head:
                                template_files.append(p.name)
                        except Exception:
                            pass
        except Exception:  # noqa: BLE001
            pass

        has_external_template = len(template_files) > 0
        has_chat_template = has_embedded_template or has_external_template

        if has_chat_template:
            detail = "embedded" if has_embedded_template else "external-file"
            self._log(
                f"💬 Chat template detected ({detail}). External files: {template_files if template_files else 'n/a'}"
            )
        else:
            self._log(
                "💬 No chat template detected (embedded or external). If you encounter 400 errors ('default chat template no longer allowed'), use --chat-template to supply one.",
                level=logging.INFO,
            )

        # Step 8: Calculate total size (now guaranteed to be accurate)
        total_size = sum((target_dir / f.path).stat().st_size for f in all_files)

        # Step 9: Register in registry (now guaranteed to be accurate)
        # Get list of downloaded files
        downloaded_files = []
        for f in all_files:
            downloaded_files.append(f.path)

        if (
            self.config.linux_wsl.root in target_dir.parents
            or self.config.linux_wsl.root == target_dir
        ):
            location_label = "linux_wsl"
        elif (
            self.config.windows_lmstudio.root in target_dir.parents
            or self.config.windows_lmstudio.root == target_dir
        ):
            location_label = "windows_lmstudio"
        else:
            location_label = "custom"

        entry = ModelEntry(
            model_name=registry_model_name,
            format_type=primary_format,
            path=str(target_dir),
            size_bytes=total_size,
            location=location_label,
            files=downloaded_files,
            downloaded_at="",  # Will be set by registry
            metadata={
                "model_type": analysis.repository.model_type,
                "library": analysis.repository.library_name,
                "huggingface_id": repository_id,
                "branch": branch,
                "files_downloaded": len(all_files),
                "verified_complete": True,  # Mark as verified
                "has_chat_template": has_chat_template,
                "has_embedded_chat_template": has_embedded_template,
                "external_chat_template_files": template_files,
                "embedded_chat_template_preview": embedded_template_snippet,
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
            self._log(f"✅ Added to registry: {registry_key}")
        except Exception as e:
            self._log(
                f"⚠️  Warning: Could not add to registry: {e}", level=logging.WARNING
            )

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
                f"--continue={'true' if self.config.enable_resume else 'false'}",
                "--auto-file-renaming=false",
                "--allow-overwrite=true",
                "--summary-interval",
                "0",
                "--truncate-console-readout=true",
                "--human-readable=true",
                "--console-log-level",
                "warn",
                # Clean, minimal progress display - warn level hides NOTICE messages
            ]

            # Execute aria2c with progress monitoring
            self._log(f"🚀 Starting parallel download of {len(files)} files...")

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

    def remove_model(
        self,
        model_name: str,
        format_type: Optional[str] = None,
        location: Optional[str] = None,
        delete_files: bool = False,
    ) -> bool:
        """Remove model from registry and optionally delete files."""
        entries = self.registry.find_model(model_name, format_type, location)
        if not entries:
            return False

        success = False
        for entry in entries:
            removed = self.registry.remove_model(
                entry.model_name, entry.format_type, entry.location
            )
            success = success or removed

            if delete_files and entry.path.exists():
                try:
                    shutil.rmtree(entry.path)

                    print(f"🗑️  Deleted files: {entry.path}")
                except Exception as e:
                    print(f"⚠️  Could not delete files: {e}")

                    return False

        return success

    def list_models(
        self,
        format_filter: Optional[str] = None,
        location_filter: Optional[str] = None,
    ) -> List[ModelEntry]:
        """List downloaded models with optional filtering."""
        models = self.registry.get_all_models()
        if format_filter:
            models = [m for m in models if m.format_type == format_filter]
        if location_filter:
            models = [m for m in models if m.location == location_filter]
        return models

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
