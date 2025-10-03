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
from imageworks.model_loader.registry import load_registry, update_entries, remove_entry
from imageworks.model_loader.download_adapter import record_download
from imageworks.model_loader.models import RegistryEntry
from .url_analyzer import URLAnalyzer
from .formats import FormatDetector
from .format_utils import detect_format_and_quant


@dataclass
class RepositoryMetadata:
    """Normalized repository metadata for download routing."""

    owner: str
    repo_name: str
    branch: str
    repository_id: str
    storage_repo_name: str
    registry_model_name: str


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

    def __init__(self, config: Optional[DownloaderConfig] = None) -> None:
        self.config = config or get_config()
        # unified registry (JSON) only
        self._registry_cache = load_registry()
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
    ) -> RegistryEntry:
        """Download a model from a URL or HuggingFace identifier."""

        analysis = self.url_analyzer.analyze_url(model_identifier)
        repo_meta = self._build_repository_metadata(analysis)
        preferred_formats = self._normalise_format_preferences(format_preference)

        existing_entry = self._handle_existing_download(
            repo_meta.registry_model_name,
            force_redownload=force_redownload,
            interactive=interactive,
        )
        if existing_entry is not None:
            return existing_entry

        primary_format = self._detect_primary_format(analysis, preferred_formats)
        target_dir = self._resolve_target_dir(
            primary_format,
            location_override,
            repo_meta.owner,
            repo_meta.storage_repo_name,
        )

        required_files, optional_files = self._partition_files(analysis)
        all_files = self._download_selected_files(
            required_files,
            optional_files,
            include_optional,
            repo_meta,
            target_dir,
        )

        self._ensure_verified(all_files, target_dir)
        chat_template_info = self._inspect_chat_templates(target_dir)
        total_size = self._calculate_total_size(all_files, target_dir)

        entry = self._register_download(
            repo_meta,
            primary_format,
            total_size,
            all_files,
            target_dir,
            analysis,
            chat_template_info,
        )

        return entry

    def _build_repository_metadata(self, analysis: Any) -> RepositoryMetadata:
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

        return RepositoryMetadata(
            owner=owner,
            repo_name=repo_name,
            branch=branch,
            repository_id=repository_id,
            storage_repo_name=storage_repo_name,
            registry_model_name=registry_model_name,
        )

    def _normalise_format_preferences(
        self, format_preference: Optional[Union[Sequence[str], str]]
    ) -> Optional[List[str]]:
        if isinstance(format_preference, str):
            return [format_preference]
        if format_preference is None:
            return None
        return [str(fmt) for fmt in format_preference]

    def _handle_existing_download(
        self,
        registry_model_name: str,
        *,
        force_redownload: bool,
        interactive: bool,
    ) -> Optional[RegistryEntry]:
        if force_redownload:
            return None

        existing = self._registry_cache.get(registry_model_name)
        if not existing or not existing.download_path:
            return None

        existing_path = Path(existing.download_path).expanduser()
        if existing_path.exists():
            self._log(f"✅ Model already downloaded: {existing.name}")
            return existing

        self._log(
            f"⚠️  Model '{existing.name}' found in registry but directory missing: {existing.download_path}",
            level=logging.WARNING,
        )

        if not interactive:
            self._log("🔄 Non-interactive mode: proceeding with re-download...")
            return None

        try:
            response = input("Directory not found. Re-download the model? (y/N): ")
        except (KeyboardInterrupt, EOFError):
            self._log("\n❌ Download cancelled by user", level=logging.WARNING)
            return existing

        if response.strip().lower() not in {"y", "yes"}:
            self._log("❌ Download cancelled by user", level=logging.WARNING)
            return existing

        self._log("🔄 Proceeding with re-download...")
        return None

    def _detect_primary_format(
        self, analysis: Any, preferred_formats: Optional[List[str]]
    ) -> str:
        files: List[Any] = []
        for file_list in analysis.files.values():
            files.extend(file_list)

        formats = self.format_detector.detect_from_filelist([f.path for f in files])
        if not formats:
            self._log(
                "⚠️  Could not detect model format, using default routing",
                level=logging.WARNING,
            )
            return "unknown"

        if preferred_formats:
            for pref in preferred_formats:
                for fmt in formats:
                    if fmt.format_type == pref:
                        self._log(f"🔧 Detected format: {pref}")
                        return pref

        primary_format = formats[0].format_type
        self._log(f"🔧 Detected format: {primary_format}")
        return primary_format

    def _resolve_target_dir(
        self,
        primary_format: str,
        location_override: Optional[str],
        owner: str,
        storage_repo_name: str,
    ) -> Path:
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
        return target_dir

    def _partition_files(self, analysis: Any) -> tuple[List[Any], List[Any]]:
        required_files: List[Any] = []
        optional_files: List[Any] = []
        for category, file_list in analysis.files.items():
            if category in ["model_weights", "config", "tokenizer"]:
                required_files.extend(file_list)
            elif category in ["optional", "large_optional"]:
                optional_files.extend(file_list)

        if not required_files:
            raise RuntimeError("No essential model files found")

        return required_files, optional_files

    def _download_selected_files(
        self,
        required_files: List[Any],
        optional_files: List[Any],
        include_optional: bool,
        repo_meta: RepositoryMetadata,
        target_dir: Path,
    ) -> List[Any]:
        base_url = (
            f"https://huggingface.co/{repo_meta.repository_id}/resolve/{repo_meta.branch}"
        )

        self._log(f"📥 Downloading {len(required_files)} required files...")
        if required_files:
            self._download_files_with_aria2c(required_files, base_url, target_dir)

        if include_optional and optional_files:
            self._log(f"📥 Downloading {len(optional_files)} optional files...")
            self._download_files_with_aria2c(optional_files, base_url, target_dir)

        return required_files + (optional_files if include_optional else [])

    def _ensure_verified(self, files: List[Any], target_dir: Path) -> None:
        if not self._verify_download_complete(files, target_dir):
            raise RuntimeError(
                "Download verification failed - some files are missing, incomplete, or corrupted. "
                "Please try the download again."
            )
        self._log("✅ Download verification passed - all files complete")

    def _inspect_chat_templates(self, target_dir: Path) -> Dict[str, Any]:
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

        template_files: List[str] = []
        candidate_patterns = [
            ".jinja",
            "chat_template.json",
            "chat_template",
            "template",
        ]

        try:
            for path in target_dir.iterdir():
                if not path.is_file():
                    continue
                name_lower = path.name.lower()
                if any(pattern in name_lower for pattern in candidate_patterns):
                    if path.stat().st_size < 2_000_000:
                        try:
                            head = path.read_text(encoding="utf-8", errors="ignore")[:400]
                            if "{{" in head and "}}" in head:
                                template_files.append(path.name)
                        except Exception:  # noqa: BLE001
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

        return {
            "has_chat_template": has_chat_template,
            "has_embedded_chat_template": has_embedded_template,
            "embedded_chat_template_preview": embedded_template_snippet,
            "external_chat_template_files": template_files,
        }

    def _calculate_total_size(self, files: List[Any], target_dir: Path) -> int:
        return sum((target_dir / f.path).stat().st_size for f in files)

    def _determine_location_label(self, target_dir: Path) -> str:
        if (
            self.config.linux_wsl.root in target_dir.parents
            or self.config.linux_wsl.root == target_dir
        ):
            return "linux_wsl"
        if (
            self.config.windows_lmstudio.root in target_dir.parents
            or self.config.windows_lmstudio.root == target_dir
        ):
            return "windows_lmstudio"
        return "custom"

    def _register_download(
        self,
        repo_meta: RepositoryMetadata,
        primary_format: str,
        total_size: int,
        files: List[Any],
        target_dir: Path,
        analysis: Any,
        chat_template_info: Dict[str, Any],
    ) -> RegistryEntry:
        location_label = self._determine_location_label(target_dir)
        downloaded_files = [f.path for f in files]

        det_fmt, det_quant = detect_format_and_quant(target_dir)
        final_format = det_fmt or primary_format

        try:
            entry = record_download(
                hf_id=repo_meta.repository_id,
                backend="unassigned",
                format_type=final_format,
                quantization=det_quant,
                path=str(target_dir),
                location=location_label,
                files=downloaded_files,
                size_bytes=total_size,
                source_provider="hf",
                roles=[],
                role_priority=None,
            )

            entry.metadata.update(
                {
                    "model_type": analysis.repository.model_type,
                    "library": analysis.repository.library_name,
                    "files_downloaded": len(files),
                    "verified_complete": True,
                    "has_chat_template": chat_template_info["has_chat_template"],
                    "has_embedded_chat_template": chat_template_info[
                        "has_embedded_chat_template"
                    ],
                    "external_chat_template_files": chat_template_info[
                        "external_chat_template_files"
                    ],
                    "embedded_chat_template_preview": chat_template_info[
                        "embedded_chat_template_preview"
                    ],
                    "branch": repo_meta.branch,
                }
            )

            update_entries([entry], save=True)
            self._registry_cache = load_registry(force=True)  # type: ignore[arg-type]
            self._log(f"✅ Registry updated for {entry.name}")
            return entry
        except Exception as exc:  # noqa: BLE001
            self._log(
                f"⚠️  Warning: Could not update unified registry via adapter: {exc}",
                level=logging.WARNING,
            )
            raise

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
        entry = self._registry_cache.get(model_name)
        if not entry or not entry.download_path:
            return False
        path = Path(entry.download_path).expanduser()
        if not path.exists():
            return False
        # optional: verify directory checksum if present
        if entry.download_directory_checksum:
            if _calculate_directory_checksum(path) != entry.download_directory_checksum:
                return False
        return True

    def remove_model(
        self,
        model_name: str,
        format_type: Optional[str] = None,
        location: Optional[str] = None,
        delete_files: bool = False,
    ) -> bool:
        """Remove model from registry and optionally delete files."""
        entry = self._registry_cache.get(model_name)
        if not entry:
            return False
        path = Path(entry.download_path) if entry.download_path else None
        if delete_files and path and path.exists():
            try:
                shutil.rmtree(path)
                print(f"🗑️  Deleted files: {path}")
            except Exception as e:  # noqa: BLE001
                print(f"⚠️  Could not delete files: {e}")
                return False
        # If entry is only a download placeholder (unassigned backend & no roles) remove entirely
        if entry.backend == "unassigned" and not entry.roles:
            removed = remove_entry(model_name, save=True)
            if removed:
                self._registry_cache = load_registry(force=True)  # type: ignore[arg-type]
            return removed
        # Otherwise just clear download fields
        entry.download_format = None
        entry.download_location = None
        entry.download_path = None
        entry.download_size_bytes = None
        entry.download_files = []
        entry.download_directory_checksum = None
        update_entries([entry], save=True)
        self._registry_cache = load_registry(force=True)  # type: ignore[arg-type]
        return True

    def list_models(
        self,
        format_filter: Optional[str] = None,
        location_filter: Optional[str] = None,
    ) -> List[RegistryEntry]:
        """List downloaded models with optional filtering."""
        models = [
            e for e in self._registry_cache.values() if e.download_path is not None
        ]
        if format_filter:
            models = [m for m in models if m.download_format == format_filter]
        if location_filter:
            models = [m for m in models if m.download_location == location_filter]
        return models

    def get_stats(self) -> Dict[str, Any]:
        """Get download statistics."""
        models = [
            e for e in self._registry_cache.values() if e.download_path is not None
        ]
        total_size = sum((m.download_size_bytes or 0) for m in models)
        format_counts = {}
        location_counts = {}

        for model in models:
            fmt = model.download_format or "unknown"
            loc = model.download_location or "unknown"
            format_counts[fmt] = format_counts.get(fmt, 0) + 1
            location_counts[loc] = location_counts.get(loc, 0) + 1

        return {
            "total_models": len(models),
            "total_size_bytes": total_size,
            "by_format": format_counts,
            "by_location": location_counts,
        }


def _calculate_directory_checksum(directory: Path) -> str:
    """Lightweight directory checksum (names + sizes) first 16 chars of sha256."""
    import hashlib

    if not directory.exists():
        return ""
    hasher = hashlib.sha256()
    files = sorted(directory.rglob("*"))
    for file_path in files:
        if file_path.is_file():
            rel_path = file_path.relative_to(directory)
            hasher.update(str(rel_path).encode())
            hasher.update(str(file_path.stat().st_size).encode())
    return hasher.hexdigest()[:16]
